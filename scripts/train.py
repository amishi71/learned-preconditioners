"""Train the GNN preconditioner (run locally; needs torch + torch_geometric).

    python scripts/train.py --loss frobenius --pattern tril --steps 4000
    python scripts/train.py --loss kaporin   --pattern tril --steps 4000
    python scripts/train.py --loss kaporin   --pattern tril2 --steps 4000

One matrix per step (batch size 1) keeps sparse index bookkeeping simple. Prints validation CG iterations on
Poisson 32x32 against the classical-FSAI target every --val_every steps."""
import argparse, pathlib, sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lpc.fsai import fsai, fsai_precond
from lpc.graph import build_graph, pattern_matrix, to_pyg, unit_diag_scale
from lpc.loss import exact_loss, hutchinson_loss, kaporin_loss
from lpc.matrices import poisson2d, sample_training_matrix
from lpc.model import PrecondGNN
from lpc.precond import gnn_preconditioner
from lpc.seed import set_seed
from lpc.solvers import cg_count

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--pattern", default="tril", choices=["tril", "tril2"])
ap.add_argument("--loss", default="frobenius", choices=["frobenius", "kaporin"])
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--exact_max_n", type=int, default=1024)
ap.add_argument("--probes", type=int, default=16)
ap.add_argument("--val_every", type=int, default=250)
ap.add_argument("--no_residual", action="store_true", help="ablation: no G = I + dG parametrization (model.residual=False)")
ap.add_argument("--out", default=None)
args = ap.parse_args()
tag = f"{args.loss}_{args.pattern}" + ("_nores" if args.no_residual else "")
out = args.out or f"checkpoints/gnn_{tag}.pt"

rng = set_seed(0)
model = PrecondGNN(residual=not args.no_residual)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

Aval = poisson2d(32)
bval = np.random.default_rng(123).normal(size=Aval.shape[0])
val_graph = to_pyg(build_graph(Aval, args.pattern))
P = pattern_matrix(unit_diag_scale(Aval)[0], args.pattern) if args.pattern == "tril2" else None
target = cg_count(Aval, bval, M=fsai_precond(fsai(Aval, pattern=P))).iters
print(f"targets on Poisson 32x32: none={cg_count(Aval, bval).iters}  classical FSAI({args.pattern})={target}", flush=True)


def loss_fn(data, n):
    G = model(data)
    if args.loss == "kaporin":
        return kaporin_loss(G, data, n, None if n <= args.exact_max_n else args.probes)
    return exact_loss(G, data, n) if n <= args.exact_max_n else hutchinson_loss(G, data, n, args.probes)


best = 1e9
for step in range(args.steps + 1):
    g = build_graph(sample_training_matrix(rng), args.pattern)
    data = to_pyg(g)
    loss = loss_fn(data, g.n)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % args.val_every == 0:
        its = cg_count(Aval, bval, M=gnn_preconditioner(model, Aval, args.pattern)).iters
        model.train()
        print(f"step {step:5d}  train {args.loss} {loss.item():+.4f}   val CG iters (32^2) {its}  [FSAI {target}]", flush=True)
        if its < best:
            best = its
            pathlib.Path(out).parent.mkdir(exist_ok=True)
            torch.save({"state": model.state_dict(), "pattern": args.pattern, "loss": args.loss, "residual": not args.no_residual, "val_iters": its}, out)
print(f"best val iters {best}; checkpoint: {out}")
