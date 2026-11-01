"""Train the GNN preconditioner. Run locally (needs torch + torch_geometric):
    python scripts/train.py --steps 4000 --pattern tril
Batch size 1 (one matrix per step) keeps sparse index bookkeeping simple."""
import argparse, pathlib, sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lpc.graph import build_graph, to_pyg
from lpc.loss import exact_loss, hutchinson_loss
from lpc.matrices import poisson2d, sample_training_matrix
from lpc.model import PrecondGNN
from lpc.seed import set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--pattern", default="tril", choices=["tril", "tril2"])
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--exact_max_n", type=int, default=1024)
ap.add_argument("--out", default="checkpoints/gnn.pt")
args = ap.parse_args()

rng = set_seed(0)
model = PrecondGNN()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
val = to_pyg(build_graph(poisson2d(32), args.pattern))


def loss_fn(data, n):
    G = model(data)
    return exact_loss(G, data, n) if n <= args.exact_max_n else hutchinson_loss(G, data, n)


for step in range(args.steps):
    g = build_graph(sample_training_matrix(rng), args.pattern)
    data = to_pyg(g)
    loss = loss_fn(data, g.n)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 100 == 0:
        with torch.no_grad():
            v = exact_loss(model(val), val, val.num_nodes).item()
        print(f"step {step:5d} train {loss.item():.4f} val(32^2 exact) {v:.4f}", flush=True)
pathlib.Path(args.out).parent.mkdir(exist_ok=True)
torch.save(model.state_dict(), args.out)
