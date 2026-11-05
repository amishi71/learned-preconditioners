"""Evaluate trained checkpoints against baselines. Needs torch + PyG.
    python scripts/evaluate_checkpoint.py --ckpt checkpoints/gnn_kaporin_tril.pt [--ckpt ...] [--mtx data/Pres_Poisson.mtx]
Writes benchmarks/results/eval_all.csv (iterations, setup/apply time, losses) and eval_curves.npz (residual histories)."""
import argparse, pathlib, sys

import numpy as np, scipy.io, scipy.sparse as sp, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lpc.evaluate import evaluate
from lpc.fsai import fsai
from lpc.graph import pattern_matrix, unit_diag_scale
from lpc.matrices import diffusion2d, poisson2d, sample_training_matrix
from lpc.model import PrecondGNN
from lpc.precond import gnn_preconditioner, operator_from_factor
from lpc.seed import set_seed
from lpc.solvers import cg_history, ic0_precond, jacobi

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", action="append", default=[])
ap.add_argument("--mtx", default=None, help="optional SuiteSparse .mtx (SPD) for the out-of-distribution test")
args = ap.parse_args()

rng = set_seed(1)
mats = {f"poisson{n}": poisson2d(n) for n in (16, 32, 64, 128)}
mats["aniso64_0.01"] = diffusion2d(64, ax=1.0, ay=0.01)
for i in range(3):
    mats[f"aug{i}"] = sample_training_matrix(rng, sizes=(32, 48, 64))
if args.mtx:
    mats["suitesparse"] = sp.csr_matrix(scipy.io.mmread(args.mtx))


def with_G(G):
    return operator_from_factor(G), G


def wide_ok(A):
    """tril2 (A^2 pattern) explodes on dense-row matrices (e.g. SuiteSparse ~48 nnz/row): only run it on sparse stencils."""
    return A.shape[0] <= 20000 and A.nnz / A.shape[0] <= 8


methods = {
    "none": lambda A: (None, None),
    "jacobi": lambda A: (jacobi(A), None),
    "ic0": lambda A: (ic0_precond(A), None) if A.shape[0] <= 4096 else (None, None),
    "fsai_tril": lambda A: with_G(fsai(A)),
    "fsai_tril2": lambda A: with_G(fsai(A, pattern=pattern_matrix(unit_diag_scale(A)[0], "tril2"))) if wide_ok(A) else (None, None),
}
models = {}
for c in args.ckpt:
    ck = torch.load(c, map_location="cpu")
    m = PrecondGNN(residual=ck.get("residual", True)); m.load_state_dict(ck["state"]); m.eval()
    name = pathlib.Path(c).stem
    models[name] = (m, ck["pattern"])
    methods[name] = (lambda A, m=m, p=ck["pattern"]: (gnn_preconditioner(m, A, p), None) if (p == "tril" or wide_ok(A)) else (None, None))

df = evaluate(mats, methods)
# rows where a method was skipped (returned no preconditioner) are really "none": blank them out
skipped = ((df.method.str.startswith("gnn") & (df.setup_s < 1e-4)) | ((df.method == "ic0") & (df.n > 4096)) | ((df.method == "fsai_tril2") & df.apply(lambda r: r.n > 20000, axis=1)))
df.loc[skipped, "iters"] = np.nan
out = ROOT / "benchmarks" / "results" / "eval_all.csv"; df.to_csv(out, index=False)
print(df.pivot_table(index="matrix", columns="method", values="iters").round(0).to_string())

# residual curves on poisson64 for the notebook
A = mats["poisson64"]; b = np.random.default_rng(0).normal(size=A.shape[0]); curves = {}
for name, build in methods.items():
    M, _ = build(A)
    curves[name], _ = cg_history(A, b, M=M)
np.savez(ROOT / "benchmarks" / "results" / "eval_curves.npz", **curves)
print("wrote", out)
