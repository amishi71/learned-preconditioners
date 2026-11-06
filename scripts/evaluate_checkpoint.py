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
from lpc.matrices import diffusion2d, fem_delaunay, q1hex, poisson2d, poisson3d, rot_aniso9, sample_training_matrix
from lpc.model import PrecondGNN
from lpc.precond import gnn_preconditioner, operator_from_factor
from lpc.seed import set_seed
from lpc.solvers import cg_history, ic0_precond, jacobi

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", action="append", default=[])
ap.add_argument("--mtx", default=None, help="optional SuiteSparse .mtx (SPD) for the out-of-distribution test")
ap.add_argument("--amg", action="store_true", help="add pyamg smoothed-aggregation AMG (V-cycle) as a baseline")
args = ap.parse_args()

rng = set_seed(1)
mats = {f"poisson{n}": poisson2d(n) for n in (16, 32, 64, 128)}
mats["aniso64_0.01"] = diffusion2d(64, ax=1.0, ay=0.01)
for i in range(3):
    mats[f"aug{i}"] = sample_training_matrix(rng, sizes=(32, 48, 64))
# out-of-distribution matrices that need no download: 3D stencil and an unstructured FEM mesh (not an M-matrix)
mats["ood_poisson3d_16"] = poisson3d(16)
mats["ood_fem_delaunay_4000"] = fem_delaunay(4000, seed=0)
mats["heldout_q1hex_16"] = q1hex(16)          # 21 nnz/row, easy; held out even for the wide-trained models
mats["heldout_rotaniso9_64"] = rot_aniso9(64)   # 9-point, positive off-diagonals (not an M-matrix), hard
if args.mtx:
    mats["suitesparse"] = sp.csr_matrix(scipy.io.mmread(args.mtx))


def with_G(G):
    return operator_from_factor(G), G


def safe_ic0(A):
    if A.shape[0] > 4096:
        return None, None
    try:
        return ic0_precond(A), None
    except np.linalg.LinAlgError:
        return None, None      # IC(0) breakdown (non-positive pivot): reported as n/a


def wide_ok(A):
    """tril2 (A^2 pattern) explodes on dense-row matrices (e.g. SuiteSparse ~48 nnz/row): only run it on sparse stencils."""
    return A.shape[0] <= 20000 and A.nnz / A.shape[0] <= 8


methods = {
    "none": lambda A: (None, None),
    "jacobi": lambda A: (jacobi(A), None),
    "ic0": safe_ic0,
    "fsai_tril": lambda A: with_G(fsai(A)),
    "fsai_tril2": lambda A: with_G(fsai(A, pattern=pattern_matrix(unit_diag_scale(A)[0], "tril2"))) if wide_ok(A) else (None, None),
}
if args.amg:
    import pyamg

    def amg(A):
        ml = pyamg.smoothed_aggregation_solver(sp.csr_matrix(A))
        return ml.aspreconditioner(cycle="V"), None

    methods["pyamg_amg"] = amg
models = {}
for c in args.ckpt:
    ck = torch.load(c, map_location="cpu")
    m = PrecondGNN(residual=ck.get("residual", True), robust=ck.get("robust", False)); m.load_state_dict(ck["state"]); m.eval()
    name = pathlib.Path(c).stem
    models[name] = (m, ck["pattern"])
    methods[name] = (lambda A, m=m, p=ck["pattern"]: (gnn_preconditioner(m, A, p), None) if (p == "tril" or wide_ok(A)) else (None, None))

df = evaluate(mats, methods)
# a method that returned no preconditioner (skipped / IC(0) breakdown / too large) is "n/a", not "none"
df.loc[(df.method != "none") & df.apply_s.isna(), "iters"] = np.nan
out = ROOT / "benchmarks" / "results" / "eval_all.csv"; df.to_csv(out, index=False)
print(df.pivot_table(index="matrix", columns="method", values="iters").round(0).to_string())

# residual curves on poisson64 for the notebook
A = mats["poisson64"]; b = np.random.default_rng(0).normal(size=A.shape[0]); curves = {}
for name, build in methods.items():
    M, _ = build(A)
    curves[name], _ = cg_history(A, b, M=M)
np.savez(ROOT / "benchmarks" / "results" / "eval_curves.npz", **curves)
print("wrote", out)
