"""Day 1-2 baseline table: CG iterations on 2D Poisson for each preconditioner."""
import pathlib, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lpc.fsai import fsai, fsai_precond, frobenius_loss
from lpc.matrices import poisson2d
from lpc.seed import set_seed
from lpc.solvers import cg_count, ic0_precond, jacobi

rng = set_seed(0)
rows = []
for n in (16, 32, 64, 128):
    A = poisson2d(n)
    b = rng.normal(size=A.shape[0])
    entry = {"grid": f"{n}x{n}", "rows": A.shape[0]}
    t = time.perf_counter(); entry["none"] = cg_count(A, b).iters
    entry["jacobi"] = cg_count(A, b, M=jacobi(A)).iters
    if n <= 64:
        t0 = time.perf_counter(); M = ic0_precond(A); entry["ic0_setup_s"] = time.perf_counter() - t0
        entry["ic0"] = cg_count(A, b, M=M).iters
    t0 = time.perf_counter(); G = fsai(A); entry["fsai_setup_s"] = time.perf_counter() - t0
    entry["fsai"] = cg_count(A, b, M=fsai_precond(G)).iters
    entry["fsai_loss"] = frobenius_loss(G, A)
    rows.append(entry)
    print(entry, flush=True)
df = pd.DataFrame(rows)
out = pathlib.Path(__file__).parent / "results" / "day1_baselines.csv"
df.to_csv(out, index=False)
print(df.to_string(index=False))
