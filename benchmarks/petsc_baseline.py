"""Day 6 baselines: PETSc CG with ILU(0) / ICC(0) / BoomerAMG; falls back to pyamg if petsc4py (or hypre) is missing.
    python benchmarks/petsc_baseline.py [--mtx data/Pres_Poisson.mtx]
Reports ITERATION COUNTS (primary metric). Wall-clock across frameworks is not a fair comparison; see README.
Not executed in the authoring sandbox (no petsc4py/pyamg there) - first run may need small API fixes."""
import argparse, pathlib, sys

import numpy as np, scipy.io, scipy.sparse as sp, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lpc.matrices import poisson2d
from lpc.solvers import cg_count

ap = argparse.ArgumentParser(); ap.add_argument("--mtx", default=None); args = ap.parse_args()
mats = {f"poisson{n}": poisson2d(n) for n in (64,)}
if args.mtx:
    mats["suitesparse"] = sp.csr_matrix(scipy.io.mmread(args.mtx))

rows = []
try:
    from petsc4py import PETSc

    def petsc_cg(A, b, pc_type, rtol=1e-8):
        A = sp.csr_matrix(A)
        Ap = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
        ksp = PETSc.KSP().create(); ksp.setOperators(Ap); ksp.setType("cg")
        ksp.setTolerances(rtol=rtol, atol=0.0, max_it=20000)
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)   # compare like-for-like with the scipy residual
        pc = ksp.getPC(); pc.setType(pc_type)
        if pc_type == "hypre":
            pc.setHYPREType("boomeramg")
        ksp.setFromOptions()
        x, bb = Ap.createVecs(); bb.setArray(b)
        ksp.solve(bb, x)
        return ksp.getIterationNumber(), ksp.getConvergedReason() > 0

    for name, A in mats.items():
        b = np.random.default_rng(0).normal(size=A.shape[0])
        for label, pc in (("petsc_ilu0", "ilu"), ("petsc_icc0", "icc"), ("petsc_boomeramg", "hypre")):
            try:
                it, ok = petsc_cg(A, b, pc); rows.append(dict(matrix=name, method=label, iters=it, converged=ok))
            except Exception as e:
                rows.append(dict(matrix=name, method=label, iters=np.nan, converged=False, note=str(e)[:80]))
except ImportError:
    print("petsc4py not available - using pyamg fallback (note this substitution in the README)")
    import pyamg
    for name, A in mats.items():
        b = np.random.default_rng(0).normal(size=A.shape[0])
        ml = pyamg.smoothed_aggregation_solver(sp.csr_matrix(A))
        r = cg_count(A, b, M=ml.aspreconditioner(cycle="V"))
        rows.append(dict(matrix=name, method="pyamg_SA_amg", iters=r.iters, converged=r.converged))
df = pd.DataFrame(rows); print(df.to_string(index=False))
df.to_csv(ROOT / "benchmarks" / "results" / "petsc_baseline.csv", index=False)
