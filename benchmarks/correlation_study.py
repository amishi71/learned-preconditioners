"""Does the Frobenius loss predict CG iterations? Stand-in study with FSAI-family preconditioners
(the GNN plugs into the same harness later). For each matrix we build preconditioners of varying quality:
Jacobi, blends between Jacobi and FSAI (t = .25/.5/.75), FSAI(tril), FSAI(tril2)."""
import pathlib, sys

import numpy as np, pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lpc.evaluate import evaluate
from lpc.fsai import fsai, frobenius_loss
from lpc.graph import pattern_matrix, unit_diag_scale
from lpc.matrices import sample_training_matrix
from lpc.precond import operator_from_factor
from lpc.seed import set_seed

rng = set_seed(0)
mats = {f"m{i:02d}": sample_training_matrix(rng, sizes=(8, 12, 16, 24, 32)) for i in range(32)}


def with_G(G):
    return operator_from_factor(G), G


def jac(A):
    return with_G(sp.diags(1.0 / np.sqrt(A.diagonal())).tocsr())


def blend(t):
    def f(A):
        J = sp.diags(1.0 / np.sqrt(A.diagonal())).tocsr()
        return with_G(sp.csr_matrix((1 - t) * J + t * fsai(A)))
    return f


methods = {
    "none": lambda A: (None, None),
    "jacobi": jac,
    "blend25": blend(0.25), "blend50": blend(0.5), "blend75": blend(0.75),
    "fsai_tril": lambda A: with_G(fsai(A)),
    "fsai_tril2": lambda A: with_G(fsai(A, pattern=pattern_matrix(unit_diag_scale(A)[0], "tril2"))),
}
df = evaluate(mats, methods)
df.to_csv(pathlib.Path(__file__).parent / "results" / "correlation_study.csv", index=False)

none = df[df.method == "none"].set_index("matrix").iters
df["rel_iters"] = df.apply(lambda r: r.iters / none[r.matrix], axis=1)
d = df[(df.method != "none")].dropna(subset=["loss", "kaporin"])
d = d[d.converged]
print(f"matrices: {df.matrix.nunique()}, preconditioners/matrix: {len(methods)-1}, converged rows: {len(d)}/{(df.method!='none').sum()}")
summary = {}
for col, tf in (("loss", np.log), ("kaporin", np.log)):
    x, y = tf(d[col]), np.log(d.rel_iters)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    within = [grp[[col, "iters"]].corr(method="spearman").iloc[0, 1] for _, grp in d.groupby("matrix") if grp[col].nunique() > 2]
    summary[col] = (r2, float(np.mean(within)), float(np.min(within)))
    print(f"{col:8s} pooled R^2 (log-log vs rel. iters) = {r2:.3f} | within-matrix Spearman mean {np.mean(within):.3f}, min {np.min(within):.3f}")
print(d.groupby("method")[["loss", "kaporin", "rel_iters"]].mean().round(3))
pd.DataFrame(summary, index=["pooled_R2", "within_spearman_mean", "within_spearman_min"]).T.to_csv(
    pathlib.Path(__file__).parent / "results" / "correlation_summary.csv")
