import pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lpc.fsai import fsai
from lpc.graph import pattern_matrix, unit_diag_scale
from lpc.matrices import poisson2d
from lpc.solvers import ic0

A = poisson2d(16); Ad = A.toarray()
J = sp.diags(1 / np.sqrt(A.diagonal()))
Gt = fsai(A)
G2 = fsai(A, pattern=pattern_matrix(unit_diag_scale(A)[0], "tril2"))
L = ic0(A).toarray(); Li = np.linalg.inv(L)
spectra = {
    "A (none)": np.linalg.eigvalsh(Ad),
    "FSAI tril": np.linalg.eigvalsh((Gt @ A @ Gt.T).toarray()),
    "FSAI tril$^2$": np.linalg.eigvalsh((G2 @ A @ G2.T).toarray()),
    "IC(0)": np.linalg.eigvalsh(Li @ Ad @ Li.T),
}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, w in spectra.items():
    ax[0].semilogy(np.sort(w), label=f"{name}  (cond {w.max()/w.min():.0f})")
    ax[1].hist(w, bins=40, alpha=0.5, label=name)
ax[0].set_xlabel("eigenvalue index"); ax[0].set_ylabel("eigenvalue"); ax[0].legend(fontsize=8)
ax[0].set_title("Spectrum of preconditioned operator, 2D Poisson 16x16")
ax[1].set_xlabel("eigenvalue"); ax[1].set_title("Clustering (tighter = fewer CG iterations)"); ax[1].set_xlim(0, 8)
plt.tight_layout()
out = pathlib.Path(__file__).parent / "results" / "spectrum_poisson16.png"
plt.savefig(out, dpi=150); print({k: round(v.max() / v.min(), 1) for k, v in spectra.items()})
