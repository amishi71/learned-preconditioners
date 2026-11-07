"""Eigenvalue spectrum of the preconditioned operator G A G^T: GNN vs classical FSAI vs IC(0) vs none.
Needs torch + PyG.   python benchmarks/gnn_spectrum_figure.py --ckpt checkpoints/gnn_kaporin_tril_final.pt [--ckpt ...]
A tighter spectrum near 1 means fewer CG iterations; this shows WHY the GNN matches FSAI on Poisson and where it does not."""
import argparse, pathlib, sys

import numpy as np, scipy.sparse as sp, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from make_results_tables import pretty          # same display names as the tables
from lpc.fsai import fsai
from lpc.graph import build_graph, pattern_matrix, to_pyg, unit_diag_scale
from lpc.matrices import fem_delaunay, poisson2d, rot_aniso9
from lpc.model import PrecondGNN
from lpc.precond import factor_from_values
from lpc.solvers import ic0

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", action="append", required=True)
ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "spectrum_gnn.png"))
args = ap.parse_args()

models = []
for c in args.ckpt:
    ck = torch.load(c, map_location="cpu")
    m = PrecondGNN(residual=ck.get("residual", True), robust=ck.get("robust", False)); m.load_state_dict(ck["state"]); m.eval()
    models.append((pretty(pathlib.Path(c).stem), m, ck["pattern"]))

mats = {"Poisson 16x16": poisson2d(16), "Delaunay FEM (400 nodes)": fem_delaunay(400, seed=0), "Rotated aniso 9-pt 16x16 (held out)": rot_aniso9(16)}
fig, axes = plt.subplots(1, len(mats), figsize=(5.2 * len(mats), 4.2))
for ax, (title, A) in zip(np.atleast_1d(axes), mats.items()):
    Ad = A.toarray()
    spectra = {"A (none, raw)": np.linalg.eigvalsh(Ad)}
    try:
        L = ic0(A).toarray(); Li = np.linalg.inv(L)
        spectra["IC(0)"] = np.linalg.eigvalsh(Li @ Ad @ Li.T)
    except np.linalg.LinAlgError:
        pass
    G = fsai(A); spectra["FSAI tril"] = np.linalg.eigvalsh((G @ A @ G.T).toarray())
    for name, m, pat in models:
        g = build_graph(A, pat); d = to_pyg(g)
        with torch.no_grad():
            vals = m(d).numpy().astype(float)
        GA = factor_from_values(vals, g.row, g.col, g.scale, g.n)
        w = np.linalg.eigvalsh((GA @ A @ GA.T).toarray())
        spectra[name] = w / np.exp(np.log(w).mean())          # K is scale-invariant: normalise so geometric mean is 1
    for name, w in spectra.items():
        if name in ("IC(0)", "FSAI tril"):
            w = w / np.exp(np.log(w).mean())
        is_gnn = name.startswith("GNN")
        ax.semilogy(np.sort(w), label=f"{name} (cond {w.max() / w.min():.0f})", ls="--" if is_gnn else "-",
                    lw=1.6 if is_gnn else (2.8 if name == "FSAI tril" else 1.4), zorder=3 if is_gnn else 2)
    ax.set_title(title, fontsize=10); ax.set_xlabel("eigenvalue index"); ax.legend(fontsize=7)
np.atleast_1d(axes)[0].set_ylabel("eigenvalue\n(preconditioned: geo. mean = 1)")
plt.tight_layout(); plt.savefig(args.out, dpi=150); print("wrote", args.out)
