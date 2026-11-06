"""Why does the GNN fail on out-of-distribution matrices?  (needs torch + PyG)
    python scripts/diagnose_ood.py --ckpt checkpoints/gnn_kaporin_tril.pt --mtx data/Pres_Poisson.mtx
Prints (1) node-feature ranges: training distribution vs each OOD matrix (is the input outside what the model saw?)
       (2) the TRAINING objective (Kaporin, Hutchinson trace) for Jacobi / GNN / classical FSAI on each matrix,
           as K ratios: if GNN's K is far above FSAI's, the model's G is genuinely bad under its own loss
           (a generalization failure), not an evaluation bug."""
import argparse, pathlib, sys

import numpy as np, scipy.io, scipy.sparse as sp, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lpc.fsai import fsai
from lpc.graph import build_graph, to_pyg, unit_diag_scale
from lpc.loss import kaporin_loss
from lpc.matrices import fem_delaunay, poisson3d, sample_training_matrix
from lpc.model import PrecondGNN
from lpc.seed import set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/gnn_kaporin_tril.pt")
ap.add_argument("--mtx", default=None)
ap.add_argument("--probes", type=int, default=64)
args = ap.parse_args()

rng = set_seed(0)
feats = np.concatenate([build_graph(sample_training_matrix(rng)).node_x for _ in range(60)])
names = ["diag(A_hat)", "degree", "sum|offdiag|"]
print("node-feature ranges (min .. max)")
print("  training  :", "  ".join(f"{n}: {feats[:, i].min():.2f}..{feats[:, i].max():.2f}" for i, n in enumerate(names)))

ck = torch.load(args.ckpt, map_location="cpu")
model = PrecondGNN(residual=ck.get("residual", True), robust=ck.get("robust", False)); model.load_state_dict(ck["state"]); model.eval()

mats = {"poisson64": None, "ood_poisson3d_16": poisson3d(16), "ood_fem_delaunay_4000": fem_delaunay(4000, seed=0)}
from lpc.matrices import poisson2d
mats["poisson64"] = poisson2d(64)
if args.mtx:
    mats["suitesparse"] = sp.csr_matrix(scipy.io.mmread(args.mtx))

for name, A in mats.items():
    g = build_graph(A, ck["pattern"]); d = to_pyg(g); n = g.n
    print(f"\n{name}: n={n}")
    print("  features  :", "  ".join(f"{nm}: {g.node_x[:, i].min():.2f}..{g.node_x[:, i].max():.2f}" for i, nm in enumerate(names)))
    Ah = unit_diag_scale(A)[0]
    Gf = fsai(Ah, pattern=sp.csr_matrix((np.ones(len(g.row)), (g.row, g.col)), shape=Ah.shape))
    vals = {
        "jacobi": torch.tensor(g.is_diag, dtype=torch.float32),
        "GNN": model(d).detach(),
        "FSAI": torch.tensor(np.asarray(Gf[g.row, g.col]).ravel(), dtype=torch.float32),
    }
    res = {}
    for label, V in vals.items():
        torch.manual_seed(0)
        res[label] = float(np.mean([kaporin_loss(V, d, n, args.probes).item() for _ in range(5)]))
    for label in ("jacobi", "GNN"):
        print(f"  K({label}) / K(FSAI) = {np.exp(res[label] - res['FSAI']):.3f}")
    gv, fv = vals["GNN"].numpy(), vals["FSAI"].numpy()
    off = ~g.is_diag
    print(f"  diag(G): GNN {gv[g.is_diag].min():.3f}..{gv[g.is_diag].max():.3f}   FSAI {fv[g.is_diag].min():.3f}..{fv[g.is_diag].max():.3f}")
    print(f"  offdiag(G) mean |.|: GNN {np.abs(gv[off]).mean():.4f}   FSAI {np.abs(fv[off]).mean():.4f}")
