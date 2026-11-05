"""Day 3 gate: run on your machine after installing torch + PyG.   python scripts/day3_checks.py
Checks: (1) torch losses == numpy, (2) Hutchinson ~ exact, (3) SPD, (4) overfit one 16x16 matrix toward FSAI."""
import pathlib, sys
import numpy as np, scipy.sparse as sp, torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lpc.fsai import fsai, frobenius_loss, kaporin_value, fsai_precond
from lpc.graph import build_graph, to_pyg, unit_diag_scale
from lpc.loss import exact_loss, hutchinson_loss, kaporin_loss
from lpc.matrices import poisson2d
from lpc.model import PrecondGNN
from lpc.precond import gnn_preconditioner
from lpc.seed import set_seed
from lpc.solvers import cg_count

set_seed(0)
A = poisson2d(16); n = A.shape[0]
Ah, _ = unit_diag_scale(A)
g = build_graph(A); data = to_pyg(g)
Gf = fsai(Ah)
print(f"FSAI reference: frobenius {frobenius_loss(Gf, Ah):.4f}  kaporin {kaporin_value(Gf, Ah):.4f}")
logdetA = np.linalg.slogdet(Ah.toarray())[1]

model = PrecondGNN()
G0 = model(data)
print("init is Jacobi (G = I):", bool(torch.allclose(G0, torch.tensor(g.is_diag, dtype=torch.float32))))
print(f"init losses: frobenius {exact_loss(G0, data, n).item():.4f}  kaporin K {np.exp(kaporin_loss(G0, data, n).item() + logdetA / n):.4f}")

# (2) estimator agreement (perturb the zero-initialised head so G != I)
for p in model.head[-1].parameters():
    torch.nn.init.normal_(p, std=0.05)
G = model(data).detach()
ex = exact_loss(G, data, n).item()
hs = np.mean([hutchinson_loss(G, data, n, 64).item() for _ in range(50)])
print(f"(2) Hutchinson vs exact: {hs:.4f} vs {ex:.4f}  rel.err {abs(hs-ex)/ex:.3%}  (want < 3%)")

for loss_name in ("frobenius", "kaporin"):
    torch.manual_seed(0)
    m = PrecondGNN(); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for step in range(400):
        Gv = m(data)
        loss = exact_loss(Gv, data, n) if loss_name == "frobenius" else kaporin_loss(Gv, data, n)
        opt.zero_grad(); loss.backward(); opt.step()
    M = gnn_preconditioner(m, A)
    Gm = sp.csr_matrix((m(data).detach().numpy().astype(float), (g.row, g.col)), shape=(n, n))
    b = np.random.default_rng(0).normal(size=n)
    print(f"(4) overfit [{loss_name}] 400 steps: frobenius {frobenius_loss(Gm, Ah):.4f}  kaporin {kaporin_value(Gm, Ah):.4f}  "
          f"CG iters {cg_count(A, b, M=M).iters}  (FSAI: {cg_count(A, b, M=fsai_precond(fsai(A))).iters}, none: {cg_count(A, b).iters})")
    d = np.diag(Gm.toarray()); print("    diag(G) > 0:", bool((d > 0).all()))
