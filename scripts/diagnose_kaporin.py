import numpy as np, scipy.sparse as sp, torch
from lpc.fsai import kaporin_value, spd_logdet
from lpc.graph import build_graph, to_pyg, unit_diag_scale
from lpc.loss import kaporin_loss
from lpc.matrices import diffusion2d
from lpc.model import PrecondGNN
A = diffusion2d(6, ax=1.0, ay=0.1); Ah, _ = unit_diag_scale(A); D = Ah.toarray()
print("finite:", np.isfinite(D).all(), " min eig:", np.linalg.eigvalsh(D).min())
print("slogdet :", np.linalg.slogdet(D))
print("cholesky:", spd_logdet(Ah))
print("eigvalsh:", np.log(np.linalg.eigvalsh(D)).sum())
torch.manual_seed(0); m = PrecondGNN()
for p in m.head[-1].parameters(): torch.nn.init.normal_(p, std=0.05)
g = build_graph(A); d = to_pyg(g); G = m(d)
Gs = sp.csr_matrix((G.detach().numpy().astype(float), (g.row, g.col)), shape=(g.n, g.n))
print("K torch:", np.exp(kaporin_loss(G, d, g.n).item() - spd_logdet(Ah) / g.n), " K numpy:", kaporin_value(Gs, Ah))
