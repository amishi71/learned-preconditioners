"""Model/loss tests. Need torch + torch_geometric; they SKIP (not fail) in environments without them."""
import unittest

import numpy as np
import scipy.sparse as sp

from lpc.fsai import frobenius_loss, kaporin_value, spd_logdet
from lpc.graph import build_graph, to_pyg, unit_diag_scale
from lpc.matrices import diffusion2d, poisson2d
from lpc.solvers import cg_count, jacobi


def _stack():
    try:
        import torch
        import torch_geometric  # noqa: F401
    except ImportError as e:
        raise unittest.SkipTest(f"torch/PyG not installed ({e})")
    from lpc import loss, model
    return torch, model, loss


def _perturbed_model(torch, model_mod, seed=0, std=0.05):
    torch.manual_seed(seed)
    m = model_mod.PrecondGNN()
    for p in m.head[-1].parameters():
        torch.nn.init.normal_(p, std=std)
    return m


def test_init_is_exactly_jacobi():
    torch, M, L = _stack()
    A = diffusion2d(6, ax=1.0, ay=0.1)
    g = build_graph(A); d = to_pyg(g)
    G = M.PrecondGNN()(d)
    assert torch.allclose(G, torch.tensor(g.is_diag, dtype=torch.float32), atol=1e-6)


def test_pattern_positive_diagonal_and_spd():
    torch, M, L = _stack()
    A = diffusion2d(6, ax=1.0, ay=0.1)
    for pattern in ("tril", "tril2"):
        g = build_graph(A, pattern); d = to_pyg(g)
        G = _perturbed_model(torch, M, std=0.3)(d).detach().numpy().astype(float)
        Gs = sp.csr_matrix((G, (g.row, g.col)), shape=(g.n, g.n))
        assert abs(sp.triu(Gs, k=1)).sum() == 0                       # lower triangular
        assert (Gs.diagonal() > 0).all()                              # diag(G) > 0
        pat = set(zip(g.row.tolist(), g.col.tolist()))
        if pattern == "tril":
            ref = sp.tril(unit_diag_scale(A)[0]).tocoo()
            assert pat == set(zip(ref.row.tolist(), ref.col.tolist()))  # pattern == tril(A)
        Md = (Gs.T @ Gs).toarray()
        assert np.linalg.eigvalsh((Md + Md.T) / 2).min() > 0


def test_torch_losses_match_numpy():
    torch, M, L = _stack()
    A = diffusion2d(6, ax=1.0, ay=0.1)
    Ah, _ = unit_diag_scale(A)
    g = build_graph(A); d = to_pyg(g)
    G = _perturbed_model(torch, M)(d)
    Gs = sp.csr_matrix((G.detach().numpy().astype(float), (g.row, g.col)), shape=(g.n, g.n))
    assert abs(L.exact_loss(G, d, g.n).item() - frobenius_loss(Gs, Ah)) < 1e-3 * max(1.0, frobenius_loss(Gs, Ah))
    k = np.exp(L.kaporin_loss(G, d, g.n).item() - spd_logdet(Ah) / g.n)
    kv = kaporin_value(Gs, Ah)
    assert abs(k - kv) < 1e-3, f"torch K={k}, numpy K={kv}"


def test_hutchinson_close_to_exact():
    torch, M, L = _stack()
    A = diffusion2d(6, ax=1.0, ay=0.1)
    g = build_graph(A); d = to_pyg(g)
    G = _perturbed_model(torch, M)(d).detach()
    ex = L.exact_loss(G, d, g.n).item()
    est = np.mean([L.hutchinson_loss(G, d, g.n, 64).item() for _ in range(40)])
    assert abs(est - ex) / ex < 0.05


def test_overfit_decreases_both_losses():
    torch, M, L = _stack()
    A = poisson2d(8)
    g = build_graph(A); d = to_pyg(g)
    for fn in (lambda G: L.exact_loss(G, d, g.n), lambda G: L.kaporin_loss(G, d, g.n)):
        torch.manual_seed(0)
        m = M.PrecondGNN(); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        first = fn(m(d)).item()
        for _ in range(80):
            loss = fn(m(d)); opt.zero_grad(); loss.backward(); opt.step()
        assert loss.item() < first - 0.05 * abs(first)


def test_untrained_model_equals_jacobi_in_cg():
    torch, M, L = _stack()
    from lpc.precond import gnn_preconditioner
    A = diffusion2d(10, ax=1.0, ay=0.05)
    b = np.random.default_rng(0).normal(size=A.shape[0])
    it_gnn = cg_count(A, b, M=gnn_preconditioner(M.PrecondGNN(), A)).iters
    assert it_gnn == cg_count(A, b, M=jacobi(A)).iters


def test_robust_variant_init_jacobi_valid_and_degree_stable():
    torch, M, L = _stack()
    A = diffusion2d(6, ax=1.0, ay=0.1)
    g = build_graph(A); d = to_pyg(g)
    m = M.PrecondGNN(robust=True)
    assert torch.allclose(m(d), torch.tensor(g.is_diag, dtype=torch.float32), atol=1e-6)      # starts at Jacobi
    G = _perturbed_model(torch, M, std=0.3)(d)                                                  # non-robust reference still fine
    mr = M.PrecondGNN(robust=True)
    for p in mr.head[-1].parameters():
        torch.nn.init.normal_(p, std=0.3)
    Gr = mr(d).detach().numpy().astype(float)
    Gs = sp.csr_matrix((Gr, (g.row, g.col)), shape=(g.n, g.n))
    assert (Gs.diagonal() > 0).all() and abs(sp.triu(Gs, k=1)).sum() == 0
    # denser rows must not blow up hidden activations: a 27-point matrix gives outputs of the same order as a 5-point one
    from lpc.matrices import q1hex
    g2 = build_graph(q1hex(6)); d2 = to_pyg(g2)
    out2 = mr(d2).detach().abs().max().item()
    assert np.isfinite(out2) and out2 < 50 * max(1.0, np.abs(Gr).max())
