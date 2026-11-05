"""numpy mirror of the index/transposition logic in src/lpc/loss.py (which needs torch to run).
If these pass, the residual identity R Z = G(A(G^T Z)) - Z and the mirroring/index conventions are right;
tests/test_gnn_torch.py then checks the torch code agrees with numpy."""
import numpy as np
import scipy.sparse as sp

from lpc.fsai import frobenius_loss, fsai, kaporin_value
from lpc.graph import build_graph, unit_diag_scale
from lpc.matrices import diffusion2d, poisson2d


def spmm_np(row, col, val, X, n):
    out = np.zeros((n, X.shape[1]))
    np.add.at(out, row, val[:, None] * X[col])
    return out


def full_coo_np(row, col, val, is_diag):
    off = ~is_diag
    return np.concatenate([row, col[off]]), np.concatenate([col, row[off]]), np.concatenate([val, val[off]])


def residual_np(g, Gvals, Z):
    Ar, Ac, Av = full_coo_np(g.a_row, g.a_col, g.a_val, g.a_is_diag)
    t = spmm_np(g.col, g.row, Gvals, Z, g.n)       # G^T Z   (swap row/col)
    t = spmm_np(Ar, Ac, Av, t, g.n)                # A_hat (G^T Z)
    t = spmm_np(g.row, g.col, Gvals, t, g.n)       # G (...)
    return t - Z


def _setup(pattern="tril", A=None):
    A = diffusion2d(6, ax=1.0, ay=0.1) if A is None else A
    g = build_graph(A, pattern)
    Ah, _ = unit_diag_scale(A)
    Gh = fsai(Ah, pattern=sp.csr_matrix((np.ones(len(g.row)), (g.row, g.col)), shape=Ah.shape))
    vals = np.asarray(Gh[g.row, g.col]).ravel()
    return A, Ah, g, Gh, vals


def test_exact_residual_matches_dense_both_patterns():
    for pattern in ("tril", "tril2"):
        A, Ah, g, Gh, vals = _setup(pattern)
        R = residual_np(g, vals, np.eye(g.n))
        dense = (Gh @ Ah @ Gh.T).toarray() - np.eye(g.n)
        assert np.allclose(R, dense, atol=1e-10)
        assert abs((R ** 2).sum() / g.n - frobenius_loss(Gh, Ah)) < 1e-10


def test_hutchinson_is_unbiased():
    A, Ah, g, Gh, vals = _setup()
    rng = np.random.default_rng(0)
    exact = frobenius_loss(Gh, Ah)
    Z = rng.choice([-1.0, 1.0], size=(g.n, 4000))
    est = (residual_np(g, vals, Z) ** 2).sum() / (4000 * g.n)
    assert abs(est - exact) / exact < 0.05


def test_trace_probe_matches_dense_trace():
    A, Ah, g, Gh, vals = _setup()
    rng = np.random.default_rng(1)
    Z = rng.choice([-1.0, 1.0], size=(g.n, 4000))
    Ar, Ac, Av = full_coo_np(g.a_row, g.a_col, g.a_val, g.a_is_diag)
    w = spmm_np(g.col, g.row, vals, Z, g.n)
    tr_est = (w * spmm_np(Ar, Ac, Av, w, g.n)).sum() / 4000
    tr = (Gh @ Ah @ Gh.T).diagonal().sum()
    assert abs(tr_est - tr) / tr < 0.05


def test_kaporin_properties():
    A = poisson2d(5)
    G = fsai(A)
    assert kaporin_value(G, A) >= 1.0
    full = fsai(A, pattern=sp.csr_matrix(np.ones(A.shape)))
    assert abs(kaporin_value(full, A) - 1.0) < 1e-9          # exact inverse factor -> K = 1
    J = sp.diags(1 / np.sqrt(A.diagonal())).tocsr()
    assert kaporin_value(G, A) < kaporin_value(J, A)


def test_fsai_minimizes_kaporin_on_its_pattern():
    A = diffusion2d(5, ax=1.0, ay=0.2)
    G = fsai(A)
    k0 = kaporin_value(G, A)
    rng = np.random.default_rng(0)
    for _ in range(25):
        P = G.copy()
        P.data = P.data * (1 + 0.05 * rng.normal(size=P.data.shape))
        assert kaporin_value(P, A) >= k0 - 1e-12


def test_cg_history_matches_count():
    from lpc.solvers import cg_count, cg_history
    A = poisson2d(10)
    b = np.random.default_rng(0).normal(size=A.shape[0])
    h, ok = cg_history(A, b)
    assert ok and len(h) == cg_count(A, b).iters and h[-1] < 1e-8
