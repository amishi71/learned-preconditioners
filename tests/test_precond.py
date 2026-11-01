import numpy as np
import scipy.sparse as sp

from lpc.fsai import fsai, frobenius_loss
from lpc.graph import build_graph, unit_diag_scale
from lpc.matrices import diffusion2d, poisson2d
from lpc.precond import factor_from_values, operator_from_factor
from lpc.solvers import cg_count


def test_unscaling_convention_matches_direct_fsai():
    """FSAI is scale-invariant: fsai(A_hat) @ S must equal fsai(A). Guards the G_A = G_hat S convention."""
    A = diffusion2d(8, ax=1.0, ay=0.05)
    g = build_graph(A)
    Ah, s = unit_diag_scale(A)
    G_hat = fsai(Ah)
    G_A = factor_from_values(G_hat.tocoo().data, G_hat.tocoo().row, G_hat.tocoo().col, g.scale, g.n)
    assert abs(G_A - fsai(A)).max() < 1e-9


def test_operator_is_spd_and_symmetric():
    A = diffusion2d(6, ax=1.0, ay=0.1)
    M = operator_from_factor(fsai(A))
    Md = M.matmat(np.eye(A.shape[0]))
    assert np.allclose(Md, Md.T, atol=1e-12)
    assert np.linalg.eigvalsh(Md).min() > 0


def test_operator_cg_converges_on_hard_matrix():
    rng = np.random.default_rng(0)
    A = diffusion2d(16, ax=1.0, ay=0.01)
    b = rng.normal(size=A.shape[0])
    res = cg_count(A, b, M=operator_from_factor(fsai(A)))
    assert res.converged and res.relres < 1e-7
