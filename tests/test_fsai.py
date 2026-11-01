import numpy as np
import scipy.sparse as sp

from lpc.fsai import frobenius_loss, fsai, fsai_precond
from lpc.matrices import poisson2d
from lpc.solvers import cg_count, ic0, ic0_precond, jacobi


def test_fsai_unit_diagonal_and_lower_triangular():
    A = poisson2d(6)
    G = fsai(A)
    assert abs(sp.triu(G, k=1)).sum() == 0
    D = (G @ A @ G.T).diagonal()
    assert np.allclose(D, 1.0)


def test_fsai_full_pattern_is_exact_inverse_factor():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(12, 12))
    A = sp.csr_matrix(B @ B.T + 12 * np.eye(12))
    G = fsai(A, pattern=sp.csr_matrix(np.ones((12, 12))))
    assert np.allclose((G @ A @ G.T).toarray(), np.eye(12), atol=1e-9)


def test_fsai_reduces_iterations_vs_none():
    A = poisson2d(16)
    b = np.random.default_rng(1).normal(size=A.shape[0])
    base = cg_count(A, b)
    res = cg_count(A, b, M=fsai_precond(fsai(A)))
    assert res.converged and base.converged
    assert res.iters < base.iters


def test_fsai_beats_identity_on_loss():
    A = poisson2d(8)
    n = A.shape[0]
    d = 1.0 / np.sqrt(A.diagonal())
    jac = sp.diags(d).tocsr()
    assert frobenius_loss(fsai(A), A) < frobenius_loss(jac, A)


def test_preconditioner_is_spd():
    A = poisson2d(5)
    G = fsai(A)
    M = (G.T @ G).toarray()
    assert np.linalg.eigvalsh((M + M.T) / 2).min() > 0


def test_ic0_matches_A_on_pattern_and_converges():
    A = poisson2d(8)
    L = ic0(A)
    LLt = (L @ L.T).toarray()
    P = (A.toarray() != 0)
    assert np.allclose(LLt[P], A.toarray()[P], atol=1e-10)
    b = np.random.default_rng(0).normal(size=A.shape[0])  # not ones: constant RHS is degenerate for CG
    assert cg_count(A, b, M=ic0_precond(A)).iters < cg_count(A, b).iters


def test_jacobi_is_noop_on_constant_diagonal():
    A = poisson2d(16)
    b = np.random.default_rng(2).normal(size=A.shape[0])
    assert cg_count(A, b, M=jacobi(A)).iters == cg_count(A, b).iters


def test_wider_pattern_lowers_loss_and_iterations():
    from lpc.graph import pattern_matrix, unit_diag_scale
    A = poisson2d(16)
    P2 = pattern_matrix(unit_diag_scale(A)[0], "tril2")
    G1, G2 = fsai(A), fsai(A, pattern=P2)
    assert frobenius_loss(G2, A) < frobenius_loss(G1, A)
    b = np.random.default_rng(3).normal(size=A.shape[0])
    assert cg_count(A, b, M=fsai_precond(G2)).iters < cg_count(A, b, M=fsai_precond(G1)).iters
