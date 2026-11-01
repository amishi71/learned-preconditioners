import numpy as np
import scipy.sparse as sp

from lpc.graph import build_graph, unit_diag_scale
from lpc.matrices import diffusion2d, poisson2d


def test_unit_diagonal_after_scaling():
    A = diffusion2d(8, ax=1.0, ay=0.01)
    Ah, s = unit_diag_scale(A)
    assert np.allclose(Ah.diagonal(), 1.0)
    assert abs(Ah - Ah.T).max() < 1e-12


def test_edges_are_lower_triangle_and_match_matrix():
    A = poisson2d(6)
    g = build_graph(A)
    assert (g.row >= g.col).all()
    Ah, _ = unit_diag_scale(A)
    Ah = Ah.tocsr()
    # row/col/val must stay aligned (the old nonzero()+data bug)
    for i, j, v in zip(g.row, g.col, g.val):
        assert abs(Ah[i, j] - v) < 1e-12
    assert g.is_diag.sum() == A.shape[0]
    assert len(g.val) == sp.tril(A).nnz


def test_explicit_zeros_do_not_misalign():
    A = poisson2d(5).tolil()
    A[0, 7] = 0.0
    A[7, 0] = 0.0
    g = build_graph(A.tocsr())
    Ah, _ = unit_diag_scale(A.tocsr())
    Ah = Ah.tocsr()
    for i, j, v in zip(g.row, g.col, g.val):
        assert abs(Ah[i, j] - v) < 1e-12


def test_node_features_shape():
    g = build_graph(poisson2d(6))
    assert g.node_x.shape == (36, 3)
    assert np.allclose(g.node_x[:, 0], 1.0)


def test_tril2_pattern_is_superset_and_aligned():
    A = poisson2d(6)
    g1, g2 = build_graph(A, "tril"), build_graph(A, "tril2")
    s1 = set(zip(g1.row.tolist(), g1.col.tolist()))
    s2 = set(zip(g2.row.tolist(), g2.col.tolist()))
    assert s1 <= s2 and len(s2) > len(s1)
    Ah, _ = unit_diag_scale(A)
    Ah = Ah.tocsr()
    for i, j, v in zip(g2.row, g2.col, g2.val):
        assert i >= j and abs(Ah[i, j] - v) < 1e-12
