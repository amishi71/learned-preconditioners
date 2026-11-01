import numpy as np
import scipy.sparse as sp

from lpc.matrices import diffusion2d, poisson2d, sample_training_matrix


def test_poisson_diag_and_symmetry():
    A = poisson2d(8)
    assert A.shape == (64, 64)
    assert np.allclose(A.diagonal(), 4.0)
    assert abs(A - A.T).max() < 1e-14


def test_spd_all_variants():
    rng = np.random.default_rng(0)
    for _ in range(10):
        A = sample_training_matrix(rng, sizes=(6, 8))
        assert abs(A - A.T).max() < 1e-12
        w = np.linalg.eigvalsh(A.toarray())
        assert w.min() > 1e-10


def test_mixed_bc_is_spd_and_has_smaller_diag():
    A = diffusion2d(8, bc="mixed")
    assert np.linalg.eigvalsh(A.toarray()).min() > 1e-10
    assert A.diagonal().min() < 4.0
