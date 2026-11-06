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


def test_ood_matrices_are_spd_and_structurally_different():
    from lpc.matrices import fem_delaunay, poisson3d
    A3 = poisson3d(6)
    assert A3.shape == (216, 216) and np.allclose(A3.diagonal(), 6.0)
    assert np.linalg.eigvalsh(A3.toarray()).min() > 0
    F = fem_delaunay(150, seed=1)
    assert F.shape == (150, 150) and abs(F - F.T).max() < 1e-10
    assert np.linalg.eigvalsh(F.toarray()).min() > 1e-10
    assert (F.diagonal() > 0).all()
    off = F - sp.diags(F.diagonal())
    assert off.max() < 1e-10                       # Delaunay => M-matrix (non-positive off-diagonals)
    assert np.diff(F.indptr).std() > 0.5           # irregular row lengths, unlike the training stencils


def test_heldout_matrices_and_wide_sampler():
    from lpc.matrices import q1hex, poisson3d, rot_aniso9, sample_wide_training_matrix
    H = q1hex(5)
    assert np.diff(H.indptr).max() == 21 and abs(H - H.T).max() < 1e-12
    assert np.linalg.eigvalsh(H.toarray()).min() > 0
    R = rot_aniso9(12)
    assert abs(R - R.T).max() < 1e-12 and np.linalg.eigvalsh(R.toarray()).min() > 0
    assert (R - sp.diags(R.diagonal())).max() > 0           # positive off-diagonals: NOT an M-matrix (checked, not assumed)
    assert np.diff(R.indptr).max() == 9
    A = poisson3d(4, 1.0, 0.1, 0.5)
    assert np.linalg.eigvalsh(A.toarray()).min() > 0
    rng = np.random.default_rng(0)
    for _ in range(12):
        W = sample_wide_training_matrix(rng)
        assert abs(W - W.T).max() < 1e-10 and (W.diagonal() > 0).all()
