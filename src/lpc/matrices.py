"""SPD test-matrix generators: 2D diffusion  -div(k grad u)  on an n x n grid (5-point, flux form).

Poisson (k=1, ax=ay=1, Dirichlet) gives the classic stencil with diagonal 4.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _harmonic(a, b):
    return 2.0 * a * b / (a + b)


def diffusion2d(
    n: int,
    ax: float = 1.0,
    ay: float = 1.0,
    k: np.ndarray | None = None,
    bc: str = "dirichlet",
) -> sp.csr_matrix:
    """Assemble the n^2 x n^2 matrix.

    ax, ay : anisotropy (coupling weights in x / y).
    k      : (n, n) positive cell coefficient field (default all ones).
    bc     : 'dirichlet' (all sides) or 'mixed' (Neumann left/right, Dirichlet top/bottom).
             Both are SPD. (Pure Neumann is singular, so it is not offered.)
    """
    if bc not in ("dirichlet", "mixed"):
        raise ValueError("bc must be 'dirichlet' or 'mixed'")
    k = np.ones((n, n)) if k is None else np.asarray(k, dtype=float)
    assert k.shape == (n, n) and (k > 0).all()
    idx = np.arange(n * n).reshape(n, n)
    diag = np.zeros((n, n))
    rows, cols, vals = [], [], []

    # horizontal faces between (i,j) and (i,j+1)
    w = ax * _harmonic(k[:, :-1], k[:, 1:])
    rows += [idx[:, :-1].ravel(), idx[:, 1:].ravel()]
    cols += [idx[:, 1:].ravel(), idx[:, :-1].ravel()]
    vals += [-w.ravel(), -w.ravel()]
    diag[:, :-1] += w
    diag[:, 1:] += w
    # vertical faces between (i,j) and (i+1,j)
    w = ay * _harmonic(k[:-1, :], k[1:, :])
    rows += [idx[:-1, :].ravel(), idx[1:, :].ravel()]
    cols += [idx[1:, :].ravel(), idx[:-1, :].ravel()]
    vals += [-w.ravel(), -w.ravel()]
    diag[:-1, :] += w
    diag[1:, :] += w

    # boundary faces: Dirichlet faces add to the diagonal, Neumann faces add nothing
    diag[0, :] += ay * k[0, :]
    diag[-1, :] += ay * k[-1, :]
    if bc == "dirichlet":
        diag[:, 0] += ax * k[:, 0]
        diag[:, -1] += ax * k[:, -1]

    rows.append(idx.ravel())
    cols.append(idx.ravel())
    vals.append(diag.ravel())
    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n * n, n * n),
    ).tocsr()
    A.sum_duplicates()
    return A


def poisson2d(n: int) -> sp.csr_matrix:
    return diffusion2d(n)


def random_coeff_field(n: int, rng: np.random.Generator, contrast: float = 100.0, smooth: int = 2):
    """Positive field with controlled contrast, lightly smoothed."""
    f = rng.uniform(0.0, 1.0, size=(n, n))
    for _ in range(smooth):
        f = (f + np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 5.0
    f = (f - f.min()) / max(f.max() - f.min(), 1e-12)
    return contrast ** (f - 0.5)


def sample_training_matrix(rng: np.random.Generator, sizes=(8, 12, 16, 24, 32, 48, 64)) -> sp.csr_matrix:
    """Random 'Poisson-like' matrix for augmentation (anisotropy, coefficients, BCs, sizes)."""
    n = int(rng.choice(sizes))
    ax = 1.0
    ay = float(10 ** rng.uniform(-2, 0)) if rng.random() < 0.5 else 1.0
    if rng.random() < 0.5:
        ax, ay = ay, ax
    k = random_coeff_field(n, rng, contrast=float(10 ** rng.uniform(0, 3))) if rng.random() < 0.5 else None
    bc = "mixed" if rng.random() < 0.3 else "dirichlet"
    return diffusion2d(n, ax=ax, ay=ay, k=k, bc=bc)
