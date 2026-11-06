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


# ---- out-of-distribution test matrices (no download needed) ----

def poisson3d(n: int, ax: float = 1.0, ay: float = 1.0, az: float = 1.0) -> sp.csr_matrix:
    """3D Poisson, 7-point stencil, Dirichlet, n^3 rows (diagonal 2(ax+ay+az), 6 when isotropic)."""
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    I = sp.identity(n)
    A = ax * sp.kron(sp.kron(T, I), I) + ay * sp.kron(sp.kron(I, T), I) + az * sp.kron(sp.kron(I, I), T)
    return sp.csr_matrix(A)


def q1hex(n: int) -> sp.csr_matrix:
    """Q1 trilinear finite-element Laplacian on a uniform n^3 grid: SPD, 21 nonzeros per interior row (the six face
    couplings cancel exactly, so the nominal 27-point stencil has 21 nonzeros).
    K (x) M (x) M + M (x) K (x) M + M (x) M (x) K with 1D stiffness K and 1D mass M. Denser rows than anything in
    training, but well conditioned and with non-positive off-diagonals (an easy held-out case)."""
    K = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    M = sp.diags([1.0, 4.0, 1.0], [-1, 0, 1], shape=(n, n)) / 6.0
    k3 = lambda a, b, c: sp.kron(sp.kron(a, b), c)
    A = sp.csr_matrix(k3(K, M, M) + k3(M, K, M) + k3(M, M, K))
    A.eliminate_zeros()
    return A


def rot_aniso9(n: int, eps: float = 0.05, theta: float = np.pi / 6) -> sp.csr_matrix:
    """Rotated anisotropic diffusion -div(K grad u), K = R diag(1, eps) R^T, 9-point finite-difference stencil on an
    n x n grid (Dirichlet). The mixed-derivative term gives POSITIVE off-diagonals, so it is SPD but not an M-matrix,
    and a different stencil from the 5-point training matrices (a hard held-out case)."""
    c_, s_ = np.cos(theta), np.sin(theta)
    a = c_ ** 2 + eps * s_ ** 2
    c = s_ ** 2 + eps * c_ ** 2
    b = (1.0 - eps) * s_ * c_
    stencil = {(0, 0): 2 * (a + c), (0, 1): -a, (0, -1): -a, (1, 0): -c, (-1, 0): -c,
               (1, 1): -b / 2, (-1, -1): -b / 2, (1, -1): b / 2, (-1, 1): b / 2}
    idx = np.arange(n * n).reshape(n, n)
    rows, cols, vals = [], [], []
    for (di, dj), w in stencil.items():
        i0, i1 = max(0, -di), min(n, n - di)
        j0, j1 = max(0, -dj), min(n, n - dj)
        r = idx[i0:i1, j0:j1]
        rows.append(r.ravel()); cols.append(idx[i0 + di:i1 + di, j0 + dj:j1 + dj].ravel())
        vals.append(np.full(r.size, w))
    return sp.csr_matrix(sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n)))


def fem_delaunay(n_interior: int = 4000, seed: int = 0, boundary_per_side: int | None = None) -> sp.csr_matrix:
    """P1 finite-element Laplacian on a random Delaunay triangulation of the unit square, Dirichlet boundary
    (boundary nodes eliminated). Unstructured with irregular row lengths (~7 nnz/row) and widely varying element sizes.
    A Delaunay mesh makes the P1 stiffness matrix an M-matrix, so it is SPD with non-positive off-diagonals; the
    difference from the training set is the irregular connectivity and coefficients, not the sign pattern."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    m = boundary_per_side or max(8, int(np.sqrt(n_interior)))
    t = np.linspace(0.0, 1.0, m, endpoint=False)
    boundary = np.concatenate([np.stack([t, 0 * t], 1), np.stack([1 + 0 * t, t], 1),
                               np.stack([1 - t, 1 + 0 * t], 1), np.stack([0 * t, 1 - t], 1)])
    interior = rng.uniform(0.02, 0.98, size=(n_interior, 2))
    pts = np.vstack([interior, boundary])
    tri = Delaunay(pts).simplices
    x, y = pts[:, 0], pts[:, 1]
    rows, cols, vals = [], [], []
    for a, b, c in tri:
        bi = np.array([y[b] - y[c], y[c] - y[a], y[a] - y[b]])
        ci = np.array([x[c] - x[b], x[a] - x[c], x[b] - x[a]])
        area2 = (x[b] - x[a]) * (y[c] - y[a]) - (x[c] - x[a]) * (y[b] - y[a])
        if abs(area2) < 1e-14:
            continue
        Kloc = (np.outer(bi, bi) + np.outer(ci, ci)) / (2.0 * abs(area2))
        idx = np.array([a, b, c])
        rows.append(np.repeat(idx, 3)); cols.append(np.tile(idx, 3)); vals.append(Kloc.ravel())
    K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(pts), len(pts))).tocsr()
    A = K[:n_interior][:, :n_interior]
    A.sum_duplicates()
    return sp.csr_matrix(A)


def sample_wide_training_matrix(rng: np.random.Generator) -> sp.csr_matrix:
    """Wider training distribution: 50% the original 2D diffusion family, 25% anisotropic 3D 7-point stencils,
    25% random Delaunay FEM meshes. (Held out for evaluation: rot_aniso9, q1hex and SuiteSparse Pres_Poisson.)"""
    r = rng.random()
    if r < 0.5:
        return sample_training_matrix(rng)
    if r < 0.75:
        n = int(rng.choice([4, 5, 6, 8, 10]))
        w = [float(10 ** rng.uniform(-1.5, 0)) if rng.random() < 0.4 else 1.0 for _ in range(3)]
        return poisson3d(n, *w)
    return fem_delaunay(int(rng.choice([100, 200, 400, 800, 1200])), seed=int(rng.integers(0, 2**31 - 1)))
