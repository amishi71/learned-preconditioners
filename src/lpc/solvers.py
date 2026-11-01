"""CG with iteration counting, plus baseline preconditioners."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class CGResult:
    iters: int
    converged: bool
    relres: float


def cg_count(A, b, M=None, rtol: float = 1e-8, maxiter: int | None = None) -> CGResult:
    """Run scipy CG and count iterations with a callback. M is a LinearOperator approximating A^-1."""
    count = [0]

    def cb(_xk):
        count[0] += 1

    maxiter = maxiter or 10 * A.shape[0]
    x, info = spla.cg(A, b, rtol=rtol, atol=0.0, M=M, maxiter=maxiter, callback=cb)
    relres = float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))
    return CGResult(iters=count[0], converged=(info == 0), relres=relres)


def jacobi(A) -> spla.LinearOperator:
    d = 1.0 / A.diagonal()
    return spla.LinearOperator(A.shape, matvec=lambda x: d * x, dtype=A.dtype)


def spilu_precond(A, drop_tol: float = 0.0, fill_factor: float = 1.0) -> spla.LinearOperator:
    """scipy SuperLU incomplete LU. NOT valid as a CG preconditioner: the result is non-symmetric
    (verified: |y.Mx - x.My| ~ 1). Kept only for GMRES comparisons. Use ic0_precond for CG."""
    lu = spla.spilu(sp.csc_matrix(A), drop_tol=drop_tol, fill_factor=fill_factor)
    return spla.LinearOperator(A.shape, matvec=lu.solve, dtype=A.dtype)


def ic0(A) -> sp.csr_matrix:
    """Incomplete Cholesky with zero fill: returns lower-triangular L with L L^T ~ A on tril(A)'s pattern.
    Pure python/numpy; fine up to ~10^4 rows. Can break down (non-positive pivot) on non-M-matrices."""
    A = sp.csr_matrix(A)
    n = A.shape[0]
    L = sp.tril(A, format="csr").astype(float)
    L.sort_indices()
    indptr, indices, data = L.indptr, L.indices, L.data
    rows = [dict(zip(indices[indptr[i]:indptr[i + 1]], range(indptr[i], indptr[i + 1]))) for i in range(n)]
    diag_pos = np.array([rows[i][i] for i in range(n)])
    for i in range(n):
        cols_i = indices[indptr[i]:indptr[i + 1]]
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            s = data[p]
            # subtract sum_k L[i,k] * L[j,k] over shared k < j
            rj = rows[j]
            for q in range(indptr[i], p):
                k = indices[q]
                pos = rj.get(k)
                if pos is not None:
                    s -= data[q] * data[pos]
            if j == i:
                if s <= 0:
                    raise np.linalg.LinAlgError(f"IC(0) breakdown at row {i}")
                data[p] = np.sqrt(s)
            else:
                data[p] = s / data[diag_pos[j]]
    return sp.csr_matrix((data, indices, indptr), shape=A.shape)


def ic0_precond(A) -> spla.LinearOperator:
    L = ic0(A)
    Lc = sp.csr_matrix(L)
    LT = sp.csr_matrix(L.T)

    def mv(x):
        y = spla.spsolve_triangular(Lc, x, lower=True)
        return spla.spsolve_triangular(LT, y, lower=False)

    return spla.LinearOperator(A.shape, matvec=mv, dtype=A.dtype)
