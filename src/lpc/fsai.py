"""Classical FSAI (Kolotilina-Yeremin): the closed-form per-row optimum on a fixed lower-triangular pattern.

G has the pattern P (default tril(A)); row i solves A[P_i, P_i] y = e_i, then g = y / sqrt(y_i).
This makes diag(G A G^T) = 1 and minimizes the Kaporin-type objective on P. Preconditioner: M^-1 = G^T G.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def tril_pattern(A) -> sp.csr_matrix:
    P = sp.tril(sp.csr_matrix(A), format="csr")
    P.sort_indices()
    return P


def fsai(A, pattern: sp.spmatrix | None = None) -> sp.csr_matrix:
    A = sp.csr_matrix(A)
    P = tril_pattern(A) if pattern is None else sp.tril(sp.csr_matrix(pattern), format="csr")
    P.sort_indices()
    n = A.shape[0]
    data = np.zeros_like(P.data, dtype=float)
    for i in range(n):
        s, e = P.indptr[i], P.indptr[i + 1]
        cols = P.indices[s:e]
        if cols[-1] != i:
            raise ValueError(f"row {i}: pattern must include the diagonal")
        sub = A[cols][:, cols].toarray()
        rhs = np.zeros(len(cols))
        rhs[-1] = 1.0
        y = np.linalg.solve(sub, rhs)
        data[s:e] = y / np.sqrt(y[-1])
    return sp.csr_matrix((data, P.indices.copy(), P.indptr.copy()), shape=A.shape)


def fsai_precond(G) -> spla.LinearOperator:
    G = sp.csr_matrix(G)
    GT = sp.csr_matrix(G.T)
    return spla.LinearOperator(G.shape, matvec=lambda x: GT @ (G @ x), dtype=float)


def frobenius_loss(G, A) -> float:
    """||G A G^T - I||_F^2 / n (exact, sparse)."""
    n = A.shape[0]
    R = G @ sp.csr_matrix(A) @ G.T - sp.identity(n)
    return float(R.multiply(R).sum() / n)
