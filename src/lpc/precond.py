"""Turn a predicted factor into a CG preconditioner for the ORIGINAL (unscaled) matrix A.

The model works on A_hat = S A S with S = diag(a_ii^-1/2) and predicts G_hat with G_hat A_hat G_hat^T ~ I.
Then G_A = G_hat S satisfies G_A A G_A^T ~ I, and M^-1 = G_A^T G_A.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def factor_from_values(vals, row, col, scale, n) -> sp.csr_matrix:
    """Build G_A from G_hat entries (vals at (row, col)) and S = diag(scale)."""
    vals = np.asarray(vals, dtype=float)
    G_hat = sp.csr_matrix((vals, (row, col)), shape=(n, n))
    return sp.csr_matrix(G_hat @ sp.diags(scale))


def operator_from_factor(G) -> spla.LinearOperator:
    """M^-1 x = G^T (G x): two sparse matvecs, SPD whenever diag(G) > 0."""
    G = sp.csr_matrix(G)
    GT = sp.csr_matrix(G.T)
    return spla.LinearOperator(G.shape, matvec=lambda x: GT @ (G @ x), matmat=lambda X: GT @ (G @ X), dtype=float)


def gnn_preconditioner(model, A, pattern: str = "tril") -> spla.LinearOperator:
    """Run a trained PrecondGNN on A and return the CG preconditioner (needs torch + PyG)."""
    import torch

    from .graph import build_graph, to_pyg

    g = build_graph(A, pattern)
    data = to_pyg(g)
    model.eval()
    with torch.no_grad():
        vals = model(data).cpu().numpy().astype(float)
    return operator_from_factor(factor_from_values(vals, g.row, g.col, g.scale, g.n))
