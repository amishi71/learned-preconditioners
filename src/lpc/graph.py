"""Matrix -> graph conversion. Homogeneous graph: node = row, edge = nonzero.

Steps: symmetric unit-diagonal scaling, COO extraction (row/col/data together), lower-triangle edges.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


def unit_diag_scale(A):
    """A_hat = D^-1/2 A D^-1/2. Returns (A_hat, d_inv_sqrt)."""
    d = A.diagonal()
    assert (d > 0).all(), "SPD matrix must have positive diagonal"
    s = 1.0 / np.sqrt(d)
    D = sp.diags(s)
    return sp.csr_matrix(D @ A @ D), s


@dataclass
class MatrixGraph:
    n: int
    row: np.ndarray          # edge sources (lower triangle incl. diagonal), i >= j  -> (row=i, col=j)
    col: np.ndarray
    val: np.ndarray          # scaled A_hat[i, j]
    is_diag: np.ndarray      # bool per edge
    node_x: np.ndarray       # (n, 3): scaled diag, degree, sum|offdiag|
    scale: np.ndarray        # d^-1/2, to undo scaling
    a_row: np.ndarray = None  # lower triangle of A_hat itself (for the loss and message passing)
    a_col: np.ndarray = None
    a_val: np.ndarray = None
    a_is_diag: np.ndarray = None


def pattern_matrix(Ah, pattern: str = "tril"):
    """Sparsity pattern of G (lower triangle). 'tril' = tril(A); 'tril2' = tril(A^2) (2-hop, superset of tril)."""
    if pattern == "tril":
        return sp.tril(Ah).tocsr()
    if pattern == "tril2":
        B = abs(Ah).astype(bool).astype(float)
        return sp.tril((B @ B).astype(bool).astype(float) + sp.identity(Ah.shape[0])).tocsr()
    raise ValueError(pattern)


def build_graph(A, pattern: str = "tril") -> MatrixGraph:
    Ah, s = unit_diag_scale(sp.csr_matrix(A))
    # prediction edges = pattern of G; val = A_hat entry there (0 where A has no entry, e.g. 2-hop fill)
    P = pattern_matrix(Ah, pattern).tocoo()
    L = sp.coo_matrix((np.asarray(Ah[P.row, P.col]).ravel(), (P.row, P.col)), shape=Ah.shape)
    full = sp.coo_matrix(Ah)
    La = sp.tril(Ah).tocoo()
    n = Ah.shape[0]
    off = full.row != full.col
    deg = np.bincount(full.row[off], minlength=n).astype(float)
    sum_off = np.bincount(full.row[off], weights=np.abs(full.data[off]), minlength=n)
    node_x = np.stack([Ah.diagonal(), deg, sum_off], axis=1)
    return MatrixGraph(
        n=n, row=L.row.astype(np.int64), col=L.col.astype(np.int64), val=L.data.astype(float),
        is_diag=(L.row == L.col), node_x=node_x, scale=s,
        a_row=La.row.astype(np.int64), a_col=La.col.astype(np.int64), a_val=La.data.astype(float),
        a_is_diag=(La.row == La.col),
    )


def to_pyg(g: MatrixGraph):
    """Convert to a torch_geometric Data (needs torch + PyG). Message passing uses both directions
    of the lower-triangle edges (symmetric A), the prediction head uses only the lower-triangle edges."""
    import torch
    from torch_geometric.data import Data

    off = ~g.a_is_diag
    src = np.concatenate([g.a_row[off], g.a_col[off]])
    dst = np.concatenate([g.a_col[off], g.a_row[off]])
    w = np.concatenate([g.a_val[off], g.a_val[off]])
    return Data(
        x=torch.tensor(g.node_x, dtype=torch.float32),
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_attr=torch.tensor(w[:, None], dtype=torch.float32),
        # prediction targets: lower-triangle entries (incl. diagonal)
        pred_row=torch.tensor(g.row), pred_col=torch.tensor(g.col),
        pred_val=torch.tensor(g.val, dtype=torch.float32),
        pred_is_diag=torch.tensor(g.is_diag),
        a_row=torch.tensor(g.a_row), a_col=torch.tensor(g.a_col),
        a_val=torch.tensor(g.a_val, dtype=torch.float32), a_is_diag=torch.tensor(g.a_is_diag),
        scale=torch.tensor(g.scale, dtype=torch.float32),
        num_nodes=g.n,
    )
