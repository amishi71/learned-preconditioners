"""Frobenius loss ||G A G^T - I||_F^2 / n for a sparse lower-triangular G (autograd-safe, no sparse-tensor autograd)."""
import torch


def spmm(row, col, val, X, n):
    """Y = S X for the sparse matrix with entries (row, col, val). X: (n, k)."""
    out = torch.zeros(n, X.shape[1], dtype=X.dtype, device=X.device)
    return out.index_add(0, row, val[:, None] * X[col])


def full_coo(row, col, val, is_diag):
    """Mirror a lower-triangle (incl. diagonal) entry list into the full symmetric matrix."""
    off = ~is_diag
    return torch.cat([row, col[off]]), torch.cat([col, row[off]]), torch.cat([val, val[off]])


def residual_apply(Grow, Gcol, Gval, Arow, Acol, Aval, Z, n):
    """R Z = G (A (G^T Z)) - Z."""
    t = spmm(Gcol, Grow, Gval, Z, n)          # G^T Z
    t = spmm(Arow, Acol, Aval, t, n)          # A (G^T Z)
    t = spmm(Grow, Gcol, Gval, t, n)          # G (...)
    return t - Z


def exact_loss(Gvals, data, n):
    """Exact ||R||_F^2 / n via R @ I (cost ~ n sparse matvecs; fine for n <= ~1024)."""
    Ar, Ac, Av = full_coo(data.a_row, data.a_col, data.a_val, data.a_is_diag)
    I = torch.eye(n, dtype=Gvals.dtype, device=Gvals.device)
    R = residual_apply(data.pred_row, data.pred_col, Gvals, Ar, Ac, Av, I, n)
    return (R ** 2).sum() / n


def hutchinson_loss(Gvals, data, n, probes: int = 8):
    """Unbiased estimate of ||R||_F^2 / n with Rademacher probes."""
    Ar, Ac, Av = full_coo(data.a_row, data.a_col, data.a_val, data.a_is_diag)
    Z = (torch.randint(0, 2, (n, probes), device=Gvals.device) * 2 - 1).to(Gvals.dtype)
    R = residual_apply(data.pred_row, data.pred_col, Gvals, Ar, Ac, Av, Z, n)
    return (R ** 2).sum() / (probes * n)


def kaporin_loss(Gvals, data, n, probes=None):
    """log K(G) minus the constant (1/n) log det(A_hat), where K is the Kaporin conditioning number
        K = (tr(G A G^T)/n) / det(G A G^T)^(1/n),  and for lower-triangular G: log det(G A G^T) = 2 sum log g_ii + log det A.
    So   loss = log(tr(G A G^T)/n) - (2/n) sum_i log g_ii.
    FSAI is the exact minimizer of K on a fixed pattern; K ranked CG iteration counts far better than the
    Frobenius loss in benchmarks/correlation_study.py. Uses only matvecs (no determinant needed).
    probes=None -> exact trace (n columns, fine for n <= ~1024); else Hutchinson trace with `probes` Rademacher vectors
    (slightly biased inside the log; use >= 16)."""
    Ar, Ac, Av = full_coo(data.a_row, data.a_col, data.a_val, data.a_is_diag)
    if probes is None:
        Z = torch.eye(n, dtype=Gvals.dtype, device=Gvals.device)
        scale = 1.0
    else:
        Z = (torch.randint(0, 2, (n, probes), device=Gvals.device) * 2 - 1).to(Gvals.dtype)
        scale = 1.0 / probes
    W = spmm(data.pred_col, data.pred_row, Gvals, Z, n)      # G^T Z
    AW = spmm(Ar, Ac, Av, W, n)
    tr = (W * AW).sum() * scale
    logdiag = torch.log(Gvals[data.pred_is_diag]).sum()
    return torch.log(tr / n) - 2.0 * logdiag / n
