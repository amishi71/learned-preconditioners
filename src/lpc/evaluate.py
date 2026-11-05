"""Evaluation harness: run CG with several preconditioner builders on a list of matrices."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .fsai import frobenius_loss, kaporin_value
from .solvers import cg_count


def evaluate(matrices: dict, methods: dict, seed: int = 0, rtol: float = 1e-8) -> pd.DataFrame:
    """matrices: name -> sparse A.  methods: name -> fn(A) returning (LinearOperator|None, G|None).
    Records iterations, convergence, setup time, per-apply time, and loss (if G is given)."""
    rng = np.random.default_rng(seed)
    rows = []
    for mname, A in matrices.items():
        b = rng.normal(size=A.shape[0])
        for name, build in methods.items():
            t0 = time.perf_counter()
            M, G = build(A)
            setup = time.perf_counter() - t0
            apply_t = np.nan
            if M is not None:
                x = rng.normal(size=A.shape[0])
                t1 = time.perf_counter()
                for _ in range(20):
                    M.matvec(x)
                apply_t = (time.perf_counter() - t1) / 20
            r = cg_count(A, b, M=M, rtol=rtol)
            rows.append(dict(matrix=mname, n=A.shape[0], method=name, iters=r.iters, converged=r.converged,
                             relres=r.relres, setup_s=setup, apply_s=apply_t,
                             loss=(frobenius_loss(G, A) if G is not None else np.nan),
                             kaporin=(kaporin_value(G, A) if (G is not None and A.shape[0] <= 2500) else np.nan)))
    return pd.DataFrame(rows)
