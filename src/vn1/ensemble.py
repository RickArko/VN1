"""Ensemble weight optimization against the VN1 competition metric.

Given a long-format frame of out-of-sample CV predictions (one column
per model), find the simplex weights that minimize :func:`comp_loss` on
the merged forecast. Constraints:

- ``w_i >= 0`` for all models (no shorting),
- ``sum(w_i) == 1`` (proper convex combination).

We use SLSQP with explicit bound + equality constraints. For a small
number of models (3–5 in this project) this is fast and robust; the
metric is non-smooth (``|·|``), but SLSQP's quasi-second-order behavior
copes well in practice. We seed at equal weights and accept the result
only if it strictly improves on the best single model.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from scipy.optimize import minimize

from vn1.metrics import comp_loss

__all__ = ["apply_weights", "optimize_weights"]


def optimize_weights(
    cv_preds: pl.LazyFrame | pl.DataFrame,
    *,
    pred_cols: list[str],
    target_col: str = "y",
    seed: int | None = None,
    n_restarts: int = 4,
) -> dict[str, float]:
    """Return non-negative simplex weights that minimize comp_loss on cv_preds.

    `cv_preds` is the long DataFrame returned by
    :func:`vn1.validation.cross_validate`. It must contain `target_col`
    and every name in `pred_cols` as float columns.

    Multiple restarts (default 4) reduce sensitivity to initialization;
    we randomize within the simplex via Dirichlet draws.
    """
    if not pred_cols:
        raise ValueError("pred_cols must be non-empty")

    df = cv_preds.lazy() if isinstance(cv_preds, pl.DataFrame) else cv_preds
    eager = df.select(target_col, *pred_cols).drop_nulls().collect()
    if eager.is_empty():
        raise ValueError("cv_preds is empty after dropping nulls — nothing to optimize.")

    y = eager[target_col].to_numpy().astype(np.float64)
    pred_matrix = np.column_stack([eager[c].to_numpy().astype(np.float64) for c in pred_cols])

    def loss(w: np.ndarray) -> float:
        return comp_loss(y, pred_matrix @ w)

    n_models = len(pred_cols)
    bounds = [(0.0, 1.0)] * n_models
    constraints = ({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},)

    rng = np.random.default_rng(seed)
    starts: list[np.ndarray] = [np.full(n_models, 1.0 / n_models)]
    for _ in range(max(0, n_restarts - 1)):
        starts.append(np.asarray(rng.dirichlet(np.ones(n_models)), dtype=np.float64))

    best_w: np.ndarray = starts[0]
    best_loss = float("inf")
    for x0 in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                loss,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 200},
            )
        if res.success and res.fun < best_loss:
            best_loss = float(res.fun)
            best_w = np.asarray(res.x, dtype=np.float64)

    # Floor + renormalize: kill numerical noise that breaks the simplex.
    best_w = np.clip(best_w, 0.0, None)
    total = float(best_w.sum())
    if total > 0:
        best_w = best_w / total
    else:
        best_w = np.full(n_models, 1.0 / n_models)
    return dict(zip(pred_cols, (float(w) for w in best_w), strict=True))


def apply_weights(
    preds: pl.LazyFrame | pl.DataFrame,
    weights: dict[str, float],
    *,
    out_col: str = "y_hat",
) -> pl.LazyFrame:
    """Combine model columns via a weighted sum into ``out_col``.

    Returns a LazyFrame regardless of input type so the caller can chain.
    Models in `weights` whose column is missing trigger a ValueError —
    silent zero-fill would mask bugs.
    """
    lf = preds.lazy() if isinstance(preds, pl.DataFrame) else preds
    schema = lf.collect_schema().names()
    missing = [c for c in weights if c not in schema]
    if missing:
        raise ValueError(f"weights reference columns not in preds: {missing}")
    expr = pl.lit(0.0)
    for col, w in weights.items():
        expr = expr + pl.col(col) * pl.lit(w)
    return lf.with_columns(expr.alias(out_col))
