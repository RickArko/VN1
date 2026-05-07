"""VN1 competition metric.

The official scoring rule normalizes a *bias* term (cumulative signed error)
plus a *dispersion* term (cumulative absolute error) by the sum of true
values:

    score = (|sum(err)| + sum(|err|)) / sum(y_true),       err = y_pred - y_true

Both terms are penalized equally, so persistent over- or under-forecasting
hurts twice. Lower is better. A series whose truth sums to zero scores 0
(no signal to forecast against).

This is a *global* metric across all `(unique_id, ds)` pairs in the
evaluation window — not a per-series mean.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = [
    "comp_loss",
    "comp_loss_mlforecast",
    "comp_loss_polars",
    "per_series_comp_loss",
]


def comp_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pure numpy implementation of the VN1 competition metric.

    `y_true` and `y_pred` must be 1-D arrays of equal length, with NaNs
    already filtered out by the caller.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape}")
    yt_sum = float(y_true.sum())
    if yt_sum == 0.0:
        return 0.0
    err = y_pred - y_true
    return (abs(float(err.sum())) + float(np.abs(err).sum())) / yt_sum


def comp_loss_mlforecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ids: np.ndarray | None = None,
    dates: np.ndarray | None = None,
) -> float:
    """`mlforecast`-compatible metric signature.

    `LightGBMCV.fit(metric=...)` calls back with `(y_true, y_pred, ids,
    dates)` per fold; we ignore ids/dates and compute the global metric.
    """
    return comp_loss(y_true, y_pred)


def comp_loss_polars(
    df: pl.LazyFrame | pl.DataFrame,
    *,
    target_col: str = "y",
    pred_col: str = "y_hat",
) -> float:
    """Compute the metric directly on a polars frame, in lazy if possible.

    The caller must ensure `pred_col` and `target_col` exist and align
    row-wise. Rows with null in either column are dropped first.
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    err = pl.col(pred_col) - pl.col(target_col)
    agg = (
        lf.drop_nulls([target_col, pred_col])
        .select(
            yt_sum=pl.col(target_col).sum(),
            err_sum=err.sum(),
            abs_err_sum=err.abs().sum(),
        )
        .collect()
    )
    yt_sum = float(agg["yt_sum"][0])
    if yt_sum == 0.0:
        return 0.0
    return (abs(float(agg["err_sum"][0])) + float(agg["abs_err_sum"][0])) / yt_sum


def per_series_comp_loss(
    df: pl.LazyFrame | pl.DataFrame,
    *,
    id_col: str = "unique_id",
    target_col: str = "y",
    pred_col: str = "y_hat",
) -> pl.DataFrame:
    """Per-`unique_id` competition score for diagnostic drill-down.

    Returned columns: `unique_id`, `comp_loss`, `n_obs`, `yt_sum`. Series
    with zero truth-sum get a `comp_loss` of 0 to match the global rule.
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    err = pl.col(pred_col) - pl.col(target_col)
    return (
        lf.drop_nulls([target_col, pred_col])
        .group_by(id_col)
        .agg(
            yt_sum=pl.col(target_col).sum(),
            err_sum=err.sum(),
            abs_err_sum=err.abs().sum(),
            n_obs=pl.len(),
        )
        .with_columns(
            comp_loss=pl.when(pl.col("yt_sum") == 0)
            .then(pl.lit(0.0))
            .otherwise((pl.col("err_sum").abs() + pl.col("abs_err_sum")) / pl.col("yt_sum")),
        )
        .select(id_col, "comp_loss", "n_obs", "yt_sum")
        .sort("comp_loss", descending=True)
        .collect()
    )
