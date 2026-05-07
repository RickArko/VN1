"""Rolling-origin cross-validation for the VN1 long-frame schema.

The competition horizon is 13 weeks. We hold out the most recent
`n_windows` non-overlapping (or step-overlapping) windows of length `h`
and produce a long DataFrame of out-of-sample predictions per model that
the ensemble layer can then optimize weights against.

Splits are computed from *distinct timestamps in the data* — no
calendar-arithmetic assumptions about frequency. That makes the helper
identical for daily, weekly, or arbitrary-step series.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import polars as pl

from vn1.metrics import comp_loss_polars

__all__ = [
    "Split",
    "cross_validate",
    "evaluate_predictions",
    "rolling_origin_splits",
]


@dataclass(frozen=True)
class Split:
    """One rolling-origin fold."""

    fold: int
    cutoff: date
    train: pl.LazyFrame
    val: pl.LazyFrame


def rolling_origin_splits(
    long: pl.LazyFrame | pl.DataFrame,
    *,
    h: int,
    n_windows: int,
    step_size: int | None = None,
    time_col: str = "ds",
) -> list[Split]:
    """Return `n_windows` rolling folds with horizon `h`.

    The most recent fold's val window ends at the maximum date in the
    data. Earlier folds step back by `step_size` (default = `h`, so
    non-overlapping). Each fold's train slice is everything strictly
    before its val slice's first timestamp.
    """
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    step = step_size if step_size is not None else h

    lf = long.lazy() if isinstance(long, pl.DataFrame) else long
    distinct = lf.select(pl.col(time_col).unique().sort()).collect()[time_col].to_list()
    n_dates = len(distinct)

    splits: list[Split] = []
    for k in range(n_windows):
        end_idx = n_dates - (n_windows - 1 - k) * step
        start_idx = end_idx - h
        if start_idx < 1:
            raise ValueError(
                f"Not enough history for n_windows={n_windows}, h={h}, step={step}: "
                f"fold {k} would need {h + (n_windows - 1 - k) * step} prior periods, "
                f"only {n_dates} distinct timestamps available."
            )
        val_first = distinct[start_idx]
        val_last = distinct[end_idx - 1]
        train = lf.filter(pl.col(time_col) < val_first)
        val = lf.filter((pl.col(time_col) >= val_first) & (pl.col(time_col) <= val_last))
        cutoff = distinct[start_idx - 1]
        splits.append(Split(fold=k, cutoff=cutoff, train=train, val=val))
    return splits


def cross_validate(
    forecast_fn: Callable[[pl.LazyFrame, int], pl.DataFrame | pl.LazyFrame],
    long: pl.LazyFrame | pl.DataFrame,
    *,
    h: int,
    n_windows: int,
    step_size: int | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pl.DataFrame:
    """Run `forecast_fn(train, h)` on each fold; return long predictions joined to truth.

    The returned frame has `[id_col, time_col, target_col, fold, cutoff,
    <model_cols...>]` and is the canonical input to
    :func:`vn1.ensemble.optimize_weights`. `forecast_fn` must return at
    least `id_col + time_col` and one or more model-named columns.
    """
    splits = rolling_origin_splits(
        long,
        h=h,
        n_windows=n_windows,
        step_size=step_size,
        time_col=time_col,
    )

    parts: list[pl.DataFrame] = []
    for split in splits:
        preds = forecast_fn(split.train, h)
        preds_df = preds.collect() if isinstance(preds, pl.LazyFrame) else preds
        truth = split.val.select(id_col, time_col, target_col).collect()
        merged = truth.join(preds_df, on=[id_col, time_col], how="inner").with_columns(
            fold=pl.lit(split.fold, dtype=pl.Int32),
            cutoff=pl.lit(split.cutoff, dtype=pl.Date),
        )
        parts.append(merged)
    return pl.concat(parts, how="vertical_relaxed")


def evaluate_predictions(
    cv_preds: pl.LazyFrame | pl.DataFrame,
    *,
    pred_cols: list[str],
    target_col: str = "y",
    fold_col: str = "fold",
) -> pl.DataFrame:
    """Per-fold and overall comp_loss for each model in `pred_cols`.

    Returns one row per (model, fold) plus an `overall` row per model
    with `fold = -1`. Useful for sanity checks before optimizing
    ensemble weights — if a model's overall score is the same magnitude
    as its per-fold scores, the CV is honest.
    """
    df = cv_preds.lazy() if isinstance(cv_preds, pl.DataFrame) else cv_preds
    eager = df.collect()

    rows: list[dict[str, object]] = []
    for model in pred_cols:
        rows.append(
            {
                "model": model,
                "fold": -1,
                "comp_loss": comp_loss_polars(eager, target_col=target_col, pred_col=model),
            }
        )
        for fold_val in eager[fold_col].unique().sort().to_list():
            sub = eager.filter(pl.col(fold_col) == fold_val)
            rows.append(
                {
                    "model": model,
                    "fold": int(fold_val),
                    "comp_loss": comp_loss_polars(sub, target_col=target_col, pred_col=model),
                }
            )
    return pl.DataFrame(rows).sort("model", "fold")
