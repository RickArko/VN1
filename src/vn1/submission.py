"""Submission-shaping utilities — long predictions → wide CSV.

The VN1 leaderboard expects a wide matrix:

    Client, Warehouse, Product, <date_1>, <date_2>, ..., <date_h>

with one row per series and one column per forecast week. Predictions
are clipped to non-negative (sales can't go below zero) and a zero-
variance naive override is applied for series whose final two
observations are constant — those generate fragile lag features that
the ML model often hallucinates noise on.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

__all__ = [
    "apply_zero_variance_override",
    "long_to_submission",
    "write_submission",
]


def apply_zero_variance_override(
    preds: pl.LazyFrame | pl.DataFrame,
    history: pl.LazyFrame | pl.DataFrame,
    *,
    n_tail: int = 2,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    pred_col: str = "y_hat",
) -> pl.LazyFrame:
    """Replace predictions with the constant tail value for zero-variance series.

    For each `unique_id`, look at the last `n_tail` observations in
    `history`. If they're identical (std == 0), we have no signal beyond
    "the value is locked" — in retail this is typically a discontinued
    SKU sitting at zero — and the ML model's noisy extrapolation hurts
    more than it helps.
    """
    history_lf = history.lazy() if isinstance(history, pl.DataFrame) else history
    preds_lf = preds.lazy() if isinstance(preds, pl.DataFrame) else preds

    overrides = (
        history_lf.sort(id_col, time_col)
        .group_by(id_col, maintain_order=True)
        .tail(n_tail)
        .group_by(id_col)
        .agg(
            std=pl.col(target_col).std(),
            fill=pl.col(target_col).last(),
        )
        .filter(pl.col("std") == 0)
        .select(id_col, "fill")
    )

    return (
        preds_lf.join(overrides, on=id_col, how="left")
        .with_columns(
            pl.coalesce(pl.col("fill"), pl.col(pred_col)).alias(pred_col),
        )
        .drop("fill")
    )


def long_to_submission(
    long_preds: pl.LazyFrame | pl.DataFrame,
    *,
    pred_col: str = "y_hat",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Pivot long predictions to the wide submission schema.

    Splits `unique_id` back into `(Client, Warehouse, Product)` integer
    columns, clips negative predictions to zero, and produces one column
    per forecast week formatted as ``YYYY-MM-DD``.
    """
    eager = long_preds.collect() if isinstance(long_preds, pl.LazyFrame) else long_preds
    parts = eager.with_columns(
        pl.col(id_col).str.split("-").alias("__parts"),
    ).with_columns(
        Client=pl.col("__parts").list.get(0).cast(pl.Int64),
        Warehouse=pl.col("__parts").list.get(1).cast(pl.Int64),
        Product=pl.col("__parts").list.get(2).cast(pl.Int64),
        clipped=pl.col(pred_col).clip(lower_bound=0.0),
    )

    wide = parts.pivot(
        on=time_col,
        index=["Client", "Warehouse", "Product"],
        values="clipped",
    )

    # Format date columns as YYYY-MM-DD strings to match the leaderboard CSV.
    date_cols = [c for c in wide.columns if c not in ("Client", "Warehouse", "Product")]
    rename_map = {c: (c if isinstance(c, str) else c.strftime("%Y-%m-%d")) for c in date_cols}
    wide = wide.rename(rename_map)
    return wide.sort("Client", "Warehouse", "Product")


def write_submission(
    long_preds: pl.LazyFrame | pl.DataFrame,
    output_path: str | Path,
    *,
    pred_col: str = "y_hat",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> Path:
    """Pivot to wide, write CSV, return the resolved path."""
    wide = long_to_submission(long_preds, pred_col=pred_col, id_col=id_col, time_col=time_col)
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    wide.write_csv(out)
    return out
