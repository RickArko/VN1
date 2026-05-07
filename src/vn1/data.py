"""VN1 challenge data loaders — polars-lazy by default.

Two phases of weekly demand are concatenated horizontally to form the wide
training matrix; :func:`process_wide_df` unpivots it to the Nixtla
``(unique_id, ds, y)`` long schema. ``unique_id`` is the
``"{Client}-{Warehouse}-{Product}"`` triple.

I/O is eager (one HTTP read per phase) but the public API returns
``LazyFrame`` so downstream transforms compose without materializing
intermediates.
"""

from __future__ import annotations

import polars as pl

PHASE1_URL = (
    "https://www.datasource.ai/attachments/"
    "eyJpZCI6Ijk4NDYxNjE2NmZmZjM0MGRmNmE4MTczOGMyMzI2ZWI2LmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIs"
    "Im1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMCAtIFNhbGVzLmNzdiIsInNpemUiOjEwODA0NjU0LCJtaW1lX3R5cGUiOiJ0ZXh0L2NzdiJ9fQ"
)
PHASE2_URL = (
    "https://www.datasource.ai/attachments/"
    "eyJpZCI6ImM2OGQxNGNmNTJkZDQ1MTUyZTg0M2FkMDAyMjVlN2NlLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIs"
    "Im1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMSAtIFNhbGVzLmNzdiIsInNpemUiOjEwMTgzOTYsIm1pbWVfdHlwZSI6InRleHQvY3N2In19"
)

ID_COLS = ("Client", "Warehouse", "Product")
ID_COL = "unique_id"
TIME_COL = "ds"
TARGET_COL = "y"

__all__ = [
    "ID_COL",
    "ID_COLS",
    "PHASE1_URL",
    "PHASE2_URL",
    "TARGET_COL",
    "TIME_COL",
    "load_full_data",
    "process_wide_df",
    "trim_leading_zeros",
]


def load_full_data() -> pl.LazyFrame:
    """Read phase 1 + phase 2 sales CSVs and concatenate horizontally.

    Returns a wide LazyFrame with ``Client, Warehouse, Product`` and one
    column per week. Sorting is stable on the ID triple so the horizontal
    concat is row-aligned.
    """
    phase1 = pl.read_csv(PHASE1_URL).sort(*ID_COLS)
    phase2 = pl.read_csv(PHASE2_URL).sort(*ID_COLS).drop(*ID_COLS)
    return pl.concat([phase1, phase2], how="horizontal").lazy()


def process_wide_df(wide: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Unpivot wide → long Nixtla schema (``unique_id, ds, y``).

    Accepts either a ``LazyFrame`` or a ``DataFrame`` and always returns
    a ``LazyFrame`` so callers can chain further transforms cheaply.
    """
    lf = wide.lazy() if isinstance(wide, pl.DataFrame) else wide
    return (
        lf.unpivot(index=list(ID_COLS), variable_name=TIME_COL, value_name=TARGET_COL)
        .with_columns(
            unique_id=(
                pl.col("Client").cast(pl.Utf8)
                + "-"
                + pl.col("Warehouse").cast(pl.Utf8)
                + "-"
                + pl.col("Product").cast(pl.Utf8)
            ),
            ds=pl.col(TIME_COL).cast(pl.Date),
        )
        .select(ID_COL, TIME_COL, TARGET_COL, *ID_COLS)
        .sort(ID_COL, TIME_COL)
    )


def trim_leading_zeros(long: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Drop leading zeros per series (pre-launch padding).

    Within each ``unique_id`` (sorted by ``ds``), keep rows from the first
    positive ``y`` onward. Series that are entirely zero collapse to empty.
    """
    lf = long.lazy() if isinstance(long, pl.DataFrame) else long
    started = pl.col(TARGET_COL).gt(0).cast(pl.Int8).cum_max().over(ID_COL).cast(pl.Boolean)
    return lf.sort(ID_COL, TIME_COL).filter(started)
