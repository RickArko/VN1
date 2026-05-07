"""Unit tests for `vn1.data` — schema conversions, no network."""

from __future__ import annotations

from datetime import date

import polars as pl

from vn1.data import (
    ID_COL,
    ID_COLS,
    TARGET_COL,
    TIME_COL,
    process_wide_df,
    trim_leading_zeros,
)


def _toy_wide() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "Client": [1, 1, 2],
            "Warehouse": [10, 10, 20],
            "Product": [100, 200, 100],
            "2023-01-02": [3.0, 0.0, 5.0],
            "2023-01-09": [4.0, 1.0, 6.0],
        }
    )


def test_process_wide_df_returns_lazyframe() -> None:
    out = process_wide_df(_toy_wide())
    assert isinstance(out, pl.LazyFrame)


def test_process_wide_df_unpivots_to_long_schema() -> None:
    out = process_wide_df(_toy_wide()).collect()
    expected_cols = {ID_COL, TIME_COL, TARGET_COL, *ID_COLS}
    assert set(out.columns) == expected_cols
    assert out.shape[0] == 3 * 2  # 3 series × 2 dates
    # `unique_id` is the dash-joined triple
    assert "1-10-100" in set(out[ID_COL].to_list())
    # `ds` is a Date dtype
    assert out[TIME_COL].dtype == pl.Date


def test_process_wide_df_accepts_dataframe_input() -> None:
    """API contract: accept either polars DataFrame or LazyFrame."""
    eager = _toy_wide().collect()
    out = process_wide_df(eager)
    assert isinstance(out, pl.LazyFrame)
    assert out.collect().shape[0] == 6


def test_trim_leading_zeros_drops_pre_launch_padding() -> None:
    """Series with [0, 0, 3, 5, 4] keeps only [3, 5, 4]."""
    long = pl.LazyFrame(
        {
            ID_COL: ["a"] * 5 + ["b"] * 5,
            TIME_COL: [date(2023, 1, 2 + 7 * i) for i in range(5)] * 2,
            TARGET_COL: [0.0, 0.0, 3.0, 5.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    out = trim_leading_zeros(long).collect()
    a = out.filter(pl.col(ID_COL) == "a")
    b = out.filter(pl.col(ID_COL) == "b")
    assert a.shape[0] == 3, "two leading zeros should be dropped from series a"
    assert a[TARGET_COL].to_list() == [3.0, 5.0, 4.0]
    assert b.shape[0] == 5, "series b has no leading zeros"


def test_trim_leading_zeros_keeps_post_launch_zeros() -> None:
    """Zeros AFTER the first positive are real (out of stock, off-season) — keep them."""
    long = pl.LazyFrame(
        {
            ID_COL: ["a"] * 4,
            TIME_COL: [date(2023, 1, 2 + 7 * i) for i in range(4)],
            TARGET_COL: [0.0, 5.0, 0.0, 3.0],
        }
    )
    out = trim_leading_zeros(long).collect()
    assert out.shape[0] == 3
    assert out[TARGET_COL].to_list() == [5.0, 0.0, 3.0]
