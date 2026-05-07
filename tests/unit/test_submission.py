"""Unit tests for submission shaping (long → wide CSV)."""

from __future__ import annotations

from datetime import date

import polars as pl

from vn1.submission import (
    apply_zero_variance_override,
    long_to_submission,
)


def _toy_preds() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["1-2-100"] * 3 + ["5-7-200"] * 3,
            "ds": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)] * 2,
            "y_hat": [10.0, -2.0, 8.0, 5.0, 5.0, 5.0],
        }
    )


def test_long_to_submission_splits_unique_id() -> None:
    wide = long_to_submission(_toy_preds())
    assert {"Client", "Warehouse", "Product"}.issubset(wide.columns)
    row = wide.filter((pl.col("Client") == 1) & (pl.col("Warehouse") == 2) & (pl.col("Product") == 100))
    assert row.shape[0] == 1


def test_long_to_submission_clips_negative_predictions() -> None:
    wide = long_to_submission(_toy_preds())
    row = wide.filter(pl.col("Client") == 1)
    # 2024-01-08 had y_hat=-2 → must be clipped to 0
    assert row["2024-01-08"][0] == 0.0


def test_long_to_submission_one_column_per_date() -> None:
    wide = long_to_submission(_toy_preds())
    date_cols = [c for c in wide.columns if c not in ("Client", "Warehouse", "Product")]
    assert sorted(date_cols) == ["2024-01-01", "2024-01-08", "2024-01-15"]


def test_long_to_submission_sorted_by_id_triple() -> None:
    wide = long_to_submission(_toy_preds())
    triples = list(
        zip(wide["Client"].to_list(), wide["Warehouse"].to_list(), wide["Product"].to_list(), strict=True)
    )
    assert triples == sorted(triples)


def test_zero_variance_override_replaces_constant_tail() -> None:
    preds = _toy_preds()
    history = pl.DataFrame(
        {
            "unique_id": ["1-2-100"] * 3 + ["5-7-200"] * 3,
            "ds": [date(2023, 12, 18), date(2023, 12, 25), date(2024, 1, 1)] * 2,
            "y": [3.0, 0.0, 0.0, 4.0, 5.0, 6.0],
        }
    )
    out = apply_zero_variance_override(preds, history, n_tail=2).collect()
    s1 = out.filter(pl.col("unique_id") == "1-2-100")["y_hat"].to_list()
    s2 = out.filter(pl.col("unique_id") == "5-7-200")["y_hat"].to_list()
    assert s1 == [0.0, 0.0, 0.0], "constant-tail series should be locked to fill value"
    assert s2 == [5.0, 5.0, 5.0], "varying-tail series should pass through unchanged"


def test_zero_variance_override_no_op_when_no_constant_tails() -> None:
    preds = _toy_preds()
    history = pl.DataFrame(
        {
            "unique_id": ["1-2-100"] * 3 + ["5-7-200"] * 3,
            "ds": [date(2023, 12, 18), date(2023, 12, 25), date(2024, 1, 1)] * 2,
            "y": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    out = apply_zero_variance_override(preds, history, n_tail=2).collect().sort("unique_id", "ds")
    expected = preds.sort("unique_id", "ds")
    assert out.select("unique_id", "ds", "y_hat").equals(expected.select("unique_id", "ds", "y_hat"))
