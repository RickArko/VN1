"""Unit tests for `vn1.models` — short-series fallback in `fit_predict_stats`.

We don't fit real statsforecast models here (that's the integration tier).
We *do* exercise the naive-fallback path that protects against
`statsforecast.AutoETS` raising ``NotImplementedError("tiny datasets")``
on series with fewer than ``2 * season_length + 1`` observations.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from vn1.models import _naive_fallback_horizon, fit_predict_stats


def _short_series(n: int, sid: str = "1-2-100") -> pl.DataFrame:
    start = date(2024, 1, 1)
    return pl.DataFrame(
        {
            "unique_id": [sid] * n,
            "ds": [start + timedelta(days=7 * i) for i in range(n)],
            "y": [float(i + 1) for i in range(n)],
        }
    )


def test_naive_fallback_horizon_emits_h_rows_per_series() -> None:
    short = pl.concat(
        [_short_series(n=4, sid="a-1-1"), _short_series(n=4, sid="b-2-2")],
        how="vertical",
    )
    out = _naive_fallback_horizon(short, h=3, step_days=7, id_col="unique_id", time_col="ds", target_col="y")
    # 2 series × 3 horizons = 6 rows
    assert out.shape[0] == 6
    assert (out.group_by("unique_id").len()["len"] == 3).all()


def test_naive_fallback_replicates_last_value_into_all_three_columns() -> None:
    short = _short_series(n=5, sid="x")
    out = _naive_fallback_horizon(short, h=4, step_days=7, id_col="unique_id", time_col="ds", target_col="y")
    last_y = 5.0  # _short_series fills y=1..n
    for col in ("Theta", "AutoETS", "SNaive"):
        assert (out[col] == last_y).all(), f"{col} should equal last_y"


def test_naive_fallback_dates_step_correctly() -> None:
    short = _short_series(n=3, sid="x")  # last_ds = 2024-01-15
    out = _naive_fallback_horizon(
        short, h=3, step_days=7, id_col="unique_id", time_col="ds", target_col="y"
    ).sort("ds")
    expected = [date(2024, 1, 15) + timedelta(days=7 * i) for i in (1, 2, 3)]
    assert out["ds"].to_list() == expected


def test_naive_fallback_emits_date_typed_ds() -> None:
    """`Date + Duration` upcasts to Datetime in polars 1.x; the fallback must coerce
    back to Date so the cross_validate join key matches the truth side."""
    short = _short_series(n=3, sid="x")
    out = _naive_fallback_horizon(short, h=2, step_days=7, id_col="unique_id", time_col="ds", target_col="y")
    assert out.schema["ds"] == pl.Date


def test_fit_predict_stats_routes_short_series_to_fallback(toy_long: pl.DataFrame) -> None:
    """Trim one series to <105 rows; it must come back via the naive path
    with the last observed value in Theta/AutoETS/SNaive, not through
    `statsforecast` (which would raise on so few observations)."""
    long_with_short = pl.concat(
        [
            toy_long.filter(pl.col("unique_id") != "1-10-100"),  # keep three full series
            toy_long.filter(pl.col("unique_id") == "1-10-100").tail(20),  # truncate to 20 rows
        ],
        how="vertical",
    )

    out = fit_predict_stats(
        long_with_short,
        h=4,
        season_length=52,
        freq="W-MON",
        n_jobs=1,
    )

    short_rows = out.filter(pl.col("unique_id") == "1-10-100")
    assert short_rows.shape[0] == 4, "fallback should emit h rows for the short series"
    last_actual = long_with_short.filter(pl.col("unique_id") == "1-10-100").sort("ds").tail(1)["y"][0]
    for col in ("Theta", "AutoETS", "SNaive"):
        assert (short_rows[col] == last_actual).all(), (
            f"short-series {col} should equal last observed y={last_actual}"
        )
