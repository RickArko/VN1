"""Integration: end-to-end ensemble pipeline on toy data.

Exercises the path the submission notebook takes — CV with multiple
models, weight optimization, blended predictions, zero-variance
override, wide-CSV write — without downloading the competition data.
"""

from __future__ import annotations

import importlib.util
from datetime import timedelta

import polars as pl
import pytest

from vn1.ensemble import apply_weights, optimize_weights
from vn1.metrics import comp_loss_polars
from vn1.submission import apply_zero_variance_override, long_to_submission
from vn1.validation import cross_validate, evaluate_predictions


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def _naive_forecast_fn(train: pl.LazyFrame, h: int) -> pl.LazyFrame:
    last = (
        train.sort("ds")
        .group_by("unique_id", maintain_order=True)
        .tail(1)
        .select("unique_id", pl.col("y").alias("Naive"))
    )
    max_ds = train.select(pl.col("ds").max()).collect().item()
    future = pl.LazyFrame({"ds": [max_ds + timedelta(days=7 * (i + 1)) for i in range(h)]})
    return last.join(future, how="cross")


def _mean13_forecast_fn(train: pl.LazyFrame, h: int) -> pl.LazyFrame:
    last_window = (
        train.sort("ds")
        .group_by("unique_id", maintain_order=True)
        .tail(13)
        .group_by("unique_id")
        .agg(pl.col("y").mean().alias("Mean13"))
    )
    max_ds = train.select(pl.col("ds").max()).collect().item()
    future = pl.LazyFrame({"ds": [max_ds + timedelta(days=7 * (i + 1)) for i in range(h)]})
    return last_window.join(future, how="cross")


def test_two_model_ensemble_pipeline_end_to_end(toy_long: pl.DataFrame) -> None:
    """Two simple forecasters → CV → weight optimization → blended preds."""
    cv_naive = cross_validate(_naive_forecast_fn, toy_long, h=4, n_windows=2)
    cv_mean = cross_validate(_mean13_forecast_fn, toy_long, h=4, n_windows=2)

    merged = cv_naive.join(
        cv_mean.select("unique_id", "ds", "fold", "Mean13"),
        on=["unique_id", "ds", "fold"],
    )
    assert {"Naive", "Mean13", "y", "fold"}.issubset(merged.columns)

    weights = optimize_weights(merged, pred_cols=["Naive", "Mean13"], seed=0)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    blended = apply_weights(merged, weights, out_col="ens").collect()
    ens_loss = comp_loss_polars(blended, pred_col="ens")
    naive_loss = comp_loss_polars(merged, pred_col="Naive")
    mean_loss = comp_loss_polars(merged, pred_col="Mean13")
    assert ens_loss <= min(naive_loss, mean_loss) + 1e-6


def test_evaluate_predictions_per_model_per_fold(toy_long: pl.DataFrame) -> None:
    cv = cross_validate(_naive_forecast_fn, toy_long, h=4, n_windows=2)
    cv = cv.join(
        cross_validate(_mean13_forecast_fn, toy_long, h=4, n_windows=2).select(
            "unique_id", "ds", "fold", "Mean13"
        ),
        on=["unique_id", "ds", "fold"],
    )
    eval_df = evaluate_predictions(cv, pred_cols=["Naive", "Mean13"])
    assert {"model", "fold", "comp_loss"}.issubset(eval_df.columns)
    # 2 models × (2 folds + 1 overall row) = 6 rows
    assert eval_df.shape[0] == 6


def test_pipeline_emits_valid_submission(toy_long: pl.DataFrame) -> None:
    """End of the line: blended predictions pivot to a clean wide CSV."""
    cv_naive = cross_validate(_naive_forecast_fn, toy_long, h=4, n_windows=2)
    cv_mean = cross_validate(_mean13_forecast_fn, toy_long, h=4, n_windows=2)
    merged = cv_naive.join(
        cv_mean.select("unique_id", "ds", "fold", "Mean13"),
        on=["unique_id", "ds", "fold"],
    )
    weights = optimize_weights(merged, pred_cols=["Naive", "Mean13"], seed=0)

    # Hold-out forecast: refit each on full data
    forecast_n = _naive_forecast_fn(toy_long.lazy(), 4).collect()
    forecast_m = _mean13_forecast_fn(toy_long.lazy(), 4).collect()
    full = forecast_n.join(forecast_m, on=["unique_id", "ds"])
    blended = apply_weights(full, weights, out_col="y_hat").collect()
    finalized = apply_zero_variance_override(blended, toy_long).collect()

    wide = long_to_submission(finalized)
    assert wide.shape[0] == toy_long["unique_id"].n_unique()
    date_cols = [c for c in wide.columns if c not in ("Client", "Warehouse", "Product")]
    assert len(date_cols) == 4  # h=4 weeks
    # All predictions non-negative after the clip.
    for c in date_cols:
        assert (wide[c] >= 0).all()


@pytest.mark.slow
@pytest.mark.skipif(not (_have("mlforecast") and _have("lightgbm")), reason="mlforecast / lightgbm missing")
def test_lgbm_inside_cross_validate(toy_long: pl.DataFrame) -> None:
    """Confirm the real LightGBM forecaster slots into the cross_validate harness."""
    from vn1.models import fit_predict_lgbm

    def lgbm_fn(train: pl.LazyFrame, h: int) -> pl.DataFrame:
        return fit_predict_lgbm(train, h=h, n_estimators=30, freq="7d", season_length=52)

    cv = cross_validate(lgbm_fn, toy_long, h=4, n_windows=2)
    assert {"LGBM", "y", "fold"}.issubset(cv.columns)
    assert cv["fold"].n_unique() == 2
    assert (cv.group_by("fold").len()["len"] == 4 * toy_long["unique_id"].n_unique()).all()
