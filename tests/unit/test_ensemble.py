"""Unit tests for the ensemble weight optimizer."""

from __future__ import annotations

import polars as pl
import pytest

from vn1.ensemble import apply_weights, optimize_weights
from vn1.metrics import comp_loss_polars


def test_weights_sum_to_one(toy_cv_preds: pl.DataFrame) -> None:
    w = optimize_weights(toy_cv_preds, pred_cols=["Perfect", "Biased", "Naive"])
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= -1e-9 for v in w.values())


def test_weights_concentrate_on_best_model(toy_cv_preds: pl.DataFrame) -> None:
    """`Perfect` has zero error, so its weight should be ~1."""
    w = optimize_weights(toy_cv_preds, pred_cols=["Perfect", "Biased", "Naive"], seed=0)
    assert w["Perfect"] == pytest.approx(1.0, abs=1e-3)
    assert w["Biased"] == pytest.approx(0.0, abs=1e-3)
    assert w["Naive"] == pytest.approx(0.0, abs=1e-3)


def test_ensemble_no_worse_than_best_single(toy_cv_preds: pl.DataFrame) -> None:
    """The optimized ensemble must score <= the best single-model score."""
    cols = ["Biased", "Naive"]  # exclude Perfect to make the test informative
    w = optimize_weights(toy_cv_preds, pred_cols=cols, seed=0)
    blended = apply_weights(toy_cv_preds, w, out_col="ens").collect()

    single_losses = {c: comp_loss_polars(toy_cv_preds, pred_col=c) for c in cols}
    ens_loss = comp_loss_polars(blended, pred_col="ens")
    assert ens_loss <= min(single_losses.values()) + 1e-6


def test_apply_weights_is_linear_combination() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    out = apply_weights(df, {"a": 0.3, "b": 0.7}, out_col="ens").collect()
    assert out["ens"].to_list() == pytest.approx([0.3 * 1 + 0.7 * 10, 0.3 * 2 + 0.7 * 20, 0.3 * 3 + 0.7 * 30])


def test_apply_weights_rejects_missing_columns() -> None:
    df = pl.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="weights reference columns not in preds"):
        apply_weights(df, {"a": 0.5, "b": 0.5}).collect()


def test_optimize_rejects_empty_pred_cols() -> None:
    with pytest.raises(ValueError, match="pred_cols must be non-empty"):
        optimize_weights(pl.DataFrame({"y": [1.0]}), pred_cols=[])


def test_optimize_rejects_empty_after_dropna() -> None:
    df = pl.DataFrame({"y": [None, None], "m1": [1.0, None]})
    with pytest.raises(ValueError, match="empty after dropping nulls"):
        optimize_weights(df, pred_cols=["m1"])
