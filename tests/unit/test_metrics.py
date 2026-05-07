"""Unit tests for the VN1 competition metric."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from vn1.metrics import (
    comp_loss,
    comp_loss_mlforecast,
    comp_loss_polars,
    per_series_comp_loss,
)


def test_perfect_forecast_scores_zero() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert comp_loss(y, y) == 0.0


def test_zero_truth_scores_zero() -> None:
    """A series whose truth sums to zero scores 0 regardless of predictions."""
    assert comp_loss(np.zeros(5), np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == 0.0


def test_canonical_value_matches_formula() -> None:
    y_true = np.array([10.0, 5.0, 0.0, 8.0])
    y_pred = np.array([12.0, 4.0, 1.0, 9.0])
    err = y_pred - y_true  # [2, -1, 1, 1]
    expected = (abs(err.sum()) + np.abs(err).sum()) / y_true.sum()  # (3 + 5) / 23
    assert comp_loss(y_true, y_pred) == pytest.approx(expected)


def test_constant_bias_scores_double_the_bias() -> None:
    """If err == c (constant), score = (|n*c| + n*|c|) / sum(y) = 2*n*|c|/sum(y)."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    bias = 0.5
    y_pred = y_true + bias
    expected = 2.0 * len(y_true) * bias / y_true.sum()
    assert comp_loss(y_true, y_pred) == pytest.approx(expected)


def test_zero_mean_error_still_penalized_by_dispersion() -> None:
    """Symmetric errors cancel in the sum but not in the abs sum."""
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([11.0, 9.0])  # err = +1, -1 → sum=0, abs_sum=2
    expected = (0.0 + 2.0) / 20.0
    assert comp_loss(y_true, y_pred) == pytest.approx(expected)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        comp_loss(np.array([1.0, 2.0]), np.array([1.0]))


def test_mlforecast_signature_ignores_ids_and_dates() -> None:
    """The mlforecast adapter must accept and ignore ids/dates kwargs."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.0, 2.5])
    via_pure = comp_loss(y_true, y_pred)
    via_mlf = comp_loss_mlforecast(y_true, y_pred, ids=np.array(["a", "b", "c"]), dates=np.array([1, 2, 3]))
    assert via_mlf == via_pure


def test_polars_agrees_with_numpy() -> None:
    y_true = np.array([10.0, 5.0, 0.0, 8.0])
    y_pred = np.array([12.0, 4.0, 1.0, 9.0])
    df = pl.DataFrame({"y": y_true, "y_hat": y_pred})
    assert comp_loss_polars(df) == pytest.approx(comp_loss(y_true, y_pred))


def test_polars_lazy_path_works() -> None:
    df = pl.LazyFrame({"y": [1.0, 2.0, 3.0], "y_hat": [1.5, 2.5, 3.5]})
    assert comp_loss_polars(df) > 0


def test_polars_drops_nulls() -> None:
    df = pl.DataFrame({"y": [1.0, None, 3.0], "y_hat": [1.0, 2.0, 3.0]})
    # Should compute on the non-null pair → perfect → 0
    assert comp_loss_polars(df) == 0.0


def test_per_series_breakdown_matches_global_for_single_id() -> None:
    df = pl.DataFrame(
        {
            "unique_id": ["a"] * 4,
            "y": [10.0, 5.0, 0.0, 8.0],
            "y_hat": [12.0, 4.0, 1.0, 9.0],
        }
    )
    per = per_series_comp_loss(df)
    glob = comp_loss_polars(df)
    assert per["comp_loss"][0] == pytest.approx(glob)


def test_per_series_zero_truth_scores_zero() -> None:
    df = pl.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "y": [0.0, 0.0, 5.0, 5.0],
            "y_hat": [3.0, 3.0, 5.0, 5.0],
        }
    )
    per = per_series_comp_loss(df).sort("unique_id")
    by_id = dict(zip(per["unique_id"].to_list(), per["comp_loss"].to_list(), strict=True))
    assert by_id["a"] == 0.0
    assert by_id["b"] == 0.0
