"""Unit tests for rolling-origin CV split generation."""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest

from vn1.validation import (
    cross_validate,
    evaluate_predictions,
    rolling_origin_splits,
)


def test_n_windows_matches_request(toy_long: pl.DataFrame) -> None:
    splits = rolling_origin_splits(toy_long, h=4, n_windows=3)
    assert len(splits) == 3
    assert [s.fold for s in splits] == [0, 1, 2]


def test_val_window_has_h_distinct_dates(toy_long: pl.DataFrame) -> None:
    h = 4
    splits = rolling_origin_splits(toy_long, h=h, n_windows=3)
    for split in splits:
        n_dates = split.val.select(pl.col("ds").n_unique()).collect().item()
        assert n_dates == h, f"fold {split.fold}: expected {h} dates, got {n_dates}"


def test_no_train_val_leakage(toy_long: pl.DataFrame) -> None:
    """Every fold's max train date must be strictly before its min val date."""
    splits = rolling_origin_splits(toy_long, h=4, n_windows=3)
    for split in splits:
        train_max = split.train.select(pl.col("ds").max()).collect().item()
        val_min = split.val.select(pl.col("ds").min()).collect().item()
        assert train_max < val_min, f"fold {split.fold} leakage: train_max={train_max}, val_min={val_min}"


def test_non_overlapping_val_windows(toy_long: pl.DataFrame) -> None:
    """Default step_size = h means consecutive val windows don't overlap."""
    h = 4
    splits = rolling_origin_splits(toy_long, h=h, n_windows=3)
    from itertools import pairwise

    for prev, curr in pairwise(splits):
        prev_max = prev.val.select(pl.col("ds").max()).collect().item()
        curr_min = curr.val.select(pl.col("ds").min()).collect().item()
        assert prev_max < curr_min


def test_overlapping_val_windows_with_smaller_step(toy_long: pl.DataFrame) -> None:
    splits = rolling_origin_splits(toy_long, h=8, n_windows=3, step_size=4)
    from itertools import pairwise

    for prev, curr in pairwise(splits):
        prev_max = prev.val.select(pl.col("ds").max()).collect().item()
        curr_min = curr.val.select(pl.col("ds").min()).collect().item()
        # 8-week val with 4-week step → 4 weeks of overlap
        assert curr_min < prev_max


def test_too_little_history_raises(toy_long: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="Not enough history"):
        rolling_origin_splits(toy_long, h=200, n_windows=3)


def test_invalid_args_rejected(toy_long: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_windows must be"):
        rolling_origin_splits(toy_long, h=4, n_windows=0)
    with pytest.raises(ValueError, match="h must be"):
        rolling_origin_splits(toy_long, h=0, n_windows=3)


def test_cross_validate_attaches_fold_and_cutoff(toy_long: pl.DataFrame) -> None:
    """A trivial naive forecast through cross_validate should produce a long
    frame with `fold`, `cutoff`, plus the model column. Truth (`y`) is joined."""

    def naive(train: pl.LazyFrame, h: int) -> pl.LazyFrame:
        last = (
            train.sort("ds")
            .group_by("unique_id", maintain_order=True)
            .tail(1)
            .select("unique_id", pl.col("y").alias("Naive"))
        )
        max_ds = train.select(pl.col("ds").max()).collect().item()
        future = pl.DataFrame({"ds": [max_ds + timedelta(days=7 * (i + 1)) for i in range(h)]})
        return last.join(future.lazy(), how="cross")

    out = cross_validate(naive, toy_long, h=4, n_windows=2)
    assert {"unique_id", "ds", "y", "Naive", "fold", "cutoff"}.issubset(out.columns)
    assert out["fold"].n_unique() == 2
    # 4 weeks × 4 series × 2 folds
    assert out.shape[0] == 4 * 4 * 2


def test_evaluate_predictions_overall_uses_fold_minus_one(toy_cv_preds: pl.DataFrame) -> None:
    eval_df = evaluate_predictions(toy_cv_preds, pred_cols=["Perfect", "Biased"])
    overall = eval_df.filter(pl.col("fold") == -1)
    assert set(overall["model"].to_list()) == {"Perfect", "Biased"}
    perfect_overall = overall.filter(pl.col("model") == "Perfect")["comp_loss"][0]
    assert perfect_overall == pytest.approx(0.0)
