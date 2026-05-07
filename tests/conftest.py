"""Shared fixtures and per-directory markers.

Layout:
    tests/smoke/        marker: smoke
    tests/unit/         marker: unit
    tests/integration/  marker: integration

Fixtures live here so any tier can use them. Polars-first to match the
production code path; statsforecast / lightgbm tests convert at the
library boundary themselves.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------
# Auto-mark by directory — keeps test files free of `pytestmark = ...`.
# ---------------------------------------------------------------------
_TIER_MARKERS = {
    "smoke": pytest.mark.smoke,
    "unit": pytest.mark.unit,
    "integration": pytest.mark.integration,
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    tests_root = Path(__file__).parent
    for item in items:
        try:
            rel = Path(item.fspath).relative_to(tests_root)
        except ValueError:
            continue
        tier = rel.parts[0] if rel.parts else ""
        marker = _TIER_MARKERS.get(tier)
        if marker is not None:
            item.add_marker(marker)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def toy_long(rng: np.random.Generator) -> pl.DataFrame:
    """A small Nixtla-shaped polars frame: 4 series × 156 weekly periods.

    Each series is `level + annual_seasonality + noise`, clipped to >=0
    so the data shape mirrors retail demand. Series IDs follow VN1's
    `Client-Warehouse-Product` integer triple convention.
    """
    n_weeks = 156  # 3 years
    start = date(2021, 1, 4)  # a Monday
    dates = [start + timedelta(days=7 * i) for i in range(n_weeks)]
    series = [
        ("1-10-100", 12.0),
        ("1-10-200", 4.0),
        ("2-20-100", 25.0),
        ("2-20-300", 0.5),  # near-zero / intermittent
    ]
    rows = []
    for sid, level in series:
        seasonal = 0.4 * level * np.sin(np.arange(n_weeks) * 2 * np.pi / 52)
        noise = rng.normal(0, max(level * 0.1, 0.5), n_weeks)
        y = np.clip(level + seasonal + noise, 0.0, None).astype(np.float64)
        rows.extend({"unique_id": sid, "ds": d, "y": float(v)} for d, v in zip(dates, y, strict=True))
    return pl.DataFrame(rows)


@pytest.fixture()
def toy_long_lazy(toy_long: pl.DataFrame) -> pl.LazyFrame:
    """Lazy version of `toy_long` for tests that exercise the lazy code path."""
    return toy_long.lazy()


@pytest.fixture()
def toy_cv_preds(toy_long: pl.DataFrame) -> pl.DataFrame:
    """Synthetic CV-prediction frame: 2 folds × 13 weeks × 4 series × 3 models.

    Forecast columns are chosen so they have an unambiguous ranking
    against ``comp_loss``:
        Perfect — equals truth
        Biased  — truth + 1.5 (positive bias term contributes 2x)
        Naive   — last-week-of-train value broadcast across the fold
    The optimizer should land on weight ≈ 1 for ``Perfect``.
    """
    h = 13
    max_ds = toy_long["ds"].max()
    folds = []
    for k in range(2):
        cutoff = max_ds - timedelta(days=7 * (2 - k) * h)
        fold_window = toy_long.filter(
            (pl.col("ds") > cutoff) & (pl.col("ds") <= cutoff + timedelta(days=7 * h))
        ).with_columns(
            fold=pl.lit(k, dtype=pl.Int32),
            cutoff=pl.lit(cutoff, dtype=pl.Date),
            Perfect=pl.col("y"),
            Biased=pl.col("y") + 1.5,
        )
        last_train = (
            toy_long.filter(pl.col("ds") <= cutoff)
            .sort("unique_id", "ds")
            .group_by("unique_id", maintain_order=True)
            .tail(1)
            .select("unique_id", pl.col("y").alias("Naive"))
        )
        fold_window = fold_window.join(last_train, on="unique_id", how="left")
        folds.append(fold_window)
    return pl.concat(folds, how="vertical")


@pytest.fixture()
def small_holdout(toy_long: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Train/holdout split with a 13-week evaluation window."""
    cutoff = toy_long["ds"].max() - timedelta(days=7 * 13)
    return (
        toy_long.filter(pl.col("ds") <= cutoff),
        toy_long.filter(pl.col("ds") > cutoff),
    )
