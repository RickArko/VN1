"""Integration: build forecasters and run small fit/predict cycles on toy data.

Each test is guarded by `_have(...)` so the suite runs even when an
optional forecasting backend isn't installed (e.g. on a slim CI runner).
"""

from __future__ import annotations

import importlib.util

import polars as pl
import pytest


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


# ---------------------------------------------------------------------
# Statistical bundle
# ---------------------------------------------------------------------
@pytest.mark.skipif(not _have("statsforecast"), reason="statsforecast not installed")
def test_stats_forecaster_has_three_models() -> None:
    from vn1.models import build_stats_forecaster

    sf = build_stats_forecaster(season_length=52)
    assert len(sf.models) == 3
    aliases = {m.alias for m in sf.models}
    assert {"Theta", "AutoETS", "SNaive"}.issubset(aliases)


@pytest.mark.slow
@pytest.mark.skipif(not _have("statsforecast"), reason="statsforecast not installed")
def test_stats_fit_predict_returns_long_polars(toy_long: pl.DataFrame) -> None:
    from vn1.models import fit_predict_stats

    out = fit_predict_stats(toy_long, h=8, season_length=52, freq="W-MON", n_jobs=1)
    assert isinstance(out, pl.DataFrame)
    assert {"unique_id", "ds", "Theta", "AutoETS", "SNaive"}.issubset(out.columns)
    # 8 forecast weeks × 4 series
    assert out["unique_id"].n_unique() == toy_long["unique_id"].n_unique()
    assert (out.group_by("unique_id").len()["len"] == 8).all()


# ---------------------------------------------------------------------
# LightGBM via mlforecast
# ---------------------------------------------------------------------
@pytest.mark.skipif(not (_have("mlforecast") and _have("lightgbm")), reason="mlforecast / lightgbm missing")
def test_lgbm_recursive_builds() -> None:
    from vn1.models import build_lgbm_recursive

    mlf = build_lgbm_recursive(n_estimators=50)
    assert "LGBM" in mlf.models


@pytest.mark.slow
@pytest.mark.skipif(not (_have("mlforecast") and _have("lightgbm")), reason="mlforecast / lightgbm missing")
def test_lgbm_fit_predict_on_toy(toy_long: pl.DataFrame) -> None:
    from vn1.models import fit_predict_lgbm

    out = fit_predict_lgbm(toy_long, h=8, n_estimators=50, freq="7d", season_length=52)
    assert isinstance(out, pl.DataFrame)
    assert {"unique_id", "ds", "LGBM"}.issubset(out.columns)
    assert (out.group_by("unique_id").len()["len"] == 8).all()
