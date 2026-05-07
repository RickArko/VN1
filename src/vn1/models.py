"""Model factories and fit/predict adapters for the VN1 ensemble.

Two model families:

1. **Statistical bundle** (`statsforecast`): AutoTheta + AutoETS +
   SeasonalNaive, run jointly via :class:`statsforecast.StatsForecast`.
   `statsforecast` requires pandas at its API boundary, so we convert
   exactly there and return polars.

2. **LightGBM via mlforecast**: a recursive multi-step forecaster with
   the starter notebook's lag/rolling-window feature stack.
   ``mlforecast`` accepts polars natively, so the LazyFrame stays
   lazy until ``fit``. :func:`build_lgbm_cv` exposes the LightGBMCV
   loop for selecting ``num_iterations`` via early stopping; the
   ``MLForecast.from_cv`` → ``fit(full_data)`` → ``predict`` pattern
   from the starter notebook is wrapped in :func:`fit_predict_lgbm`.

Every public function takes a ``LazyFrame`` and returns a long
``DataFrame`` with at least ``[id_col, time_col, <model alias(es)>]``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

__all__ = [
    "build_lgbm_cv",
    "build_lgbm_recursive",
    "build_stats_forecaster",
    "fit_predict_lgbm",
    "fit_predict_stats",
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _to_pandas_long(
    long: pl.LazyFrame | pl.DataFrame,
    *,
    id_col: str,
    time_col: str,
    target_col: str,
) -> Any:  # `pd.DataFrame` — typed as Any to avoid module-level import.
    """Materialize a polars long frame as a pandas DataFrame at a library boundary.

    Only the three Nixtla-canonical columns are kept; downstream
    statsforecast / mlforecast code paths don't need anything else.
    """
    eager = long.collect() if isinstance(long, pl.LazyFrame) else long
    return eager.select(id_col, time_col, target_col).to_pandas()


# ----------------------------------------------------------------------
# Statistical bundle: Theta + AutoETS + SeasonalNaive
# ----------------------------------------------------------------------
def build_stats_forecaster(
    *,
    season_length: int = 52,
    freq: str = "W-MON",
    n_jobs: int = -1,
):
    """Build a `StatsForecast` with three weekly baselines.

    Aliases (used as column names in the output frame):
        Theta     — `AutoTheta(season_length)` — captures level + trend +
                    seasonality with auto-selected variant.
        AutoETS   — `AutoETS(season_length)` — exponential smoothing
                    auto-selected over (A, M, N) error/trend/seasonality.
        SNaive    — `SeasonalNaive(season_length)` — last-season-ago value;
                    a strong reference for stable retail series.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, AutoTheta, SeasonalNaive

    return StatsForecast(
        models=[
            AutoTheta(season_length=season_length, alias="Theta"),
            AutoETS(season_length=season_length, alias="AutoETS"),
            SeasonalNaive(season_length=season_length, alias="SNaive"),
        ],
        freq=freq,
        n_jobs=n_jobs,
    )


def fit_predict_stats(
    long: pl.LazyFrame | pl.DataFrame,
    *,
    h: int,
    season_length: int = 52,
    freq: str = "W-MON",
    n_jobs: int = -1,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pl.DataFrame:
    """Fit Theta + AutoETS + SNaive and return long-format polars predictions.

    Output schema: ``[id_col, time_col, "Theta", "AutoETS", "SNaive"]``.
    Fits on the full training frame and forecasts ``h`` steps forward.
    """
    sf = build_stats_forecaster(season_length=season_length, freq=freq, n_jobs=n_jobs)
    pdf = _to_pandas_long(long, id_col=id_col, time_col=time_col, target_col=target_col)
    fcst = sf.forecast(df=pdf, h=h)
    out = pl.from_pandas(fcst.reset_index() if "unique_id" not in fcst.columns else fcst)
    return out.select(id_col, time_col, "Theta", "AutoETS", "SNaive")


# ----------------------------------------------------------------------
# LightGBM via mlforecast — feature stack
# ----------------------------------------------------------------------
def _default_lag_transforms(season_length: int = 52):
    """Lag-window feature stack from the LightGBM starter notebook.

    Mirrors that notebook 1:1 so an ensemble built from this module is
    comparable to standalone runs of `LightGbmStarter.ipynb`.
    """
    from mlforecast.lag_transforms import ExpandingMean, RollingMean, RollingStd

    rmean4 = RollingMean(window_size=4, min_samples=1)
    rstd4 = RollingStd(window_size=4, min_samples=2)
    return {
        1: [
            ExpandingMean(),
            rmean4,
            rstd4,
            RollingMean(window_size=13, min_samples=1),
            RollingMean(window_size=26, min_samples=1),
            RollingMean(window_size=season_length, min_samples=1),
        ],
        **{k: [rmean4, rstd4, RollingMean(window_size=season_length - k, min_samples=1)] for k in (4, 8, 13)},
    }


def build_lgbm_recursive(
    *,
    freq: str = "7d",
    season_length: int = 52,
    lags: tuple[int, ...] = (13, 52),
    n_estimators: int = 500,
    learning_rate: float = 0.2,
    num_leaves: int = 128,
    num_threads: int = 4,
    random_state: int = 42,
):
    """Build an `MLForecast` with a single recursive LightGBM model.

    Use this in the *inner* CV loop where speed matters and an honest
    comparison across folds is required (no nested CV). For the final
    submission, prefer the :func:`build_lgbm_cv` flow — it tunes
    `num_iterations` via early stopping.
    """
    from lightgbm import LGBMRegressor
    from mlforecast import MLForecast
    from mlforecast.target_transforms import LocalRobustScaler

    return MLForecast(
        models={
            "LGBM": LGBMRegressor(
                verbosity=-1,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                n_estimators=n_estimators,
                num_threads=num_threads,
                random_state=random_state,
            ),
        },
        freq=freq,
        lags=list(lags),
        lag_transforms=_default_lag_transforms(season_length),
        target_transforms=[LocalRobustScaler(scale="iqr")],
        date_features=["year", "month"],
        num_threads=num_threads,
    )


def build_lgbm_cv(
    *,
    freq: str = "7d",
    season_length: int = 52,
    lags: tuple[int, ...] = (13, 52),
    learning_rate: float = 0.2,
    num_leaves: int = 128,
    num_threads: int = 4,
):
    """Build a `LightGBMCV` for selecting `num_iterations` via early stopping.

    Pair with `MLForecast.from_cv(lgb_cv).fit(full_data)` to refit on the
    full training set with the iteration count the CV chose.
    """
    from mlforecast.lgb_cv import LightGBMCV
    from mlforecast.target_transforms import LocalRobustScaler

    return LightGBMCV(
        freq=freq,
        lags=list(lags),
        lag_transforms=_default_lag_transforms(season_length),
        target_transforms=[LocalRobustScaler(scale="iqr")],
        date_features=["year", "month"],
        num_threads=num_threads,
    )


def fit_predict_lgbm(
    long: pl.LazyFrame | pl.DataFrame,
    *,
    h: int,
    n_estimators: int = 500,
    use_cv: bool = False,
    cv_n_windows: int = 4,
    cv_num_iterations: int = 10_000,
    cv_early_stopping_evals: int = 5,
    cv_metric: Callable[..., float] | None = None,
    cv_params: dict[str, Any] | None = None,
    freq: str = "7d",
    season_length: int = 52,
    lags: tuple[int, ...] = (13, 52),
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pl.DataFrame:
    """Fit LightGBM via mlforecast and forecast `h` steps.

    `use_cv=False` (default) is the fast path: a recursive LightGBM with
    fixed `n_estimators`. Use this in the ensemble CV loop.

    `use_cv=True` runs `LightGBMCV` first to choose `num_iterations`,
    then refits via `MLForecast.from_cv(...)`. Slower but stronger; this
    is the submission path.

    Output schema: ``[id_col, time_col, "LGBM"]``.
    """
    lf = long.lazy() if isinstance(long, pl.DataFrame) else long
    sales = lf.select(id_col, time_col, target_col).collect()

    if use_cv:
        from mlforecast import MLForecast

        lgb_cv = build_lgbm_cv(
            freq=freq,
            season_length=season_length,
            lags=lags,
            num_threads=4,
        )
        lgb_cv.fit(
            sales.to_pandas(),  # LightGBMCV historically pandas; cheaper than lazy here
            n_windows=cv_n_windows,
            h=h,
            dropna=False,
            metric=cv_metric,
            num_iterations=cv_num_iterations,
            params=cv_params or {"verbosity": -1, "learning_rate": 0.2, "num_leaves": 128},
            early_stopping_evals=cv_early_stopping_evals,
        )
        mlf = MLForecast.from_cv(lgb_cv)
        mlf.fit(sales, dropna=False)
        preds = mlf.predict(h)
    else:
        mlf = build_lgbm_recursive(
            freq=freq,
            season_length=season_length,
            lags=lags,
            n_estimators=n_estimators,
        )
        mlf.fit(sales, dropna=False)
        preds = mlf.predict(h)

    if isinstance(preds, pl.DataFrame):
        return preds.select(id_col, time_col, pl.col("LGBM"))
    return pl.from_pandas(preds).select(id_col, time_col, pl.col("LGBM"))
