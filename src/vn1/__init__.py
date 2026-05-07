"""VN1 forecasting experiments — package entry point.

Notebooks and scripts should import from this package:

    from vn1.data import load_full_data, process_wide_df
    from vn1.metrics import comp_loss
    from vn1.models import build_stats_forecaster, build_lgbm_forecaster
    from vn1.validation import rolling_origin_splits
    from vn1.ensemble import optimize_weights

Plotting style is opt-in (no side effects on import):

    from vn1.style import apply_style
    apply_style()
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vn1")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
