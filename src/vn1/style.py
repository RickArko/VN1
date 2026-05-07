"""Notebook plotting style — opt-in via :func:`apply_style`."""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler

_COLORS = cycler(color=["black", "003DFD", "b512b8", "11a9ba", "0d780f", "f77f07", "ba0f0f"])
_STYLE = {
    "font.family": "sans serif",
    "axes.edgecolor": "black",
    "axes.grid": True,
    "axes.labelcolor": "#333333",
    "axes.labelweight": 600,
    "axes.linewidth": 1,
    "axes.prop_cycle": _COLORS,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.bottom": False,
    "axes.spines.left": False,
    "grid.color": "#dedede",
    "legend.frameon": False,
    "lines.linewidth": 1.3,
    "xtick.color": "#333333",
    "xtick.labelsize": "small",
    "ytick.color": "#333333",
    "ytick.labelsize": "small",
    "xtick.bottom": False,
    "figure.figsize": (24, 6),
    "figure.titlesize": 18,
}

__all__ = ["apply_style"]


def apply_style() -> None:
    """Apply the VN1 matplotlib style. Call once near the top of a notebook."""
    mpl.rcParams.update(_STYLE)
