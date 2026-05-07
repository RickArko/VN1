"""Smoke: every public surface must import without side-effects."""

from __future__ import annotations

import importlib

import pytest

PUBLIC_MODULES = [
    "vn1",
    "vn1.data",
    "vn1.metrics",
    "vn1.validation",
    "vn1.models",
    "vn1.ensemble",
    "vn1.submission",
    "vn1.style",
]


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_module_importable(module: str) -> None:
    importlib.import_module(module)


def test_top_level_exports_version() -> None:
    import vn1

    assert isinstance(vn1.__version__, str)
    assert vn1.__version__  # non-empty


def test_data_constants_match_nixtla_schema() -> None:
    from vn1.data import ID_COL, ID_COLS, TARGET_COL, TIME_COL

    assert ID_COL == "unique_id"
    assert TIME_COL == "ds"
    assert TARGET_COL == "y"
    assert ID_COLS == ("Client", "Warehouse", "Product")
