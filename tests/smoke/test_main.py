"""Smoke: `main.py` must compile and resolve every imported name.

This catches the most common breakage: someone renames a public
function in `vn1.*` and forgets to update `main.py`. We don't actually
run the pipeline (it'd download competition data and fit real models);
we just compile the file and probe attributes on the imported modules.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = REPO_ROOT / "main.py"


def test_main_py_exists() -> None:
    assert MAIN_PY.is_file(), f"{MAIN_PY} not found"


def test_main_py_compiles() -> None:
    """Parses cleanly — catches syntax errors without executing the script."""
    source = MAIN_PY.read_text(encoding="utf-8")
    ast.parse(source, filename=str(MAIN_PY))


def test_main_py_imports_resolve() -> None:
    """Every `from vn1.X import a, b` in main.py must resolve in the installed package."""
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vn1"):
            mod = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(mod, alias.name), (
                    f"main.py imports {alias.name!r} from {node.module!r} "
                    f"but the module has no such attribute"
                )
