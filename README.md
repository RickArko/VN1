# VN1 Forecasting Experiments

Forecasting + inventory-optimization experiments on the
[VN1 Challenge](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge)
weekly demand data. Includes a LightGBM (`mlforecast`) starter and an AutoETS
(`statsforecast`) baseline.

## Requirements

- Python 3.10 (pinned in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) for environment + dependency management
- Make (Windows users: run under WSL — see `Makefile` header)

## Quickstart

```bash
make install
```

That single target does two things:

1. **`make sync`** — `uv sync` resolves `pyproject.toml` against `uv.lock`,
   creates `.venv/` if needed, and installs both runtime and `dev`
   dependencies (which include `jupyter` + `ipykernel`).
2. **`make kernel`** — registers the project venv as a Jupyter kernel via
   `python -m ipykernel install --user`, exposing it as:
   - kernel name: `vn1`
   - display name: **VN1 (uv)**

After `make install`, any Jupyter front-end on your machine (JupyterLab,
classic Notebook, VS Code, Cursor, PyCharm) will see **VN1 (uv)** in the
kernel picker and run notebooks against this project's `.venv`.

## Launching notebooks

```bash
make lab        # JupyterLab
make notebook   # classic Jupyter Notebook
```

Both run via `uv run`, so you don't need to manually activate the venv. If
you launch from an external IDE instead, just pick the **VN1 (uv)** kernel.

The two starter notebooks at the repo root are:

- `LightGbmStarter.ipynb` — `mlforecast` + LightGBM with rolling/expanding
  lag transforms, robust scaling, CV-driven early stopping, and zero-variance
  naive overrides.
- `AutoETS.ipynb` — `statsforecast` AutoETS baseline.

## All Make targets

```text
make install   Create venv, install deps, register Jupyter kernel
make sync      Resolve and install dependencies into .venv
make lock      Refresh uv.lock without installing
make upgrade   Upgrade all dependencies to latest compatible versions
make kernel    (Re)register the project venv as a Jupyter kernel
make lab       Launch JupyterLab using the project venv
make notebook  Launch classic Jupyter Notebook using the project venv
make fmt       Format src/ with isort + black
make lint      Check formatting without modifying files
make clean     Remove caches and build artifacts
make nuke      clean + remove .venv
```

Run `make help` to print this list from the Makefile itself.

## Re-registering or removing the kernel

If you rename the venv, change Python versions, or want to clean up:

```bash
make kernel                       # re-register (idempotent)
jupyter kernelspec list           # show installed kernels
jupyter kernelspec uninstall vn1  # remove the VN1 kernel
```

## Project layout

```text
.
├── LightGbmStarter.ipynb   # mlforecast + LightGBM pipeline
├── AutoETS.ipynb           # statsforecast AutoETS baseline
├── src/
│   ├── data.py             # load_full_data + wide→long reshaping
│   └── style.py
├── Makefile                # uv + ipykernel workflow
├── pyproject.toml          # deps; dev group has jupyter + ipykernel
├── uv.lock
└── .python-version         # 3.10
```
