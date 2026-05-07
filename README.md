# VN1 Forecasting Experiments

[![CI](https://github.com/RickArko/VN1/actions/workflows/ci.yml/badge.svg)](https://github.com/RickArko/VN1/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10-blue)
![uv](https://img.shields.io/badge/managed%20by-uv-261230)

A reproducible solution for the [VN1 Forecasting Accuracy Challenge](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge):
weekly retail demand at the **Client × Warehouse × Product** grain, 13-week
horizon. The repo ships an installable Python package (`vn1`) with a
Theta + AutoETS + SeasonalNaive + LightGBM ensemble whose weights are
learned out-of-sample against the official competition metric.

---

## Quickstart (30 seconds)

```bash
git clone git@github.com:RickArko/VN1.git
cd VN1
make install
```

`make install` creates the venv, installs the package + dev + notebook
deps via `uv`, and registers a Jupyter kernel called **VN1 (uv)**.
Now pick how you want to work:

| You prefer…                    | Do this                                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------------------------ |
| **VS Code / Cursor**           | `code .`  → click *Install* on the recommended extensions → open `EnsembleSubmission.ipynb`      |
| **JupyterLab in the browser**  | `make lab`                                                                                       |
| **Classic Jupyter Notebook**   | `make notebook`                                                                                  |

---

## Don't have `uv` or `make` yet?

```bash
# uv (one-line installer):
curl -LsSf https://astral.sh/uv/install.sh | sh

# make on WSL Ubuntu:
sudo apt install build-essential
```

`uv` will fetch Python 3.10 automatically — you don't need to install
Python yourself. Windows users: please run under WSL.

---

## What the submission does

`EnsembleSubmission.ipynb` is the canonical entry point. Pipeline:

1. **Load + preprocess** — fetch Phase 0/1 sales CSVs from datasource.ai,
   unpivot to `(unique_id, ds, y)` long format, trim leading zeros (pre-launch padding).
2. **Cross-validate four models on identical folds**:
   - **AutoTheta** — captures level + trend + auto-selected seasonality
   - **AutoETS** — exponential smoothing, auto-selected (E, T, S) variant
   - **SeasonalNaive** — strong reference for stable retail series
   - **LightGBM** via `mlforecast` — cross-series + lag interactions
3. **Optimize a non-negative simplex of weights** against the official
   competition metric `(|Σ err| + Σ|err|) / Σ y_true` on out-of-sample
   CV predictions (no in-sample leakage).
4. **Refit each model on the full training history**, blend with the
   CV-tuned weights, apply a zero-variance naive override for locked-tail
   SKUs, and write the wide-format submission CSV.

The metric penalizes both cumulative bias *and* dispersion equally, which
rewards model diversity — one model overshoots, another undershoots, the
blend cancels. The four bases were chosen for complementary failure modes.

---

## Package layout

```text
.
├── EnsembleSubmission.ipynb    # canonical submission flow
├── LightGbmStarter.ipynb       # standalone LightGBM exploration
├── AutoETS.ipynb               # standalone AutoETS exploration
├── src/vn1/                    # installable package (`pip install vn1`)
│   ├── data.py                 # polars-lazy loaders + preprocess
│   ├── metrics.py              # comp_loss + per-series breakdown
│   ├── validation.py           # rolling-origin CV helpers
│   ├── models.py               # Theta/ETS/SNaive bundle + LightGBM factories
│   ├── ensemble.py             # SLSQP weight optimizer
│   ├── submission.py           # zero-variance override + wide-CSV writer
│   └── style.py                # opt-in matplotlib rcParams
├── tests/
│   ├── smoke/                  # imports, package metadata
│   ├── unit/                   # pure-function tests
│   └── integration/            # model fit/predict on toy data
├── .github/workflows/ci.yml    # quality + tests + wheel build
├── .vscode/                    # interpreter, formatter, tasks, extensions
├── Makefile                    # all developer entry points
└── pyproject.toml              # deps, ruff, mypy, pytest, hatchling build
```

`vn1` is **polars-first**. Pandas appears only at the `statsforecast`
library boundary (it requires pandas) and inside `LightGBMCV` where the
upstream API is historically pandas. Everything else stays lazy.

---

## Programmatic use

The full pipeline is accessible from any script:

```python
from vn1.data import load_full_data, process_wide_df, trim_leading_zeros
from vn1.validation import cross_validate, evaluate_predictions
from vn1.models import fit_predict_stats, fit_predict_lgbm
from vn1.ensemble import optimize_weights, apply_weights
from vn1.submission import apply_zero_variance_override, write_submission

sales = trim_leading_zeros(process_wide_df(load_full_data())).collect()

cv_stats = cross_validate(lambda tr, h: fit_predict_stats(tr, h=h), sales, h=13, n_windows=4)
cv_lgbm  = cross_validate(lambda tr, h: fit_predict_lgbm(tr, h=h), sales, h=13, n_windows=4)
cv = cv_stats.join(cv_lgbm.select("unique_id", "ds", "fold", "LGBM"), on=["unique_id", "ds", "fold"])

weights = optimize_weights(cv, pred_cols=["Theta", "AutoETS", "SNaive", "LGBM"])

# refit on full data, blend, override, write
stats_full = fit_predict_stats(sales, h=13)
lgbm_full = fit_predict_lgbm(sales, h=13, use_cv=True)
preds = apply_weights(stats_full.join(lgbm_full, on=["unique_id", "ds"]), weights, out_col="y_hat").collect()
write_submission(apply_zero_variance_override(preds, sales).collect(), "artifacts/submission.csv")
```

---

## Make targets

```text
make install        Create venv, install deps + dev + notebook groups, register Jupyter kernel
make sync           Resolve and install dependencies into .venv
make lock           Refresh uv.lock without installing
make upgrade        Upgrade all dependencies to latest compatible versions
make kernel         (Re)register the project venv as a Jupyter kernel
make lab            Launch JupyterLab using the project venv
make notebook       Launch classic Jupyter Notebook using the project venv

make fmt            Format with ruff (replaces black + isort)
make lint           Lint without modifying files
make typecheck      mypy on src/vn1

make test           Full pytest suite (smoke + unit + integration; skips `slow`)
make test-smoke     Smoke tier — every public surface imports
make test-unit      Unit tier — pure-function tests
make test-integration   Integration tier — model fit/predict + CV pipeline
make test-fast      Smoke + unit only (no integration, no `slow`)
make cov            Suite with coverage (terminal + htmlcov/)

make check          Lint + types + tests (CI entry point)
make build          Build sdist + wheel into dist/
make verify-build   Build, install into a clean venv, smoke-test imports

make clean          Remove caches and build artifacts
make nuke           clean + remove .venv
```

`make help` prints this list from the Makefile itself.

---

## Continuous integration

GitHub Actions runs three jobs on every PR + push to main
(`.github/workflows/ci.yml`):

1. **quality** — ruff lint + ruff format check + mypy on `src/vn1`
2. **test** — pytest with coverage, skipping `slow`-marked tests
3. **build** — `uv build`, then install the wheel into a clean venv and
   smoke-test the public API

Slow tests (real `statsforecast` / `lightgbm` fits on the toy fixture)
run locally via `make test`, not in CI — they take ~10s each and don't
catch anything CI's faster tests miss.

---

## VS Code / Cursor setup

The repo ships a `.vscode/` folder so the editor is configured out of the box:

- `extensions.json` — prompts to install Python + Jupyter + Ruff extensions
- `settings.json` — auto-points the interpreter at `.venv/bin/python`,
  enables ruff format on save, sets sensible notebook defaults
- `tasks.json` — every `make` target shows up under
  *Ctrl/Cmd+Shift+P → Tasks: Run Task*

After `make install`: `code .`, install recommended extensions when
prompted, open `EnsembleSubmission.ipynb`, pick **VN1 (uv)** in the
kernel dropdown.

---

## Troubleshooting

**Kernel hangs in the browser.**  Make sure `ipykernel<7` is installed
(`uv run python -c "import ipykernel; print(ipykernel.__version__)"`).
Then `make kernel` to re-register and restart the Jupyter server.

**`AttributeError: can't set attribute 'feature_names_in_'` from LightGBM.**
That's a `lightgbm 4.5+` ↔ `scikit-learn` clash. The project pins
`lightgbm<4.5` to avoid it; if you bumped it, revert and `make sync`.

**Re-registering / removing the kernel.**

```bash
make kernel                       # (re)register, idempotent
jupyter kernelspec list           # show installed kernels
jupyter kernelspec uninstall vn1  # remove the VN1 kernel
```

**Wipe and start over.**

```bash
make nuke && make install
```
