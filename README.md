# VN1 Forecasting Experiments

Forecasting + inventory-optimization experiments on the
[VN1 Challenge](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge)
weekly demand data. Includes a LightGBM (`mlforecast`) starter and an AutoETS
(`statsforecast`) baseline.

---

## Quickstart (30 seconds)

```bash
git clone git@github.com:RickArko/VN1.git
cd VN1
make install
```

That's it. `make install` creates the virtual environment, installs every
dependency, and registers a Jupyter kernel called **VN1 (uv)**.

Now pick how you want to work:

| You prefer…                    | Do this                                                  |
| ------------------------------ | -------------------------------------------------------- |
| **VS Code / Cursor**           | `code .`  → click *Yes* on the recommended extensions prompt → open any `.ipynb` and pick **VN1 (uv)** in the kernel dropdown |
| **JupyterLab in the browser**  | `make lab`                                               |
| **Classic Jupyter Notebook**   | `make notebook`                                          |

---

## Don't have `uv` or `make` yet?

**`uv`** (one line, official installer):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via Homebrew: `brew install uv`. Windows users: please run this project
under WSL — see the note at the top of the `Makefile`.

**`make`** is preinstalled on macOS and Linux. On WSL Ubuntu:
`sudo apt install build-essential`.

You don't need to install Python yourself — `uv` will fetch the version
pinned in `.python-version` (3.10) automatically.

---

## What `make install` actually does

```
make install  =  make sync  +  make kernel
```

1. **`uv sync`** — resolves `pyproject.toml` against `uv.lock`, creates
   `.venv/`, and installs every runtime + dev dependency (which includes
   `jupyter` and `ipykernel`).
2. **`ipykernel install`** — registers the venv as a Jupyter kernel:
   - kernel name: `vn1`
   - display name: **VN1 (uv)**

After this, every Jupyter front-end on your machine (JupyterLab, classic
Notebook, VS Code, Cursor, PyCharm) will see **VN1 (uv)** in the kernel
picker and run notebooks against this project's `.venv`.

---

## VS Code / Cursor setup

The repo ships a `.vscode/` folder so the editor is configured out of the box:

- **`.vscode/extensions.json`** — VS Code prompts you to install the
  recommended Python + Jupyter extensions on first open.
- **`.vscode/settings.json`** — points the Python interpreter at
  `.venv/bin/python`, enables black + isort on save (line length 120, matching
  `pyproject.toml`), and sets sensible notebook defaults.
- **`.vscode/tasks.json`** — exposes every `make` target via
  *Ctrl/Cmd+Shift+P → Tasks: Run Task* so you don't have to memorize them.

If you've never opened the repo before:

1. `make install` (creates the venv the editor will discover).
2. `code .`
3. Click **Install** when prompted to add the recommended extensions.
4. Open `LightGbmStarter.ipynb`, pick **VN1 (uv)** in the kernel dropdown
   (top-right of the notebook), and run cells.

---

## Browser Jupyter

```bash
make lab        # JupyterLab
make notebook   # classic Jupyter Notebook
```

Both run via `uv run`, so you don't need to manually activate the venv.
When the page opens, pick the **VN1 (uv)** kernel.

---

## Notebooks in the repo

- **`LightGbmStarter.ipynb`** — `mlforecast` + LightGBM with rolling/expanding
  lag transforms, robust scaling, CV-driven early stopping, and a zero-variance
  naive override.
- **`AutoETS.ipynb`** — `statsforecast` AutoETS baseline.

Shared data loading lives in `src/data.py` (`load_full_data`,
`process_wide_df`).

---

## Make targets

| Target          | What it does                                           |
| --------------- | ------------------------------------------------------ |
| `make install`  | Create venv, install deps, register Jupyter kernel     |
| `make sync`     | Resolve and install dependencies into `.venv`          |
| `make lock`     | Refresh `uv.lock` without installing                   |
| `make upgrade`  | Upgrade all dependencies to latest compatible versions |
| `make kernel`   | (Re)register the project venv as a Jupyter kernel      |
| `make lab`      | Launch JupyterLab using the project venv               |
| `make notebook` | Launch classic Jupyter Notebook using the project venv |
| `make fmt`      | Format `src/` with isort + black                       |
| `make lint`     | Check formatting without modifying files               |
| `make clean`    | Remove caches and build artifacts                      |
| `make nuke`     | `clean` + remove `.venv`                               |

Run `make help` to print this list from the Makefile itself.

---

## Project layout

```text
.
├── LightGbmStarter.ipynb   # mlforecast + LightGBM pipeline
├── AutoETS.ipynb           # statsforecast AutoETS baseline
├── src/
│   ├── data.py             # load_full_data + wide→long reshaping
│   └── style.py
├── .vscode/                # interpreter, formatter, tasks, extensions
├── Makefile                # uv + ipykernel workflow
├── pyproject.toml          # deps + tooling config (black, isort)
├── uv.lock
└── .python-version         # 3.10
```

---

## Troubleshooting

**Kernel hangs in the browser / "stuck" connecting.**
Make sure `ipykernel<7` is installed (`uv run python -c "import ipykernel; print(ipykernel.__version__)"`); ipykernel 7 is currently incompatible with the bundled `jupyter_client`/`notebook` versions. Then `make kernel` to re-register and restart the Jupyter server.

**`AttributeError: can't set attribute 'feature_names_in_'` from LightGBM.**
That's a `lightgbm` 4.5+ vs `scikit-learn` 1.6+ clash. The project pins `lightgbm<4.5` to avoid it; if you bumped it, revert and `make sync`.

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
