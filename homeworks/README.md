# ELEC5340 - Compressed Sensing and Sparse Recovery

Homeworks for ELEC5340 - Compressed Sensing.

## Overview

This repository contains the homework notebooks for ELEC5340. The recommended workflow is:

1. Manage the Python environment with `uv`.
2. Work inside the notebooks in `notebooks/`.
3. Render `.ipynb` files to PDF with Quarto.

## Prerequisites

Install the following tools before working in the repository:

- Python 3.13 or newer
- `uv` for Python environment management
- Quarto CLI for rendering notebooks
- A TeX distribution for PDF output, such as MiKTeX or TinyTeX

Verify your setup from PowerShell:

```powershell
python --version
uv --version
quarto --version
quarto check
```

## Environment Setup

This project is configured for Python 3.13 as declared in [.python-version](.python-version). Use `uv` to create and manage the environment.

From the repository root:

```powershell
uv sync
```

If you want to activate the virtual environment manually in PowerShell:

```powershell
& .\.venv\Scripts\Activate.ps1
```

If script execution is blocked by PowerShell, run this once for your user account:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### macOS / Linux

On macOS and Linux, use the same `uv sync` workflow and activate the environment with a POSIX shell:

```bash
uv sync
source .venv/bin/activate
```

## Project Structure

```text
homeworks/
├── data/        # Input data files used by the notebooks
├── notebooks/   # Homework notebooks and generated artifacts
├── pyproject.toml
├── uv.lock
└── README.md
```

The main notebooks are:

- [notebooks/HW1.ipynb](notebooks/HW1.ipynb)
- [notebooks/HW2.ipynb](notebooks/HW2.ipynb)

## Working With Notebooks

Open the notebooks in VS Code and run them with the Python kernel from the `uv` environment.

If you need a Python script version of the first homework, use [notebooks/HW1.py](notebooks/HW1.py).

## Render To PDF

Use Quarto to render a notebook to PDF:

```powershell
quarto render notebooks/HW2.ipynb --to pdf
```

You can render the other notebook the same way:

```powershell
quarto render notebooks/HW1.ipynb --to pdf
```

If you want to preview HTML output instead of PDF:

```powershell
quarto render notebooks/HW2.ipynb --to html
```

The same Quarto commands work on macOS and Linux after activating the environment.

## Dependency Management

The repository uses `uv.lock` to keep dependencies reproducible. Common commands:

```powershell
uv sync
uv add <package>
uv lock
```

## Troubleshooting

- If `uv` is not recognized, reinstall it or restart PowerShell after installation.
- If `quarto render` fails on PDF output, run `quarto check` and confirm that a TeX distribution is installed.
- If PowerShell blocks `Activate.ps1`, use `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- If a notebook cannot import a package, run `uv sync` again to restore the environment.

## Recommended Workflow

1. Run `uv sync`.
2. Activate the environment if needed.
3. Edit the notebook in `notebooks/`.
4. Render with `quarto render notebooks/HW2.ipynb --to pdf`.
5. Repeat for the other homework notebook as needed.