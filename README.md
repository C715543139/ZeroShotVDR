# ZeroShotVDR

Research project for zero-shot visual document retrieval with ColPali.

## Setup

Run the following commands from the repository root:

```powershell
conda create -n zeroshotvdr python=3.10 -y
conda activate zeroshotvdr
uv sync
.\.venv\Scripts\Activate.ps1
```

## Why This Works

`uv sync` does two separate things in this project:

1. It creates `.venv` and installs third-party dependencies.
2. It installs the current repository in editable mode, so code under `src/` is importable from the project virtual environment.

The compatibility patch for the current `colpali-engine + transformers + peft`
combination lives in the repository root as `sitecustomize.py`.

When you activate `.venv` and start `python` from the repository root, Python's
standard startup process imports `sitecustomize.py` automatically because the
current working directory is on `sys.path`. This is what makes the compatibility
patch apply with no extra command.

So the minimal stable usage path is:

```powershell
conda activate zeroshotvdr
uv sync
.\.venv\Scripts\Activate.ps1
python
```

The important constraint is that `python` should be launched from the repository
root and from the project `.venv`.