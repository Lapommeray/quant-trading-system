# Python Dependency Management

This repository is Python-first and uses `pip` + requirements files for environment setup.

## Dependency Files

- `requirements.txt`  
  Core runtime + test dependencies for the main repository.

- `requirements_qc.txt`  
  QuantConnect/dashboard extras layered on top of `requirements.txt`.

- `requirements_institutional.txt`  
  Institutional/research extras layered on top of `requirements.txt`.

- `pyproject.toml`  
  Packaging metadata and base install dependencies for `quant-trading-system`.

## Installation Options

### 1) Core setup (recommended default)

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Core + QuantConnect extras

```bash
pip install -r requirements_qc.txt
```

### 3) Core + institutional extras

```bash
pip install -r requirements_institutional.txt
```

### 4) Package install from `pyproject.toml`

```bash
pip install .
```

## Notes

- Extra requirement files are intentionally layered with `-r requirements.txt` to reduce drift.
- Historical subfolders (for archived variants) may contain their own requirements files; treat them as isolated snapshots.
