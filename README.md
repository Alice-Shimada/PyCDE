# PySR Symbolic Regression Toolkit for d-log Integrals

A small, reusable toolkit extracted from the 3l4p1m workflow in the paper source.

It supports:

- config-driven PySR runs,
- automatic post-fit equation analysis,
- Mathematica-generated sample datasets from `f = \sum_i c_i \log(W_i)`,
- a one-click bash pipeline,
- switching between **log-only** and **log+sqrt** operator/nesting modes.

## Repository layout

```text
Release/
├── analyze_hall_of_fame.py
├── configs/
│   └── 3l4p1m_multivar.json
├── examples/
│   ├── 3l4p1m_function.m
│   └── 3l4p1m_function_modified_demo.m
├── generate_log_dataset.wl
├── pysr_experiment.py
├── reproduce_3l4p1m.py
├── run_log_pipeline.sh
└── source/
    ├── cord_1.txt
    ├── cord_2.txt
    ├── dinv_1.txt
    ├── dinv_2.txt
    ├── hall_of_fame.csv
    └── regression.py
```

Notes:

- `source/regression.py` is the original paper-side reference script for the 3l4p1m example.
- `source/*.txt` and `source/hall_of_fame.csv` are the minimal paper example data needed for the packaged demos here.

## Installation

This release was tested against the following `SymReg` environment snapshot:

- Python `3.12.9`
- PySR `1.5.9`
- JuliaCall `0.9.24`
- NumPy `2.2.4`
- pandas `2.2.3`
- SymPy `1.13.1`
- scikit-learn `1.6.1`
- click `8.1.8`
- typing-extensions `4.13.0`

### Option A: create the conda environment directly (recommended)

```bash
git clone <your-repo-url>
cd Release
conda env create -f environment.yml
conda activate SymReg
```

This is the easiest route for clone-and-run usage.

### Option B: create a conda env manually, then install with `requirements.txt`

```bash
git clone <your-repo-url>
cd Release
conda create -n SymReg python=3.12.9 pip -y
conda activate SymReg
pip install -r requirements.txt
```

### First PySR / Julia startup note

PySR requires a working Julia backend. On first import, `juliacall` / `pysr` may download or initialize Julia-related assets automatically.

### Mathematica / Wolfram requirement

`generate_log_dataset.wl` requires Mathematica or Wolfram Engine when you want to generate new datasets from a `.m` function file.

### Pipeline environment detection

The one-click script `run_log_pipeline.sh` assumes a conda environment with PySR installed.

By default it tries:

- `wolfram` from your `PATH`, otherwise `/usr/local/Wolfram/Mathematica/14.3/Executables/wolfram`
- `conda info --base` to locate `conda.sh`, otherwise `~/Software/miniconda3/etc/profile.d/conda.sh`
- environment name `SymReg`

You can override these with environment variables:

```bash
export WOLFRAM_BIN=/path/to/wolfram
export CONDA_SH=/path/to/conda.sh
export ENV_NAME=MySymRegEnv
```

### Quick smoke test after install

```bash
python reproduce_3l4p1m.py --skip-fit
python analyze_hall_of_fame.py --output-dir /tmp/pysr_analysis
```

## Operator / nesting modes

This repo supports two regressor modes.

### 1) Default: log-only mode

This matches the actual 3l4p1m `source/regression.py` setup:

```python
unary_operators = ["log"]
nested_constraints = {
    "log": {"log": 0}
}
```

### 2) Optional: log+sqrt mode

This matches the more general paper example for algebraic letters:

```python
unary_operators = ["log", "sqrt"]
nested_constraints = {
    "sqrt": {"sqrt": 0, "log": 0},
    "log": {"sqrt": 1, "log": 0},
}
```

To enable it in the generator / pipeline, add:

```bash
--allow-sqrt
```

## Quick start

### A. Reproduce the packaged paper example

From this directory:

```bash
python reproduce_3l4p1m.py --skip-fit
```

Small smoke run:

```bash
python reproduce_3l4p1m.py --preset smoke
```

More serious run:

```bash
python reproduce_3l4p1m.py --preset balanced
```

If `bumper=True` causes a Julia-side backend error on your machine, rerun with:

```bash
python reproduce_3l4p1m.py --preset balanced --no-bumper
```

## B. Analyze a hall-of-fame table directly

```bash
python analyze_hall_of_fame.py \
  --config configs/3l4p1m_multivar.json \
  --equations source/hall_of_fame.csv \
  --output-dir /tmp/pysr_analysis
```

This writes:

- `analysis_summary.json`
- `candidate_analysis.csv`

## C. Generate a new dataset from a Mathematica `.m` function

Example input file:

```mathematica
<|
  "Expression" -> (4/3) Log[1 - x] - (2/5) Log[1 - x - y] + (2/5) Log[y],
  "Variables" -> {x, y}
|>
```

Run:

```bash
wolfram -noprompt -script generate_log_dataset.wl \
  --function-file examples/3l4p1m_function.m \
  --output-dir /tmp/pysr_generated \
  --samples 200 \
  --working-precision 60 \
  --min-precision 30
```

Switch to log+sqrt mode if needed:

```bash
wolfram -noprompt -script generate_log_dataset.wl \
  --function-file examples/3l4p1m_function.m \
  --output-dir /tmp/pysr_generated \
  --allow-sqrt
```

Generated files include:

- `x.txt`, `y.txt`, `dx.txt`, `dy.txt`, ...
- `generated_config.json`
- `dataset_manifest.json`

Then run PySR on the generated config:

```bash
python pysr_experiment.py --config /tmp/pysr_generated/generated_config.json
```

## D. One-click pipeline

```bash
./run_log_pipeline.sh \
  --function-file examples/3l4p1m_function.m \
  --output-root /tmp/pysr_pipeline \
  --samples 200 \
  --niterations 50 \
  --no-bumper
```

This performs:

1. Mathematica dataset generation,
2. PySR regression,
3. automatic equation analysis.

Useful variants:

```bash
# data generation only
./run_log_pipeline.sh --function-file examples/3l4p1m_function.m --skip-fit

# enable log+sqrt operator mode
./run_log_pipeline.sh --function-file examples/3l4p1m_function.m --allow-sqrt --skip-fit

# modified coefficient demo
./run_log_pipeline.sh \
  --function-file examples/3l4p1m_function_modified_demo.m \
  --samples 200 \
  --niterations 50 \
  --no-bumper
```

## Outputs

A typical run directory contains:

- `resolved_config.json`
- `summary.json`
- `equations.csv`
- `analysis_summary.json`
- `candidate_analysis.csv`

## Current known issues / workarounds

### 1) `bumper=True` backend failure

On some PySR / Julia / DynamicExpressions combinations, `bumper=True` may fail with a Julia-side error such as:

```text
MethodError: objects of type Module are not callable
```

In that case, keep the paper-style template / operator constraints, but rerun with:

```bash
--no-bumper
```

### 2) `GLIBCXX_3.4.30 not found`

On some Linux machines, importing `pysr` / `juliacall` may fail with a `libstdc++` mismatch. A practical workaround is:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

For example:

```bash
conda activate SymReg
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python reproduce_3l4p1m.py --skip-fit
```

## Publishing note

This folder is structured to be a clean GitHub repo. Before publishing publicly, you may still want to add:

- a `LICENSE`
- a short `CITATION` note referencing the paper
- GitHub Actions / CI if you want automated checks
