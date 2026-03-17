#!/usr/bin/env python3
"""Reproduce the PySR stage of arXiv:2510.10099 (3l4p1m example).

This is a paper-specific wrapper around the reusable config-driven runner in
``pysr_experiment.py``. It keeps the convenient preset-based CLI for the
published example while delegating the generic data/template/model plumbing to
shared code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from pysr_experiment import (
    DEFAULT_RUNS_DIR,
    RegressionDataset,
    apply_common_overrides,
    load_dataset,
    load_experiment_config,
    run_experiment,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "3l4p1m_multivar.json"

x_sym, y_sym = sp.symbols("x y")
PAPER_EXPR_RAW = (
    sp.Rational(14, 15) * sp.log(1 - x_sym)
    - sp.Rational(2, 5) * sp.log((1 - x_sym - y_sym) / (1 - x_sym))
    + sp.Rational(2, 5) * sp.log(y_sym)
)
PAPER_EXPR_SIMPLIFIED = sp.expand_log(PAPER_EXPR_RAW, force=True)
ENTRY_LETTERS = ["1 - x", "1 - x - y", "y"]
FULL_ALPHABET = ["x", "1 - x", "y", "1 - y", "x + y", "1 - x - y"]

PRESETS: dict[str, dict[str, int]] = {
    "smoke": {
        "procs": 4,
        "populations": 8,
        "population_size": 30,
        "niterations": 5,
        "maxsize": 30,
    },
    "balanced": {
        "procs": 8,
        "populations": 24,
        "population_size": 80,
        "niterations": 50,
        "maxsize": 80,
    },
    "article": {
        "procs": 40,
        "populations": 120,
        "population_size": 120,
        "niterations": 500,
        "maxsize": 100,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="balanced")

    parser.add_argument("--procs", type=int, default=None)
    parser.add_argument("--populations", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--niterations", type=int, default=None)
    parser.add_argument("--maxsize", type=int, default=None)

    parser.add_argument("--random-seed", type=int, default=20260317)
    parser.add_argument(
        "--skip-fit",
        action="store_true",
        help="Only run the reference checks (no PySR search).",
    )
    bumper_group = parser.add_mutually_exclusive_group()
    bumper_group.add_argument(
        "--bumper",
        action="store_true",
        help="Enable DynamicExpressions bumper backend.",
    )
    bumper_group.add_argument(
        "--no-bumper",
        action="store_true",
        help="Disable DynamicExpressions bumper backend.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use serial deterministic evolution for reproducibility (slower).",
    )
    return parser.parse_args()


def resolve_hyperparams(args: argparse.Namespace) -> dict[str, int]:
    params = PRESETS[args.preset].copy()
    for key in ["procs", "populations", "population_size", "niterations", "maxsize"]:
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    return params


def evaluate_reference(dataset: RegressionDataset) -> dict[str, Any]:
    x_values = dataset.raw_columns["x"]
    y_values = dataset.raw_columns["y"]
    dx_values = dataset.raw_columns["dx"]
    dy_values = dataset.raw_columns["dy"]

    dfdx = sp.lambdify((x_sym, y_sym), sp.diff(PAPER_EXPR_SIMPLIFIED, x_sym), "numpy")
    dfdy = sp.lambdify((x_sym, y_sym), sp.diff(PAPER_EXPR_SIMPLIFIED, y_sym), "numpy")

    pred_dx = np.asarray(dfdx(x_values, y_values), dtype=np.float64)
    pred_dy = np.asarray(dfdy(x_values, y_values), dtype=np.float64)

    return {
        "paper_expr_raw": str(PAPER_EXPR_RAW),
        "paper_expr_simplified": str(PAPER_EXPR_SIMPLIFIED),
        "entry_letters": ENTRY_LETTERS,
        "full_alphabet": FULL_ALPHABET,
        "max_abs_dx_err": float(np.max(np.abs(pred_dx - dx_values))),
        "max_abs_dy_err": float(np.max(np.abs(pred_dy - dy_values))),
        "rmse_dx": float(np.sqrt(np.mean((pred_dx - dx_values) ** 2))),
        "rmse_dy": float(np.sqrt(np.mean((pred_dy - dy_values) ** 2))),
    }


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.data_dir is not None:
        config.data_dir = args.data_dir.resolve()

    params = resolve_hyperparams(args)
    bumper_override = True if args.bumper else (False if args.no_bumper else None)

    apply_common_overrides(
        config,
        procs=params["procs"],
        populations=params["populations"],
        population_size=params["population_size"],
        niterations=params["niterations"],
        maxsize=params["maxsize"],
        deterministic=args.deterministic,
        bumper=bumper_override,
    )

    dataset = load_dataset(config)
    reference = evaluate_reference(dataset)

    result = run_experiment(
        config,
        output_root=args.runs_dir,
        run_name=f"3l4p1m_{args.preset}",
        random_seed=args.random_seed,
        skip_fit=args.skip_fit,
        dataset=dataset,
        summary_overrides={
            "paper": "arXiv:2510.10099",
            "example": "planar three-loop four-point one-mass",
            "preset": args.preset,
            "hyperparameters": params,
            "reference_check": reference,
            "notes": config.notes
            + [
                "Reproduces the PySR regression + post-processing stage using provided supplemental data.",
                "Does not reproduce upstream Kira/IBP generation of the data points.",
            ],
        },
    )

    print(f"Run directory: {result.run_dir}")
    print("Reference check:")
    print(json.dumps(reference, indent=2, ensure_ascii=False))
    if "pysr" in result.summary:
        print("Best PySR candidate:")
        print(json.dumps(result.summary["pysr"], indent=2, ensure_ascii=False))
    if "analysis" in result.summary:
        print("Selected d-log candidate:")
        print(
            json.dumps(
                result.summary["analysis"]["selected_candidate"],
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
