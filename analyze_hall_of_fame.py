#!/usr/bin/env python3
"""Analyze PySR hall-of-fame equations and extract the physically correct one.

This follows the workflow described in the paper:

1. parse candidate equations from PySR output,
2. verify them against the sampled numerical derivatives,
3. identify d-log candidates from their symbolic derivatives,
4. extract symbol letters from the derivative denominators,
5. reconstruct a canonical sum-of-logs expression from those letters,
6. choose the simplest exact-fit candidate.

The script is intentionally written for the paper's current log/d-log workflow,
but the interface is generic enough to reuse on other cases by changing the JSON
config.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from pysr_experiment import RegressionDataset, load_dataset, load_experiment_config, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "3l4p1m_multivar.json"
DEFAULT_EQUATIONS = ROOT / "source" / "hall_of_fame.csv"


@dataclass
class DerivativeColumnSpec:
    variable: str
    column: str


@dataclass
class AnalysisSpec:
    expression_variables: list[str]
    derivative_columns: list[DerivativeColumnSpec]
    exact_loss_threshold: float = 1e-20
    max_abs_error_threshold: float = 1e-10
    rmse_threshold: float = 1e-10
    rational_tolerance: float = 1e-10


@dataclass
class CandidateResult:
    complexity: int
    loss: float
    equation: str
    parsed_expression: sp.Expr
    derivative_rmse: float
    derivative_max_abs: float
    letters: list[sp.Expr]
    reconstructed_expression: sp.Expr | None
    reconstructed_coefficients: list[sp.Expr]
    reconstructed_rmse: float | None
    reconstructed_max_abs: float | None
    clean_letters: bool
    selected: bool = False
    selection_reason: str | None = None


class AnalysisError(ValueError):
    """Raised when the analysis config or equation file is malformed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--equations", type=Path, default=DEFAULT_EQUATIONS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--exact-loss-threshold", type=float, default=None)
    parser.add_argument("--max-abs-threshold", type=float, default=None)
    parser.add_argument("--rmse-threshold", type=float, default=None)
    parser.add_argument("--rational-tolerance", type=float, default=None)
    return parser.parse_args()


def load_analysis_spec_from_block(block: dict[str, Any] | None) -> AnalysisSpec:
    if not block:
        raise AnalysisError(
            "Config has no 'analysis' block. Please add expression_variables and derivative_columns."
        )

    if "expression_variables" not in block or "derivative_columns" not in block:
        raise AnalysisError(
            "analysis block must define both 'expression_variables' and 'derivative_columns'"
        )

    derivative_columns = [DerivativeColumnSpec(**item) for item in block["derivative_columns"]]
    if not derivative_columns:
        raise AnalysisError("analysis.derivative_columns cannot be empty")

    return AnalysisSpec(
        expression_variables=list(block["expression_variables"]),
        derivative_columns=derivative_columns,
        exact_loss_threshold=float(block.get("exact_loss_threshold", 1e-20)),
        max_abs_error_threshold=float(block.get("max_abs_error_threshold", 1e-10)),
        rmse_threshold=float(block.get("rmse_threshold", 1e-10)),
        rational_tolerance=float(block.get("rational_tolerance", 1e-10)),
    )


def load_analysis_spec(config_path: Path) -> AnalysisSpec:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return load_analysis_spec_from_block(raw.get("analysis"))


def find_column(df: pd.DataFrame, *candidates: str) -> str:
    normalized = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise AnalysisError(f"Could not find any of columns {candidates} in {list(df.columns)}")


def load_equations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    complexity_col = find_column(df, "Complexity", "complexity")
    loss_col = find_column(df, "Loss", "loss")
    equation_col = find_column(df, "Equation", "equation")
    out = df[[complexity_col, loss_col, equation_col]].copy()
    out.columns = ["complexity", "loss", "equation"]
    out["complexity"] = out["complexity"].astype(int)
    out["loss"] = out["loss"].astype(float)
    return out.sort_values(["complexity", "loss"]).reset_index(drop=True)


def build_symbol_table(variable_names: list[str]) -> tuple[list[sp.Symbol], dict[str, sp.Symbol]]:
    symbols = [sp.Symbol(name, real=True) for name in variable_names]
    return symbols, {name: symbol for name, symbol in zip(variable_names, symbols)}


def parse_pysr_equation(text: str, variable_names: list[str]) -> sp.Expr:
    rhs = text.split("=", 1)[1].strip() if "=" in text else text.strip()
    rhs = rhs.replace("^", "**")
    placeholder_names = {f"__var{i}": name for i, name in enumerate(variable_names, start=1)}
    rhs = re.sub(r"#(\d+)", lambda match: f"__var{match.group(1)}", rhs)

    locals_dict: dict[str, Any] = {"log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs}
    for placeholder, name in placeholder_names.items():
        locals_dict[placeholder] = sp.Symbol(name, real=True)

    return parse_expr(rhs, local_dict=locals_dict, evaluate=True)


def as_real_array(values: Any, like: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.shape == ():
        array = np.full_like(like, float(np.real(array)), dtype=np.float64)
    array = np.real_if_close(array, tol=1000)
    return np.asarray(array, dtype=np.float64).reshape(-1)


def evaluate_expression_against_derivatives(
    expr: sp.Expr,
    dataset: RegressionDataset,
    *,
    expression_variables: list[str],
    derivative_columns: list[DerivativeColumnSpec],
) -> tuple[dict[str, dict[str, float]], float, float]:
    symbols, symbol_table = build_symbol_table(expression_variables)
    values = [dataset.raw_columns[name] for name in expression_variables]

    metrics: dict[str, dict[str, float]] = {}
    rmses: list[float] = []
    max_abs_values: list[float] = []

    for spec in derivative_columns:
        derivative_expr = sp.simplify(sp.diff(expr, symbol_table[spec.variable]))
        fn = sp.lambdify(symbols, derivative_expr, "numpy")
        predicted = as_real_array(fn(*values), like=values[0])
        target = np.asarray(dataset.raw_columns[spec.column], dtype=np.float64).reshape(-1)
        diff = predicted - target
        rmse = float(np.sqrt(np.mean(diff**2)))
        max_abs = float(np.max(np.abs(diff)))
        metrics[spec.variable] = {"rmse": rmse, "max_abs": max_abs}
        rmses.append(rmse)
        max_abs_values.append(max_abs)

    total_rmse = float(math.sqrt(np.mean(np.square(rmses)))) if rmses else 0.0
    total_max_abs = float(max(max_abs_values)) if max_abs_values else 0.0
    return metrics, total_rmse, total_max_abs


def strip_numeric_content(expr: sp.Expr) -> sp.Expr:
    expr = sp.expand(expr)
    coeff, factors = expr.as_coeff_mul()
    stripped = sp.Mul(*factors) if factors else sp.Integer(1)
    return sp.expand(stripped)


def canonicalize_factor(expr: sp.Expr, variables: list[sp.Symbol]) -> sp.Expr:
    expr = sp.simplify(expr)
    expr = strip_numeric_content(expr)

    if expr.is_Pow:
        expr = expr.base

    try:
        poly = sp.Poly(sp.expand(expr), *variables)
        primitive = poly.primitive()[1].as_expr()
        expr = sp.expand(primitive)
        constant_term = sp.simplify(expr.subs({symbol: 0 for symbol in variables}))
        if constant_term.is_number and constant_term != 0:
            if float(sp.N(constant_term)) < 0:
                expr = -expr
        else:
            leading = sp.N(sp.Poly(expr, *variables).LC())
            if leading.is_real and float(leading) < 0:
                expr = -expr
        expr = sp.factor(expr)
    except Exception:
        coeff, _ = expr.as_coeff_mul()
        if coeff.is_number and coeff != 0 and float(sp.N(coeff)) < 0:
            expr = -expr

    return sp.simplify(expr)


def extract_base_factors(expr: sp.Expr, variables: list[sp.Symbol]) -> list[sp.Expr]:
    expr = sp.simplify(expr)
    if expr.is_Number:
        return []
    if expr.is_Pow:
        return extract_base_factors(expr.base, variables)
    if expr.is_Mul:
        factors: list[sp.Expr] = []
        for factor in expr.args:
            factors.extend(extract_base_factors(factor, variables))
        return factors

    try:
        coeff, factor_list = sp.factor_list(sp.expand(expr))
        if factor_list and not (len(factor_list) == 1 and factor_list[0][0] == expr):
            factors: list[sp.Expr] = []
            for factor, _power in factor_list:
                factors.extend(extract_base_factors(factor, variables))
            return factors
    except Exception:
        pass

    if expr.free_symbols & set(variables):
        return [canonicalize_factor(expr, variables)]
    return []


def deduplicate_letters(letters: list[sp.Expr]) -> list[sp.Expr]:
    seen: dict[str, sp.Expr] = {}
    for letter in letters:
        key = sp.srepr(letter)
        if key not in seen:
            seen[key] = letter
    return list(seen.values())


def extract_symbol_letters(
    expr: sp.Expr,
    *,
    expression_variables: list[str],
    derivative_columns: list[DerivativeColumnSpec],
) -> list[sp.Expr]:
    symbols, symbol_table = build_symbol_table(expression_variables)
    letters: list[sp.Expr] = []

    for spec in derivative_columns:
        derivative_expr = sp.simplify(sp.diff(expr, symbol_table[spec.variable]))
        _num, den = sp.fraction(sp.together(derivative_expr))
        letters.extend(extract_base_factors(den, symbols))

    return deduplicate_letters(letters)


def contains_floats(expr: sp.Expr) -> bool:
    return bool(expr.atoms(sp.Float))


def reconstruct_from_letters(
    letters: list[sp.Expr],
    dataset: RegressionDataset,
    *,
    expression_variables: list[str],
    derivative_columns: list[DerivativeColumnSpec],
    rational_tolerance: float,
) -> tuple[sp.Expr | None, list[sp.Expr], float | None, float | None]:
    if not letters:
        return None, [], None, None

    symbols, symbol_table = build_symbol_table(expression_variables)
    values = [dataset.raw_columns[name] for name in expression_variables]

    blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for spec in derivative_columns:
        cols: list[np.ndarray] = []
        for letter in letters:
            derivative_expr = sp.diff(sp.log(letter), symbol_table[spec.variable])
            fn = sp.lambdify(symbols, derivative_expr, "numpy")
            cols.append(as_real_array(fn(*values), like=values[0]).reshape(-1, 1))
        blocks.append(np.hstack(cols))
        targets.append(np.asarray(dataset.raw_columns[spec.column], dtype=np.float64).reshape(-1, 1))

    matrix = np.vstack(blocks)
    rhs = np.vstack(targets)
    solution, _residuals, rank, _singular_values = np.linalg.lstsq(matrix, rhs, rcond=None)
    if rank < len(letters):
        return None, [], None, None

    coefficients: list[sp.Expr] = []
    for value in solution.reshape(-1):
        coefficients.append(sp.nsimplify(float(value), rational=True, tolerance=rational_tolerance))

    reconstructed = sp.simplify(
        sum(coeff * sp.log(letter) for coeff, letter in zip(coefficients, letters))
    )
    _per_variable, total_rmse, total_max_abs = evaluate_expression_against_derivatives(
        reconstructed,
        dataset,
        expression_variables=expression_variables,
        derivative_columns=derivative_columns,
    )
    return reconstructed, coefficients, total_rmse, total_max_abs


def evaluate_candidate(
    row: pd.Series,
    dataset: RegressionDataset,
    analysis_spec: AnalysisSpec,
) -> CandidateResult:
    expr = parse_pysr_equation(row["equation"], analysis_spec.expression_variables)
    _per_variable, derivative_rmse, derivative_max_abs = evaluate_expression_against_derivatives(
        expr,
        dataset,
        expression_variables=analysis_spec.expression_variables,
        derivative_columns=analysis_spec.derivative_columns,
    )
    letters = extract_symbol_letters(
        expr,
        expression_variables=analysis_spec.expression_variables,
        derivative_columns=analysis_spec.derivative_columns,
    )
    reconstructed_expression, reconstructed_coefficients, reconstructed_rmse, reconstructed_max_abs = reconstruct_from_letters(
        letters,
        dataset,
        expression_variables=analysis_spec.expression_variables,
        derivative_columns=analysis_spec.derivative_columns,
        rational_tolerance=analysis_spec.rational_tolerance,
    )
    clean_letters = not any(contains_floats(letter) for letter in letters)

    return CandidateResult(
        complexity=int(row["complexity"]),
        loss=float(row["loss"]),
        equation=str(row["equation"]),
        parsed_expression=expr,
        derivative_rmse=derivative_rmse,
        derivative_max_abs=derivative_max_abs,
        letters=letters,
        reconstructed_expression=reconstructed_expression,
        reconstructed_coefficients=reconstructed_coefficients,
        reconstructed_rmse=reconstructed_rmse,
        reconstructed_max_abs=reconstructed_max_abs,
        clean_letters=clean_letters,
    )


def is_exact_candidate(candidate: CandidateResult, spec: AnalysisSpec) -> bool:
    return (
        candidate.loss <= spec.exact_loss_threshold
        and candidate.reconstructed_expression is not None
        and candidate.reconstructed_rmse is not None
        and candidate.reconstructed_max_abs is not None
        and candidate.reconstructed_rmse <= spec.rmse_threshold
        and candidate.reconstructed_max_abs <= spec.max_abs_error_threshold
    )


def select_best_candidate(candidates: list[CandidateResult], spec: AnalysisSpec) -> CandidateResult:
    if not candidates:
        raise AnalysisError("No candidate equations were available for analysis")

    exact_clean = [candidate for candidate in candidates if is_exact_candidate(candidate, spec) and candidate.clean_letters]
    if exact_clean:
        chosen = min(exact_clean, key=lambda item: (item.complexity, item.loss))
        chosen.selected = True
        chosen.selection_reason = "simplest exact-fit candidate with clean extracted letters"
        return chosen

    exact_any = [candidate for candidate in candidates if is_exact_candidate(candidate, spec)]
    if exact_any:
        chosen = min(exact_any, key=lambda item: (item.complexity, item.loss))
        chosen.selected = True
        chosen.selection_reason = "simplest exact-fit candidate"
        return chosen

    dlog_candidates = [candidate for candidate in candidates if candidate.reconstructed_expression is not None]
    if dlog_candidates:
        chosen = min(dlog_candidates, key=lambda item: (item.loss, item.complexity))
        chosen.selected = True
        chosen.selection_reason = "best available d-log candidate (no exact-fit row crossed the threshold)"
        return chosen

    chosen = min(candidates, key=lambda item: (item.loss, item.complexity))
    chosen.selected = True
    chosen.selection_reason = "fallback to lowest-loss candidate"
    return chosen


def candidate_to_dict(candidate: CandidateResult) -> dict[str, Any]:
    return {
        "complexity": candidate.complexity,
        "loss": candidate.loss,
        "equation": candidate.equation,
        "parsed_expression": str(candidate.parsed_expression),
        "derivative_rmse": candidate.derivative_rmse,
        "derivative_max_abs": candidate.derivative_max_abs,
        "letters": [str(letter) for letter in candidate.letters],
        "reconstructed_expression": None
        if candidate.reconstructed_expression is None
        else str(candidate.reconstructed_expression),
        "reconstructed_coefficients": [str(value) for value in candidate.reconstructed_coefficients],
        "reconstructed_rmse": candidate.reconstructed_rmse,
        "reconstructed_max_abs": candidate.reconstructed_max_abs,
        "clean_letters": candidate.clean_letters,
        "selected": candidate.selected,
        "selection_reason": candidate.selection_reason,
    }


def build_analysis_summary(
    selected: CandidateResult,
    candidates: list[CandidateResult],
    analysis_spec: AnalysisSpec,
    *,
    config_path: Path | None = None,
    equations_path: Path | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "selection_reason": selected.selection_reason,
        "selected_candidate": candidate_to_dict(selected),
        "thresholds": {
            "exact_loss_threshold": analysis_spec.exact_loss_threshold,
            "max_abs_error_threshold": analysis_spec.max_abs_error_threshold,
            "rmse_threshold": analysis_spec.rmse_threshold,
            "rational_tolerance": analysis_spec.rational_tolerance,
        },
        "candidate_count": len(candidates),
        "exact_candidate_count": sum(is_exact_candidate(candidate, analysis_spec) for candidate in candidates),
    }
    if config_path is not None:
        summary["config"] = str(config_path.resolve())
    if equations_path is not None:
        summary["equations"] = str(equations_path.resolve())
    return summary


def analyze_equations_dataframe(
    equations: pd.DataFrame,
    dataset: RegressionDataset,
    analysis_spec: AnalysisSpec,
    *,
    config_path: Path | None = None,
    equations_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [evaluate_candidate(row, dataset, analysis_spec) for _, row in equations.iterrows()]
    selected = select_best_candidate(candidates, analysis_spec)
    summary = build_analysis_summary(
        selected,
        candidates,
        analysis_spec,
        config_path=config_path,
        equations_path=equations_path,
    )
    candidate_rows = [candidate_to_dict(candidate) for candidate in candidates]
    return summary, candidate_rows


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    analysis_spec = load_analysis_spec(args.config)

    if args.exact_loss_threshold is not None:
        analysis_spec.exact_loss_threshold = args.exact_loss_threshold
    if args.max_abs_threshold is not None:
        analysis_spec.max_abs_error_threshold = args.max_abs_threshold
    if args.rmse_threshold is not None:
        analysis_spec.rmse_threshold = args.rmse_threshold
    if args.rational_tolerance is not None:
        analysis_spec.rational_tolerance = args.rational_tolerance

    dataset = load_dataset(config)
    equations = load_equations(args.equations)
    summary, candidate_rows = analyze_equations_dataframe(
        equations,
        dataset,
        analysis_spec,
        config_path=args.config,
        equations_path=args.equations,
    )

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "analysis_summary.json", summary)
        pd.DataFrame(candidate_rows).to_csv(output_dir / "candidate_analysis.csv", index=False)

    print("Selected candidate:")
    print(json.dumps(summary["selected_candidate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
