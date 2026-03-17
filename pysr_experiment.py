#!/usr/bin/env python3
"""Reusable config-driven PySR runner.

This module turns the paper's one-off ``source/regression.py`` script into a
small library + CLI that can be reused across different symbolic-regression
problems.

Typical usage:

    python pysr_experiment.py --config configs/3l4p1m_multivar.json

The config specifies:
- which text files provide the feature columns,
- how to construct the target passed to ``model.fit()``,
- the template/combine expression,
- the PySRRegressor keyword arguments.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pysr import PySRRegressor, TemplateExpressionSpec


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = ROOT / "runs"


@dataclass
class ColumnSpec:
    name: str
    file: str


@dataclass
class TargetSpec:
    mode: str = "feature"
    name: str | None = None
    file: str | None = None
    constant: float = 0.0


@dataclass
class TemplateSpec:
    expressions: list[str]
    variable_names: list[str]
    combine: str


@dataclass
class ExperimentConfig:
    data_dir: Path
    feature_columns: list[ColumnSpec]
    target: TargetSpec
    template: TemplateSpec
    regressor_kwargs: dict[str, Any]
    run_name: str | None = None
    notes: list[str] = field(default_factory=list)
    analysis: dict[str, Any] | None = None

    @property
    def feature_names(self) -> list[str]:
        return [column.name for column in self.feature_columns]


@dataclass
class RegressionDataset:
    feature_names: list[str]
    features: np.ndarray
    target: np.ndarray
    raw_columns: dict[str, np.ndarray]

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])


@dataclass
class ExperimentResult:
    run_dir: Path
    summary: dict[str, Any]
    equations_path: Path | None = None


class ConfigError(ValueError):
    """Raised when a config file is malformed or internally inconsistent."""


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)


def _require_keys(data: dict[str, Any], keys: list[str], *, where: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ConfigError(f"Missing keys in {where}: {', '.join(missing)}")


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    _require_keys(
        raw,
        ["data_dir", "feature_columns", "target", "template", "regressor_kwargs"],
        where=str(config_path),
    )

    feature_columns = [ColumnSpec(**item) for item in raw["feature_columns"]]
    if not feature_columns:
        raise ConfigError("feature_columns cannot be empty")

    target = TargetSpec(**raw["target"])
    template = TemplateSpec(**raw["template"])
    data_dir = (config_path.parent / raw["data_dir"]).resolve()

    if len({column.name for column in feature_columns}) != len(feature_columns):
        raise ConfigError("feature column names must be unique")

    return ExperimentConfig(
        data_dir=data_dir,
        feature_columns=feature_columns,
        target=target,
        template=template,
        regressor_kwargs=dict(raw["regressor_kwargs"]),
        run_name=raw.get("run_name"),
        notes=list(raw.get("notes", [])),
        analysis=dict(raw["analysis"]) if raw.get("analysis") is not None else None,
    )


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "data_dir": config.data_dir,
        "feature_columns": [column.__dict__ for column in config.feature_columns],
        "target": config.target.__dict__,
        "template": config.template.__dict__,
        "regressor_kwargs": config.regressor_kwargs,
        "notes": config.notes,
        "analysis": config.analysis,
    }


def _load_vector(path: Path) -> np.ndarray:
    array = np.loadtxt(path)
    return np.asarray(array, dtype=np.float64).reshape(-1)


def _resolve_target(
    target_spec: TargetSpec,
    *,
    data_dir: Path,
    raw_columns: dict[str, np.ndarray],
    n_samples: int,
) -> np.ndarray:
    mode = target_spec.mode.lower()

    if mode in {"feature", "column"}:
        if not target_spec.name:
            raise ConfigError("target.mode=feature requires target.name")
        if target_spec.name not in raw_columns:
            available = ", ".join(sorted(raw_columns))
            raise ConfigError(
                f"Unknown target feature '{target_spec.name}'. Available: {available}"
            )
        target = raw_columns[target_spec.name]
    elif mode == "file":
        if not target_spec.file:
            raise ConfigError("target.mode=file requires target.file")
        target = _load_vector(data_dir / target_spec.file)
    elif mode == "zeros":
        target = np.zeros(n_samples, dtype=np.float64)
    elif mode == "constant":
        target = np.full(n_samples, float(target_spec.constant), dtype=np.float64)
    else:
        raise ConfigError(
            "Unsupported target.mode. Expected one of: feature, column, file, zeros, constant"
        )

    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if len(target) != n_samples:
        raise ConfigError(
            f"Target has {len(target)} samples, but feature matrix has {n_samples} rows"
        )
    return target.reshape(-1, 1)


def load_dataset(config: ExperimentConfig) -> RegressionDataset:
    if not config.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {config.data_dir}")

    raw_columns: dict[str, np.ndarray] = {}
    lengths: dict[str, int] = {}

    for column in config.feature_columns:
        path = config.data_dir / column.file
        if not path.exists():
            raise FileNotFoundError(f"Missing feature column file: {path}")
        values = _load_vector(path)
        raw_columns[column.name] = values
        lengths[column.name] = int(len(values))

    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ConfigError(f"Feature columns have inconsistent lengths: {lengths}")

    feature_names = [column.name for column in config.feature_columns]
    features = np.column_stack([raw_columns[name] for name in feature_names])
    target = _resolve_target(
        config.target,
        data_dir=config.data_dir,
        raw_columns=raw_columns,
        n_samples=features.shape[0],
    )

    return RegressionDataset(
        feature_names=feature_names,
        features=features,
        target=target,
        raw_columns=raw_columns,
    )


def validate_config_against_dataset(
    config: ExperimentConfig, dataset: RegressionDataset
) -> None:
    if config.template.variable_names != dataset.feature_names:
        raise ConfigError(
            "template.variable_names must exactly match the feature column order. "
            f"Expected {dataset.feature_names}, got {config.template.variable_names}"
        )

    forbidden = {"expression_spec", "random_state"}
    collisions = sorted(forbidden & set(config.regressor_kwargs))
    if collisions:
        joined = ", ".join(collisions)
        raise ConfigError(
            f"regressor_kwargs should not define {joined}; those are injected by the runner"
        )


def _import_pysr() -> tuple[type[PySRRegressor], type[TemplateExpressionSpec]]:
    try:
        from pysr import PySRRegressor, TemplateExpressionSpec
    except OSError as exc:
        raise RuntimeError(
            "Failed to import PySR / Julia backend. "
            "If you see a GLIBCXX mismatch in the SymReg environment on this machine, try: "
            "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
        ) from exc
    return PySRRegressor, TemplateExpressionSpec


def build_expression_spec(config: ExperimentConfig) -> TemplateExpressionSpec:
    _, template_expression_spec = _import_pysr()
    return template_expression_spec(
        expressions=config.template.expressions,
        variable_names=config.template.variable_names,
        combine=config.template.combine,
    )


def build_model(
    config: ExperimentConfig,
    *,
    random_seed: int | None,
) -> PySRRegressor:
    pysr_regressor, _ = _import_pysr()

    kwargs = dict(config.regressor_kwargs)
    kwargs["expression_spec"] = build_expression_spec(config)
    if random_seed is not None:
        kwargs.setdefault("random_state", random_seed)
    return pysr_regressor(**kwargs)


def save_equations(model: PySRRegressor, run_dir: Path) -> pd.DataFrame:
    equations = model.equations_.copy()
    equations.to_csv(run_dir / "equations.csv", index=False)
    return equations


def summarize_best_equation(equations: pd.DataFrame) -> dict[str, Any]:
    if equations.empty:
        return {"equation_count": 0}

    idx = equations["loss"].astype(float).idxmin()
    best = equations.loc[idx]
    summary: dict[str, Any] = {
        "equation_count": int(len(equations)),
        "best_complexity": int(best["complexity"]),
        "best_loss": float(best["loss"]),
        "best_equation": str(best["equation"]),
    }
    if "score" in equations.columns:
        summary["best_score"] = float(best["score"])
    if "julia_expression" in equations.columns:
        summary["best_julia_expression"] = (
            None if pd.isna(best["julia_expression"]) else str(best["julia_expression"])
        )
    return summary


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = slug.strip("._")
    return slug or "run"


def make_run_dir(output_root: Path, name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root.resolve() / f"{timestamp}_{_slugify(name)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def apply_common_overrides(
    config: ExperimentConfig,
    *,
    procs: int | None = None,
    populations: int | None = None,
    population_size: int | None = None,
    niterations: int | None = None,
    maxsize: int | None = None,
    deterministic: bool = False,
    bumper: bool | None = None,
) -> None:
    kwargs = config.regressor_kwargs
    if procs is not None:
        kwargs["procs"] = procs
    if populations is not None:
        kwargs["populations"] = populations
    if population_size is not None:
        kwargs["population_size"] = population_size
    if niterations is not None:
        kwargs["niterations"] = niterations
    if maxsize is not None:
        kwargs["maxsize"] = maxsize
    if bumper is not None:
        kwargs["bumper"] = bumper
    if deterministic:
        kwargs["deterministic"] = True
        kwargs["parallelism"] = "serial"
        kwargs["procs"] = 1


def run_postfit_analysis(
    config: ExperimentConfig,
    dataset: RegressionDataset,
    equations: pd.DataFrame,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    if not config.analysis:
        raise ConfigError("Cannot run post-fit analysis: config.analysis is missing")

    from analyze_hall_of_fame import analyze_equations_dataframe, load_analysis_spec_from_block

    analysis_spec = load_analysis_spec_from_block(config.analysis)
    summary, candidate_rows = analyze_equations_dataframe(
        equations,
        dataset,
        analysis_spec,
        config_path=run_dir / "resolved_config.json",
        equations_path=run_dir / "equations.csv",
    )
    write_json(run_dir / "analysis_summary.json", summary)
    pd.DataFrame(candidate_rows).to_csv(run_dir / "candidate_analysis.csv", index=False)
    return summary


def run_experiment(
    config: ExperimentConfig,
    *,
    output_root: Path = DEFAULT_RUNS_DIR,
    run_name: str | None = None,
    random_seed: int | None = None,
    skip_fit: bool = False,
    dataset: RegressionDataset | None = None,
    summary_overrides: dict[str, Any] | None = None,
) -> ExperimentResult:
    dataset = dataset or load_dataset(config)
    validate_config_against_dataset(config, dataset)

    resolved_name = run_name or config.run_name or "pysr_experiment"
    run_dir = make_run_dir(output_root, resolved_name)

    summary: dict[str, Any] = {
        "run_name": resolved_name,
        "run_dir": run_dir,
        "data_dir": config.data_dir,
        "random_seed": random_seed,
        "skip_fit": skip_fit,
        "dataset": {
            "n_samples": dataset.n_samples,
            "n_features": dataset.n_features,
            "feature_names": dataset.feature_names,
            "target_mode": config.target.mode,
        },
        "config": experiment_config_to_dict(config),
    }

    if summary_overrides:
        summary.update(summary_overrides)

    write_json(run_dir / "resolved_config.json", experiment_config_to_dict(config))

    equations_path: Path | None = None
    if not skip_fit:
        model = build_model(config, random_seed=random_seed)
        fit_kwargs: dict[str, Any] = {}
        if "variable_names" not in config.regressor_kwargs:
            fit_kwargs["variable_names"] = dataset.feature_names
        model.fit(
            dataset.features,
            dataset.target,
            **fit_kwargs,
        )
        equations = save_equations(model, run_dir)
        equations_path = run_dir / "equations.csv"
        summary["pysr"] = summarize_best_equation(equations)
        if config.analysis is not None:
            summary["analysis"] = run_postfit_analysis(
                config,
                dataset,
                equations,
                run_dir=run_dir,
            )

    write_json(run_dir / "summary.json", summary)
    return ExperimentResult(run_dir=run_dir, summary=summary, equations_path=equations_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON experiment config")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--procs", type=int, default=None)
    parser.add_argument("--populations", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--niterations", type=int, default=None)
    parser.add_argument("--maxsize", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    bumper_group = parser.add_mutually_exclusive_group()
    bumper_group.add_argument(
        "--bumper",
        action="store_true",
        help="Override config and enable bumper=True",
    )
    bumper_group.add_argument(
        "--no-bumper",
        action="store_true",
        help="Override config and disable bumper=True",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    if args.data_dir is not None:
        config.data_dir = args.data_dir.resolve()

    bumper_override = True if args.bumper else (False if args.no_bumper else None)

    apply_common_overrides(
        config,
        procs=args.procs,
        populations=args.populations,
        population_size=args.population_size,
        niterations=args.niterations,
        maxsize=args.maxsize,
        deterministic=args.deterministic,
        bumper=bumper_override,
    )

    result = run_experiment(
        config,
        output_root=args.output_root,
        run_name=args.run_name,
        random_seed=args.random_seed,
        skip_fit=args.skip_fit,
    )

    print(f"Run directory: {result.run_dir}")
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
