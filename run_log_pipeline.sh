#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WOLFRAM_BIN="/usr/local/Wolfram/Mathematica/14.3/Executables/wolfram"
if command -v wolfram >/dev/null 2>&1; then
  WOLFRAM_BIN_DEFAULT_FROM_PATH="$(command -v wolfram)"
else
  WOLFRAM_BIN_DEFAULT_FROM_PATH="$DEFAULT_WOLFRAM_BIN"
fi
WOLFRAM_BIN="${WOLFRAM_BIN:-$WOLFRAM_BIN_DEFAULT_FROM_PATH}"

if [[ -z "${CONDA_SH:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
  else
    CONDA_SH="${HOME}/Software/miniconda3/etc/profile.d/conda.sh"
  fi
fi
ENV_NAME="${ENV_NAME:-SymReg}"

FUNCTION_FILE=""
OUTPUT_ROOT="${SCRIPT_DIR}/pipeline_runs"
SAMPLES=200
WORKING_PRECISION=60
MIN_PRECISION=30
INTEGER_RANGE=9
DENOMINATOR_MAX=13
RANDOM_SEED=""
ALLOW_SQRT=0
SKIP_FIT=0

PY_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  run_log_pipeline.sh --function-file path/to/f.m [options]

This one-click pipeline does:
  1) Mathematica 14.3 generates high-precision derivative samples
  2) PySR runs symbolic regression in the configured conda env
  3) the integrated analyzer selects the best d-log candidate automatically

Required:
  --function-file PATH         Mathematica .m file returning either an expression
                               or <|"Expression" -> expr, "Variables" -> {...}|>

Pipeline options:
  --output-root DIR            Root directory for timestamped pipeline outputs
  --samples N                  Number of sampled points for dataset generation
  --working-precision N        Export precision for numeric values (default: 60)
  --min-precision N            Minimum accepted derivative precision (default: 30)
  --integer-range N            Random rational numerators sampled from [-N, N]
  --denominator-max N          Random rational denominators sampled from [1, N]
  --seed N                     Random seed for dataset generation
  --allow-sqrt                 Switch generator config to the paper's log+sqrt mode
  --env-name NAME              Conda environment name (default: SymReg)
  --skip-fit                   Only generate dataset + generated_config.json

Environment overrides:
  WOLFRAM_BIN                  Path to the wolfram executable
  CONDA_SH                     Path to conda.sh for shell activation
  ENV_NAME                     Conda environment name (alternative to --env-name)

PySR override options:
  --procs N
  --populations N
  --population-size N
  --niterations N
  --maxsize N
  --deterministic
  --bumper
  --no-bumper

Example:
  ./run_log_pipeline.sh \
    --function-file examples/3l4p1m_function.m \
    --output-root /tmp/pysr_pipeline \
    --samples 200 \
    --niterations 50 \
    --populations 24 \
    --population-size 80 \
    --procs 8
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_arg() {
  local flag="$1"
  local value="${2-}"
  [[ -n "$value" ]] || die "$flag requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --function-file)
      need_arg "$1" "${2-}"
      FUNCTION_FILE="$2"
      shift 2
      ;;
    --output-root)
      need_arg "$1" "${2-}"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --samples)
      need_arg "$1" "${2-}"
      SAMPLES="$2"
      shift 2
      ;;
    --working-precision)
      need_arg "$1" "${2-}"
      WORKING_PRECISION="$2"
      shift 2
      ;;
    --min-precision)
      need_arg "$1" "${2-}"
      MIN_PRECISION="$2"
      shift 2
      ;;
    --integer-range)
      need_arg "$1" "${2-}"
      INTEGER_RANGE="$2"
      shift 2
      ;;
    --denominator-max)
      need_arg "$1" "${2-}"
      DENOMINATOR_MAX="$2"
      shift 2
      ;;
    --seed)
      need_arg "$1" "${2-}"
      RANDOM_SEED="$2"
      shift 2
      ;;
    --allow-sqrt)
      ALLOW_SQRT=1
      shift
      ;;
    --env-name)
      need_arg "$1" "${2-}"
      ENV_NAME="$2"
      shift 2
      ;;
    --skip-fit)
      SKIP_FIT=1
      shift
      ;;
    --procs|--populations|--population-size|--niterations|--maxsize)
      need_arg "$1" "${2-}"
      PY_ARGS+=("$1" "$2")
      shift 2
      ;;
    --deterministic|--bumper|--no-bumper)
      PY_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[[ -n "$FUNCTION_FILE" ]] || {
  usage
  die "--function-file is required"
}

command -v python3 >/dev/null 2>&1 || die "python3 not found"
[[ -f "$WOLFRAM_BIN" ]] || die "Mathematica 14.3 executable not found: $WOLFRAM_BIN"
[[ -f "$CONDA_SH" ]] || die "conda activation script not found: $CONDA_SH"

FUNCTION_FILE="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$FUNCTION_FILE")"
[[ -f "$FUNCTION_FILE" ]] || die "function file does not exist: $FUNCTION_FILE"

RUN_NAME="$(basename "${FUNCTION_FILE%.*}")"
STAMP="$(date +%Y%m%d_%H%M%S)"
PIPELINE_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_ROOT")/${STAMP}_${RUN_NAME}"
DATASET_DIR="${PIPELINE_DIR}/dataset"
RUNS_DIR="${PIPELINE_DIR}/runs"
LOG_FILE="${PIPELINE_DIR}/pipeline.log"

mkdir -p "$PIPELINE_DIR" "$RUNS_DIR"

exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "[1/2] Generating dataset with Mathematica 14.3"
echo "      function file : $FUNCTION_FILE"
echo "      dataset dir   : $DATASET_DIR"
echo "      operator mode : $([[ "$ALLOW_SQRT" -eq 1 ]] && echo log-sqrt || echo log-only)"

WL_CMD=(
  "$WOLFRAM_BIN" -noprompt -script "$SCRIPT_DIR/generate_log_dataset.wl"
  --function-file "$FUNCTION_FILE"
  --output-dir "$DATASET_DIR"
  --samples "$SAMPLES"
  --working-precision "$WORKING_PRECISION"
  --min-precision "$MIN_PRECISION"
  --integer-range "$INTEGER_RANGE"
  --denominator-max "$DENOMINATOR_MAX"
)

if [[ -n "$RANDOM_SEED" ]]; then
  WL_CMD+=(--random-seed "$RANDOM_SEED")
fi
if [[ "$ALLOW_SQRT" -eq 1 ]]; then
  WL_CMD+=(--allow-sqrt)
fi

"${WL_CMD[@]}"

CONFIG_FILE="$DATASET_DIR/generated_config.json"
[[ -f "$CONFIG_FILE" ]] || die "generator did not produce config: $CONFIG_FILE"

echo ""
echo "Dataset generation complete."
echo "  config : $CONFIG_FILE"
echo "  data   : $DATASET_DIR"

if [[ "$SKIP_FIT" -eq 1 ]]; then
  echo ""
  echo "--skip-fit enabled, stopping after data generation."
  echo "You can run later with:"
  echo "  python $SCRIPT_DIR/pysr_experiment.py --config $CONFIG_FILE"
  exit 0
fi

echo ""
echo "[2/2] Running PySR + automatic equation analysis"
source "$CONDA_SH"
conda activate "$ENV_NAME"

if [[ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]]; then
  if [[ -n "${LD_PRELOAD:-}" ]]; then
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:${LD_PRELOAD}"
  else
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
  fi
fi

PY_CMD=(python "$SCRIPT_DIR/pysr_experiment.py" --config "$CONFIG_FILE" --output-root "$RUNS_DIR")
if [[ ${#PY_ARGS[@]} -gt 0 ]]; then
  PY_CMD+=("${PY_ARGS[@]}")
fi

"${PY_CMD[@]}"

LATEST_RUN_DIR="$(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
[[ -n "$LATEST_RUN_DIR" ]] || die "PySR finished but no run directory was created in $RUNS_DIR"

echo ""
echo "Pipeline finished."
echo "  pipeline dir           : $PIPELINE_DIR"
echo "  dataset dir            : $DATASET_DIR"
echo "  run dir                : $LATEST_RUN_DIR"
echo "  summary                : $LATEST_RUN_DIR/summary.json"
echo "  equations              : $LATEST_RUN_DIR/equations.csv"
echo "  selected equation      : $LATEST_RUN_DIR/analysis_summary.json"
echo "  candidate table        : $LATEST_RUN_DIR/candidate_analysis.csv"
echo "  pipeline log           : $LOG_FILE"
