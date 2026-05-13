#!/bin/bash
#SBATCH --job-name=pcqvnn-pcn-search
#SBATCH --partition=h200
#SBATCH --array=0-3%2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_ROOT/cluster/pcqvnn.sif}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results}"
RUN_NAME="${RUN_NAME:-pcn-h200-search}"
TRIALS="${TRIALS:-24}"

BENCHMARKS=(ackley griewank rastrigin schwefel)
BENCHMARK="${BENCHMARKS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "$RESULTS_ROOT" "$PROJECT_ROOT/logs"

if command -v apptainer >/dev/null 2>&1; then
    RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    RUNTIME="singularity"
else
    echo "Neither apptainer nor singularity is available on PATH." >&2
    exit 1
fi

if [[ ! -f "$CONTAINER_IMAGE" ]]; then
    echo "Container image not found at $CONTAINER_IMAGE" >&2
    echo "Build it with: apptainer build cluster/pcqvnn.sif cluster/pcqvnn.def" >&2
    exit 1
fi

to_container_path() {
    local host_path="$1"
    if [[ "$host_path" == "$PROJECT_ROOT"* ]]; then
        printf '/workspace%s\n' "${host_path#$PROJECT_ROOT}"
    else
        printf '%s\n' "$host_path"
    fi
}

RESULTS_ROOT_IN_CONTAINER="$(to_container_path "$RESULTS_ROOT")"
OUTPUT_PATH_IN_CONTAINER="$RESULTS_ROOT_IN_CONTAINER/$RUN_NAME/$BENCHMARK/pcn_hyperparameter_search.csv"

echo "Job started at $(date)"
echo "Benchmark: $BENCHMARK"
echo "Trials: $TRIALS"
echo "Run output: $OUTPUT_PATH_IN_CONTAINER"

"$RUNTIME" exec \
    --nv \
    --bind "$PROJECT_ROOT:/workspace" \
    "$CONTAINER_IMAGE" \
    python -m qvnn_pc.pcn_hyperparameter_search \
    --benchmark "$BENCHMARK" \
    --device cuda \
    --trials "$TRIALS" \
    --output "$OUTPUT_PATH_IN_CONTAINER" \
    "$@"
