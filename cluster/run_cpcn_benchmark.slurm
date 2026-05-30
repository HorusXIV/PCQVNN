#!/bin/bash
#SBATCH --job-name=pcqvnn-cpcn
#SBATCH --partition=performance
#SBATCH --array=0-3%1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_ROOT/cluster/pcqvnn.sif}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results}"
RUN_NAME="${RUN_NAME:-cpcn-h200}"

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
OUTPUT_PATH_IN_CONTAINER="$RESULTS_ROOT_IN_CONTAINER/$RUN_NAME/$BENCHMARK/cpcn_benchmark.csv"

echo "Job started at $(date)"
echo "Benchmark: $BENCHMARK"
echo "Run output: $OUTPUT_PATH_IN_CONTAINER"

"$RUNTIME" exec \
    --nv \
    --bind "$PROJECT_ROOT:/workspace" \
    "$CONTAINER_IMAGE" \
    python -m qvnn_pc.cpcn_benchmark \
    --benchmark "$BENCHMARK" \
    --device cuda \
    --output "$OUTPUT_PATH_IN_CONTAINER" \
    "$@"
