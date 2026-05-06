#!/bin/bash
#SBATCH --job-name=pcqvnn-archcmp
#SBATCH -M cluster
#SBATCH --partition=performance
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_ROOT/cluster/pcqvnn.sif}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/hyperparameters.yml}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results}"
DEVICE="${DEVICE:-cpu}"

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

CONFIG_PATH_IN_CONTAINER="$(to_container_path "$CONFIG_PATH")"
RESULTS_ROOT_IN_CONTAINER="$(to_container_path "$RESULTS_ROOT")"
RUN_NAME="${RUN_NAME:-slurm_${SLURM_JOB_ID}}"

"$RUNTIME" exec \
    --bind "$PROJECT_ROOT:/workspace" \
    "$CONTAINER_IMAGE" \
    python -m qvnn_pc.architecture_comparison \
    --config "$CONFIG_PATH_IN_CONTAINER" \
    --device "$DEVICE" \
    --results-root "$RESULTS_ROOT_IN_CONTAINER" \
    --run-name "$RUN_NAME" \
    "$@"
