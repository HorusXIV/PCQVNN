#!/bin/bash
#SBATCH --job-name=pcqvnn-archcmp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_ROOT/cluster/pcqvnn.sif}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/hyperparameters.yml}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/slurm}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "$RESULTS_DIR" "$PROJECT_ROOT/logs"

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

OUTPUT_PATH="${OUTPUT_PATH:-$RESULTS_DIR/architecture_comparison_${SLURM_JOB_ID}.csv}"

"$RUNTIME" exec \
    --bind "$PROJECT_ROOT:/workspace" \
    "$CONTAINER_IMAGE" \
    python -m qvnn_pc.architecture_comparison \
    --config "$CONFIG_PATH" \
    --device "$DEVICE" \
    --output "$OUTPUT_PATH" \
    "$@"

