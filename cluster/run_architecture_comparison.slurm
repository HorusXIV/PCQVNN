#!/bin/bash
#SBATCH --job-name=pcqvnn-archcmp
#SBATCH -M cluster
#SBATCH --partition=performance
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@60
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_ROOT/cluster/pcqvnn.sif}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results}"

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
RUN_NAME="${RUN_NAME:-archcmp_resume}"
OUTPUT_PATH_IN_CONTAINER="$RESULTS_ROOT_IN_CONTAINER/$RUN_NAME/architecture_comparison.csv"

REQUEUE_REQUESTED=0
CHILD_PID=""

cleanup_and_requeue() {
    echo "--- Signal caught at $(date) ---"
    REQUEUE_REQUESTED=1
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        echo "Forwarding SIGUSR1 to training process (pid $CHILD_PID)"
        kill -USR1 "$CHILD_PID" || true
        wait "$CHILD_PID" || true
    fi
    echo "Requeueing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}

trap 'cleanup_and_requeue' USR1

echo "Job started at $(date) (restart count: ${SLURM_RESTART_COUNT:-0})"
echo "Run output will be written to $OUTPUT_PATH_IN_CONTAINER"

timeout --signal=SIGUSR1 23h50m \
    "$RUNTIME" exec \
    --bind "$PROJECT_ROOT:/workspace" \
    "$CONTAINER_IMAGE" \
    python -m qvnn_pc.architecture_comparison \
    --output "$OUTPUT_PATH_IN_CONTAINER" \
    "$@" &
CHILD_PID=$!

EXIT_CODE=0
wait "$CHILD_PID" || EXIT_CODE=$?

if [[ "$EXIT_CODE" -eq 124 ]]; then
    echo "Timeout reached at $(date); requeueing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

exit "$EXIT_CODE"
