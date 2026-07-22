#!/bin/bash

# Try the largest single-node Spot allocation first while keeping the effective
# batch close to the centered-gate baseline (global batch 32).
set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
TRAIN_SCRIPT=$PROJ/scripts/version5/qwen3_4b/run_train_gate_d4_shifted_sigmoid.sh
WAIT_SECONDS=${WAIT_SECONDS:-30}
GPU_CANDIDATES=${GPU_CANDIDATES:-"8 7 6 5 4 3"}
mkdir -p "$PROJ/logs/v5_qwen3_4b"

active_job=
cleanup() {
  if [[ -n "$active_job" ]] \
      && squeue -h -j "$active_job" -o %T | grep -q .; then
    scancel "$active_job"
  fi
  active_job=
}
on_signal() {
  cleanup
  trap - EXIT
  exit "$1"
}
trap cleanup EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

declare -A ACCUM_BY_GPUS=(
  [8]=4
  [7]=5
  [6]=5
  [5]=6
  [4]=8
  [3]=11
)

for gpus in $GPU_CANDIDATES; do
  [[ -n "${ACCUM_BY_GPUS[$gpus]:-}" ]] || {
    echo "FATAL: unsupported GPU candidate: $gpus" >&2
    exit 2
  }
  accum=${ACCUM_BY_GPUS[$gpus]}
  echo "Trying Spot: GPUs=$gpus ACCUM=$accum global_batch=$((gpus * accum))"
  job_id=$(sbatch --parsable \
    --gres="gpu:$gpus" \
    --quotatype=spot \
    --job-name=e09_q3g3sm1 \
    --export="ALL,GPUS=$gpus,ACCUM=$accum" \
    "$TRAIN_SCRIPT")
  job_id=${job_id%%;*}
  active_job=$job_id

  for ((elapsed=0; elapsed<WAIT_SECONDS; elapsed+=2)); do
    state=$(squeue -h -j "$job_id" -o %T | head -1)
    if [[ "$state" == RUNNING ]]; then
      echo "STARTED job=$job_id GPUs=$gpus ACCUM=$accum"
      active_job=
      exit 0
    fi
    if [[ -z "$state" ]]; then
      final=$(sacct -n -X -j "$job_id" -o State | awk 'NF {print $1; exit}')
      echo "FATAL: job=$job_id left queue before RUNNING (state=${final:-unknown})" >&2
      exit 1
    fi
    sleep 2
  done

  state=$(squeue -h -j "$job_id" -o %T | head -1)
  if [[ "$state" == RUNNING ]]; then
    echo "STARTED job=$job_id GPUs=$gpus ACCUM=$accum"
    active_job=
    exit 0
  fi
  echo "No start within ${WAIT_SECONDS}s; cancelling job=$job_id"
  scancel "$job_id"
  active_job=
done

echo "FATAL: Spot allocations from 8 through 3 GPUs all remained pending" >&2
exit 1
