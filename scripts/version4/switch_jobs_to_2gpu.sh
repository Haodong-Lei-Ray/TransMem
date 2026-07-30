#!/bin/bash
#SBATCH -J e09_switch2g
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:0
#SBATCH --quotatype=reserved
#SBATCH -t 02:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/%j_switch_jobs_to_2gpu.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/%j_switch_jobs_to_2gpu.err

# One-shot controller for preserving the last partial intervals of jobs
# 10258721 and 10258691 before restarting them with two GPUs.
set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4

wait_for_step() {
  local result_file=$1
  local target_step=$2
  local old_job=$3

  while true; do
    if grep -Eq '"global_step"[[:space:]]*:[[:space:]]*'"${target_step}"'([,[:space:]]|$)' "$result_file" 2>/dev/null; then
      echo "checkpoint ready: job=$old_job step=$target_step"
      return 0
    fi
    if ! squeue -h -j "$old_job" | grep -q .; then
      echo "old job $old_job disappeared before checkpoint step $target_step" >&2
      return 1
    fi
    sleep 30
  done
}

stop_and_wait() {
  local job=$1
  scancel "$job"
  while squeue -h -j "$job" | grep -q .; do
    sleep 2
  done
}

switch_mix() {
  local result=$PROJ/checkpoints/v4_inloop_tf_mix_d4/result.json
  wait_for_step "$result" 750 10258721
  stop_and_wait 10258721
  sbatch "$PROJ/scripts/version4/run_mix_d4_2gpu.sh"
}

switch_s22() {
  local result=$PROJ/checkpoints/v4_qwen3_4b_inloop_tf_s22_d4_n4/result.json
  wait_for_step "$result" 500 10258691
  stop_and_wait 10258691
  sbatch \
    --gres=gpu:2 \
    --quotatype=reserved \
    --job-name=e09_v4_s22d4 \
    --export=ALL,GPUS=2,ACCUM=16 \
    "$PROJ/scripts/version4/dynamic_layer/run_s22_d4.sh"
}

case "${ONLY:-both}" in
  mix)
    switch_mix
    ;;
  s22)
    switch_s22
    ;;
  both)
    switch_mix &
    mix_pid=$!
    switch_s22 &
    s22_pid=$!

    status=0
    wait "$mix_pid" || status=1
    wait "$s22_pid" || status=1
    exit "$status"
    ;;
  *)
    echo "ONLY 必须是 mix、s22 或 both" >&2
    exit 2
    ;;
esac
