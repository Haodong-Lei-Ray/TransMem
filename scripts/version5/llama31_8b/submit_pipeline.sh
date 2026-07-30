#!/bin/bash
set -euo pipefail
ROOT=/mnt/petrelfs/leihaodong/Project4
mkdir -p "$ROOT/logs/v5_llama31_8b"
stage0_job=$(sbatch --parsable "$ROOT/scripts/version5/llama31_8b/run_stage0.sh")
train_job=$(sbatch --parsable --dependency="afterok:${stage0_job}" "$ROOT/scripts/version5/llama31_8b/run_train_gate_d4.sh")
echo "STAGE0_JOB=$stage0_job"
echo "TRAIN_JOB=$train_job"
