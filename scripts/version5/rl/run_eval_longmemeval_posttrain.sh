#!/bin/bash
#SBATCH -J e09_v5glme
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_longmemeval/%j_v5grpo.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_longmemeval/%j_v5grpo.err

set -euo pipefail
PROJ=/mnt/petrelfs/leihaodong/Project4
: "${MODEL_NAME:?MODEL_NAME is required}"
: "${S3_CKPT_REL:?S3_CKPT_REL is required}"
: "${OUT_ROOT:?OUT_ROOT is required}"
export GPU_COUNT=${GPU_COUNT:-4}
export WORKERS=${WORKERS:-$GPU_COUNT}
export THINKING=${THINKING:-1}
export MAX_ANS=${MAX_ANS:-256}
export MAXQ=${MAXQ:-100}
export RUN_OFFICIAL_JUDGE=${RUN_OFFICIAL_JUDGE:-1}
export DATA_FILE=${DATA_FILE:-$PROJ/data/LongMemEval/data/longmemeval_dev.json}

exec bash "$PROJ/scripts/eval/run_eval_longmemeval_parallel.sh"
