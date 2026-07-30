#!/bin/bash
#SBATCH -J e09_v38_mab
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b/%j.err
set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
export MODE=${MODE:-student}
export ModelName=Qwen/Qwen3-8B
export OUT_ROOT=${OUT_ROOT:?Set OUT_ROOT}
export SOURCES=${SOURCES:?Set SOURCES}
if [[ "$MODE" == paired ]]; then
  export CKPT=${CKPT:?Set CKPT for paired mode}
fi
exec bash "$PROJ/scripts/eval/Qwen3-4B-Instruct-2507/run_eval_memory_agent_bench.sh"
