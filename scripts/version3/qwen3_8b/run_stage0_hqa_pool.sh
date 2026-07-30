#!/bin/bash
#SBATCH -J e09_v38_hqa_s0
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/data/logs/qwen3_8b/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/data/logs/qwen3_8b/%j.err
set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
export ModelName=Qwen/Qwen3-8B
export POOL_NS=4,8
export MAX_ANS=200
export WORKERS=${WORKERS:-8}
export OUT_SUFFIX=_qwen3_8b_n4_n8
export OUT_ROOT=${OUT_ROOT:-$PROJ/data/hotpotqa_data/Qwen3-8B-pool-n4-n8}
exec bash "$PROJ/scripts/stage0/hotpotqa/run_stage0_pool.sh"
