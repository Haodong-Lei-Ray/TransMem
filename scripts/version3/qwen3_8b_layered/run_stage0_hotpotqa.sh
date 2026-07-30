#!/bin/bash
#SBATCH -J e09_v38l8_s0
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/data/logs/qwen3_8b_layered/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/data/logs/qwen3_8b_layered/%j.err

# Qwen3-8B / HotpotQA Stage0. 续用已有 qwen3_8b_n4_n8 manifest；
# in-loop 只消费 answer_ids+hq_tea，pool HM 不进入新架构。
set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
mkdir -p "$PROJ/data/logs/qwen3_8b_layered"

export ModelName=Qwen/Qwen3-8B
export POOL_NS=4,8
export N=4
export MAX_ANS=200
export THINKING=false
export WORKERS=${WORKERS:-4}
export OUT_SUFFIX=_qwen3_8b_n4_n8
export OUT_ROOT=$PROJ/data/hotpotqa_data/Qwen3-8B-pool-n4-n8

exec bash "$PROJ/scripts/stage0/hotpotqa/run_stage0_pool.sh"
