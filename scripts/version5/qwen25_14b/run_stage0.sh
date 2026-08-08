#!/bin/bash
#SBATCH -J e09_v5q2514_s0
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 48:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/v5_14b/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/v5_14b/%j.err

# Qwen2.5-14B-Instruct / HotpotQA Stage0. 注意: 该模型 max_position_embeddings=32768,
# hotpotqa_train_32k 按 32k 预算构建, 极限样本贴边; extract_features 的
# _supports_enable_thinking 守卫会自动跳过 qwen2 不支持的 enable_thinking 开关.
set -euo pipefail

export ModelName=Qwen/Qwen2.5-14B-Instruct
export POOL_NS=4,8
export N=4
export MAX_ANS=200
export WORKERS=4
export OUT_SUFFIX=_qwen25_14b_n4_n8

exec bash /mnt/petrelfs/leihaodong/Project4/scripts/train/stage0/hotpotqa/run_stage0_pool.sh
