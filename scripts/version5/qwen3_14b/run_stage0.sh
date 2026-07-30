#!/bin/bash
#SBATCH -J e09_v5q314_s0
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 48:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/v5_14b/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/v5_14b/%j.err

# Qwen3-14B / HotpotQA Stage0: 复用通用 pool 提取脚本, 输出直写 s3mount
# (runner 默认 OUT_ROOT=<mount>/leihaodong/Project4/data/hotpotqa_pool${OUT_SUFFIX}),
# manifest 留 petrelfs 支持断点续抽. in-loop 只消费 answer_ids+hq_tea.
set -euo pipefail

export ModelName=Qwen/Qwen3-14B
export POOL_NS=4,8
export N=4
export MAX_ANS=200
export WORKERS=4
export OUT_SUFFIX=_qwen3_14b_n4_n8

exec bash /mnt/petrelfs/leihaodong/Project4/scripts/stage0/hotpotqa/run_stage0_pool.sh
