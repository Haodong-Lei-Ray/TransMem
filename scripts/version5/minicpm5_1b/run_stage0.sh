#!/bin/bash
#SBATCH -J e09_mcp1_s0
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 48:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/v5_minicpm5_1b/%j_stage0.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/v5_minicpm5_1b/%j_stage0.err

# MiniCPM5-1B-SFT / HotpotQA-agent Stage0 teacher trajectory.
# The model and extracted features are read/written through the DataFrontier mount.
set -euo pipefail

export ModelName=openbmb/MiniCPM5-1B-SFT
export POOL_NS=4,8
export N=4
export MAX_ANS=200
export WORKERS=4
export OUT_SUFFIX=_minicpm5_1b_n4_n8
export ATTN=sdpa

exec bash /mnt/petrelfs/leihaodong/Project4/scripts/train/stage0/hotpotqa/run_stage0_pool.sh
