#!/bin/bash
#SBATCH -J e09_l318_s0
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 48:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/v5_llama31_8b/%j_stage0.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/v5_llama31_8b/%j_stage0.err

set -euo pipefail
export ModelName=meta-llama/Llama-3.1-8B-Instruct
export POOL_NS=4,8
export N=4
export MAX_ANS=200
export WORKERS=4
export OUT_SUFFIX=_llama31_8b_n4_n8
export ATTN=sdpa
exec bash /mnt/petrelfs/leihaodong/Project4/scripts/train/stage0/hotpotqa/run_stage0_pool.sh
