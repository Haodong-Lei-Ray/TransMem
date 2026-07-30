#!/bin/bash
#SBATCH -J e09_loc0_8b
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/data/logs/locomo/stage0_8b_%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/data/logs/locomo/stage0_8b_%j.err

# Qwen3-8B LoCoMo Stage0. The shared implementation supplies the same audited
# teacher evidence policy as 4B: labelled evidence +/- 5 turns within its session,
# with speaker_a/speaker_b and session date retained. Output is model-separated.
set -euo pipefail

export ModelName=Qwen/Qwen3-8B
export OUT_ROOT=/mnt/petrelfs/leihaodong/Project4/data/locomo_data/Qwen3-8B

exec bash /mnt/petrelfs/leihaodong/Project4/scripts/stage0/locomo/run_stage0_qwen3_4b_d4.sh
