#!/bin/bash
#SBATCH -J e09_locfull_4b
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/data/logs/locomo/full_%A_%a.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/data/logs/locomo/full_%A_%a.err

# Full locomo10.json Stage0, intentionally isolated from locomo-train Stage0.
# Array index 0/1/2 extracts N=4/8/16 into separate directories.
set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
NS=(4 8 16)
export N=${NS[$SLURM_ARRAY_TASK_ID]}
export DATA=/mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo10.json
export OUT_ROOT=$PROJ/data/locomo10_data/Qwen3-4B-Instruct-2507
export OUTPUT_TAG=stage0_full_short50_n${N}
export ModelName=Qwen/Qwen3-4B-Instruct-2507

exec bash "$PROJ/scripts/train/stage0/locomo/run_stage0_qwen3_4b_d4.sh"
