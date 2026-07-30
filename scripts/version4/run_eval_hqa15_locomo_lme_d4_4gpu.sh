#!/bin/bash
#SBATCH -J e09_h15loc_eval
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_h15loc.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_h15loc.err

set -euo pipefail
PROJ=/mnt/petrelfs/leihaodong/Project4
export S3_CKPT_REL=leihaodong/Project4/checkpoints/v4_inloop_tf_hqa15_locomo_lme_d4/best.pt
export OUT_ROOT=$PROJ/eval_outputs/locomo_v4_hqa15_locomo_lme_d4
export GPU_COUNT=4
export WORKERS_PER_GPU=2
exec bash "$PROJ/scripts/version4/run_eval_locomo_s32_parallel.sh"
