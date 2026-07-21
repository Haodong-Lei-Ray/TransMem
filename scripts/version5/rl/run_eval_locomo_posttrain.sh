#!/bin/bash
#SBATCH -J e09_v5rlev
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_v5rlev.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_v5rlev.err

set -euo pipefail
PROJ=/mnt/petrelfs/leihaodong/Project4
RUN_NAME=${RUN_NAME:-v5_opd_qwen3_4b_hqa_d4_rkl_s250}
export S3_CKPT_REL=${S3_CKPT_REL:-leihaodong/Project4/checkpoints/$RUN_NAME/best.pt}
export OUT_ROOT=${OUT_ROOT:-$PROJ/eval_results/locomo_$RUN_NAME}
export GPU_COUNT=${GPU_COUNT:-4}
export WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
export DATA_FILE=/mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo10.json

exec bash "$PROJ/scripts/version4/run_eval_locomo_s32_parallel.sh"
