#!/bin/bash
#SBATCH -J e09_v38l8_loc
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b_layered/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b_layered/%j.err
set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
mkdir -p "$PROJ/logs/qwen3_8b_layered"
export MODE=transmem
export CKPT=${CKPT:-$PROJ/checkpoints/v3_2_qwen3_8b_inloop_tf_d8_n4/best.pt}
export N=4
export OUT_ROOT=${OUT_ROOT:-$PROJ/eval_outputs/locomo_v3_2_qwen3_8b_inloop_tf_d8_n4}
exec bash "$PROJ/scripts/version3/qwen3_8b/run_eval_locomo.sh"
