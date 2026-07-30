#!/bin/bash
#SBATCH -J e09_lc_l318
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_llama31_8b_gate_d4.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_llama31_8b_gate_d4.err

set -euo pipefail
PROJ=/mnt/petrelfs/leihaodong/Project4
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export S3_CKPT_REL=leihaodong/Project4/checkpoints/v5_gate_llama31_8b_scratch_joint_d4/best.pt
export OUT_ROOT=$PROJ/eval_results/locomo_llama31_8b_dynamic_gate_d4
export DATA_FILE=/mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo10.json
export GPU_COUNT=${GPU_COUNT:-1}
export WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
export MAX_ANS=50
export THINKING=0
exec bash "$PROJ/scripts/version4/run_eval_locomo_s32_parallel.sh"
