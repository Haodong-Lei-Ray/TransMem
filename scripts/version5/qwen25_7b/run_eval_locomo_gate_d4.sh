#!/bin/bash
#SBATCH -J e09_lc_q257
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_qwen25_7b_gate_d4.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_qwen25_7b_gate_d4.err

set -euo pipefail
PROJ=/mnt/petrelfs/leihaodong/Project4
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export S3_CKPT_REL=leihaodong/Project4/checkpoints/v5_gate_qwen25_7b_scratch_joint_d4/best.pt
export OUT_ROOT=$PROJ/eval_results/locomo_qwen25_7b_dynamic_gate_d4
export DATA_FILE=/mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo10.json
export GPU_COUNT=1
export WORKERS_PER_GPU=2
export MAX_ANS=50
export THINKING=0
exec bash "$PROJ/scripts/version4/run_eval_locomo_s32_parallel.sh"
