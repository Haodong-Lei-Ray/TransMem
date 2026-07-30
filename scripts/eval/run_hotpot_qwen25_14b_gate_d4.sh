#!/bin/bash
#SBATCH -J e09_hp_q25d4
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --quotatype=reserved
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_hotpot_official/%j_q25.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_hotpot_official/%j_q25.err

set -euo pipefail
export GPUS=4 WORKERS_PER_GPU=1
export MODEL_REL=leihaodong/Qwen/Qwen2.5-14B-Instruct
export CKPT_REL=leihaodong/Project4/checkpoints/v5_gate_qwen25_14b_scratch_joint_d4/best.pt
export OUTPUT_DIR=/mnt/petrelfs/leihaodong/Project4/eval_results/hotpot_official_qwen25_14b_gate_d4
exec /mnt/petrelfs/leihaodong/Project4/scripts/eval/run_hotpot_official_parallel.sh
