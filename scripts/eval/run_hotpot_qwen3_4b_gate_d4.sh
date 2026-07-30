#!/bin/bash
#SBATCH -J e09_hp_q3d4
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --quotatype=reserved
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_hotpot_official/%j_q3.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_hotpot_official/%j_q3.err

set -euo pipefail
export GPUS=4 WORKERS_PER_GPU=2
export MODEL_REL=leihaodong/Qwen/Qwen3-4B-Instruct-2507
export CKPT_REL=leihaodong/Project4/checkpoints/v4_gate_layered_scratch_joint_s36_d4/best.pt
export OUTPUT_DIR=/mnt/petrelfs/leihaodong/Project4/eval_results/hotpot_official_qwen3_4b_gate_d4
exec /mnt/petrelfs/leihaodong/Project4/scripts/eval/run_hotpot_official_parallel.sh
