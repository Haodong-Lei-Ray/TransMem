#!/bin/bash
#SBATCH -J e09_mab_q3g
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH --cpus-per-task=32
#SBATCH -t 48:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_memory_agent_bench/%j_q3_gate_d4.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_memory_agent_bench/%j_q3_gate_d4.err

set -euo pipefail
PROJ=/mnt/petrelfs/leihaodong/Project4
export MODE=paired
export MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
export S3_CKPT_REL=leihaodong/Project4/checkpoints/v4_gate_layered_scratch_joint_s36_d4/best.pt
export CHECKPOINT_ID=s3://datafrontier/$S3_CKPT_REL
export OUT_ROOT=${OUT_ROOT:-$PROJ/eval_results/mab_qwen3_4b_dynamic_gate_d4}
export WORKERS_PER_GPU=${WORKERS_PER_GPU:-1}

exec bash "$PROJ/scripts/eval/Qwen3-4B-Instruct-2507/run_eval_memory_agent_bench_parallel.sh"
