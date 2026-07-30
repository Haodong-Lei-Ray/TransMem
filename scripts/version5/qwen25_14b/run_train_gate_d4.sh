#!/bin/bash
#SBATCH -J e09_v5q2514_tr
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 48:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/v5_14b/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/v5_14b/%j.err

# Qwen2.5-14B-Instruct D=4 动态 gate (scratch_joint), v5 统一配方:
# lr=5e-5, max_steps=1250, 全局批 4x8=32, seed 20260714.
# backbone 为 qwen2 架构: rollout 只用 model.model.layers 前向 hook, 结构同名兼容.
set -euo pipefail

export D=4
export GPUS=${GPUS:-4}
export ACCUM=${ACCUM:-8}
export LR=5e-5
export CONFIG=/mnt/petrelfs/leihaodong/Project4/transmem/config_layered_qwen25_14b_dynamic_gate.json
export MODEL_REL=leihaodong/Qwen/Qwen2.5-14B-Instruct
export FEAT_REL=leihaodong/Project4/data/hotpotqa_pool_qwen25_14b_n4_n8
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/petrelfs/leihaodong/Project4/checkpoints/v5_gate_qwen25_14b_scratch_joint_d4}
export S3_CKPT=${S3_CKPT:-s3://datafrontier/leihaodong/Project4/checkpoints/v5_gate_qwen25_14b_scratch_joint_d4}
export EXTRA="--init_scheme scratch_joint --gate_calibration_steps 0 --gate_prior_weight 0.0 --max_steps 1250 --seed 20260714 ${EXTRA:-}"

exec bash /mnt/petrelfs/leihaodong/Project4/scripts/version5/run_train_inloop_generic.sh
