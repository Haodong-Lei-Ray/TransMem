#!/bin/bash
#SBATCH -J e09_v38gl4lr5
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b_layered/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b_layered/%j.err

# Qwen3-8B gate-D4 重调参臂 B: 与 10263754 (臂 A: lr=1e-4, 3 epoch≈2958 步) 唯一
# 区别是 lr=5e-5 + max_steps=1250 (v5 统一配方, 同 14B 训练). 同 seed, 同 4x8=32 批.
# 动机: 8B 借用 4B 配方时 in-domain 收敛快但 LoCoMo 无增益 → 降 lr + 缩步数抗过拟合.
set -euo pipefail

export D=4
export GPUS=${GPUS:-4}
export ACCUM=${ACCUM:-8}
export LR=5e-5
export CONFIG=/mnt/petrelfs/leihaodong/Project4/transmem/config_layered_8b_dynamic_gate.json
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/petrelfs/leihaodong/Project4/checkpoints/v5_gate_qwen3_8b_scratch_joint_d4_lr5e5_ms1250}
export S3_CKPT=${S3_CKPT:-s3://datafrontier/leihaodong/Project4/checkpoints/v5_gate_qwen3_8b_scratch_joint_d4_lr5e5_ms1250}
export EXTRA="--init_scheme scratch_joint --gate_calibration_steps 0 --gate_prior_weight 0.0 --max_steps 1250 --seed 20260714 ${EXTRA:-}"

exec bash /mnt/petrelfs/leihaodong/Project4/scripts/version3/qwen3_8b_layered/run_train_inloop_d8.sh
