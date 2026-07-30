#!/bin/bash
#SBATCH -J e09_v38gl4_tr
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b_layered/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b_layered/%j.err

# Qwen3-8B D=4 动态 gate (方案 B / scratch_joint): TransMem 与 centered-sigmoid
# gate_proj 从零联合训练. 复用 D=8 主线 runner (--D/--config/EXTRA 透传),
# 只切换 config 到 8B 的 gate 版本并注入 gate CLI 参数.
set -euo pipefail

export D=4
export CONFIG=/mnt/petrelfs/leihaodong/Project4/transmem/config_layered_8b_dynamic_gate.json
# 默认 OUTPUT_DIR 会和固定-gate 的 v3_2_qwen3_8b_inloop_tf_d4_n4 撞车, 必须另起目录
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/petrelfs/leihaodong/Project4/checkpoints/v4_gate_qwen3_8b_layered_scratch_joint_d4}
export S3_CKPT=${S3_CKPT:-s3://datafrontier/leihaodong/Project4/checkpoints/v4_gate_qwen3_8b_layered_scratch_joint_d4}
export EXTRA="--init_scheme scratch_joint --gate_calibration_steps 0 --gate_prior_weight 0.0 --seed 20260714 ${EXTRA:-}"
exec bash /mnt/petrelfs/leihaodong/Project4/scripts/version3/qwen3_8b_layered/run_train_inloop_d8.sh
