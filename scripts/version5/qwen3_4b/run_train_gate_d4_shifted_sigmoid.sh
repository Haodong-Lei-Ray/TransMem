#!/bin/bash
#SBATCH -J e09_q3g3sm1
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH -t 48:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/v5_qwen3_4b/%j_shifted_gate_d4.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/v5_qwen3_4b/%j_shifted_gate_d4.err

# Qwen3-4B, D=4, hotpotqa-agent, scratch-joint ablation:
#   z = (W_g h^m) / tau
#   g = 3 * sigmoid(z) - 1, range (-1, 2), identity gate init g=1.
# This wrapper opts into a new config; the historical centered-sigmoid scripts
# and checkpoints remain byte/behavior compatible.
set -euo pipefail

echo "=== allocated GPU capacity before shifted-gate training ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free \
  --format=csv || true

PROJ=/mnt/petrelfs/leihaodong/Project4
export D=4
export GPUS=${GPUS:-8}
export ACCUM=${ACCUM:-4}
export LR=1e-4
export CONFIG=$PROJ/transmem/config_layered_shifted_sigmoid.json
export MODEL_REL=leihaodong/Qwen/Qwen3-4B-Instruct-2507
export FEAT_REL=leihaodong/Project4/data/hotpotqa_data/Qwen3-4B-Instruct-2507
export TRAIN_DIR=$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507/stage0_train_short200
export VAL_DIR=$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507/stage0_dev_short200
export DATA_PATH=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem/hotpotqa_train_32k.parquet
export VAL_DATA_PATH=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem/hotpotqa_dev.parquet
export OUTPUT_DIR=$PROJ/checkpoints/v5_gate_qwen3_4b_shifted_sigmoid_d4
export S3_CKPT=s3://datafrontier/leihaodong/Project4/checkpoints/v5_gate_qwen3_4b_shifted_sigmoid_d4
export SAVE_NVME_S3=1
export EXTRA="--init_scheme scratch_joint --gate_calibration_steps 0 --gate_prior_weight 0.0 --max_steps 1250 --gate_lr 1e-4 --seed 20260714 ${EXTRA:-}"

exec bash $PROJ/scripts/version5/run_train_inloop_generic.sh
