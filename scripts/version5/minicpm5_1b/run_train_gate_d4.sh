#!/bin/bash
#SBATCH -J e09_mcp1_d4g
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 48:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/v5_minicpm5_1b/%j_train.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/v5_minicpm5_1b/%j_train.err

# MiniCPM5-1B-SFT / HotpotQA-agent / last-D=4 dynamic-gate TransMem.
# Global optimization batch remains 32: 2 ranks x grad_accum 16.
set -euo pipefail

export D=4
export GPUS=2
export ACCUM=16
export LR=5e-5
export CONFIG=/mnt/petrelfs/leihaodong/Project4/transmem/config_layered_minicpm5_1b_dynamic_gate.json
export MODEL_REL=leihaodong/openbmb/MiniCPM5-1B-SFT
export FEAT_REL=leihaodong/Project4/data/hotpotqa_pool_minicpm5_1b_n4_n8
export OUTPUT_DIR=/mnt/petrelfs/leihaodong/Project4/checkpoints/v5_gate_minicpm5_1b_sft_hqa_d4
export S3_CKPT=s3://datafrontier/leihaodong/Project4/checkpoints/v5_gate_minicpm5_1b_sft_hqa_d4
export EXTRA="--init_scheme scratch_joint --gate_calibration_steps 0 --gate_prior_weight 0.0 --max_steps 1250 --seed 20260722 ${EXTRA:-}"

exec bash /mnt/petrelfs/leihaodong/Project4/scripts/version5/run_train_inloop_generic.sh
