#!/bin/bash
#SBATCH -J e09_l318_d4g
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 48:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/v5_llama31_8b/%j_train.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/v5_llama31_8b/%j_train.err

set -euo pipefail
export D=4 GPUS=4 ACCUM=8 LR=5e-5
export CONFIG=/mnt/petrelfs/leihaodong/Project4/transmem/config_layered_llama31_8b_dynamic_gate.json
export MODEL_REL=leihaodong/meta-llama/Llama-3.1-8B-Instruct
export FEAT_REL=leihaodong/Project4/data/hotpotqa_pool_llama31_8b_n4_n8
export OUTPUT_DIR=/mnt/petrelfs/leihaodong/Project4/checkpoints/v5_gate_llama31_8b_scratch_joint_d4
export S3_CKPT=s3://datafrontier/leihaodong/Project4/checkpoints/v5_gate_llama31_8b_scratch_joint_d4
export EXTRA="--init_scheme scratch_joint --gate_calibration_steps 0 --gate_prior_weight 0.0 --max_steps 1250 --seed 20260722 ${EXTRA:-}"
exec bash /mnt/petrelfs/leihaodong/Project4/scripts/version5/run_train_inloop_generic.sh
