#!/bin/bash
#SBATCH -J e09_h15loclme_d4
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/run_inloop/%j_h15loclme.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/run_inloop/%j_h15loclme.err

# Qwen3-4B layered in-loop D=4, fixed-gate TransMem.
# Train = fixed random 15% of available HQA Stage0 + all LoCoMo-train + all LME.
# HQA's literal 15% is still ~91% of the concatenated samples; log reports counts.
set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
export D=4
export GPUS=8
export ACCUM=4                   # global batch = 8 * 4 = 32
export POLICY=tf
export EPOCHS=3
export VAL_MAX=128
export VAL_INTERVAL=100
export TRAIN_DIR=$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507/stage0_train_short200,$PROJ/data/locomo_data/Qwen3-4B-Instruct-2507/stage0_train_short50_n4,$PROJ/data/longmemeval_data/Qwen3-4B-Instruct-2507/stage0_train_short200
export DATA_PATH=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem/hotpotqa_train_32k.parquet,/mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo-train.json,$PROJ/data/LongMemEval/data/longmemeval_train.json
export DATA_FORMAT=hotpotqa-agentmem,locomo,longmemeval
export VAL_DIR=$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507/stage0_dev_short200,$PROJ/data/longmemeval_data/Qwen3-4B-Instruct-2507/stage0_dev_short200
export VAL_DATA_PATH=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem/hotpotqa_dev.parquet,$PROJ/data/LongMemEval/data/longmemeval_dev.json
export VAL_DATA_FORMAT=hotpotqa-agentmem,longmemeval
export OUTPUT_DIR=$PROJ/checkpoints/v4_inloop_tf_hqa15_locomo_lme_d4
export EXTRA="--data_fractions 0.15,1,1 --data_sample_seed 42 --seed 42 --save_nvme_s3"

exec bash "$PROJ/scripts/train/run_inloop.sh"
