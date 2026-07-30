#!/bin/bash
#SBATCH -J e09_inloop_mixd4
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/run_inloop/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/run_inloop/%j.err

set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
export D=4
export GPUS=${GPUS:-2}
export ACCUM=${ACCUM:-15}
export POLICY=tf
export EPOCHS=3
export VAL_MAX=128
export TRAIN_DIR=$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507/stage0_train_short200,$PROJ/data/longmemeval_data/Qwen3-4B-Instruct-2507/stage0_train_short200
export DATA_PATH=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem/hotpotqa_train_32k.parquet,$PROJ/data/LongMemEval/data/longmemeval_train.json
export DATA_FORMAT=hotpotqa-agentmem,longmemeval
export VAL_DIR=$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507/stage0_dev_short200,$PROJ/data/longmemeval_data/Qwen3-4B-Instruct-2507/stage0_dev_short200
export VAL_DATA_PATH=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem/hotpotqa_dev.parquet,$PROJ/data/LongMemEval/data/longmemeval_dev.json
export VAL_DATA_FORMAT=hotpotqa-agentmem,longmemeval
export OUTPUT_DIR=$PROJ/checkpoints/v4_inloop_tf_mix_d4

exec bash "$PROJ/scripts/run_inloop.sh"
