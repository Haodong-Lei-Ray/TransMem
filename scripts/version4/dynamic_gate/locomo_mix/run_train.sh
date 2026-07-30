#!/bin/bash

# Shared Qwen3-4B dynamic-gate trainer for the two LoCoMo mixtures.
# Submit one of the four sbatch entrypoints in this directory.
set -euo pipefail

: "${CORPUS:?需要 CORPUS=locomo_train|locomo_full}"
: "${D:?需要 D=4|8}"

PROJ=/mnt/petrelfs/leihaodong/Project4
LOCOMO_ROOT=/mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data
LME_FEAT=$PROJ/data/longmemeval_data/Qwen3-4B-Instruct-2507
LME_DATA=$PROJ/data/LongMemEval/data

case "$CORPUS" in
  locomo_train)
    LOCOMO_FEAT=$PROJ/data/locomo_data/Qwen3-4B-Instruct-2507/stage0_train_short50_n4
    LOCOMO_DATA=$LOCOMO_ROOT/locomo-train.json
    RUN_TAG=lme_locomo_train
    ;;
  locomo_full)
    LOCOMO_FEAT=$PROJ/data/locomo10_data/Qwen3-4B-Instruct-2507/stage0_full_short50_n4
    LOCOMO_DATA=$LOCOMO_ROOT/locomo10.json
    RUN_TAG=lme_locomo_full
    ;;
  *)
    echo "FATAL: CORPUS=$CORPUS 不是 locomo_train|locomo_full" >&2
    exit 2
    ;;
esac

export INIT_SCHEME=scratch_joint
export S=36
export D
export GPUS=4
export ACCUM=8                 # global batch = 4 * 8 = 32
export JOINT_STEPS=${JOINT_STEPS:-1250}
export SEED=20260716
export TRAIN_DIR=$LME_FEAT/stage0_train_short200,$LOCOMO_FEAT
export DATA_PATH=$LME_DATA/longmemeval_train.json,$LOCOMO_DATA
export DATA_FORMAT=longmemeval,locomo
# Keep validation independent of the LoCoMo training split.
export VAL_DIR=$LME_FEAT/stage0_dev_short200
export VAL_DATA_PATH=$LME_DATA/longmemeval_dev.json
export VAL_DATA_FORMAT=longmemeval
export OUTPUT_DIR=$PROJ/checkpoints/v4_gate_${RUN_TAG}_d${D}
export EXTRA="--save_nvme_s3 --gradient_checkpointing ${EXTRA_ARGS:-}"

exec "$PROJ/scripts/version4/dynamic_gate/run_layered_inloop.sh"
