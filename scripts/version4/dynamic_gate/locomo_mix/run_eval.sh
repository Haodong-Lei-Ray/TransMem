#!/bin/bash

# Full LoCoMo category 1-4 evaluation.  This intentionally does not exclude
# questions present in locomo-train.json.
set -euo pipefail

: "${CORPUS:?需要 CORPUS=locomo_train|locomo_full}"
: "${D:?需要 D=4|8}"

PROJ=/mnt/petrelfs/leihaodong/Project4
case "$CORPUS" in
  locomo_train) RUN_TAG=lme_locomo_train ;;
  locomo_full) RUN_TAG=lme_locomo_full ;;
  *) echo "FATAL: CORPUS=$CORPUS 不是 locomo_train|locomo_full" >&2; exit 2 ;;
esac

export S3_CKPT_REL=leihaodong/Project4/checkpoints/v4_gate_${RUN_TAG}_d${D}/best.pt
export OUT_ROOT=$PROJ/eval_outputs/locomo_v4_gate_${RUN_TAG}_d${D}
export GPU_COUNT=4
export WORKERS_PER_GPU=2
export DATA_FILE=/mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo10.json

exec "$PROJ/scripts/version4/run_eval_locomo_s32_parallel.sh"
