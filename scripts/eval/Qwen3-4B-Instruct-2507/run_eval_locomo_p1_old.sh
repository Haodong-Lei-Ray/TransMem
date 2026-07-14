#!/bin/bash
#SBATCH -J e09_locp1
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo_p1_old/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo_p1_old/%j.err

# Dedicated P1 evaluation on the original LoCoMo checkout requested for v3.
# Defaults run the frozen student baseline and the P1 TransMem checkpoint.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY=("$UV" run --python "$VENV/bin/python" python)
export NLTK_DATA=/mnt/petrelfs/leihaodong/nltk_data
cd "$PROJ"

DATA_FILE=${DATA_FILE:-/mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo10.json}
CKPT=${CKPT:-$PROJ/checkpoints/offpolicy_v3_p1_lmehqa_d4e60_forward_kl/best.pt}
MODES=${MODES:-"student transmem"}
N=${N:-4}
MAX_ANS=${MAX_ANS:-50}
CATS=${CATS:-"1 2 3 4"}
MAXQ=${MAXQ:-}
ATTN=${ATTN:-sdpa}
OUT_ROOT=${OUT_ROOT:-$PROJ/eval_outputs/locomo_v3_p1_olddata}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}

if [[ ! -f "$DATA_FILE" ]]; then
  echo "FATAL: LoCoMo data not found: $DATA_FILE" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "FATAL: P1 checkpoint not found: $CKPT" >&2
  exit 1
fi

# Mount the model inside the compute job. A restart-specific cache keeps every
# s3mount invocation on an empty directory, including Slurm requeues.
RUN_ID="${SLURM_JOB_ID:-manual_$$_$(date +%s)}_${SLURM_RESTART_COUNT:-0}"
MOUNT_POINT="/mnt/petrelfs/leihaodong/tmp/s3_locomo_p1_old_${RUN_ID}"
CACHE_DIR="/nvme/leihaodong/s3cache_locomo_p1_old_${RUN_ID}"
S3_LOG_DIR=/mnt/petrelfs/leihaodong/s3mount_logs
mkdir -p "$MOUNT_POINT" "$CACHE_DIR" "$S3_LOG_DIR" "$OUT_ROOT"

/mnt/petrelfs/leihaodong/app/s3mount datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" \
  --endpoint-url http://d-ceph-ssd-inside.pjlab.org.cn \
  --force-path-style \
  --log-directory "$S3_LOG_DIR" &
S3_PID=$!

cleanup() {
  fusermount -u "$MOUNT_POINT" 2>/dev/null \
    || umount "$MOUNT_POINT" 2>/dev/null \
    || true
  kill "$S3_PID" 2>/dev/null || true
  rm -rf "$MOUNT_POINT" "$CACHE_DIR" || true
}
trap cleanup EXIT

sleep 20
if ! kill -0 "$S3_PID" 2>/dev/null; then
  echo "FATAL: s3mount exited before the model became available" >&2
  exit 1
fi

MODEL_PATH=${MODEL_PATH:-$MOUNT_POINT/leihaodong/$MODEL_NAME}
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "FATAL: model is not visible at $MODEL_PATH" >&2
  exit 1
fi

read -r -a MODE_ARGS <<< "$MODES"
read -r -a CATEGORY_ARGS <<< "$CATS"
echo "Data : $DATA_FILE"
echo "Model: $MODEL_PATH"
echo "CKPT : $CKPT"
echo "Modes: ${MODE_ARGS[*]}"
echo "Out  : $OUT_ROOT"

for MODE in "${MODE_ARGS[@]}"; do
  case "$MODE" in
    teacher|student|transmem) ;;
    *)
      echo "FATAL: unsupported LoCoMo mode: $MODE" >&2
      exit 1
      ;;
  esac

  ARGS=(
    --data_file "$DATA_FILE"
    --model_path "$MODEL_PATH"
    --mode "$MODE"
    --N "$N"
    --max_answer_tokens "$MAX_ANS"
    --categories "${CATEGORY_ARGS[@]}"
    --attn_impl "$ATTN"
    --output_json "$OUT_ROOT/locomo_$MODE.json"
  )
  if [[ "$MODE" == transmem ]]; then
    ARGS+=(--ckpt "$CKPT")
  fi
  if [[ -n "$MAXQ" ]]; then
    ARGS+=(--max_questions "$MAXQ")
  fi

  echo "############ LoCoMo P1 mode=$MODE ############"
  "${PY[@]}" scripts/eval/eval_locomo.py "${ARGS[@]}"
done

echo "LoCoMo P1 evaluation complete: $OUT_ROOT"
