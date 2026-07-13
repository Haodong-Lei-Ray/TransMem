#!/bin/bash
#SBATCH -J e09_curveev
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH -t 12:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_checkpoint_curve/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_checkpoint_curve/%j.err

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"
cd "$PROJ"

CKPT_DIR=${CKPT_DIR:-$PROJ/checkpoints/diagnostics/p1_curve_seed20260713_prefix4980}
FINAL_CKPT=${FINAL_CKPT:-$CKPT_DIR/step_0004980.pt}
EVAL_FILE=${EVAL_FILE:?set EVAL_FILE}
DATA_FORMAT=${DATA_FORMAT:?set DATA_FORMAT}
MAXQ=${MAXQ:-16}
OUT_JSON=${OUT_JSON:?set OUT_JSON}
MAX_ANS=${MAX_ANS:-50}
ATTN=${ATTN:-sdpa}

mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs "$(dirname "$OUT_JSON")"
JOB_ID="${SLURM_JOB_ID:-$(date +%s)_$$}"
MOUNT_POINT="/mnt/petrelfs/leihaodong/tmp/s3_curveev_${JOB_ID}"
CACHE_DIR="/nvme/leihaodong/s3cache_curveev_${JOB_ID}"
mkdir -p "$MOUNT_POINT" "$CACHE_DIR"

/mnt/petrelfs/leihaodong/app/s3mount datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" \
  --allow-delete --allow-overwrite \
  --endpoint-url http://d-ceph-ssd-inside.pjlab.org.cn \
  --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &
S3PID=$!
sleep 20

cleanup() {
  fusermount -u "$MOUNT_POINT" 2>/dev/null || umount "$MOUNT_POINT" 2>/dev/null || true
  kill "$S3PID" 2>/dev/null || true
  rm -rf "$MOUNT_POINT" "$CACHE_DIR" || true
}
trap cleanup EXIT

MODEL_PATH=${MODEL_PATH:-$MOUNT_POINT/leihaodong/Qwen/Qwen3-4B-Instruct-2507}
test -f "$MODEL_PATH/config.json"
test -f "$FINAL_CKPT"

$PY scripts/eval/eval_checkpoint_curve.py \
  --eval_file "$EVAL_FILE" --data_format "$DATA_FORMAT" \
  --model_path "$MODEL_PATH" --checkpoint_dir "$CKPT_DIR" \
  --checkpoints "$FINAL_CKPT" --max_samples "$MAXQ" \
  --max_answer_tokens "$MAX_ANS" --attn_impl "$ATTN" \
  --output_json "$OUT_JSON"

echo "Checkpoint curve evaluation complete: $OUT_JSON"
