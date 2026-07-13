#!/bin/bash
#SBATCH -J e09_hmabl
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH -t 12:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_hm_ablation/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_hm_ablation/%j.err

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"
cd "$PROJ"

VARIANT=${VARIANT:?set VARIANT=student|real|shuffled|zero}
EVAL_FILE=${EVAL_FILE:-$PROJ/data/LongMemEval/data/longmemeval_dev.json}
DATA_FORMAT=${DATA_FORMAT:-longmemeval}
STAGE0_DIR=${STAGE0_DIR:-$PROJ/data/longmemeval_data/Qwen3-4B-Instruct-2507/stage0_dev_short200}
CKPT=${CKPT:-$PROJ/checkpoints/offpolicy_v3_p1_lmehqa_d4e60_forward_kl/best.pt}
OUT_ROOT=${OUT_ROOT:-$PROJ/eval_outputs/diagnostics/p1_lme_hm}
MAXQ=${MAXQ:-100}
MAX_ANS=${MAX_ANS:-50}
SEED=${SEED:-20260713}
ATTN=${ATTN:-sdpa}

mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs "$OUT_ROOT"
JOB_ID="${SLURM_JOB_ID:-$(date +%s)_$$}"
MOUNT_POINT="/mnt/petrelfs/leihaodong/tmp/s3_hmabl_${JOB_ID}"
CACHE_DIR="/nvme/leihaodong/s3cache_hmabl_${JOB_ID}"
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
test -f "$CKPT"

$PY scripts/eval/eval_hm_ablation.py \
  --eval_file "$EVAL_FILE" --data_format "$DATA_FORMAT" \
  --stage0_dir "$STAGE0_DIR" --model_path "$MODEL_PATH" --ckpt "$CKPT" \
  --variant "$VARIANT" --max_samples "$MAXQ" --max_answer_tokens "$MAX_ANS" \
  --seed "$SEED" --attn_impl "$ATTN" \
  --output_json "$OUT_ROOT/${VARIANT}.json"

echo "HM ablation complete: $OUT_ROOT/${VARIANT}.json"
