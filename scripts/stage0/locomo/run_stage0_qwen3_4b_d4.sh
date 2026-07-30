#!/bin/bash
#SBATCH -J e09_loc0_4b
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/data/logs/locomo/stage0_%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/data/logs/locomo/stage0_%j.err

# LoCoMo train Stage0 for Qwen3-4B, used by later layered TransMem D=4 in-loop training.
# Teacher C_S contains every labelled evidence turn plus five turns before/after
# within the same session, retaining speaker names and session date. One flattened
# QA is one sample; N=4 memory slots. D=4 does not require --dump_layers:
# in-loop training recomputes the four injected LLM layers and reads only answer_ids/hq_tea.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=

PROJ=/mnt/petrelfs/leihaodong/Project4
DATA=${DATA:-/mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo-train.json}
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"

N=${N:-4}
MAX_ANS=${MAX_ANS:-50}
ATTN=${ATTN:-sdpa}
MAXN=${MAXN:-}
ModelName=${ModelName:-Qwen/Qwen3-4B-Instruct-2507}
OUT_ROOT=${OUT_ROOT:-$PROJ/data/locomo_data/$(basename "$ModelName")}
OUTPUT_TAG=${OUTPUT_TAG:-stage0_train_short${MAX_ANS}_n${N}}
OUTPUT_DIR="$OUT_ROOT/$OUTPUT_TAG"
MANIFEST_DIR="$OUTPUT_DIR/.worker_manifests"

mkdir -p "$PROJ/data/logs/locomo" "$OUTPUT_DIR" /mnt/petrelfs/leihaodong/s3mount_logs
JOB_ID="${SLURM_JOB_ID:-$(date +%s)_$$}"
MOUNT_POINT="/mnt/petrelfs/leihaodong/tmp/s3_locomo0_${JOB_ID}"
CACHE_DIR="/nvme/leihaodong/s3cache_locomo0_${JOB_ID}"
mkdir -p "$MOUNT_POINT" "$CACHE_DIR"

/mnt/petrelfs/leihaodong/app/s3mount datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" \
  --allow-delete --allow-overwrite \
  --endpoint-url http://d-ceph-ssd-inside.pjlab.org.cn \
  --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &
S3PID=$!

cleanup() {
  fusermount -u "$MOUNT_POINT" 2>/dev/null || umount "$MOUNT_POINT" 2>/dev/null || true
  kill "$S3PID" 2>/dev/null || true
  rm -rf "$MOUNT_POINT" "$CACHE_DIR" || true
}
trap cleanup EXIT

for _ in $(seq 1 12); do
  MODEL_PATH=${MODEL_PATH:-$MOUNT_POINT/leihaodong/$ModelName}
  [[ -f "$MODEL_PATH/config.json" ]] && break
  sleep 5
done
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "FATAL: model is not visible at $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$DATA" ]]; then
  echo "FATAL: LoCoMo train file is missing: $DATA" >&2
  exit 1
fi

requeue_or_die() {
  if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_RESTART_COUNT:-0}" -lt 4 ]]; then
    echo "Stage0 failed; requeue job $SLURM_JOB_ID for resumable extraction"
    scontrol requeue "$SLURM_JOB_ID" && sleep 120
  fi
  exit 1
}

cd "$PROJ"
$PY -m transmem.extract_features \
  --data_path "$DATA" --data_format locomo \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --manifest_dir "$MANIFEST_DIR" \
  --N "$N" --max_answer_tokens "$MAX_ANS" --num_workers 1 \
  --attn_impl "$ATTN" --save_dtype bfloat16 \
  ${MAXN:+--max_samples "$MAXN"} || requeue_or_die

echo "Stage0 complete: $OUTPUT_DIR"
