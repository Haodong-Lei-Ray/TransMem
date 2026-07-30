#!/bin/bash
# Shared multi-GPU HotpotQA official-dev evaluator. Invoked by an sbatch entrypoint.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

PROJ=/mnt/petrelfs/leihaodong/Project4
PY=$PROJ/.venv-transmem/bin/python
ENDPOINT=http://d-ceph-ssd-inside.pjlab.org.cn
: "${MODEL_REL:?missing MODEL_REL}"
: "${OUTPUT_DIR:?missing OUTPUT_DIR}"
MODE=${MODE:-transmem}
case "$MODE" in
  student) ;;
  transmem) : "${CKPT_REL:?missing CKPT_REL in transmem mode}" ;;
  *) echo "FATAL: MODE must be student or transmem, got $MODE" >&2; exit 2 ;;
esac

GPUS=${GPUS:-${SLURM_GPUS_ON_NODE:-4}}
GPUS=${GPUS%%\(*}
GPUS=${GPUS%%,*}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-1}
NUM_WORKERS=$((GPUS * WORKERS_PER_GPU))
DATA_FILE=$PROJ/data/hotpotqa-benchmark/hotpot/hotpot_dev_distractor_v1_eval.json
[[ -s "$DATA_FILE" ]] || { echo "FATAL: prepared HotpotQA dev missing: $DATA_FILE" >&2; exit 1; }

JOB_ID=${SLURM_JOB_ID:-$(date +%s)_$$}
MOUNT_POINT=/mnt/petrelfs/leihaodong/tmp/s3_hotpot_eval_${JOB_ID}
CACHE_DIR=/nvme/leihaodong/s3cache_hotpot_eval_${JOB_ID}
SNAPSHOT_DIR=/nvme/leihaodong/hotpot_ckpt_${JOB_ID}
mkdir -p "$MOUNT_POINT" "$CACHE_DIR" "$SNAPSHOT_DIR" "$OUTPUT_DIR" \
  /mnt/petrelfs/leihaodong/s3mount_logs

/mnt/petrelfs/leihaodong/app/s3mount datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" --endpoint-url "$ENDPOINT" --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &
S3PID=$!
cleanup() {
  fusermount -u "$MOUNT_POINT" 2>/dev/null || umount "$MOUNT_POINT" 2>/dev/null || true
  kill "$S3PID" 2>/dev/null || true
  rm -rf "$CACHE_DIR" "$SNAPSHOT_DIR"
  rmdir "$MOUNT_POINT" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do [[ -f "$MOUNT_POINT/$MODEL_REL/config.json" ]] && break; sleep 2; done
MODEL_PATH=$MOUNT_POINT/$MODEL_REL
[[ -f "$MODEL_PATH/config.json" ]] || { echo "FATAL: model missing $MODEL_PATH" >&2; exit 1; }
EVAL_MODE_ARGS=(--mode "$MODE")
CKPT=none
if [[ "$MODE" == transmem ]]; then
  SOURCE_CKPT=$MOUNT_POINT/$CKPT_REL
  [[ -f "$SOURCE_CKPT" ]] || { echo "FATAL: checkpoint missing $SOURCE_CKPT" >&2; exit 1; }
  cp "$SOURCE_CKPT" "$SNAPSHOT_DIR/best.pt"
  CKPT=$SNAPSHOT_DIR/best.pt
  EVAL_MODE_ARGS+=(--ckpt "$CKPT")
fi

export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=4
cd "$PROJ"
echo "HOTPOT_EVAL mode=$MODE model=$MODEL_PATH ckpt=$CKPT gpus=$GPUS workers=$NUM_WORKERS output=$OUTPUT_DIR"

pids=()
for worker in $(seq 0 $((NUM_WORKERS - 1))); do
  gpu=$((worker % GPUS))
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$PROJ/scripts/eval/eval_hotpot_official.py" \
    --data-file "$DATA_FILE" --model-path "$MODEL_PATH" \
    "${EVAL_MODE_ARGS[@]}" \
    --output-json "$OUTPUT_DIR/shard_${worker}.json" \
    --num-shards "$NUM_WORKERS" --shard-id "$worker" \
    --max-answer-tokens 50 --attn-impl sdpa \
    >"$OUTPUT_DIR/shard_${worker}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" = 0 ]] || { echo "FATAL: at least one HotpotQA worker failed" >&2; exit 1; }

"$PY" "$PROJ/scripts/eval/merge_hotpot_official_shards.py" \
  --input-dir "$OUTPUT_DIR" --num-shards "$NUM_WORKERS" \
  --output-json "$OUTPUT_DIR/hotpot_official_results.json"
