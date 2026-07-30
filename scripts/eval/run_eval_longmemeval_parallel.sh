#!/bin/bash
#SBATCH -J e09_lme_par
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH --cpus-per-task=32
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_longmemeval/%j_parallel.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_longmemeval/%j_parallel.err

# Multi-GPU LongMemEval dev evaluation.  One independent model worker is
# pinned to each allocated GPU; questions are round-robin sharded and merged.
# The merged hypotheses are then scored with the benchmark's official GPT-4o
# judge in Project4/data/longmemeval.
set -euo pipefail

: "${OUT_ROOT:?Set a unique OUT_ROOT}"
: "${MODEL_NAME:?Set the S3 model path below leihaodong/, e.g. Qwen/Qwen3-4B-Instruct-2507}"
: "${S3_CKPT_REL:?Set the checkpoint path inside the datafrontier bucket}"

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY=("$UV" run --python "$VENV/bin/python" python)
ENDPOINT=http://d-ceph-ssd-inside.pjlab.org.cn
DATA_FILE=${DATA_FILE:-$PROJ/data/LongMemEval/data/longmemeval_dev.json}
MODE=${MODE:-transmem}
MAX_ANS=${MAX_ANS:-50}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-}
MAXQ=${MAXQ:-100}
RUN_OFFICIAL_JUDGE=${RUN_OFFICIAL_JUDGE:-1}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-4o}

if [[ -n "${GPU_COUNT:-}" ]]; then
  :
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]]; then
  IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
  GPU_COUNT=${#visible_gpus[@]}
else
  GPU_COUNT=1
fi
WORKERS=${WORKERS:-$GPU_COUNT}
if ((GPU_COUNT < 1 || WORKERS < 1)); then
  echo "FATAL: GPU_COUNT and WORKERS must be positive" >&2
  exit 2
fi

# The official judge needs outbound network access.  Save proxies while the
# generation phase disables them for the internal S3 endpoint.
SAVED_http_proxy=${http_proxy-}
SAVED_https_proxy=${https_proxy-}
SAVED_HTTP_PROXY=${HTTP_PROXY-}
SAVED_HTTPS_PROXY=${HTTPS_PROXY-}
SAVED_all_proxy=${all_proxy-}
SAVED_ALL_PROXY=${ALL_PROXY-}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

cd "$PROJ"
mkdir -p "$OUT_ROOT" "$PROJ/logs/eval_longmemeval" \
  /mnt/petrelfs/leihaodong/s3mount_logs
MOUNT_POINT=/mnt/petrelfs/leihaodong/tmp/s3_lme_parallel_${SLURM_JOB_ID}
CACHE_DIR=/nvme/leihaodong/s3cache_lme_parallel_${SLURM_JOB_ID}
LOCAL_CKPT_DIR=/nvme/leihaodong/lme_ckpt_${SLURM_JOB_ID}
fusermount -u "$MOUNT_POINT" 2>/dev/null || true
rm -rf "$MOUNT_POINT" "$CACHE_DIR" "$LOCAL_CKPT_DIR" 2>/dev/null || true
mkdir -p "$MOUNT_POINT" "$CACHE_DIR" "$LOCAL_CKPT_DIR"

/mnt/petrelfs/leihaodong/app/s3mount datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" --allow-delete --allow-overwrite \
  --endpoint-url "$ENDPOINT" --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &
S3PID=$!
cleanup() {
  fusermount -u "$MOUNT_POINT" 2>/dev/null || umount "$MOUNT_POINT" 2>/dev/null || true
  kill "$S3PID" 2>/dev/null || true
  rm -rf "$MOUNT_POINT" "$CACHE_DIR" "$LOCAL_CKPT_DIR" || true
}
trap cleanup EXIT
sleep 20

MODEL_PATH=$MOUNT_POINT/leihaodong/$MODEL_NAME
SOURCE_CKPT=$MOUNT_POINT/$S3_CKPT_REL
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "FATAL: model is not visible: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_CKPT" ]]; then
  echo "FATAL: checkpoint is not visible: $SOURCE_CKPT" >&2
  exit 1
fi
EVAL_CKPT=$LOCAL_CKPT_DIR/$(basename "$SOURCE_CKPT")
cp "$SOURCE_CKPT" "$EVAL_CKPT"

echo "LongMemEval parallel eval: model=$MODEL_NAME checkpoint=$S3_CKPT_REL gpus=$GPU_COUNT workers=$WORKERS data=$DATA_FILE output=$OUT_ROOT thinking=${THINKING:-0} max_prompt_tokens=${MAX_PROMPT_TOKENS:-none}"
pids=()
for ((worker=0; worker<WORKERS; worker++)); do
  device=$((worker % GPU_COUNT))
  echo "worker=$worker shard=$worker/$WORKERS device=cuda:$device"
  worker_cmd=("${PY[@]}" scripts/eval/eval_longmemeval.py \
    --data_file "$DATA_FILE" --model_path "$MODEL_PATH" \
    --mode "$MODE" --ckpt "$EVAL_CKPT" --N 4 \
    --max_answer_tokens "$MAX_ANS" --max_samples "$MAXQ")
  [[ "${THINKING:-0}" == "1" ]] && worker_cmd+=(--thinking)
  [[ -n "$MAX_PROMPT_TOKENS" ]] && worker_cmd+=(--max_prompt_tokens "$MAX_PROMPT_TOKENS")
  worker_cmd+=( \
    --attn_impl sdpa --device "cuda:$device" \
    --num_shards "$WORKERS" --shard_index "$worker" \
    --output_jsonl "$OUT_ROOT/shard_${worker}.jsonl" \
    --summary_json "$OUT_ROOT/shard_${worker}.summary.json")
  "${worker_cmd[@]}" \
    >"$OUT_ROOT/shard_${worker}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "FATAL: at least one LongMemEval GPU worker failed; progress files are retained" >&2
  exit "$status"
fi

shards=()
for ((worker=0; worker<WORKERS; worker++)); do
  shards+=("$OUT_ROOT/shard_${worker}.jsonl")
done
"${PY[@]}" scripts/eval/merge_longmemeval_shards.py \
  --reference "$DATA_FILE" \
  --output_jsonl "$OUT_ROOT/hypotheses.jsonl" \
  --metrics_json "$OUT_ROOT/internal_metrics.json" \
  "${shards[@]}"

if [[ "$RUN_OFFICIAL_JUDGE" == "1" ]]; then
  if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
    # The lab gateway is directly reachable; inherited public-network proxies
    # can make requests to its IP hang indefinitely.
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    export all_proxy= ALL_PROXY=
  else
    export http_proxy="$SAVED_http_proxy" https_proxy="$SAVED_https_proxy"
    export HTTP_PROXY="$SAVED_HTTP_PROXY" HTTPS_PROXY="$SAVED_HTTPS_PROXY"
    export all_proxy="$SAVED_all_proxy" ALL_PROXY="$SAVED_ALL_PROXY"
  fi
  echo "Running official LongMemEval judge: $JUDGE_MODEL"
  export OUT_ROOT DATA_FILE JUDGE_MODEL
  bash scripts/eval/run_longmemeval_official_judge.sh
fi
echo "LongMemEval parallel evaluation complete: $OUT_ROOT"
