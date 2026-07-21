#!/bin/bash
#SBATCH -J e09_loc_s32
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_s32_parallel.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_s32_parallel.err

set -euo pipefail

: "${OUT_ROOT:?需要传入独立的 OUT_ROOT}"
# MODE=transmem(默认)|student|teacher; 非 transmem 模式不需要 checkpoint
MODE=${MODE:-transmem}
if [[ "$MODE" == "transmem" && -z "${CKPT:-}" && -z "${S3_CKPT_REL:-}" ]]; then
  echo "FATAL: transmem 模式需要传入本地 CKPT 或桶内相对路径 S3_CKPT_REL" >&2
  exit 2
fi

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY=("$UV" run --python "$VENV/bin/python" python)
# Slurm remaps allocated devices to logical cuda:0..N-1.  Count the visible
# entries instead of trusting physical GPU ids or a hard-coded resource count.
if [[ -n "${GPU_COUNT:-}" ]]; then
  : # explicit override, mainly for local debugging
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]]; then
  IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
  GPU_COUNT=${#visible_gpus[@]}
else
  GPU_COUNT=1
fi
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
WORKERS=${WORKERS:-$((GPU_COUNT * WORKERS_PER_GPU))}
MAX_ANS=${MAX_ANS:-50}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-}
# THINKING=1: --thinking (hybrid 模型 enable_thinking / instruct 模型 prompt hack);
# 配合调大 MAX_ANS (思考链要预算, 如 1024), 结果 json 里 thinking 与 answer 分开存
THINKING=${THINKING:-0}
THINK_ARGS=()
[[ "$THINKING" == "1" ]] && THINK_ARGS=(--thinking)
PROMPT_BUDGET_ARGS=()
[[ -n "$MAX_PROMPT_TOKENS" ]] && PROMPT_BUDGET_ARGS=(--max_prompt_tokens "$MAX_PROMPT_TOKENS")
if ((GPU_COUNT < 1 || WORKERS < 1)); then
  echo "FATAL: GPU_COUNT 和 WORKERS 必须为正数" >&2
  exit 2
fi
DATA_FILE=${DATA_FILE:-/mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo10.json}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}
ENDPOINT=http://d-ceph-ssd-inside.pjlab.org.cn

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
export NLTK_DATA=/mnt/petrelfs/leihaodong/nltk_data
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
cd "$PROJ"
mkdir -p "$OUT_ROOT" "$PROJ/logs/eval_locomo" /mnt/petrelfs/leihaodong/s3mount_logs

MOUNT_POINT=/mnt/petrelfs/leihaodong/tmp/s3_locomo_parallel_${SLURM_JOB_ID}
CACHE_DIR=/nvme/leihaodong/s3cache_locomo_parallel_${SLURM_JOB_ID}
fusermount -u "$MOUNT_POINT" 2>/dev/null || true
rm -rf "$MOUNT_POINT" "$CACHE_DIR" 2>/dev/null || true
mkdir -p "$MOUNT_POINT" "$CACHE_DIR"

/mnt/petrelfs/leihaodong/app/s3mount datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" --allow-delete --allow-overwrite \
  --endpoint-url "$ENDPOINT" --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &
S3PID=$!
cleanup() {
  fusermount -u "$MOUNT_POINT" 2>/dev/null || umount "$MOUNT_POINT" 2>/dev/null || true
  kill "$S3PID" 2>/dev/null || true
  rm -rf "$MOUNT_POINT" "$CACHE_DIR" || true
  rm -rf "${LOCAL_CKPT_DIR:-}" || true
}
trap cleanup EXIT
sleep 20

MODEL_PATH=$MOUNT_POINT/leihaodong/$MODEL_NAME
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "FATAL: 模型不可见 $MODEL_PATH" >&2
  exit 1
fi

# Freeze the selected training checkpoint at job start.  The training job may
# atomically replace best.pt/latest.pt while this evaluation is queued/running.
CKPT_ARGS=()
if [[ "$MODE" == "transmem" ]]; then
  SOURCE_CKPT=${CKPT:-$MOUNT_POINT/$S3_CKPT_REL}
  if [[ ! -f "$SOURCE_CKPT" ]]; then
    echo "FATAL: checkpoint 不可见 $SOURCE_CKPT" >&2
    exit 1
  fi
  LOCAL_CKPT_DIR=/nvme/leihaodong/locomo_ckpt_${SLURM_JOB_ID}
  mkdir -p "$LOCAL_CKPT_DIR"
  EVAL_CKPT=$LOCAL_CKPT_DIR/$(basename "$SOURCE_CKPT")
  cp "$SOURCE_CKPT" "$EVAL_CKPT"
  CKPT_ARGS=(--ckpt "$EVAL_CKPT")
fi
echo "LoCoMo parallel eval: mode=$MODE model=$MODEL_NAME source_ckpt=${SOURCE_CKPT:-none} snapshot=${EVAL_CKPT:-none} gpus=$GPU_COUNT workers=$WORKERS workers_per_gpu=$WORKERS_PER_GPU output=$OUT_ROOT"
pids=()
for ((worker=0; worker<WORKERS; worker++)); do
  device=$((worker % GPU_COUNT))
  echo "worker=$worker shard=$worker/$WORKERS device=cuda:$device"
  "${PY[@]}" scripts/eval/eval_locomo.py \
    --data_file "$DATA_FILE" --model_path "$MODEL_PATH" \
    --mode "$MODE" ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} --N 4 \
    --max_answer_tokens "$MAX_ANS" ${THINK_ARGS[@]+"${THINK_ARGS[@]}"} \
    ${PROMPT_BUDGET_ARGS[@]+"${PROMPT_BUDGET_ARGS[@]}"} \
    --categories 1 2 3 4 --attn_impl sdpa --print_examples 1 \
    --device "cuda:$device" \
    --num_shards "$WORKERS" --shard_index "$worker" \
    --output_json "$OUT_ROOT/shard_${worker}.json" \
    >"$OUT_ROOT/shard_${worker}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "至少一个 LoCoMo worker 失败，保留 progress 文件供断点续跑" >&2
  # spot 抢占把 worker SIGTERM 掉后 batch 会先于 SLURM 自然退出, 自动 requeue
  # 不触发 → 手动兜底 (progress.jsonl 断点续跑, 真代码 bug 最多循环 6 次后停)
  if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_RESTART_COUNT:-0}" -lt 6 ]]; then
    echo "scontrol requeue 兜底 (restart_count=${SLURM_RESTART_COUNT:-0})" >&2
    scontrol requeue "$SLURM_JOB_ID" && sleep 60
  fi
  exit "$status"
fi

shards=()
for ((worker=0; worker<WORKERS; worker++)); do
  shards+=("$OUT_ROOT/shard_${worker}.json")
done
"${PY[@]}" scripts/eval/merge_locomo_shards.py \
  --output "$OUT_ROOT/locomo_${MODE}.json" "${shards[@]}"
echo "LoCoMo parallel eval complete: $OUT_ROOT/locomo_${MODE}.json"
