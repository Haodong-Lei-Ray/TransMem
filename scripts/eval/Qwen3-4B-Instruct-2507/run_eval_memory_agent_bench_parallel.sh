#!/bin/bash
#SBATCH -J e09_mab_mc
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH --cpus-per-task=32
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_memory_agent_bench/%j_parallel.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_memory_agent_bench/%j_parallel.err

# Multi-GPU, multi-process MemoryAgentBench main-13 evaluation.  Sources are
# assigned whole to workers, so every progress file has exactly one writer and
# the existing per-source resume contract remains unchanged.
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/mnt/petrelfs/leihaodong/nltk_data
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY=("$UV" run --python "$VENV/bin/python" python)
cd "$PROJ"

CKPT=${CKPT:-$PROJ/checkpoints/offpolicy_v3_p1_lmehqa_d4e60_forward_kl/best.pt}
S3_CKPT_REL=${S3_CKPT_REL:-leihaodong/Project4/checkpoints/offpolicy_v3_p1_lmehqa_d4e60_forward_kl/best.pt}
CHECKPOINT_ID=${CHECKPOINT_ID:-}
MODE=${MODE:-paired}
MODEL_NAME=${MODEL_NAME:-${ModelName:-Qwen/Qwen3-4B-Instruct-2507}}
OUT_ROOT=${OUT_ROOT:-$PROJ/eval_outputs/memory_agent_bench_v3_p1_main13_parallel}
MAB_ROOT=${MAB_ROOT:-/mnt/petrelfs/leihaodong/Project1/MemoryAgentBenchProject/MemoryAgentBench}
MAXQ=${MAXQ:-}
SOURCES=${SOURCES:-}
ATTN=${ATTN:-sdpa}
NO_PREFIX_CACHE=${NO_PREFIX_CACHE:-0}
FORCE=${FORCE:-0}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-1}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]]; then
  IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  requested=${GPU_COUNT:-4}
  VISIBLE_GPUS=()
  for ((gpu=0; gpu<requested; gpu++)); do VISIBLE_GPUS+=("$gpu"); done
fi
GPU_COUNT=${GPU_COUNT:-${#VISIBLE_GPUS[@]}}
if ((GPU_COUNT < 1 || GPU_COUNT > ${#VISIBLE_GPUS[@]})); then
  echo "FATAL: GPU_COUNT=$GPU_COUNT but visible GPUs=${VISIBLE_GPUS[*]}" >&2
  exit 2
fi
if ((WORKERS_PER_GPU < 1)); then
  echo "FATAL: WORKERS_PER_GPU must be positive" >&2
  exit 2
fi
WORKERS=${WORKERS:-$((GPU_COUNT * WORKERS_PER_GPU))}
if ((WORKERS < 1 || WORKERS > GPU_COUNT * WORKERS_PER_GPU)); then
  echo "FATAL: WORKERS=$WORKERS exceeds GPU_COUNT*WORKERS_PER_GPU=$((GPU_COUNT * WORKERS_PER_GPU))" >&2
  exit 2
fi

ALL_SOURCES=(
  ruler_qa1_197K ruler_qa2_421K 'longmemeval_s*' eventqa_full
  icl_banking77_5900shot_balance icl_clinic150_7050shot_balance
  icl_nlu_8296shot_balance icl_trec_coarse_6600shot_balance
  icl_trec_fine_6400shot_balance recsys_redial_full
  infbench_sum_eng_shots2 factconsolidation_sh_262k factconsolidation_mh_262k
)
if [[ -n "$SOURCES" ]]; then read -r -a ALL_SOURCES <<< "$SOURCES"; fi

mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs \
  "$PROJ/logs/eval_memory_agent_bench" "$OUT_ROOT"
JOB_ID=${SLURM_JOB_ID:-$(date +%s)_$$}
MOUNT_POINT=/mnt/petrelfs/leihaodong/tmp/s3_mab_parallel_${JOB_ID}
CACHE_DIR=/nvme/leihaodong/s3cache_mab_parallel_${JOB_ID}
LOCAL_CKPT_DIR=/nvme/leihaodong/mab_ckpt_${JOB_ID}
fusermount -u "$MOUNT_POINT" 2>/dev/null || true
rm -rf "$MOUNT_POINT" "$CACHE_DIR" "$LOCAL_CKPT_DIR" 2>/dev/null || true
mkdir -p "$MOUNT_POINT" "$CACHE_DIR" "$LOCAL_CKPT_DIR"

/mnt/petrelfs/leihaodong/app/s3mount datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" --allow-delete --allow-overwrite \
  --endpoint-url http://d-ceph-ssd-inside.pjlab.org.cn \
  --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &
S3PID=$!
pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  fusermount -u "$MOUNT_POINT" 2>/dev/null || umount "$MOUNT_POINT" 2>/dev/null || true
  kill "$S3PID" 2>/dev/null || true
  rm -rf "$MOUNT_POINT" "$CACHE_DIR" "$LOCAL_CKPT_DIR" || true
}
trap cleanup EXIT
sleep 20

MODEL_PATH=${MODEL_PATH:-$MOUNT_POINT/leihaodong/$MODEL_NAME}
[[ -f "$MODEL_PATH/config.json" ]] || { echo "FATAL: missing model: $MODEL_PATH" >&2; exit 1; }
[[ -d "$MAB_ROOT" ]] || { echo "FATAL: missing MAB root: $MAB_ROOT" >&2; exit 1; }
if [[ "$MODE" == paired && ! -f "$CKPT" ]]; then
  S3_CKPT=$MOUNT_POINT/$S3_CKPT_REL
  [[ -f "$S3_CKPT" ]] || {
    echo "FATAL: paired checkpoint missing locally ($CKPT) and on S3 ($S3_CKPT_REL)" >&2
    exit 1
  }
  CKPT=$LOCAL_CKPT_DIR/$(basename "$S3_CKPT")
  echo "Copy paired checkpoint from S3 to node-local NVMe: $S3_CKPT_REL"
  cp "$S3_CKPT" "$CKPT"
  CHECKPOINT_ID=${CHECKPOINT_ID:-s3://datafrontier/$S3_CKPT_REL}
fi

PLAN_ARGS=(--workers "$WORKERS" --sources "${ALL_SOURCES[@]}" --format tsv)
[[ -n "$MAXQ" ]] && PLAN_ARGS+=(--max_questions_per_source "$MAXQ")
mapfile -t PLAN_LINES < <("${PY[@]}" -m scripts.eval.plan_mab_parallel "${PLAN_ARGS[@]}")
echo "MemoryAgentBench parallel eval: mode=$MODE model=$MODEL_NAME gpus=$GPU_COUNT workers=${#PLAN_LINES[@]} workers_per_gpu=$WORKERS_PER_GPU output=$OUT_ROOT"

for line in "${PLAN_LINES[@]}"; do
  IFS=$'\t' read -r worker question_count source_text <<< "$line"
  read -r -a worker_sources <<< "$source_text"
  gpu_slot=$((worker % GPU_COUNT))
  gpu_label=${VISIBLE_GPUS[$gpu_slot]}
  echo "worker=$worker gpu_slot=$gpu_slot questions=$question_count sources=${worker_sources[*]}"
  ARGS=(
    --model_path "$MODEL_PATH" --mode "$MODE" --mab_root "$MAB_ROOT"
    --output_dir "$OUT_ROOT" --attn_impl "$ATTN" --device cuda:0
    --sources "${worker_sources[@]}"
  )
  if [[ "$MODE" == paired ]]; then
    ARGS+=(--ckpt "$CKPT")
    [[ -n "$CHECKPOINT_ID" ]] && ARGS+=(--checkpoint_id "$CHECKPOINT_ID")
  fi
  [[ -n "$MAXQ" ]] && ARGS+=(--max_questions_per_source "$MAXQ")
  [[ "$NO_PREFIX_CACHE" == 1 ]] && ARGS+=(--no_prefix_cache)
  [[ "$FORCE" == 1 ]] && ARGS+=(--force)
  CUDA_VISIBLE_DEVICES="$gpu_label" "${PY[@]}" \
    scripts/eval/eval_memory_agent_bench.py "${ARGS[@]}" \
    >"$OUT_ROOT/worker_${worker}.log" \
    2>"$OUT_ROOT/worker_${worker}.err" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if ((status != 0)); then
  echo "FATAL: at least one MemoryAgentBench worker failed; progress is retained" >&2
  exit 1
fi

"${PY[@]}" - "$OUT_ROOT/summary.json" "${ALL_SOURCES[@]}" <<'PY'
import json
import sys

summary_path, *expected = sys.argv[1:]
with open(summary_path, encoding="utf-8") as handle:
    summary = json.load(handle)
actual = summary.get("sources", {})
missing = [source for source in expected if source not in actual]
incomplete = [source for source in expected if not actual.get(source, {}).get("complete")]
if missing or incomplete:
    raise SystemExit(f"incomplete MAB merge: missing={missing}, incomplete={incomplete}")
print(f"MemoryAgentBench evaluation complete: sources={len(expected)} output={summary_path}")
PY
