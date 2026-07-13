#!/bin/bash
#SBATCH -J e09_mab_p1
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_memory_agent_bench/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_memory_agent_bench/%j.err

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/mnt/petrelfs/leihaodong/nltk_data

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"
cd "$PROJ"

CKPT=${CKPT:-$PROJ/checkpoints/offpolicy_v3_p1_lmehqa_d4e60_forward_kl/best.pt}
OUT_ROOT=${OUT_ROOT:-$PROJ/eval_outputs/memory_agent_bench_v3_p1_main13}
MAB_ROOT=${MAB_ROOT:-/mnt/petrelfs/leihaodong/Project1/MemoryAgentBenchProject/MemoryAgentBench}
MAXQ=${MAXQ:-}
SOURCES=${SOURCES:-}
ATTN=${ATTN:-sdpa}
NO_PREFIX_CACHE=${NO_PREFIX_CACHE:-0}
FORCE=${FORCE:-0}

mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs "$OUT_ROOT"
JOB_ID="${SLURM_JOB_ID:-$(date +%s)_$$}"
MOUNT_POINT="/mnt/petrelfs/leihaodong/tmp/s3_mab_${JOB_ID}"
CACHE_DIR="/nvme/leihaodong/s3cache_mab_${JOB_ID}"
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
test -d "$MAB_ROOT"

ARGS=(
  --model_path "$MODEL_PATH"
  --ckpt "$CKPT"
  --mab_root "$MAB_ROOT"
  --output_dir "$OUT_ROOT"
  --attn_impl "$ATTN"
)
if [[ -n "$MAXQ" ]]; then ARGS+=(--max_questions_per_source "$MAXQ"); fi
if [[ "$NO_PREFIX_CACHE" == 1 ]]; then ARGS+=(--no_prefix_cache); fi
if [[ "$FORCE" == 1 ]]; then ARGS+=(--force); fi
if [[ -n "$SOURCES" ]]; then
  read -r -a SOURCE_ARGS <<< "$SOURCES"
  ARGS+=(--sources "${SOURCE_ARGS[@]}")
fi

$PY scripts/eval/eval_memory_agent_bench.py "${ARGS[@]}"

echo "MemoryAgentBench evaluation complete: $OUT_ROOT"
