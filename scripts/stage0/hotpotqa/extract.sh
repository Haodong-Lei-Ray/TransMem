#!/bin/bash
#SBATCH -J e09_transmem_stage0
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/data/hotpotqa-benchmark/hotpotqa-agentmem/logs/stage0_%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/data/hotpotqa-benchmark/hotpotqa-agentmem/logs/stage0_%j.err

# ── Stage 0: 离线特征抽取 (hotpotqa-agentmem) ──
# 数据格式: hotpotqa-agentmem (通过 golden_titles_map.json 回填 golden_index)
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=

UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=/mnt/petrelfs/leihaodong/Project4/.venv-transmem
PY="$UV run --python $VENV/bin/python python"
PROJ=/mnt/petrelfs/leihaodong/Project4
HERE=/mnt/petrelfs/leihaodong/Project4/data/hotpotqa-benchmark/hotpotqa-agentmem
mkdir -p "$HERE/logs" "$HERE/stage0_output"

# ═══════════════════════════════════════════════════════════════════════════
# 超参数 (环境变量覆盖)
# ═══════════════════════════════════════════════════════════════════════════
N=${N:-4}
MAX_ANS=${MAX_ANS:-128}
ATTN=${ATTN:-sdpa}
MAXN=${MAXN:-}              # 空=全量; 填数字只抽前 N 条
THINKING=${THINKING:-false}
# ═══════════════════════════════════════════════════════════════════════════

# ── s3mount: 挂载 Qwen3-4B 模型 ──────────────────────────────────────────
mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs
JOB_ID="${SLURM_JOB_ID:-$(date +%s)_$$}"
MOUNT_POINT="/mnt/petrelfs/leihaodong/tmp/s3_stage0_${JOB_ID}"
CACHE_DIR="/nvme/leihaodong/s3cache_stage0_${JOB_ID}"
mkdir -p "${MOUNT_POINT}" "${CACHE_DIR}"

/mnt/petrelfs/leihaodong/app/s3mount datafrontier "${MOUNT_POINT}" \
  --cache "${CACHE_DIR}" \
  --allow-delete --allow-overwrite \
  --endpoint-url http://d-ceph-ssd-inside.pjlab.org.cn \
  --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &
S3PID=$!
sleep 20

cleanup() {
  fusermount -u "${MOUNT_POINT}" 2>/dev/null || umount "${MOUNT_POINT}" 2>/dev/null || true
  kill "${S3PID}" 2>/dev/null || true
  rm -rf "${MOUNT_POINT}" "${CACHE_DIR}" || true
}
trap cleanup EXIT

MODEL_PATH="${MOUNT_POINT}/leihaodong/Qwen/Qwen3-4B-Instruct-2507"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "FATAL: 模型不可见 ${MODEL_PATH} (s3mount 失败?)" >&2
  ls -la "${MOUNT_POINT}/leihaodong/Qwen/" >&2 || true
  exit 1
fi
echo "Model (s3mount): ${MODEL_PATH}"

cd "$PROJ"

# ── 训练集 (32768 QA) ────────────────────────────────────────────────────
MODEL_PATH="$MODEL_PATH" OUTPUT_DIR="$PROJ/data/hotpotqa_data/stage0_train" \
  N="$N" MAX_ANS="$MAX_ANS" ATTN="$ATTN" MAXN="$MAXN" THINKING="$THINKING" \
  $PY -m transmem.extract_features \
    --data_path "$HERE/hotpotqa_train_32k.parquet" \
    --data_format hotpotqa-agentmem \
    --model_path "$MODEL_PATH" \
    --output_dir "$PROJ/data/hotpotqa_data/stage0_train" \
    --N "$N" --max_answer_tokens "$MAX_ANS" \
    --attn_impl "$ATTN" --save_dtype bfloat16 \
    ${MAXN:+--max_samples "$MAXN"} \
    $([ "$THINKING" = "true" ] && echo --thinking)

echo "✅ Stage 0 train 完成: $PROJ/data/hotpotqa_data/stage0_train"
