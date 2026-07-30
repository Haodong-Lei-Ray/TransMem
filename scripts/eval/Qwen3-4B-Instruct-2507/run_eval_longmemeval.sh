#!/bin/bash
#SBATCH -J e09_lme_eval
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH -t 12:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_longmemeval/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_longmemeval/%j.err

# ── LongMemEval 域内自由生成评测: teacher / student / transmem ──
# 默认 dev100; 死记诊断用 EVAL_FILE=longmemeval_train.json MAXQ=100 (train 已 shuffle,
# 前 100 = 随机子集). student/transmem 每题 prefill ~128k token (KV ~19G, 单卡可跑).
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"
cd $PROJ

EVAL_FILE=${EVAL_FILE:-$PROJ/data/LongMemEval/data/longmemeval_dev.json}
DATA_FORMAT=longmemeval
CKPT=${CKPT:-}
MODES=${MODES:-"teacher student"}
MAX_ANS=${MAX_ANS:-50}
MAXQ=${MAXQ:-100}
ATTN=${ATTN:-sdpa}

# ── s3mount: 挂载 Qwen3-4B 模型 ─────────────────────────────────────────
mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs
JOB_ID="${SLURM_JOB_ID:-$(date +%s)_$$}"
MOUNT_POINT="/mnt/petrelfs/leihaodong/tmp/s3_lmee_${JOB_ID}"
CACHE_DIR="/nvme/leihaodong/s3cache_lmee_${JOB_ID}"
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

MODEL_PATH=${MODEL_PATH:-${MOUNT_POINT}/leihaodong/Qwen/Qwen3-4B-Instruct-2507}
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "FATAL: 模型不可见 ${MODEL_PATH} (s3mount 失败?)" >&2
  exit 1
fi
echo "Model: ${MODEL_PATH}"
echo "CKPT : ${CKPT:-<none>}"
echo "EVAL : ${EVAL_FILE} (MAXQ=${MAXQ})"

for MODE in $MODES; do
  echo "############ longmemeval eval mode=$MODE ############"
  EXTRA=""
  if [[ "$MODE" == "transmem" ]]; then
    if [[ ! -f "$CKPT" ]]; then echo "(跳过 transmem: 缺 ckpt $CKPT)"; continue; fi
    EXTRA="--ckpt $CKPT"
  fi
  $PY -m transmem.evaluate \
    --eval_file $EVAL_FILE --data_format $DATA_FORMAT \
    --model_path $MODEL_PATH --mode $MODE $EXTRA \
    --max_answer_tokens $MAX_ANS --max_samples $MAXQ \
    --attn_impl $ATTN --print_examples 5
done

echo "✅ longmemeval 评测完成"
