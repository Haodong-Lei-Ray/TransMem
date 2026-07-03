#!/bin/bash
#SBATCH -J transmem_locomo
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j.err

# ── LoCoMo 评测: teacher(上界) / student(基线) / transmem(本方法) ──
# 先证 teacher >> student (plan §9.6 同款 sanity), 再看 transmem 拉近多少.
# 数据: Project1/data/locomo10.json (10 段对话, cat1-4 共 1540 题).
# 冒烟: MAXQ=20 MODES=student sbatch ...; 全量默认三模式顺序跑.
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"
export NLTK_DATA=/mnt/petrelfs/leihaodong/nltk_data   # F1 评分要 Porter stemmer
cd $PROJ

DATA_FILE=${DATA_FILE:-/mnt/petrelfs/leihaodong/Project1/data/locomo10.json}
CKPT=${CKPT:-$PROJ/checkpoints/offpolicy_short128_forward_kl/latest.pt}
MODES=${MODES:-"teacher student transmem"}
N=${N:-4}
MAX_ANS=${MAX_ANS:-50}        # LoCoMo 官方 50 new tokens
CATS=${CATS:-"1 2 3 4"}       # 官方基线同款 (不含 5=adversarial)
MAXQ=${MAXQ:-}                # 可选: 总题数上限 (冒烟)
ATTN=${ATTN:-sdpa}
OUT_ROOT=${OUT_ROOT:-$PROJ/eval_outputs/locomo_$(basename $(dirname ${CKPT}))}

# ── s3mount: 挂载 Qwen3-4B 模型 (权重在 S3, 本地无) ──────────────────────
mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs "$OUT_ROOT"
JOB_ID="${SLURM_JOB_ID:-$(date +%s)_$$}"
MOUNT_POINT="/mnt/petrelfs/leihaodong/tmp/s3_locomo_${JOB_ID}"
CACHE_DIR="/nvme/leihaodong/s3cache_locomo_${JOB_ID}"
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
  ls -la "${MOUNT_POINT}/leihaodong/Qwen/" >&2 || true
  exit 1
fi
echo "Model: ${MODEL_PATH}"
echo "CKPT : ${CKPT}"
echo "Out  : ${OUT_ROOT}"

for MODE in $MODES; do
  echo "############ locomo eval mode=$MODE ############"
  EXTRA=""
  if [[ "$MODE" == "transmem" ]]; then
    if [[ ! -f "$CKPT" ]]; then echo "(跳过 transmem: 缺 ckpt $CKPT)"; continue; fi
    EXTRA="--ckpt $CKPT"
  fi
  $PY scripts/eval/eval_locomo.py \
    --data_file $DATA_FILE --model_path $MODEL_PATH --mode $MODE $EXTRA \
    --N $N --max_answer_tokens $MAX_ANS --categories $CATS \
    ${MAXQ:+--max_questions $MAXQ} --attn_impl $ATTN \
    --output_json $OUT_ROOT/locomo_${MODE}.json
done

echo "✅ LoCoMo 评测完成: $OUT_ROOT"
