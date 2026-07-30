#!/bin/bash
#SBATCH -J transmem_hotpotqa_dev
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=spot
#SBATCH -t 6:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_hotpotqa/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_hotpotqa/%j.err

# ── HotpotQA dev 域内评测: teacher / student / transmem 三模式 ──
# 目的: 判别 LoCoMo 零迁移是 (a) final-hidden 机制在自由生成下根本无效,
# 还是 (b) 域内有效但跨域风格冲突. transmem>student(域内) → (b); ≈/< → (a).
# 指标: Exact / Contains (hotpotqa 答案为短实体, 够判别).
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"
cd $PROJ

EVAL_FILE=${EVAL_FILE:-$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem/hotpotqa_dev.parquet}
DATA_FORMAT=${DATA_FORMAT:-hotpotqa-agentmem}  # qasper: DATA_FORMAT=qasper EVAL_FILE=$PROJ/data/qasper/qasper_dev.json
CKPT=${CKPT:-$PROJ/checkpoints/offpolicy_v2_hotpotqa_short200_forward_kl/best.pt}
MODES=${MODES:-"teacher student transmem"}
N=${N:-4}
MAX_ANS=${MAX_ANS:-50}
MAXQ=${MAXQ:-128}             # dev 全量 128
ATTN=${ATTN:-sdpa}

# ── s3mount: 挂载 Qwen3-4B 模型 ─────────────────────────────────────────
mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs
JOB_ID="${SLURM_JOB_ID:-$(date +%s)_$$}"
MOUNT_POINT="/mnt/petrelfs/leihaodong/tmp/s3_hpqa_${JOB_ID}"
CACHE_DIR="/nvme/leihaodong/s3cache_hpqa_${JOB_ID}"
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
echo "CKPT : ${CKPT}"

for MODE in $MODES; do
  echo "############ hotpotqa-dev eval mode=$MODE ############"
  EXTRA=""
  if [[ "$MODE" == "transmem" ]]; then
    if [[ ! -f "$CKPT" ]]; then echo "(跳过 transmem: 缺 ckpt $CKPT)"; continue; fi
    EXTRA="--ckpt $CKPT"
  fi
  $PY -m transmem.evaluate \
    --eval_file $EVAL_FILE --data_format $DATA_FORMAT \
    --model_path $MODEL_PATH --mode $MODE $EXTRA \
    --N $N --max_answer_tokens $MAX_ANS --max_samples $MAXQ \
    --attn_impl $ATTN --print_examples 5
done

echo "✅ hotpotqa dev 域内评测完成"
