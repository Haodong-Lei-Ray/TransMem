#!/bin/bash
#SBATCH -J e09_lme_stage0
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/data/logs/longmemeval/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/data/logs/longmemeval/%j.err

# ── Stage 0: LongMemEval-S 特征抽取 (v3 计划 1/3/4/5 的训练特征) ──
# 数据: data/LongMemEval/data/longmemeval_{train,dev}.json (build_split.py 产物,
#       train 370 / dev 100, 已剔除 30 道弃权题, train 已 shuffle).
# C_L ~128k token/条 (36 层 KV ~19G, 80G 卡单 worker 放得下); C_S=证据 session 并集.
# 与 hotpotqa cs2 对齐: N=4, hm_mode=floor(默认), MAX_ANS=200 -> TAG short200.
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # 内网/S3, 关代理
export all_proxy= ALL_PROXY=

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"
DATA=$PROJ/data/LongMemEval/data

N=${N:-4}
POOL_NS=${POOL_NS:-}
HM_MODE=${HM_MODE:-floor}
MAX_ANS=${MAX_ANS:-200}
ATTN=${ATTN:-sdpa}          # flash_attention_2 在本 venv import 失败, 默认 sdpa
MAXN=${MAXN:-}              # 可选: 只抽前 MAXN 条 (冒烟); 空=全量
ModelName=${ModelName:-Qwen/Qwen3-4B-Instruct-2507}
WORKERS=${WORKERS:-}        # 空=作业内可见 GPU 数

# ── s3mount: 挂载模型 ──────────────────────────────────────────────────
mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs
JOB_ID="${SLURM_JOB_ID:-$(date +%s)_$$}"
MOUNT_POINT="/mnt/petrelfs/leihaodong/tmp/s3_lme0_${JOB_ID}"
CACHE_DIR="/nvme/leihaodong/s3cache_lme0_${JOB_ID}"
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

MODEL_PATH=${MODEL_PATH:-${MOUNT_POINT}/leihaodong/${ModelName}}
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "FATAL: 模型不可见 ${MODEL_PATH} (s3mount 失败?)" >&2
  ls -la "${MOUNT_POINT}/leihaodong/Qwen/" >&2 || true
  exit 1
fi
echo "Model (s3mount): ${MODEL_PATH}"

cd $PROJ

if [ -z "$WORKERS" ]; then WORKERS=$(nvidia-smi -L 2>/dev/null | wc -l); fi
[ "$WORKERS" -ge 1 ] 2>/dev/null || WORKERS=1
echo "WORKERS=$WORKERS"

OUT_ROOT=${OUT_ROOT:-$PROJ/data/longmemeval_data/$(basename "$ModelName")}
mkdir -p "$OUT_ROOT"

requeue_or_die() {
  if [ -n "$SLURM_JOB_ID" ] && [ "${SLURM_RESTART_COUNT:-0}" -lt 4 ]; then
    echo "⚠️ extract 失败 (restart_count=${SLURM_RESTART_COUNT:-0}), scontrol requeue 续抽"
    scontrol requeue "$SLURM_JOB_ID" && sleep 120
  fi
  exit 1
}

# 训练集 (370)
$PY -m transmem.extract_features \
  --data_path $DATA/longmemeval_train.json --data_format longmemeval \
  --model_path $MODEL_PATH \
  --output_dir $OUT_ROOT/stage0_train_short${MAX_ANS} \
  --N $N --max_answer_tokens $MAX_ANS --num_workers $WORKERS \
  --attn_impl $ATTN --save_dtype bfloat16 --hm_mode $HM_MODE ${POOL_NS:+--pool_ns $POOL_NS} ${MAXN:+--max_samples $MAXN} || requeue_or_die

# 验证集 (100)
$PY -m transmem.extract_features \
  --data_path $DATA/longmemeval_dev.json --data_format longmemeval \
  --model_path $MODEL_PATH \
  --output_dir $OUT_ROOT/stage0_dev_short${MAX_ANS} \
  --N $N --max_answer_tokens $MAX_ANS --num_workers $WORKERS \
  --attn_impl $ATTN --save_dtype bfloat16 --hm_mode $HM_MODE ${POOL_NS:+--pool_ns $POOL_NS} ${MAXN:+--max_samples $MAXN} || requeue_or_die

echo "✅ Stage 0 完成: $OUT_ROOT/stage0_train_short${MAX_ANS} , stage0_dev_short${MAX_ANS}"
