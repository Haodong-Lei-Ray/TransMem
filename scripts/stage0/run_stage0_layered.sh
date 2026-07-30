#!/bin/bash
#SBATCH -J e09_stage0_layered
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/data/logs/layered/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/data/logs/layered/%j.err

# ── Stage 0 layered (v3 计划 6): --dump_layers 8, 最后 8 层 HM/HQ_stu/HQ_tea 超集,
#    D∈{1,2,4,6,8} 全部复用. DATASET=hotpotqa|longmemeval 选数据集.
#    hotpotqa 产物 ~25-30G → 直写 s3mount (manifest 留本地); longmemeval ~1G → 本地.
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"

DATASET=${DATASET:-hotpotqa}
DUMP_LAYERS=${DUMP_LAYERS:-8}
N=${N:-4}
MAX_ANS=${MAX_ANS:-200}
ATTN=${ATTN:-sdpa}
MAXN=${MAXN:-}
ModelName=${ModelName:-Qwen/Qwen3-4B-Instruct-2507}
WORKERS=${WORKERS:-}

mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs
JOB_ID="${SLURM_JOB_ID:-$(date +%s)_$$}"
MOUNT_POINT="/mnt/petrelfs/leihaodong/tmp/s3_lay_${JOB_ID}"
CACHE_DIR="/nvme/leihaodong/s3cache_lay_${JOB_ID}"
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
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "FATAL: 模型不可见 ${MODEL_PATH}" >&2; exit 1; }
echo "Model: ${MODEL_PATH}  DATASET=${DATASET}  DUMP_LAYERS=${DUMP_LAYERS}"

cd $PROJ
if [ -z "$WORKERS" ]; then WORKERS=$(nvidia-smi -L 2>/dev/null | wc -l); fi
[ "$WORKERS" -ge 1 ] 2>/dev/null || WORKERS=1

MODEL_BASE=$(basename "$ModelName")
TAG="short${MAX_ANS}_layered${DUMP_LAYERS}"
if [ "$DATASET" = "hotpotqa" ]; then
  DATA=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem
  TRAIN_PATH=$DATA/hotpotqa_train_32k.parquet
  DEV_PATH=$DATA/hotpotqa_dev.parquet
  FORMAT=hotpotqa-agentmem
  # 大产物直写 s3mount; manifest (JSONL 追加写) 必须留本地
  OUT_ROOT=${OUT_ROOT:-${MOUNT_POINT}/leihaodong/Project4/data/hotpotqa_layered/$MODEL_BASE}
  MANIFEST_ROOT=$PROJ/data/hotpotqa_data/$MODEL_BASE/.manifests_layered${DUMP_LAYERS}
elif [ "$DATASET" = "longmemeval" ]; then
  DATA=$PROJ/data/LongMemEval/data
  TRAIN_PATH=$DATA/longmemeval_train.json
  DEV_PATH=$DATA/longmemeval_dev.json
  FORMAT=longmemeval
  OUT_ROOT=${OUT_ROOT:-$PROJ/data/longmemeval_data/$MODEL_BASE}
  MANIFEST_ROOT=""
else
  echo "FATAL: 未知 DATASET=$DATASET"; exit 1
fi
mkdir -p "$OUT_ROOT"

requeue_or_die() {
  if [ -n "$SLURM_JOB_ID" ] && [ "${SLURM_RESTART_COUNT:-0}" -lt 4 ]; then
    echo "⚠️ extract 失败 (restart_count=${SLURM_RESTART_COUNT:-0}), scontrol requeue 续抽"
    scontrol requeue "$SLURM_JOB_ID" && sleep 120
  fi
  exit 1
}

for SPLIT in train dev; do
  if [ "$SPLIT" = "train" ]; then DP=$TRAIN_PATH; else DP=$DEV_PATH; fi
  OUT=$OUT_ROOT/stage0_${SPLIT}_${TAG}
  MARG=""
  if [ -n "$MANIFEST_ROOT" ]; then
    mkdir -p "$MANIFEST_ROOT"
    MARG="--manifest_dir $MANIFEST_ROOT/${SPLIT}"
  fi
  $PY -m transmem.extract_features \
    --data_path "$DP" --data_format $FORMAT \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUT" \
    --N $N --max_answer_tokens $MAX_ANS --num_workers $WORKERS \
    --dump_layers $DUMP_LAYERS $MARG \
    --attn_impl $ATTN --save_dtype bfloat16 ${MAXN:+--max_samples $MAXN} || requeue_or_die
done

echo "✅ Stage0 layered 完成: $OUT_ROOT/stage0_{train,dev}_${TAG}"
