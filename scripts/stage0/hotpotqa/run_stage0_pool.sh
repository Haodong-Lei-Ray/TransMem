#!/bin/bash
#SBATCH -J e09_stage0_pool
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/data/logs/hotpotqa/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/data/logs/hotpotqa/%j.err

# ── Stage 0 记忆池提取 (N 消融): 一次 forward, 存全部 N∈{4..384} 取位并集 ──
# 与 run_stage0_short.sh 同骨架, 三点不同:
#   1) --pool_ns + --hm_mode frac: hm_stu 存 [P≈512, dim] 并集池 + 每 N 行索引表;
#   2) 输出直写 s3mount (本地配额只剩 ~13G, 池 ~85G 放不下);
#   3) --manifest_dir 指到 petrelfs 本地 (JSONL 追加写对象存储不支持), 断点续抽跨节点.
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
# 30k ctx 长样本的 worker 峰值可到 30-35G (10209170 实测 OOM): 密度勿超 2 worker/卡
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"
DATA=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem

N=${N:-4}                         # 仅占位, --pool_ns 生效时被忽略
POOL_NS=${POOL_NS:-4,8,16,32,64,128,256,384}
MAX_ANS=${MAX_ANS:-200}
ATTN=${ATTN:-sdpa}
MAXN=${MAXN:-}                    # 冒烟: MAXN=16
OUT_SUFFIX=${OUT_SUFFIX:-}        # 冒烟: OUT_SUFFIX=_smoke16 (避免污染正式目录)
ModelName=${ModelName:-Qwen/Qwen3-4B-Instruct-2507}
WORKERS=${WORKERS:-}

# ── s3mount: 模型(读) + 池输出(写) 走同一挂载 ────────────────────────────
mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs
JOB_ID="${SLURM_JOB_ID:-$(date +%s)_$$}"
MOUNT_POINT="/mnt/petrelfs/leihaodong/tmp/s3_pool_${JOB_ID}"
CACHE_DIR="/nvme/leihaodong/s3cache_pool_${JOB_ID}"
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
  exit 1
fi
echo "Model (s3mount): ${MODEL_PATH}"

cd $PROJ

if [ -z "$WORKERS" ]; then WORKERS=$(nvidia-smi -L 2>/dev/null | wc -l); fi
[ "$WORKERS" -ge 1 ] 2>/dev/null || WORKERS=1
echo "WORKERS=$WORKERS  POOL_NS=$POOL_NS  OUT_SUFFIX=$OUT_SUFFIX"

# 输出: 默认 S3 (经挂载点写); OUT_ROOT 可覆写到本地 petrelfs
# (datafrontier 桶满时的降级路径, 2026-07-09: 网格剪到 {4..128} 后 16GB 本地放得下).
# 训练侧 S3 部分用 aws s3 sync 拉到 /nvme, 见 scripts/run_offpolicy_pool.sh.
OUT_ROOT=${OUT_ROOT:-${MOUNT_POINT}/leihaodong/Project4/data/hotpotqa_pool${OUT_SUFFIX}}
mkdir -p "$OUT_ROOT"
# manifest 留 petrelfs: 追加写 + 抢占 requeue 跨节点续抽
MANI_ROOT=$PROJ/data/hotpotqa_data/.pool_manifests${OUT_SUFFIX}

requeue_or_die() {
  if [ -n "$SLURM_JOB_ID" ] && [ "${SLURM_RESTART_COUNT:-0}" -lt 4 ]; then
    echo "⚠️ extract 失败 (restart_count=${SLURM_RESTART_COUNT:-0}), scontrol requeue 续抽"
    scontrol requeue "$SLURM_JOB_ID" && sleep 120
  fi
  exit 1
}

# 训练集 (32768 QA, 池 ~85G)
$PY -m transmem.extract_features \
  --data_path $DATA/hotpotqa_train_32k.parquet --data_format hotpotqa-agentmem \
  --model_path $MODEL_PATH \
  --output_dir $OUT_ROOT/stage0_train_short${MAX_ANS}_pool \
  --manifest_dir $MANI_ROOT/train \
  --pool_ns $POOL_NS --hm_mode frac \
  --N $N --max_answer_tokens $MAX_ANS --num_workers $WORKERS \
  --attn_impl $ATTN --save_dtype bfloat16 ${MAXN:+--max_samples $MAXN} || requeue_or_die

# 验证集 (128 QA)
$PY -m transmem.extract_features \
  --data_path $DATA/hotpotqa_dev.parquet --data_format hotpotqa-agentmem \
  --model_path $MODEL_PATH \
  --output_dir $OUT_ROOT/stage0_dev_short${MAX_ANS}_pool \
  --manifest_dir $MANI_ROOT/dev \
  --pool_ns $POOL_NS --hm_mode frac \
  --N $N --max_answer_tokens $MAX_ANS --num_workers $WORKERS \
  --attn_impl $ATTN --save_dtype bfloat16 ${MAXN:+--max_samples $MAXN} || requeue_or_die

echo "✅ Stage 0 池提取完成 -> s3://datafrontier/leihaodong/Project4/data/hotpotqa_pool${OUT_SUFFIX}/"
