#!/bin/bash
#SBATCH -J e09_transmem_stage0
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/data/logs/qasper/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/data/logs/qasper/%j.err

# ── Stage 0: 离线特征抽取 (冻结 LLM forward -> HM_stu/HQ_stu/HQ_tea + lm_head) ──
# 数据集: Qasper (每条=一个有 evidence 的 QA; C_S=evidence 直供, C_L=全文, Q=question).
# 数据需先由 data/build_qasper_json.py 生成 qasper_{train,dev}.json.
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # 内网/S3, 关代理
export all_proxy= ALL_PROXY=

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"
DATA=/mnt/petrelfs/leihaodong/Project4/data/qasper

N=${N:-4}
MAX_ANS=${MAX_ANS:-128}
ATTN=${ATTN:-sdpa}          # flash_attention_2 在本 venv import 失败, 默认 sdpa
MAXN=${MAXN:-}              # 可选: 只抽前 MAXN 条 (先小跑); 空=全量
THINKING=${THINKING:-false} # true=开启 thinking 系统提示 (build_chat_prompt_ids thinking=True)

# ── s3mount: 挂载 Qwen3-4B 模型 (权重是 S3 存储对象, 本地无) ──────────────
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

MODEL_PATH=${MODEL_PATH:-${MOUNT_POINT}/leihaodong/Qwen/Qwen3-4B-Instruct-2507}
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "FATAL: 模型不可见 ${MODEL_PATH} (s3mount 失败?)" >&2
  ls -la "${MOUNT_POINT}/leihaodong/Qwen/" >&2 || true
  exit 1
fi
echo "Model (s3mount): ${MODEL_PATH}"

cd $PROJ

# 输出根目录: 与 run_offpolicy.sh 的 DATA_ROOT 默认值对齐 (按模型名分目录)
OUT_ROOT=${OUT_ROOT:-$PROJ/data/qasper_data/Qwen3-4B-Instruct-2507}
mkdir -p "$OUT_ROOT"

# 训练集 (2240 QA)
$PY -m transmem.extract_features \
  --data_path $DATA/qasper_train.json --data_format qasper \
  --model_path $MODEL_PATH \
  --output_dir $OUT_ROOT/stage0_train_short128 \
  --N $N --max_answer_tokens $MAX_ANS \
  --attn_impl $ATTN --save_dtype bfloat16 ${MAXN:+--max_samples $MAXN} \
  $([ "$THINKING" = "true" ] && echo --thinking)

# 验证集 (927 QA)
$PY -m transmem.extract_features \
  --data_path $DATA/qasper_dev.json --data_format qasper \
  --model_path $MODEL_PATH \
  --output_dir $OUT_ROOT/stage0_dev_short128 \
  --N $N --max_answer_tokens $MAX_ANS \
  --attn_impl $ATTN --save_dtype bfloat16 ${MAXN:+--max_samples $MAXN} \
  $([ "$THINKING" = "true" ] && echo --thinking)

echo "✅ Stage 0 完成: $OUT_ROOT/stage0_train_short128 , stage0_dev_short128"
