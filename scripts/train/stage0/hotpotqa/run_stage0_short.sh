#!/bin/bash
#SBATCH -J e09_stage0_hotpotqa
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/data/logs/hotpotqa/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/data/logs/hotpotqa/%j.err

# ── Stage 0: 离线特征抽取 (冻结 LLM forward -> HM_stu/HQ_stu/HQ_tea + lm_head) ──
# 数据集: hotpotqa-agentmem (MemAgent parquet; golden_titles_map.json 回填 golden_index,
#         C_S=extract_cs 按 golden_index 切 Document, C_L=200篇拼接全文, Q=question).
# 与 qasper/run_stage0_short128.sh 同一骨架, 仅 --data_path/--data_format/OUT_ROOT 不同.
# ⚠️ train 全量 32768 条 × ~30k token 上下文, 24h 跑不完; 先用 MAXN 小跑或加大 -t.
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # 内网/S3, 关代理
export all_proxy= ALL_PROXY=

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"
DATA=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem

N=${N:-4}
MAX_ANS=${MAX_ANS:-128}
ATTN=${ATTN:-sdpa}          # flash_attention_2 在本 venv import 失败, 默认 sdpa
MAXN=${MAXN:-}              # 可选: 只抽前 MAXN 条 (先小跑); 空=全量
THINKING=${THINKING:-false} # true=开启 thinking 系统提示 (build_chat_prompt_ids thinking=True)
ModelName=${ModelName:-Qwen/Qwen3-4B-Instruct-2507}  # S3 上 leihaodong/ 下的模型相对路径
WORKERS=${WORKERS:-}        # 并发 worker 数 (每个一份模型副本); 空=作业内可见 GPU 数.
                            # 例: WORKERS=8 sbatch --gres=gpu:8 run_stage0_short.sh

# ── s3mount: 挂载模型 (权重是 S3 存储对象, 本地无) ──────────────────────────
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

MODEL_PATH=${MODEL_PATH:-${MOUNT_POINT}/leihaodong/${ModelName}}
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "FATAL: 模型不可见 ${MODEL_PATH} (s3mount 失败?)" >&2
  ls -la "${MOUNT_POINT}/leihaodong/Qwen/" >&2 || true
  exit 1
fi
echo "Model (s3mount): ${MODEL_PATH}"

cd $PROJ

# 并发数: 未指定则用作业内可见 GPU 数 (gpu:1 时即 1, 行为同顺序版)
if [ -z "$WORKERS" ]; then WORKERS=$(nvidia-smi -L 2>/dev/null | wc -l); fi
[ "$WORKERS" -ge 1 ] 2>/dev/null || WORKERS=1
echo "WORKERS=$WORKERS"

# 输出根目录: 与 run_offpolicy.sh 的 DATA_ROOT 约定对齐 (按模型名分目录);
# 下游用 DATA_ROOT=$PROJ/data/hotpotqa_data/<模型名> 接入.
OUT_ROOT=${OUT_ROOT:-$PROJ/data/hotpotqa_data/$(basename "$ModelName")}
mkdir -p "$OUT_ROOT"

# extract 失败 (watchdog 检测坏卡/CUDA 挂死后 fail-fast) → requeue 换节点断点续抽,
# 最多重排 4 次. 需 sbatch --requeue 提交.
requeue_or_die() {
  if [ -n "$SLURM_JOB_ID" ] && [ "${SLURM_RESTART_COUNT:-0}" -lt 4 ]; then
    echo "⚠️ extract 失败 (restart_count=${SLURM_RESTART_COUNT:-0}), scontrol requeue 续抽"
    scontrol requeue "$SLURM_JOB_ID" && sleep 120
  fi
  exit 1
}

# 训练集 (32768 QA); 输出 tag = short${MAX_ANS}, 不同 MAX_ANS 不互相覆盖
$PY -m transmem.extract_features \
  --data_path $DATA/hotpotqa_train_32k.parquet --data_format hotpotqa-agentmem \
  --model_path $MODEL_PATH \
  --output_dir $OUT_ROOT/stage0_train_short${MAX_ANS} \
  --N $N --max_answer_tokens $MAX_ANS --num_workers $WORKERS \
  --attn_impl $ATTN --save_dtype bfloat16 ${MAXN:+--max_samples $MAXN} \
  $([ "$THINKING" = "true" ] && echo --thinking) || requeue_or_die

# 验证集 (128 QA)
$PY -m transmem.extract_features \
  --data_path $DATA/hotpotqa_dev.parquet --data_format hotpotqa-agentmem \
  --model_path $MODEL_PATH \
  --output_dir $OUT_ROOT/stage0_dev_short${MAX_ANS} \
  --N $N --max_answer_tokens $MAX_ANS --num_workers $WORKERS \
  --attn_impl $ATTN --save_dtype bfloat16 ${MAXN:+--max_samples $MAXN} \
  $([ "$THINKING" = "true" ] && echo --thinking) || requeue_or_die

echo "✅ Stage 0 完成: $OUT_ROOT/stage0_train_short${MAX_ANS} , stage0_dev_short${MAX_ANS}"
