#!/bin/bash
#SBATCH -J e09_inloop
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/run_inloop/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/run_inloop/%j.err

# ── v3.2: TransMem-Layer 在环训练 (LLM 层进训练环). 必填 D.
#   例: D=8 sbatch scripts/train/run_inloop.sh ;  POLICY=onpolicy D=8 sbatch ...
#   数据 = 本地 stage0 基础特征 (answer_ids+hq_tea) + 原始 parquet; 无需 layered8 特征.
#   spot 抢占 → requeue → latest.pt 原子档自动断点续训.
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED
export AWS_RESPONSE_CHECKSUM_VALIDATION=WHEN_REQUIRED
export TOKENIZERS_PARALLELISM=false

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
ENDPOINT="http://d-ceph-ssd-inside.pjlab.org.cn"
cd $PROJ

: "${D:?需要 D (注入最后 D 层, 如 2/4/8)}"
GPUS=${GPUS:-8}
POLICY=${POLICY:-tf}
EPOCHS=${EPOCHS:-3}
ACCUM=${ACCUM:-4}
LR=${LR:-1e-4}
VAL_MAX=${VAL_MAX:-128}
CONFIG=${CONFIG:-$PROJ/transmem/config_layered.json}
TRANSMEM_ARGS=()
case "${TRANSMEM_BEFORE:-0}" in
  1|true|TRUE|yes|YES) TRANSMEM_ARGS=(--transmem_before) ;;
  0|false|FALSE|no|NO|"") ;;
  *) echo "FATAL: TRANSMEM_BEFORE 只接受 0/1/true/false" >&2; exit 2 ;;
esac

FEAT=$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507
BENCH=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem
TRAIN_DIR=${TRAIN_DIR:-$FEAT/stage0_train_short200}
VAL_DIR=${VAL_DIR:-$FEAT/stage0_dev_short200}
DATA_PATH=${DATA_PATH:-$BENCH/hotpotqa_train_32k.parquet}
VAL_DATA_PATH=${VAL_DATA_PATH:-$BENCH/hotpotqa_dev.parquet}
DATA_FORMAT=${DATA_FORMAT:-hotpotqa-agentmem}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJ/checkpoints/v3_2_inloop_${POLICY}_d${D}}
OUTPUT_DIR=${OUTPUT_DIR%/}
if [[ ${#TRANSMEM_ARGS[@]} -gt 0 && "$OUTPUT_DIR" != *_transmem_before ]]; then
  OUTPUT_DIR="${OUTPUT_DIR}_transmem_before"
fi

# ── s3mount: 挂载 Qwen3-4B 模型 (requeue 重入时在新节点重挂) ────────────
mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs
JOB_ID="${SLURM_JOB_ID:-$(date +%s)_$$}"
MOUNT_POINT="/mnt/petrelfs/leihaodong/tmp/s3_inloop_${JOB_ID}"
CACHE_DIR="/nvme/leihaodong/s3cache_inloop_${JOB_ID}"
fusermount -u "${MOUNT_POINT}" 2>/dev/null || true
rm -rf "${MOUNT_POINT}" "${CACHE_DIR}" 2>/dev/null || true
mkdir -p "${MOUNT_POINT}" "${CACHE_DIR}"

/mnt/petrelfs/leihaodong/app/s3mount datafrontier "${MOUNT_POINT}" \
  --cache "${CACHE_DIR}" \
  --allow-delete --allow-overwrite \
  --endpoint-url $ENDPOINT \
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
echo "Model: ${MODEL_PATH} | D=$D POLICY=$POLICY EPOCHS=$EPOCHS ACCUM=$ACCUM"
echo "Runner --transmem_before: $([[ ${#TRANSMEM_ARGS[@]} -gt 0 ]] && echo enabled || echo not-added)"

$UV run --python $VENV/bin/python python -m torch.distributed.run \
  --standalone --nproc_per_node=$GPUS -m transmem.train_inloop \
  --data_dir "$TRAIN_DIR" --data_path "$DATA_PATH" --data_format $DATA_FORMAT \
  --val_data_dir "$VAL_DIR" --val_data_path "$VAL_DATA_PATH" --val_max $VAL_MAX \
  ${VAL_DATA_FORMAT:+--val_data_format $VAL_DATA_FORMAT} \
  --model_path "$MODEL_PATH" --config $CONFIG --D $D \
  --policy $POLICY --divergence forward_kl \
  --output_dir "$OUTPUT_DIR" \
  --grad_accum $ACCUM --lr $LR --epochs $EPOCHS \
  --warmup_steps 100 --grad_clip 1.0 --num_workers 2 \
  --log_interval 25 --val_interval ${VAL_INTERVAL:-250} --save_interval 500 \
  ${TRANSMEM_ARGS[@]+"${TRANSMEM_ARGS[@]}"} \
  ${EXTRA:-}

# ── 归档 + 腾空间 (成功训练后才走到这) ──────────────────────────────────
S3_CKPT=s3://datafrontier/leihaodong/Project4/checkpoints/$(basename "$OUTPUT_DIR")
for f in best.pt result.json; do
  [ -f "$OUTPUT_DIR/$f" ] && aws s3 cp "$OUTPUT_DIR/$f" "$S3_CKPT/$f" --endpoint-url "$ENDPOINT" --only-show-errors
done
LOCAL_SZ=$(stat -c %s "$OUTPUT_DIR/best.pt" 2>/dev/null || echo 0)
S3_SZ=$(aws s3 ls "$S3_CKPT/best.pt" --endpoint-url "$ENDPOINT" 2>/dev/null | awk '{print $3}' | head -1)
if [ -n "$S3_SZ" ] && [ "$S3_SZ" = "$LOCAL_SZ" ]; then
  rm -f "$OUTPUT_DIR"/latest.pt "$OUTPUT_DIR"/step_*.pt
  echo "已归档 $S3_CKPT (best.pt $S3_SZ B), 本地删 latest/step_*"
else
  echo "⚠️ S3 校验不一致 (local=$LOCAL_SZ s3=$S3_SZ), 保留本地全部 ckpt"
fi
echo "✅ 在环训练 D=$D POLICY=$POLICY 完成"
