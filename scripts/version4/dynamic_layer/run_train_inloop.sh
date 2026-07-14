#!/bin/bash

# Shared Qwen3-4B runner for the v4 intermediate-layer ablations.
# Submit one of run_s{32,26,22}_d4.sh instead of invoking this file directly.
set -euo pipefail

: "${SLURM_JOB_ID:?请通过 run_s32_d4.sh、run_s26_d4.sh 或 run_s22_d4.sh 用 sbatch 运行}"
: "${S:?缺少注入窗口上界 S}"
: "${D:?缺少注入窗口深度 D}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED
export AWS_RESPONSE_CHECKSUM_VALIDATION=WHEN_REQUIRED
export TOKENIZERS_PARALLELISM=false

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
ENDPOINT=http://d-ceph-ssd-inside.pjlab.org.cn
cd "$PROJ"

GPUS=${GPUS:-8}
POLICY=tf
EPOCHS=${EPOCHS:-3}
ACCUM=${ACCUM:-4}
LR=${LR:-1e-4}
VAL_MAX=${VAL_MAX:-128}
CONFIG=${CONFIG:-$PROJ/transmem/config_layered.json}

FEAT=$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507
BENCH=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem
TRAIN_DIR=${TRAIN_DIR:-$FEAT/stage0_train_short200}
VAL_DIR=${VAL_DIR:-$FEAT/stage0_dev_short200}
DATA_PATH=${DATA_PATH:-$BENCH/hotpotqa_train_32k.parquet}
VAL_DATA_PATH=${VAL_DATA_PATH:-$BENCH/hotpotqa_dev.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJ/checkpoints/v4_qwen3_4b_inloop_tf_s${S}_d${D}_n4}

# Requeue may move the job to a new node, so mount the model on every entry.
mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs
JOB_ID=${SLURM_JOB_ID}
MOUNT_POINT=/mnt/petrelfs/leihaodong/tmp/s3_v4_dynamic_${JOB_ID}
CACHE_DIR=/nvme/leihaodong/s3cache_v4_dynamic_${JOB_ID}
fusermount -u "$MOUNT_POINT" 2>/dev/null || true
rm -rf "$MOUNT_POINT" "$CACHE_DIR" 2>/dev/null || true
mkdir -p "$MOUNT_POINT" "$CACHE_DIR"

/mnt/petrelfs/leihaodong/app/s3mount datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" --allow-delete --allow-overwrite \
  --endpoint-url "$ENDPOINT" --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &
S3PID=$!
sleep 20

cleanup() {
  fusermount -u "$MOUNT_POINT" 2>/dev/null || umount "$MOUNT_POINT" 2>/dev/null || true
  kill "$S3PID" 2>/dev/null || true
  rm -rf "$MOUNT_POINT" "$CACHE_DIR" || true
}
trap cleanup EXIT

MODEL_PATH=${MODEL_PATH:-$MOUNT_POINT/leihaodong/Qwen/Qwen3-4B-Instruct-2507}
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "FATAL: 模型不可见 $MODEL_PATH" >&2
  exit 1
fi

echo "Model=$MODEL_PATH S=$S D=$D layers=$((S-D))..$((S-1)) N=4 POLICY=$POLICY"
"$UV" run --python "$VENV/bin/python" python -m torch.distributed.run \
  --standalone --nproc_per_node="$GPUS" -m transmem.train_inloop \
  --data_dir "$TRAIN_DIR" --data_path "$DATA_PATH" --data_format hotpotqa-agentmem \
  --val_data_dir "$VAL_DIR" --val_data_path "$VAL_DATA_PATH" --val_max "$VAL_MAX" \
  --model_path "$MODEL_PATH" --config "$CONFIG" --D "$D" --S "$S" \
  --policy "$POLICY" --divergence forward_kl \
  --output_dir "$OUTPUT_DIR" --grad_accum "$ACCUM" --lr "$LR" --epochs "$EPOCHS" \
  --warmup_steps 100 --grad_clip 1.0 --num_workers 2 \
  --log_interval 25 --val_interval "${VAL_INTERVAL:-250}" --save_interval 500 \
  ${EXTRA:-}

# Keep the best/result artifacts in object storage, then remove only redundant
# local restart snapshots after verifying the uploaded best checkpoint size.
S3_CKPT=s3://datafrontier/leihaodong/Project4/checkpoints/$(basename "$OUTPUT_DIR")
for file in best.pt result.json; do
  if [[ -f "$OUTPUT_DIR/$file" ]]; then
    aws s3 cp "$OUTPUT_DIR/$file" "$S3_CKPT/$file" \
      --endpoint-url "$ENDPOINT" --only-show-errors
  fi
done
LOCAL_SIZE=$(stat -c %s "$OUTPUT_DIR/best.pt" 2>/dev/null || echo 0)
S3_SIZE=$(aws s3 ls "$S3_CKPT/best.pt" --endpoint-url "$ENDPOINT" 2>/dev/null \
  | awk 'NR == 1 {print $3}' || true)
if [[ -n "$S3_SIZE" && "$S3_SIZE" = "$LOCAL_SIZE" ]]; then
  rm -f "$OUTPUT_DIR"/latest.pt "$OUTPUT_DIR"/step_*.pt
  echo "已归档 $S3_CKPT (best.pt $S3_SIZE B)，本地删除 latest/step_*"
else
  echo "WARNING: S3 校验不一致 (local=$LOCAL_SIZE s3=$S3_SIZE)，保留全部 ckpt" >&2
fi

echo "Qwen3-4B dynamic-layer 训练完成: S=$S D=$D"
