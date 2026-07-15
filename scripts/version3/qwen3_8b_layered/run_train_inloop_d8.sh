#!/bin/bash
#SBATCH -J e09_v38l8_tr
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b_layered/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b_layered/%j.err

# Qwen3-8B v3.2 主线: frozen LLM 最后 8 层各插一个 1-layer TransMem，
# teacher-forced in-loop forward-KL。旧 final-hidden train_offpolicy 不用于此方案。
set -euo pipefail
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
mkdir -p "$PROJ/logs/qwen3_8b_layered"

D=8
GPUS=${GPUS:-8}
POLICY=tf
EPOCHS=${EPOCHS:-3}
ACCUM=${ACCUM:-4}
LR=${LR:-1e-4}
VAL_MAX=${VAL_MAX:-128}
CONFIG=${CONFIG:-$PROJ/transmem/config_layered_8b.json}

BENCH=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem
FEAT=${FEAT:-}
TRAIN_DIR=${TRAIN_DIR:-}
VAL_DIR=${VAL_DIR:-}
DATA_PATH=${DATA_PATH:-$BENCH/hotpotqa_train_32k.parquet}
VAL_DATA_PATH=${VAL_DATA_PATH:-$BENCH/hotpotqa_dev.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJ/checkpoints/v3_2_qwen3_8b_inloop_tf_d8_n4}
SAVE_NVME_S3=${SAVE_NVME_S3:-1}
S3_CKPT=${S3_CKPT:-s3://datafrontier/leihaodong/Project4/checkpoints/$(basename "$OUTPUT_DIR")}

mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs
JOB_ID=${SLURM_JOB_ID:-$(date +%s)_$$}
MOUNT_POINT=/mnt/petrelfs/leihaodong/tmp/s3_inloop_8b_${JOB_ID}
CACHE_DIR=/nvme/leihaodong/s3cache_inloop_8b_${JOB_ID}
NVME_CKPT_DIR=${NVME_CKPT_DIR:-/nvme/leihaodong/Project4/checkpoints/$(basename "$OUTPUT_DIR")_${JOB_ID}}
fusermount -u "$MOUNT_POINT" 2>/dev/null || true
rm -rf "$MOUNT_POINT" "$CACHE_DIR" "$NVME_CKPT_DIR" 2>/dev/null || true
mkdir -p "$MOUNT_POINT" "$CACHE_DIR" "$NVME_CKPT_DIR"

/mnt/petrelfs/leihaodong/app/s3mount datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" --allow-delete --allow-overwrite \
  --endpoint-url "$ENDPOINT" --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &
S3PID=$!
sleep 20

cleanup() {
  fusermount -u "$MOUNT_POINT" 2>/dev/null || umount "$MOUNT_POINT" 2>/dev/null || true
  kill "$S3PID" 2>/dev/null || true
  rm -rf "$MOUNT_POINT" "$CACHE_DIR" "$NVME_CKPT_DIR" || true
}
trap cleanup EXIT

MODEL_PATH=${MODEL_PATH:-$MOUNT_POINT/leihaodong/Qwen/Qwen3-8B}
FEAT=${FEAT:-$MOUNT_POINT/leihaodong/Project4/data/hotpotqa_data/Qwen3-8B-pool-n4-n8}
TRAIN_DIR=${TRAIN_DIR:-$FEAT/stage0_train_short200_pool}
VAL_DIR=${VAL_DIR:-$FEAT/stage0_dev_short200_pool}
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "FATAL: 模型不可见 $MODEL_PATH" >&2
  exit 1
fi
for required in "$TRAIN_DIR/meta.json" "$VAL_DIR/meta.json"; do
  if [[ ! -f "$required" ]]; then
    echo "FATAL: S3 Stage0 特征不可见 $required" >&2
    exit 1
  fi
done

STORAGE_ARGS=()
if [[ "$SAVE_NVME_S3" = 1 ]]; then
  STORAGE_ARGS=(
    --save_nvme_s3
    --nvme_checkpoint_dir "$NVME_CKPT_DIR"
    --s3_checkpoint_uri "$S3_CKPT"
    --s3_endpoint_url "$ENDPOINT"
  )
fi

echo "Model=$MODEL_PATH D=$D N=4 POLICY=$POLICY CONFIG=$CONFIG"
echo "Stage0=$FEAT"
echo "Checkpoint mode=$([[ $SAVE_NVME_S3 = 1 ]] && echo NVMe-S3 || echo local) target=$S3_CKPT"
$UV run --python "$VENV/bin/python" python -m torch.distributed.run \
  --standalone --nproc_per_node="$GPUS" -m transmem.train_inloop \
  --data_dir "$TRAIN_DIR" --data_path "$DATA_PATH" --data_format hotpotqa-agentmem \
  --val_data_dir "$VAL_DIR" --val_data_path "$VAL_DATA_PATH" --val_max "$VAL_MAX" \
  --model_path "$MODEL_PATH" --config "$CONFIG" --D "$D" \
  --policy "$POLICY" --divergence forward_kl \
  --output_dir "$OUTPUT_DIR" --grad_accum "$ACCUM" --lr "$LR" --epochs "$EPOCHS" \
  --warmup_steps 100 --grad_clip 1.0 --num_workers 2 \
  --log_interval 25 --val_interval ${VAL_INTERVAL:-250} \
  --save_interval ${SAVE_INTERVAL:-500} \
  "${STORAGE_ARGS[@]}" \
  ${EXTRA:-}

if [[ "$SAVE_NVME_S3" = 1 ]]; then
  for f in latest.pt best.pt result.json; do
    if ! aws s3 ls "$S3_CKPT/$f" --endpoint-url "$ENDPOINT"; then
      echo "WARNING: 训练完成但 S3 缺少 $S3_CKPT/$f" >&2
    fi
  done
else
  for f in best.pt result.json; do
    [[ -f "$OUTPUT_DIR/$f" ]] && aws s3 cp "$OUTPUT_DIR/$f" "$S3_CKPT/$f" \
      --endpoint-url "$ENDPOINT" --only-show-errors
  done
fi
