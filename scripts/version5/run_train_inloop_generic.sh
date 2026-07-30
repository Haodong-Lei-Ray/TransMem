#!/bin/bash

# 通用 layered in-loop 训练 runner (v5): run_train_inloop_d8.sh 的模型无关版.
# 为什么复制而不是改造 d8 runner: 在跑作业会从 petrelfs live 读取被 exec 的脚本,
# 原地编辑会造成 bash 读偏移错乱 (10262495 事故), 故 8B 主线脚本保持字节不动.
#
# 必须由 sbatch 包装脚本 exec, 并提供:
#   MODEL_REL   桶内模型相对路径, 如 leihaodong/Qwen/Qwen3-14B
#   FEAT_REL    桶内 stage0 特征相对路径, 如 leihaodong/Project4/data/hotpotqa_pool_qwen3_14b_n4_n8
#   CONFIG      TransMem 配置 json 绝对路径
#   OUTPUT_DIR  本地输出目录 (SAVE_NVME_S3=1 时只放少量元数据)
# 可选: D GPUS EPOCHS ACCUM LR VAL_MAX S3_CKPT SAVE_NVME_S3 EXTRA
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

: "${MODEL_REL:?缺少 MODEL_REL (桶内模型相对路径)}"
: "${FEAT_REL:?缺少 FEAT_REL (桶内 stage0 特征相对路径)}"
: "${CONFIG:?缺少 CONFIG}"
: "${OUTPUT_DIR:?缺少 OUTPUT_DIR}"

D=${D:-4}
GPUS=${GPUS:-4}
POLICY=tf
EPOCHS=${EPOCHS:-3}
ACCUM=${ACCUM:-8}
LR=${LR:-5e-5}
VAL_MAX=${VAL_MAX:-128}
TRANSMEM_ARGS=()
case "${TRANSMEM_BEFORE:-0}" in
  1|true|TRUE|yes|YES) TRANSMEM_ARGS=(--transmem_before) ;;
  0|false|FALSE|no|NO|"") ;;
  *) echo "FATAL: TRANSMEM_BEFORE 只接受 0/1/true/false" >&2; exit 2 ;;
esac
OUTPUT_DIR=${OUTPUT_DIR%/}
if [[ ${#TRANSMEM_ARGS[@]} -gt 0 && "$OUTPUT_DIR" != *_transmem_before ]]; then
  OUTPUT_DIR="${OUTPUT_DIR}_transmem_before"
fi

BENCH=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem
DATA_PATH=${DATA_PATH:-$BENCH/hotpotqa_train_32k.parquet}
VAL_DATA_PATH=${VAL_DATA_PATH:-$BENCH/hotpotqa_dev.parquet}
SAVE_NVME_S3=${SAVE_NVME_S3:-1}
S3_CKPT=${S3_CKPT:-s3://datafrontier/leihaodong/Project4/checkpoints/$(basename "$OUTPUT_DIR")}
S3_CKPT=${S3_CKPT%/}
if [[ ${#TRANSMEM_ARGS[@]} -gt 0 && "$S3_CKPT" != *_transmem_before ]]; then
  S3_CKPT="${S3_CKPT}_transmem_before"
fi

mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs
JOB_ID=${SLURM_JOB_ID:-$(date +%s)_$$}
MOUNT_POINT=/mnt/petrelfs/leihaodong/tmp/s3_inloop_v5_${JOB_ID}
CACHE_DIR=/nvme/leihaodong/s3cache_inloop_v5_${JOB_ID}
NVME_CKPT_DIR=${NVME_CKPT_DIR:-/nvme/leihaodong/Project4/checkpoints/$(basename "$OUTPUT_DIR")_${JOB_ID}}
if mountpoint -q "$MOUNT_POINT"; then
  if ! fusermount -u "$MOUNT_POINT" 2>/dev/null \
      && ! umount "$MOUNT_POINT" 2>/dev/null; then
    echo "FATAL: 无法卸载已有 S3 挂载点 $MOUNT_POINT；拒绝清理或复用" >&2
    exit 1
  fi
fi
rmdir "$MOUNT_POINT" 2>/dev/null || true
rm -rf "$CACHE_DIR"
mkdir -p "$MOUNT_POINT" "$CACHE_DIR" "$NVME_CKPT_DIR"

/mnt/petrelfs/leihaodong/app/s3mount datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" \
  --endpoint-url "$ENDPOINT" --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &
S3PID=$!
sleep 20

cleanup() {
  if mountpoint -q "$MOUNT_POINT"; then
    fusermount -u "$MOUNT_POINT" 2>/dev/null \
      || umount "$MOUNT_POINT" 2>/dev/null \
      || echo "WARNING: S3 挂载点仍在，保留目录且绝不递归删除: $MOUNT_POINT" >&2
  fi
  kill "$S3PID" 2>/dev/null || true
  wait "$S3PID" 2>/dev/null || true
  if ! mountpoint -q "$MOUNT_POINT"; then
    rmdir "$MOUNT_POINT" 2>/dev/null || true
  fi
  rm -rf "$CACHE_DIR"
  if ! rmdir "$NVME_CKPT_DIR" 2>/dev/null; then
    echo "NVMe 中仍有未归档文件，保留供人工恢复: $NVME_CKPT_DIR" >&2
  fi
}
trap cleanup EXIT

MODEL_PATH=$MOUNT_POINT/$MODEL_REL
FEAT=$MOUNT_POINT/$FEAT_REL
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
echo "Runner --transmem_before=$([[ ${#TRANSMEM_ARGS[@]} -gt 0 ]] && echo enabled || echo not-added)"
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
  ${STORAGE_ARGS[@]+"${STORAGE_ARGS[@]}"} \
  ${TRANSMEM_ARGS[@]+"${TRANSMEM_ARGS[@]}"} \
  ${EXTRA:-}

if [[ "$SAVE_NVME_S3" = 1 ]]; then
  for f in latest.pt best.pt result.json; do
    if ! aws s3 ls "$S3_CKPT/$f" --endpoint-url "$ENDPOINT"; then
      echo "FATAL: 训练完成但 S3 缺少 $S3_CKPT/$f" >&2
      exit 1
    fi
  done
fi
