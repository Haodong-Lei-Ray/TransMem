#!/bin/bash
#SBATCH -J e09_layered
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/run_layered/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/run_layered/%j.err

# ── v3 计划 6: TransMem-Layer 训练 (D 消融). 必填 D; 特征在 S3 时填 S3_FEAT 先 sync.
#   例: D=4 TRAIN_DIR=... VAL_DIR=... OUTPUT_DIR=... sbatch scripts/run_layered.sh
#   结束后 best.pt+result.json 归档 S3, 删 latest/step_* (配额纪律).
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED
export AWS_RESPONSE_CHECKSUM_VALIDATION=WHEN_REQUIRED

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
ENDPOINT="http://d-ceph-ssd-inside.pjlab.org.cn"
cd $PROJ

: "${D:?需要 D (注入最后 D 层, 1/2/4/6/8)}"
GPUS=${GPUS:-8}
CONFIG=${CONFIG:-$PROJ/transmem/config_layered.json}
DIV=${DIV:-forward_kl}
MSE_W=${MSE_W:-1.0}
ALPHA=${ALPHA:-uniform}
BS=${BS:-16}
LR=${LR:-1e-4}
EpochNum=${EpochNum:-30}
WARMUP=${WARMUP:-50}
SaveInterval=${SaveInterval:-$((2000 / GPUS))}
LogInterval=${LogInterval:-$((50 / GPUS > 0 ? 50 / GPUS : 1))}
Val_interval=${Val_interval:-$((2000 / GPUS))}

# 特征来源: S3_FEAT 非空则 aws sync 到 /nvme 固定路径 (每次作业重拉, /nvme 是节点本地盘)
if [ -n "$S3_FEAT" ]; then
  NVME_ROOT=/nvme/leihaodong/layered_feat
  mkdir -p "$NVME_ROOT"
  echo "sync $S3_FEAT -> $NVME_ROOT ..."
  aws s3 sync "$S3_FEAT" "$NVME_ROOT" --endpoint-url "$ENDPOINT" --only-show-errors
  TRAIN_DIR=$NVME_ROOT/$(basename "${S3_TRAIN_SUB:-stage0_train_short200_layered8}")
  VAL_DIR=$NVME_ROOT/$(basename "${S3_VAL_SUB:-stage0_dev_short200_layered8}")
fi
: "${TRAIN_DIR:?需要 TRAIN_DIR (或 S3_FEAT)}"
: "${VAL_DIR:?需要 VAL_DIR (或 S3_FEAT)}"
OUTPUT_DIR=${OUTPUT_DIR:-$PROJ/checkpoints/v3_p6_layered_d${D}_${DIV}}

if [ "$GPUS" -gt 1 ]; then
  LAUNCH="$UV run --python $VENV/bin/python python -m torch.distributed.run --standalone --nproc_per_node=$GPUS -m transmem.train_layered"
else
  LAUNCH="$UV run --python $VENV/bin/python python -m transmem.train_layered"
fi

$LAUNCH \
  --data_dir "$TRAIN_DIR" --val_data_dir "$VAL_DIR" \
  --config $CONFIG --D $D \
  --output_dir $OUTPUT_DIR \
  --divergence $DIV --mse_weight $MSE_W --alpha_mix $ALPHA \
  --batch_size $BS --lr $LR --epochs $EpochNum \
  --warmup_steps $WARMUP --grad_clip 1.0 \
  --dtype float32 --num_workers 2 \
  --log_interval $LogInterval --val_interval $Val_interval --save_interval $SaveInterval

# 归档 + 腾空间 (成功训练后才走到这)
S3_CKPT=s3://datafrontier/leihaodong/Project4/checkpoints/$(basename "$OUTPUT_DIR")
for f in best.pt result.json; do
  [ -f "$OUTPUT_DIR/$f" ] && aws s3 cp "$OUTPUT_DIR/$f" "$S3_CKPT/$f" --endpoint-url "$ENDPOINT" --only-show-errors
done
# 校验 best.pt 尺寸一致后再删 latest/step_*
LOCAL_SZ=$(stat -c %s "$OUTPUT_DIR/best.pt" 2>/dev/null || echo 0)
S3_SZ=$(aws s3 ls "$S3_CKPT/best.pt" --endpoint-url "$ENDPOINT" 2>/dev/null | awk '{print $3}' | head -1)
if [ -n "$S3_SZ" ] && [ "$S3_SZ" = "$LOCAL_SZ" ]; then
  rm -f "$OUTPUT_DIR"/latest.pt "$OUTPUT_DIR"/step_*.pt
  echo "已归档 $S3_CKPT (best.pt $S3_SZ B), 本地删 latest/step_*"
else
  echo "⚠️ S3 校验不一致 (local=$LOCAL_SZ s3=$S3_SZ), 保留本地全部 ckpt"
fi
echo "✅ layered D=$D 训练完成"
