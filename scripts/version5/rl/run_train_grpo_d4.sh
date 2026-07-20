#!/bin/bash
#SBATCH -J e09_v5grp4
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 48:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/%j_v5grp4.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/%j_v5grp4.err

# Qwen3-4B D=4 dynamic-gate GRPO. Submit only after OPD's LoCoMo gate passes.
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
RUN_NAME=${RUN_NAME:-v5_grpo_qwen3_4b_hqa_d4_g4_e2_s200}
OUTPUT_DIR=$PROJ/checkpoints/$RUN_NAME
S3_CKPT=s3://datafrontier/leihaodong/Project4/checkpoints/$RUN_NAME
PARENT_REL=${PARENT_REL:-leihaodong/Project4/checkpoints/v5_opd_qwen3_4b_hqa_d4_rkl_s250/best.pt}
PARENT_ETAG=${PARENT_ETAG:-}
REFERENCE_ID=${REFERENCE_ID:-s3://datafrontier/$PARENT_REL}
MODEL_REL=leihaodong/Qwen/Qwen3-4B-Instruct-2507
TRAIN_DIR=$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507/stage0_train_short200
VAL_DIR=$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507/stage0_dev_short200
BENCH=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem
CONFIG=$PROJ/transmem/config_layered_dynamic_gate.json
GPUS=${GPUS:-4}
ACCUM=${ACCUM:-2}
MAX_STEPS=${MAX_STEPS:-200}
LR=${LR:-1e-5}
cd "$PROJ"

JOB_ID=${SLURM_JOB_ID:-$(date +%s)_$$}
MOUNT_POINT=/mnt/petrelfs/leihaodong/tmp/s3_grpo_${JOB_ID}
CACHE_DIR=/nvme/leihaodong/s3cache_grpo_${JOB_ID}
PARENT_DIR=/nvme/leihaodong/Project4/grpo_parent_${JOB_ID}
NVME_CKPT_DIR=/nvme/leihaodong/Project4/checkpoints/${RUN_NAME}_${JOB_ID}
mkdir -p "$MOUNT_POINT" "$CACHE_DIR" "$PARENT_DIR" "$NVME_CKPT_DIR"

/mnt/petrelfs/leihaodong/app/s3mount datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" --endpoint-url "$ENDPOINT" --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &
S3PID=$!
cleanup() {
  fusermount -u "$MOUNT_POINT" 2>/dev/null \
    || umount "$MOUNT_POINT" 2>/dev/null || true
  kill "$S3PID" 2>/dev/null || true
  wait "$S3PID" 2>/dev/null || true
  rm -rf "$CACHE_DIR" "$PARENT_DIR"
  rmdir "$MOUNT_POINT" 2>/dev/null || true
  if ! rmdir "$NVME_CKPT_DIR" 2>/dev/null; then
    echo "NVMe 中仍有未归档 checkpoint，保留: $NVME_CKPT_DIR" >&2
  fi
}
trap cleanup EXIT
sleep 20

MODEL_PATH=$MOUNT_POINT/$MODEL_REL
PARENT_SOURCE=$MOUNT_POINT/$PARENT_REL
[[ -f "$MODEL_PATH/config.json" ]] || {
  echo "FATAL: model 不可见: $MODEL_PATH" >&2; exit 1; }
[[ -f "$PARENT_SOURCE" ]] || {
  echo "FATAL: GRPO parent 不可见: $PARENT_SOURCE" >&2; exit 1; }
if [[ -n "$PARENT_ETAG" ]]; then
  actual_etag=$(aws s3api head-object --bucket datafrontier --key "$PARENT_REL" \
    --query ETag --output text --endpoint-url "$ENDPOINT")
  actual_etag=${actual_etag//\"/}
  if [[ "$actual_etag" != "$PARENT_ETAG" ]]; then
    echo "FATAL: GRPO parent ETag 已变化: actual=$actual_etag expected=$PARENT_ETAG" >&2
    exit 1
  fi
fi
PARENT_LOCAL=$PARENT_DIR/reference.pt
cp "$PARENT_SOURCE" "$PARENT_LOCAL"

echo "GRPO run=$RUN_NAME parent=$PARENT_REL GPUs=$GPUS group=4 accum=$ACCUM"
"$UV" run --python "$VENV/bin/python" python -m torch.distributed.run \
  --standalone --nproc_per_node="$GPUS" -m transmem.train_grpo \
  --data_dir "$TRAIN_DIR" --data_path "$BENCH/hotpotqa_train_32k.parquet" \
  --data_format hotpotqa-agentmem \
  --val_data_dir "$VAL_DIR" --val_data_path "$BENCH/hotpotqa_dev.parquet" \
  --val_data_format hotpotqa-agentmem --val_max 128 \
  --model_path "$MODEL_PATH" --attn_impl sdpa \
  --config "$CONFIG" --D 4 --S 36 \
  --init_scheme scratch_joint --gate_calibration_steps 0 \
  --policy grpo --divergence forward_kl \
  --sample_temp 0.7 --max_answer_tokens 50 \
  --warm_start_checkpoint "$PARENT_LOCAL" \
  --warm_start_id "$REFERENCE_ID" \
  --reference_checkpoint "$PARENT_LOCAL" \
  --reference_id "$REFERENCE_ID" \
  --group_size 4 --grpo_epochs 2 --clip_eps 0.2 --reference_kl_beta 0.02 \
  --reward_em_weight 0.25 --reward_verbosity_weight 0.05 \
  --reward_verbosity_start 32 --reward_verbosity_cap 64 \
  --output_dir "$OUTPUT_DIR" --grad_accum "$ACCUM" \
  --lr "$LR" --gate_lr "$LR" --weight_decay 0.0 \
  --max_steps "$MAX_STEPS" --warmup_steps 20 --grad_clip 1.0 \
  --seed 20260714 --num_workers 2 \
  --log_interval 5 --val_interval 50 --save_interval 50 \
  --save_nvme_s3 --nvme_checkpoint_dir "$NVME_CKPT_DIR" \
  --s3_checkpoint_uri "$S3_CKPT" --s3_endpoint_url "$ENDPOINT"

for object in latest.pt best.pt result.json; do
  aws s3 ls "$S3_CKPT/$object" --endpoint-url "$ENDPOINT" >/dev/null
done
echo "GRPO complete: $S3_CKPT"
