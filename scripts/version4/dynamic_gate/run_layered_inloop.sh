#!/bin/bash

# Shared Qwen3-4B layered dynamic-gate runner. Submit an sbatch entrypoint.
set -euo pipefail

: "${SLURM_JOB_ID:?请通过本目录的 run_layered_*.sh 用 sbatch 运行}"
: "${INIT_SCHEME:?缺少 INIT_SCHEME=legacy_gate|scratch_joint}"

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
S=${S:-36}
D=${D:-8}
ACCUM=${ACCUM:-4}
BASE_LR=${BASE_LR:-1e-4}
GATE_LR=${GATE_LR:-1e-4}
PRIOR_WEIGHT=${PRIOR_WEIGHT:-0.0}
PRIOR_STEPS=${PRIOR_STEPS:-125}
SEED=${SEED:-20260714}
CONFIG=${CONFIG:-$PROJ/transmem/config_layered_dynamic_gate.json}

FEAT=$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507
BENCH=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem
TRAIN_DIR=${TRAIN_DIR:-$FEAT/stage0_train_short200}
VAL_DIR=${VAL_DIR:-$FEAT/stage0_dev_short200}
DATA_PATH=${DATA_PATH:-$BENCH/hotpotqa_train_32k.parquet}
VAL_DATA_PATH=${VAL_DATA_PATH:-$BENCH/hotpotqa_dev.parquet}
DATA_FORMAT=${DATA_FORMAT:-hotpotqa-agentmem}
VAL_DATA_FORMAT=${VAL_DATA_FORMAT:-$DATA_FORMAT}

INIT_ARGS=()
case "$INIT_SCHEME" in
  legacy_gate)
    CALIBRATION_STEPS=${CALIBRATION_STEPS:-625}
    JOINT_STEPS=${JOINT_STEPS:-625}
    INIT_CHECKPOINT=${INIT_CHECKPOINT:-$PROJ/checkpoints/v3_2_inloop_tf_d8/best.pt}
    [[ -f "$INIT_CHECKPOINT" ]] || {
      echo "FATAL: legacy parent checkpoint 不存在: $INIT_CHECKPOINT" >&2
      exit 1
    }
    INIT_ARGS=(--init_checkpoint "$INIT_CHECKPOINT")
    ;;
  scratch_joint)
    CALIBRATION_STEPS=0
    JOINT_STEPS=${JOINT_STEPS:-1250}
    ;;
  *)
    echo "FATAL: INIT_SCHEME=$INIT_SCHEME 不是 legacy_gate|scratch_joint" >&2
    exit 1
    ;;
esac
TOTAL_STEPS=$((CALIBRATION_STEPS + JOINT_STEPS))
OUTPUT_DIR=${OUTPUT_DIR:-$PROJ/checkpoints/v4_gate_layered_${INIT_SCHEME}_s${S}_d${D}}

RESUME_ARGS=()
if [[ -f "$OUTPUT_DIR/latest.pt" ]]; then
  RESUME_ARGS=(--resume "$OUTPUT_DIR/latest.pt")
fi

mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs
MOUNT_POINT=/mnt/petrelfs/leihaodong/tmp/s3_v4_gate_${SLURM_JOB_ID}
CACHE_DIR=/nvme/leihaodong/s3cache_v4_gate_${SLURM_JOB_ID}
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
[[ -f "$MODEL_PATH/config.json" ]] || {
  echo "FATAL: 模型不可见: $MODEL_PATH" >&2
  exit 1
}

echo "dynamic gate layered: init=$INIT_SCHEME S=$S D=$D seed=$SEED"
echo "calibration=$CALIBRATION_STEPS joint=$JOINT_STEPS prior=$PRIOR_WEIGHT/$PRIOR_STEPS"
"$UV" run --python "$VENV/bin/python" python -m torch.distributed.run \
  --standalone --nproc_per_node="$GPUS" -m transmem.train_inloop \
  --data_dir "$TRAIN_DIR" --data_path "$DATA_PATH" --data_format "$DATA_FORMAT" \
  --val_data_dir "$VAL_DIR" --val_data_path "$VAL_DATA_PATH" \
  --val_data_format "$VAL_DATA_FORMAT" --val_max "${VAL_MAX:-128}" \
  --model_path "$MODEL_PATH" --config "$CONFIG" --D "$D" --S "$S" \
  --policy tf --divergence forward_kl \
  --init_scheme "$INIT_SCHEME" ${INIT_ARGS[@]+"${INIT_ARGS[@]}"} \
  --gate_calibration_steps "$CALIBRATION_STEPS" \
  --joint_finetune_steps "$JOINT_STEPS" --max_steps "$TOTAL_STEPS" \
  --gate_prior_weight "$PRIOR_WEIGHT" --gate_prior_anneal_steps "$PRIOR_STEPS" \
  --output_dir "$OUTPUT_DIR" --grad_accum "$ACCUM" \
  --seed "$SEED" \
  --lr "$BASE_LR" --gate_lr "$GATE_LR" --weight_decay 0.0 \
  --warmup_steps 100 --grad_clip 1.0 --num_workers 2 \
  --log_interval 25 --val_interval 250 --save_interval 500 \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  ${EXTRA:-}

echo "dynamic gate layered 完成: $OUTPUT_DIR"
