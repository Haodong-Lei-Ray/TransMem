#!/bin/bash

# Shared Qwen3-4B final-hidden dynamic-gate runner. Stage0 is reused as-is.
set -euo pipefail

: "${SLURM_JOB_ID:?请通过本目录的 run_final_*.sh 用 sbatch 运行}"
: "${INIT_SCHEME:?缺少 INIT_SCHEME=legacy_gate|scratch_joint}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
cd "$PROJ"

GPUS=${GPUS:-8}
BASE_LR=${BASE_LR:-1e-4}
GATE_LR=${GATE_LR:-1e-4}
PRIOR_WEIGHT=${PRIOR_WEIGHT:-0.0}
PRIOR_STEPS=${PRIOR_STEPS:-125}
SEED=${SEED:-20260714}
CONFIG=${CONFIG:-$PROJ/transmem/config_dynamic_gate.json}
FEAT=${FEAT:-$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507}
TRAIN_DIR=${TRAIN_DIR:-$FEAT/stage0_train_short200}
VAL_DIR=${VAL_DIR:-$FEAT/stage0_dev_short200}

INIT_ARGS=()
case "$INIT_SCHEME" in
  legacy_gate)
    CALIBRATION_STEPS=${CALIBRATION_STEPS:-625}
    JOINT_STEPS=${JOINT_STEPS:-625}
    : "${INIT_CHECKPOINT:?legacy_gate 必须通过 sbatch --export=ALL,INIT_CHECKPOINT=/path/to/best.pt 指定父 checkpoint}"
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
OUTPUT_DIR=${OUTPUT_DIR:-$PROJ/checkpoints/v4_gate_final_${INIT_SCHEME}_d4}

RESUME_ARGS=()
if [[ -f "$OUTPUT_DIR/latest.pt" ]]; then
  RESUME_ARGS=(--resume "$OUTPUT_DIR/latest.pt")
fi

echo "dynamic gate final-hidden: init=$INIT_SCHEME seed=$SEED"
"$UV" run --python "$VENV/bin/python" python -m torch.distributed.run \
  --standalone --nproc_per_node="$GPUS" -m transmem.train_offpolicy \
  --data_dir "$TRAIN_DIR" --val_data_dir "$VAL_DIR" \
  --config "$CONFIG" --output_dir "$OUTPUT_DIR" \
  --init_scheme "$INIT_SCHEME" "${INIT_ARGS[@]}" \
  --gate_calibration_steps "$CALIBRATION_STEPS" \
  --joint_finetune_steps "$JOINT_STEPS" --max_steps "$TOTAL_STEPS" \
  --gate_prior_weight "$PRIOR_WEIGHT" --gate_prior_anneal_steps "$PRIOR_STEPS" \
  --divergence forward_kl --temperature 1.0 --reg_weight 0.0 --loss kd \
  --batch_size 16 --lr "$BASE_LR" --gate_lr "$GATE_LR" --weight_decay 0.0 \
  --seed "$SEED" \
  --warmup_steps 50 --grad_clip 1.0 --dtype float32 --num_workers 2 \
  --log_interval 6 --val_interval 250 --save_interval 500 \
  "${RESUME_ARGS[@]}"

echo "dynamic gate final-hidden 完成: $OUTPUT_DIR"
