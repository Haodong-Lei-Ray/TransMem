#!/bin/bash
#SBATCH -J e09_p1curve
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH -t 04:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/p1_curve/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/p1_curve/%j.err

# Fixed-seed 20-epoch prefix of the P1 recipe.  The LR cosine horizon remains
# the full original 14,940 steps, so this is a genuine prefix rather than a
# short-run schedule.  Snapshots are model-only; latest.pt retains optimizer
# state for spot requeue/resume.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
cd "$PROJ"

HQA=$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507
LME=$PROJ/data/longmemeval_data/Qwen3-4B-Instruct-2507
TRAIN_DIRS="$LME/stage0_train_short200,$HQA/stage0_train_short200"
VAL_DIRS="$LME/stage0_dev_short200,$HQA/stage0_dev_short200"
OUTPUT_DIR=${OUTPUT_DIR:-$PROJ/checkpoints/diagnostics/p1_curve_seed20260713_prefix4980}
SEED=${SEED:-20260713}
STOP_STEPS=${STOP_STEPS:-4980}
SCHEDULE_STEPS=${SCHEDULE_STEPS:-14940}
CURVE_STEPS=${CURVE_STEPS:-"249 498 996 1992 2988 4750"}

mkdir -p "$OUTPUT_DIR"
RESUME_ARGS=()
if [[ -f "$OUTPUT_DIR/latest.pt" ]]; then
  RESUME_ARGS=(--resume "$OUTPUT_DIR/latest.pt")
fi

$UV run --python "$VENV/bin/python" python -m torch.distributed.run \
  --standalone --nproc_per_node=8 -m transmem.train_offpolicy \
  --data_dir "$TRAIN_DIRS" --val_data_dir "$VAL_DIRS" \
  --config "$PROJ/transmem/config.json" --output_dir "$OUTPUT_DIR" \
  --divergence forward_kl --temperature 1.0 --reg_weight 0.0 --loss kd \
  --batch_size 16 --lr 1e-4 --epochs 60 --max_steps "$STOP_STEPS" \
  --schedule_total_steps "$SCHEDULE_STEPS" --curve_steps $CURVE_STEPS \
  --seed "$SEED" --warmup_steps 50 --grad_clip 1.0 \
  --dtype float32 --num_workers 2 \
  --log_interval 6 --val_interval 249 --save_interval 249 \
  "${RESUME_ARGS[@]}"

echo "P1 fixed-seed curve prefix complete: $OUTPUT_DIR"
