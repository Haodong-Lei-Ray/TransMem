#!/bin/bash
#SBATCH -J e09_v5gmem
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/grpo_memory/%j_train.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/grpo_memory/%j_train.err

# Domain GRPO for a frozen LLM + D=4 dynamic-gate TransMem.  Rollouts contain
# reasoning and a marked final answer; only the parsed answer receives reward.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED
export AWS_RESPONSE_CHECKSUM_VALIDATION=WHEN_REQUIRED
export TOKENIZERS_PARALLELISM=false

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
ENDPOINT=http://d-ceph-ssd-inside.pjlab.org.cn
: "${MODEL_KIND:?MODEL_KIND=qwen3_4b|qwen25_14b is required}"
: "${TRAIN_KIND:?TRAIN_KIND=locomo|longmemeval|locomo_train is required}"
: "${RUN_NAME:?RUN_NAME is required}"
GPUS=${GPUS:-8}
MAX_STEPS=${MAX_STEPS:-50}
MAX_RESPONSE_TOKENS=${MAX_RESPONSE_TOKENS:-256}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-}
LR=${LR:-5e-6}
REFERENCE_KL_BETA=${REFERENCE_KL_BETA:-0.05}
GROUP_SIZE=${GROUP_SIZE:-4}
# Preserve approximately the original global prompt batch of eight when a
# launch is downshifted.  Nearest-integer accumulation gives 8/7/6 -> 1,
# 5/4 -> 2, 3 -> 3, and 2 -> 4.
DEFAULT_ACCUM=$(((8 + GPUS / 2) / GPUS))
ACCUM=${ACCUM:-$DEFAULT_ACCUM}

case "$MODEL_KIND" in
  qwen3_4b)
    MODEL_REL=leihaodong/Qwen/Qwen3-4B-Instruct-2507
    PARENT_REL=leihaodong/Project4/checkpoints/v4_gate_layered_scratch_joint_s36_d4/best.pt
    CONFIG=$PROJ/transmem/config_layered_dynamic_gate.json
    STOP_LAYER=36
    ;;
  qwen25_14b)
    MODEL_REL=leihaodong/Qwen/Qwen2.5-14B-Instruct
    PARENT_REL=leihaodong/Project4/checkpoints/v5_gate_qwen25_14b_scratch_joint_d4/best.pt
    CONFIG=$PROJ/transmem/config_layered_qwen25_14b_dynamic_gate.json
    STOP_LAYER=48
    # Qwen2.5-14B config is 32K.  Reserve room for a 256-token response and
    # make the unavoidable LME truncation explicit/reproducible.
    MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-30000}
    ;;
  *) echo "FATAL: unknown MODEL_KIND=$MODEL_KIND" >&2; exit 2 ;;
esac

case "$TRAIN_KIND" in
  locomo)
    DATA_PATH=/mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo10.json
    DATA_FORMAT=locomo
    REWARD_SCORER=locomo
    REWARD_CATEGORIES=1,2,3,4
    ;;
  locomo_train)
    DATA_PATH=/mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo-train.json
    DATA_FORMAT=locomo
    REWARD_SCORER=locomo
    REWARD_CATEGORIES=1,2,3,4
    ;;
  longmemeval)
    DATA_PATH=$PROJ/data/LongMemEval/data/longmemeval_train.json
    DATA_FORMAT=longmemeval
    REWARD_SCORER=longmemeval
    REWARD_CATEGORIES=
    ;;
  *) echo "FATAL: unknown TRAIN_KIND=$TRAIN_KIND" >&2; exit 2 ;;
esac

OUTPUT_DIR=$PROJ/checkpoints/$RUN_NAME
S3_CKPT=s3://datafrontier/leihaodong/Project4/checkpoints/$RUN_NAME
JOB_ID=${SLURM_JOB_ID:-$(date +%s)_$$}
MOUNT_POINT=/mnt/petrelfs/leihaodong/tmp/s3_grpo_memory_${JOB_ID}
CACHE_DIR=/nvme/leihaodong/s3cache_grpo_memory_${JOB_ID}
PARENT_DIR=/nvme/leihaodong/Project4/grpo_memory_parent_${JOB_ID}
NVME_CKPT_DIR=/nvme/leihaodong/Project4/checkpoints/${RUN_NAME}_${JOB_ID}
mkdir -p "$MOUNT_POINT" "$CACHE_DIR" "$PARENT_DIR" "$NVME_CKPT_DIR" \
  "$PROJ/logs/grpo_memory"

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
    echo "NVMe retains unarchived checkpoint: $NVME_CKPT_DIR" >&2
  fi
}
trap cleanup EXIT
sleep 20

MODEL_PATH=$MOUNT_POINT/$MODEL_REL
PARENT_SOURCE=$MOUNT_POINT/$PARENT_REL
[[ -f "$MODEL_PATH/config.json" ]] || { echo "FATAL: model missing $MODEL_PATH"; exit 1; }
[[ -f "$PARENT_SOURCE" ]] || { echo "FATAL: parent missing $PARENT_SOURCE"; exit 1; }
PARENT_LOCAL=$PARENT_DIR/reference.pt
cp "$PARENT_SOURCE" "$PARENT_LOCAL"
REFERENCE_ID=s3://datafrontier/$PARENT_REL

init_args=(--warm_start_checkpoint "$PARENT_LOCAL" --warm_start_id "$REFERENCE_ID")
REMOTE_LATEST=$MOUNT_POINT/leihaodong/Project4/checkpoints/$RUN_NAME/latest.pt
if [[ -f "$REMOTE_LATEST" ]]; then
  RESUME_LOCAL=$PARENT_DIR/resume_latest.pt
  cp "$REMOTE_LATEST" "$RESUME_LOCAL"
  init_args=(--resume "$RESUME_LOCAL")
  echo "Resume GRPO from $S3_CKPT/latest.pt"
fi
category_args=()
[[ -n "$REWARD_CATEGORIES" ]] && category_args=(--reward_categories "$REWARD_CATEGORIES")
prompt_budget_args=()
[[ -n "$MAX_PROMPT_TOKENS" ]] && prompt_budget_args=(--max_prompt_tokens "$MAX_PROMPT_TOKENS")

echo "GRPO memory run=$RUN_NAME model=$MODEL_KIND train=$TRAIN_KIND GPUs=$GPUS accum=$ACCUM"
echo "parent=$REFERENCE_ID steps=$MAX_STEPS response_tokens=$MAX_RESPONSE_TOKENS thinking=1 answer_reward_only=1"
if [[ -n "$MAX_PROMPT_TOKENS" ]]; then
  echo "prompt_budget=$MAX_PROMPT_TOKENS (head/tail retention with explicit omission marker)"
fi
cd "$PROJ"
"$UV" run --python "$VENV/bin/python" python -m torch.distributed.run \
  --standalone --nproc_per_node="$GPUS" -m transmem.train_grpo \
  --data_dir "$DATA_PATH" --data_path "$DATA_PATH" --data_format "$DATA_FORMAT" \
  --model_path "$MODEL_PATH" --attn_impl sdpa \
  --config "$CONFIG" --D 4 --S "$STOP_LAYER" \
  --init_scheme scratch_joint --gate_calibration_steps 0 \
  --policy grpo --divergence forward_kl \
  --thinking --require_answer_marker --require_thinking \
  --reward_scorer "$REWARD_SCORER" \
  "${category_args[@]}" "${prompt_budget_args[@]}" \
  --sample_temp 0.7 --max_answer_tokens "$MAX_RESPONSE_TOKENS" \
  "${init_args[@]}" \
  --reference_checkpoint "$PARENT_LOCAL" --reference_id "$REFERENCE_ID" \
  --group_size "$GROUP_SIZE" --grpo_epochs 2 --clip_eps 0.2 \
  --reference_kl_beta "$REFERENCE_KL_BETA" \
  --reward_em_weight 0.25 --reward_verbosity_weight 0.05 \
  --reward_verbosity_start 32 --reward_verbosity_cap 64 \
  --output_dir "$OUTPUT_DIR" --grad_accum "$ACCUM" \
  --lr "$LR" --gate_lr "$LR" --weight_decay 0.0 \
  --max_steps "$MAX_STEPS" --warmup_steps 5 --grad_clip 1.0 \
  --gradient_checkpointing --seed 20260721 --num_workers 0 \
  --log_interval 2 --val_interval 10 --save_interval 10 \
  --save_nvme_s3 --nvme_checkpoint_dir "$NVME_CKPT_DIR" \
  --s3_checkpoint_uri "$S3_CKPT" --s3_endpoint_url "$ENDPOINT"

FINAL_NAME=$(printf 'step_%07d.pt' "$MAX_STEPS")
for object in latest.pt "$FINAL_NAME" result.json; do
  aws s3 ls "$S3_CKPT/$object" --endpoint-url "$ENDPOINT" >/dev/null
done
echo "GRPO memory complete: $S3_CKPT"
