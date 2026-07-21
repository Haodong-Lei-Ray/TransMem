#!/bin/bash
# Submit the 2-model x 3-domain GRPO matrix and its held dependency evals.
# Training probes 8 -> 7 -> 6 -> 5 -> 4 reserved GPUs.  A pending 4-GPU
# request is intentionally retained; cancelled probes never share an S3 run.
set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
PARTITION=DataFrontier_Explore
PROBE_SECONDS=${PROBE_SECONDS:-30}
MAX_STEPS=${MAX_STEPS:-50}
TRAIN_SCRIPT=$PROJ/scripts/version5/rl/run_train_grpo_memory_d4.sh
LOCOMO_EVAL=$PROJ/scripts/version5/rl/run_eval_locomo_posttrain.sh
LME_EVAL=$PROJ/scripts/version5/rl/run_eval_longmemeval_posttrain.sh
STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST=$PROJ/logs/grpo_memory/submission_${STAMP}.tsv
mkdir -p "$PROJ/logs/grpo_memory" "$PROJ/logs/eval_locomo" \
  "$PROJ/logs/eval_longmemeval"
printf 'model\ttrain\teval\ttrain_job\tgpus\teval_job\trun_name\n' >"$MANIFEST"

wait_until_gone() {
  local job=$1
  for _ in $(seq 1 20); do
    [[ -z "$(squeue -h -u leihaodong -p "$PARTITION" -j "$job" -o '%i' || true)" ]] && return 0
    sleep 1
  done
  echo "FATAL: cancelled probe $job did not leave squeue" >&2
  return 1
}

submit_training() {
  local model=$1 train=$2 run=$3 job_name=$4
  local job="" selected_gpus=""
  for gpus in 8 7 6 5 4; do
    job=$(sbatch --parsable --quotatype=reserved --job-name="$job_name" \
      --gres="gpu:$gpus" \
      --export="ALL,MODEL_KIND=$model,TRAIN_KIND=$train,RUN_NAME=$run,GPUS=$gpus,MAX_STEPS=$MAX_STEPS" \
      "$TRAIN_SCRIPT")
    echo "probe run=$run job=$job gpus=$gpus"
    for _ in $(seq 1 "$PROBE_SECONDS"); do
      state=$(squeue -h -u leihaodong -p "$PARTITION" -j "$job" -o '%T' | head -1 || true)
      [[ "$state" == "RUNNING" ]] && break
      [[ -z "$state" ]] && break
      sleep 1
    done
    state=$(squeue -h -u leihaodong -p "$PARTITION" -j "$job" -o '%T' | head -1 || true)
    if [[ "$state" == "RUNNING" || "$state" == "CONFIGURING" || "$gpus" == "4" ]]; then
      selected_gpus=$gpus
      break
    fi
    scancel "$job"
    wait_until_gone "$job"
  done
  [[ -n "$job" && -n "$selected_gpus" ]] || {
    echo "FATAL: failed to retain a training job for $run" >&2
    return 1
  }
  printf '%s %s\n' "$job" "$selected_gpus"
}

submit_eval() {
  local model=$1 eval_kind=$2 run=$3 train_job=$4 job_name=$5
  local model_name prompt_budget checkpoint output script export_spec
  checkpoint="leihaodong/Project4/checkpoints/$run/step_$(printf '%07d' "$MAX_STEPS").pt"
  output="$PROJ/eval_results/${eval_kind}_${run}"
  case "$model" in
    qwen3_4b)
      model_name=Qwen/Qwen3-4B-Instruct-2507
      prompt_budget=""
      ;;
    qwen25_14b)
      model_name=Qwen/Qwen2.5-14B-Instruct
      prompt_budget=30000
      ;;
    *) echo "FATAL: unknown model=$model" >&2; return 2 ;;
  esac
  export_spec="ALL,MODEL_NAME=$model_name,S3_CKPT_REL=$checkpoint,OUT_ROOT=$output,GPU_COUNT=4,WORKERS=4,WORKERS_PER_GPU=1,THINKING=1,MAX_ANS=256"
  [[ -n "$prompt_budget" ]] && export_spec+=",MAX_PROMPT_TOKENS=$prompt_budget"
  if [[ "$eval_kind" == "longmemeval" ]]; then
    script=$LME_EVAL
    export_spec+=",RUN_OFFICIAL_JUDGE=1"
  else
    script=$LOCOMO_EVAL
  fi
  sbatch --parsable --dependency="afterok:$train_job" \
    --quotatype=reserved --job-name="$job_name" --gres=gpu:4 \
    --export="$export_spec" "$script"
}

submit_one() {
  local model=$1 train=$2 eval_kind=$3 run=$4 train_name=$5 eval_name=$6
  read -r train_job gpus < <(submit_training "$model" "$train" "$run" "$train_name" | tail -1)
  eval_job=$(submit_eval "$model" "$eval_kind" "$run" "$train_job" "$eval_name")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$model" "$train" "$eval_kind" "$train_job" "$gpus" "$eval_job" "$run" \
    | tee -a "$MANIFEST"
}

submit_one qwen3_4b locomo longmemeval \
  v5_grpo_q3_4b_d4g_locomo_to_lme e09_g3lc2lm e09_e3lc2lm
submit_one qwen3_4b longmemeval locomo \
  v5_grpo_q3_4b_d4g_lme_to_locomo e09_g3lm2lc e09_e3lm2lc
submit_one qwen3_4b locomo_train locomo \
  v5_grpo_q3_4b_d4g_ltrain_to_locomo e09_g3lt2lc e09_e3lt2lc
submit_one qwen25_14b locomo longmemeval \
  v5_grpo_q25_14b_d4g_locomo_to_lme e09_g25lc2lm e09_e25lc2lm
submit_one qwen25_14b longmemeval locomo \
  v5_grpo_q25_14b_d4g_lme_to_locomo e09_g25lm2lc e09_e25lm2lc
submit_one qwen25_14b locomo_train locomo \
  v5_grpo_q25_14b_d4g_ltrain_to_locomo e09_g25lt2lc e09_e25lt2lc

echo "submission manifest: $MANIFEST"
