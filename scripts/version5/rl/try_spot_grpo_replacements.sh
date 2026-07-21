#!/bin/bash
# Safely race Spot capacity against an already-submitted Reserved GRPO matrix.
# Reserved jobs are held during each probe so only one writer can reach a run's
# S3 checkpoint prefix.  A successful Spot job replaces the Reserved job and
# gets a freshly wired afterok evaluation; otherwise Reserved is released.
set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
PARTITION=DataFrontier_Explore
PROBE_SECONDS=${PROBE_SECONDS:-30}
MAX_STEPS=${MAX_STEPS:-50}
GPU_LEVELS=${GPU_LEVELS:-"8 7 6 5 4 3 2"}
TARGET_QUOTATYPE=${TARGET_QUOTATYPE:-spot}
RETAIN_LAST_PENDING=${RETAIN_LAST_PENDING:-0}
LAST_GPU_LEVEL=${GPU_LEVELS##* }
[[ "$TARGET_QUOTATYPE" == "spot" || "$TARGET_QUOTATYPE" == "reserved" ]] || {
  echo "FATAL: TARGET_QUOTATYPE must be spot or reserved" >&2
  exit 2
}
TRAIN_SCRIPT=$PROJ/scripts/version5/rl/run_train_grpo_memory_d4.sh
LOCOMO_EVAL=$PROJ/scripts/version5/rl/run_eval_locomo_posttrain.sh
LME_EVAL=$PROJ/scripts/version5/rl/run_eval_longmemeval_posttrain.sh
INPUT_MANIFEST=${1:?usage: $0 submission_manifest.tsv}
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_MANIFEST=$PROJ/logs/grpo_memory/submission_${TARGET_QUOTATYPE}_${STAMP}.tsv

declare -a models trains evals reserved_jobs old_eval_jobs runs
declare -a spot_jobs selected_gpus

while IFS=$'\t' read -r model train eval_kind train_job gpus eval_job run; do
  [[ "$model" == "model" ]] && continue
  models+=("$model")
  trains+=("$train")
  evals+=("$eval_kind")
  reserved_jobs+=("$train_job")
  old_eval_jobs+=("$eval_job")
  runs+=("$run")
  spot_jobs+=("")
  selected_gpus+=("")
done <"$INPUT_MANIFEST"

release_unreplaced() {
  local i
  for i in "${!reserved_jobs[@]}"; do
    if [[ -z "${selected_gpus[$i]}" ]]; then
      scontrol release "${reserved_jobs[$i]}" >/dev/null 2>&1 || true
    fi
  done
}
trap release_unreplaced EXIT

for job in "${reserved_jobs[@]}"; do
  scontrol hold "$job"
done

for gpus in $GPU_LEVELS; do
  submitted=()
  for i in "${!reserved_jobs[@]}"; do
    [[ -n "${selected_gpus[$i]}" ]] && continue
    case "${models[$i]}" in
      qwen3_4b) short=g3 ;;
      qwen25_14b) short=g25 ;;
      *) echo "FATAL: unknown model=${models[$i]}" >&2; exit 2 ;;
    esac
    case "${trains[$i]}" in
      locomo) suffix=lc2lm ;;
      longmemeval) suffix=lm2lc ;;
      locomo_train) suffix=lt2lc ;;
      *) echo "FATAL: unknown train=${trains[$i]}" >&2; exit 2 ;;
    esac
    if [[ "$TARGET_QUOTATYPE" == "spot" ]]; then prefix=s; else prefix=r; fi
    job=$(sbatch --parsable --quotatype="$TARGET_QUOTATYPE" --job-name="e09_${prefix}${short}${suffix}" \
      --gres="gpu:$gpus" \
      --export="ALL,MODEL_KIND=${models[$i]},TRAIN_KIND=${trains[$i]},RUN_NAME=${runs[$i]},GPUS=$gpus,MAX_STEPS=$MAX_STEPS" \
      "$TRAIN_SCRIPT")
    spot_jobs[$i]=$job
    submitted+=("$i")
    echo "$TARGET_QUOTATYPE probe run=${runs[$i]} job=$job gpus=$gpus"
  done

  ((${#submitted[@]} == 0)) && break
  sleep "$PROBE_SECONDS"
  for i in "${submitted[@]}"; do
    job=${spot_jobs[$i]}
    state=$(squeue -h -u leihaodong -p "$PARTITION" -j "$job" -o '%T' | head -1 || true)
    if [[ "$state" == "RUNNING" || "$state" == "CONFIGURING" || \
          ( "$RETAIN_LAST_PENDING" == "1" && "$gpus" == "$LAST_GPU_LEVEL" && "$state" == "PENDING" ) ]]; then
      selected_gpus[$i]=$gpus
      echo "$TARGET_QUOTATYPE selected run=${runs[$i]} job=$job gpus=$gpus state=$state"
    else
      scancel "$job" || true
      spot_jobs[$i]=""
    fi
  done
done

submit_eval() {
  local model=$1 eval_kind=$2 run=$3 train_job=$4 job_name=$5
  local model_name prompt_budget checkpoint output script export_spec
  checkpoint="leihaodong/Project4/checkpoints/$run/step_$(printf '%07d' "$MAX_STEPS").pt"
  output="$PROJ/eval_results/${eval_kind}_${run}"
  case "$model" in
    qwen3_4b) model_name=Qwen/Qwen3-4B-Instruct-2507; prompt_budget="" ;;
    qwen25_14b) model_name=Qwen/Qwen2.5-14B-Instruct; prompt_budget=30000 ;;
  esac
  export_spec="ALL,MODEL_NAME=$model_name,S3_CKPT_REL=$checkpoint,OUT_ROOT=$output,GPU_COUNT=4,WORKERS=4,WORKERS_PER_GPU=1,THINKING=1,MAX_ANS=256"
  [[ -n "$prompt_budget" ]] && export_spec+=",MAX_PROMPT_TOKENS=$prompt_budget"
  if [[ "$eval_kind" == "longmemeval" ]]; then
    script=$LME_EVAL
    export_spec+=",RUN_OFFICIAL_JUDGE=1"
  else
    script=$LOCOMO_EVAL
  fi
  sbatch --parsable --dependency="afterok:$train_job" --quotatype=reserved \
    --job-name="$job_name" --gres=gpu:4 --export="$export_spec" "$script"
}

printf 'model\ttrain\teval\ttrain_job\tgpus\tquotatype\teval_job\trun_name\n' >"$OUTPUT_MANIFEST"
for i in "${!reserved_jobs[@]}"; do
  if [[ -n "${selected_gpus[$i]}" ]]; then
    scancel "${old_eval_jobs[$i]}" || true
    scancel "${reserved_jobs[$i]}" || true
    train_job=${spot_jobs[$i]}
    quotatype=$TARGET_QUOTATYPE
  else
    scontrol release "${reserved_jobs[$i]}"
    train_job=${reserved_jobs[$i]}
    quotatype=reserved
  fi
  case "${models[$i]}:${trains[$i]}" in
    qwen3_4b:locomo) eval_name=e09_e3lc2lm ;;
    qwen3_4b:longmemeval) eval_name=e09_e3lm2lc ;;
    qwen3_4b:locomo_train) eval_name=e09_e3lt2lc ;;
    qwen25_14b:locomo) eval_name=e09_e25lc2lm ;;
    qwen25_14b:longmemeval) eval_name=e09_e25lm2lc ;;
    qwen25_14b:locomo_train) eval_name=e09_e25lt2lc ;;
  esac
  if [[ "$train_job" != "${reserved_jobs[$i]}" ]]; then
    eval_job=$(submit_eval "${models[$i]}" "${evals[$i]}" "${runs[$i]}" "$train_job" "$eval_name")
  else
    eval_job=${old_eval_jobs[$i]}
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${models[$i]}" "${trains[$i]}" "${evals[$i]}" "$train_job" \
    "${selected_gpus[$i]:-4}" "$quotatype" "$eval_job" "${runs[$i]}" \
    | tee -a "$OUTPUT_MANIFEST"
done

trap - EXIT
echo "$TARGET_QUOTATYPE replacement manifest: $OUTPUT_MANIFEST"
