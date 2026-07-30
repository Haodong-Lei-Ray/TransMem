#!/bin/bash
#SBATCH -J e09_lme_jdg
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:0
#SBATCH --quotatype=reserved
#SBATCH --cpus-per-task=2
#SBATCH -t 02:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_longmemeval/%j_judge.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_longmemeval/%j_judge.err

set -euo pipefail

: "${OUT_ROOT:?Set OUT_ROOT containing hypotheses.jsonl}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY=("$UV" run --python "$VENV/bin/python" python)
DATA_FILE=${DATA_FILE:-$PROJ/data/LongMemEval/data/longmemeval_dev.json}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-4o}
JUDGE_WORKERS=${JUDGE_WORKERS:-8}
LONGMEMEVAL_JUDGE_MAX_ATTEMPTS=${LONGMEMEVAL_JUDGE_MAX_ATTEMPTS:-5}
LONGMEMEVAL_JUDGE_MAX_BACKOFF_SECONDS=${LONGMEMEVAL_JUDGE_MAX_BACKOFF_SECONDS:-30}
HYPOTHESES=$OUT_ROOT/hypotheses.jsonl
JUDGE_DIR=$OUT_ROOT/official_judge_shards

export LONGMEMEVAL_JUDGE_MAX_ATTEMPTS LONGMEMEVAL_JUDGE_MAX_BACKOFF_SECONDS

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
cd "$PROJ"

if [[ ! -s "$HYPOTHESES" ]]; then
  echo "FATAL: missing or empty hypotheses: $HYPOTHESES" >&2
  exit 1
fi
if ((JUDGE_WORKERS < 1)); then
  echo "FATAL: JUDGE_WORKERS must be positive" >&2
  exit 2
fi
rm -f "$HYPOTHESES.eval-results-$JUDGE_MODEL"
rm -rf "$JUDGE_DIR"
mkdir -p "$JUDGE_DIR"
awk -v n="$JUDGE_WORKERS" -v dir="$JUDGE_DIR" \
  '{print > (dir "/hyp_" ((NR - 1) % n) ".jsonl")}' "$HYPOTHESES"

pids=()
for shard in "$JUDGE_DIR"/hyp_*.jsonl; do
  "${PY[@]}" data/longmemeval/src/evaluation/evaluate_qa.py \
    "$JUDGE_MODEL" "$shard" "$DATA_FILE" \
    >"$shard.judge.log" 2>"$shard.judge.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "FATAL: at least one official judge shard failed" >&2
  exit "$status"
fi
cat "$JUDGE_DIR"/hyp_*.jsonl.eval-results-"$JUDGE_MODEL" \
  >"$HYPOTHESES.eval-results-$JUDGE_MODEL"
cat "$JUDGE_DIR"/hyp_*.jsonl.judge.log >"$OUT_ROOT/official_judge.log"
"${PY[@]}" data/longmemeval/src/evaluation/print_qa_metrics.py \
  "$HYPOTHESES.eval-results-$JUDGE_MODEL" "$DATA_FILE" \
    | tee "$OUT_ROOT/official_metrics.txt"
