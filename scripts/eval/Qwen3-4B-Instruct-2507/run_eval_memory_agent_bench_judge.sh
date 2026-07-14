#!/bin/bash
#SBATCH -J p4_mab_judge
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --quotatype=spot
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/mab_judge_%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/mab_judge_%j.err
set -euo pipefail

# Program/evaluation parameters. Resource overrides belong on the sbatch command.
PROJECT=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
PYTHON=$PROJECT/.venv-transmem/bin/python
MAB_ROOT=${MAB_ROOT:-/mnt/petrelfs/leihaodong/Project1/MemoryAgentBenchProject/MemoryAgentBench}
SOURCE=${SOURCE:?Set SOURCE to longmemeval_s* or infbench_sum_eng_shots2}
MODE=${MODE:?Set MODE to student or transmem}
INPUT=${INPUT:?Set INPUT to one completed per-source prediction progress JSONL}
OUT=${OUT:?Set OUT to a mode-specific judge output directory}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-4o}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-6}
INITIAL_BACKOFF=${INITIAL_BACKOFF:-2}
MAX_BACKOFF=${MAX_BACKOFF:-30}
DRY_RUN=${DRY_RUN:-0}

COMMAND=(
  "$UV" run --python "$PYTHON" python
  "$PROJECT/scripts/eval/eval_memory_agent_bench_judge.py"
  --source "$SOURCE"
  --mode "$MODE"
  --input "$INPUT"
  --output_dir "$OUT"
  --mab_root "$MAB_ROOT"
  --judge_model "$JUDGE_MODEL"
  --max_attempts "$MAX_ATTEMPTS"
  --initial_backoff "$INITIAL_BACKOFF"
  --max_backoff "$MAX_BACKOFF"
)

printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
  exit 0
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in the submission environment}"
if [[ ! -f "$INPUT" ]]; then
  echo "Prediction input does not exist: $INPUT" >&2
  exit 2
fi
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
exec "${COMMAND[@]}"
