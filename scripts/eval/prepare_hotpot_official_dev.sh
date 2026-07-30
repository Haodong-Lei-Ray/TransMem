#!/bin/bash
#SBATCH -J e09_hpdevprep
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --quotatype=reserved
#SBATCH -t 01:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_hotpot_official/%j_prepare.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_hotpot_official/%j_prepare.err

set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
HOTPOT=$PROJ/data/hotpotqa-benchmark/hotpot
RAW=$HOTPOT/hotpot_dev_distractor_v1.parquet
READY=$HOTPOT/hotpot_dev_distractor_v1_eval.json
URL=https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/validation/0.parquet
PY=/mnt/petrelfs/leihaodong/anaconda3/envs/qwen3/bin/python

mkdir -p "$HOTPOT" "$PROJ/logs/eval_hotpot_official"
if [[ ! -s "$RAW" ]]; then
  rm -f "$RAW.tmp"
  curl --fail --location --retry 5 --retry-delay 5 \
    --connect-timeout 30 --max-time 600 "$URL" --output "$RAW.tmp"
  mv "$RAW.tmp" "$RAW"
fi
"$PY" "$PROJ/scripts/eval/prepare_hotpot_official_dev.py" \
  --official-dev "$RAW" \
  --agent-train "$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem/hotpotqa_train_32k.parquet" \
  --output "$READY"
