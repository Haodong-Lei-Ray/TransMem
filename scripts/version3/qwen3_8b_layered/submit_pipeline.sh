#!/bin/bash
# 提交 Stage0 -> in-loop D=8 -> LoCoMo 的 afterok DAG。
set -euo pipefail
PROJ=/mnt/petrelfs/leihaodong/Project4
cd "$PROJ"

S0=$(sbatch --parsable scripts/version3/qwen3_8b_layered/run_stage0_hotpotqa.sh)
TR=$(sbatch --parsable --dependency=afterok:$S0 \
  scripts/version3/qwen3_8b_layered/run_train_inloop_d8.sh)
EV=$(sbatch --parsable --dependency=afterok:$TR \
  scripts/version3/qwen3_8b_layered/run_eval_locomo.sh)

echo "stage0=$S0 train=$TR locomo=$EV"
