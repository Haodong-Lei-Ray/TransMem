#!/bin/bash
set -euo pipefail
PROJ=/mnt/petrelfs/leihaodong/Project4
mkdir -p "$PROJ/logs/eval_hotpot_official"
prep=$(sbatch --parsable "$PROJ/scripts/eval/prepare_hotpot_official_dev.sh")
q3=$(sbatch --parsable --dependency="afterok:$prep" "$PROJ/scripts/eval/run_hotpot_qwen3_4b_gate_d4.sh")
q25=$(sbatch --parsable --dependency="afterok:$prep" "$PROJ/scripts/eval/run_hotpot_qwen25_14b_gate_d4.sh")
echo "PREP_JOB=$prep"
echo "QWEN3_4B_JOB=$q3"
echo "QWEN25_14B_JOB=$q25"
