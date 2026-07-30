#!/bin/bash
#SBATCH -J e09_h15loc_sum
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:0
#SBATCH --quotatype=reserved
#SBATCH -t 01:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_h15loc_sum.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_h15loc_sum.err

set -euo pipefail
PROJ=/mnt/petrelfs/leihaodong/Project4
PY=/mnt/petrelfs/leihaodong/Project4/.venv-transmem/bin/python
OUT=$PROJ/eval_outputs/locomo_v4_hqa15_locomo_lme_d4

"$PY" "$PROJ/scripts/eval/compare_locomo_heldout.py" \
  --full_data /mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo10.json \
  --train_data /mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo-train.json \
  --new_result "$OUT/locomo_transmem.json" \
  --student_result "$PROJ/eval_outputs/locomo_offpolicy_short128_forward_kl_student/locomo_student.json" \
  --prior_result "$PROJ/eval_outputs/locomo_v3_2_inloop_tf_d4/locomo_transmem.json" \
  --output "$OUT/heldout_comparison.json"
