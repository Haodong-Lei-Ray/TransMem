#!/bin/bash
#SBATCH -J e09_ge_ltr_d4
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_gate_ltr_d4.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_gate_ltr_d4.err

set -euo pipefail
export CORPUS=locomo_train D=4
exec /mnt/petrelfs/leihaodong/Project4/scripts/version4/dynamic_gate/locomo_mix/run_eval.sh
