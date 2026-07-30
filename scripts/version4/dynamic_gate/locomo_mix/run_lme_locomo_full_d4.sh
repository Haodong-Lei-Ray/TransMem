#!/bin/bash
#SBATCH -J e09_g_lfull_d4
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/dynamic_gate/%j_lme_locomo_full_d4.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/dynamic_gate/%j_lme_locomo_full_d4.err

set -euo pipefail
export CORPUS=locomo_full D=4
exec /mnt/petrelfs/leihaodong/Project4/scripts/version4/dynamic_gate/locomo_mix/run_train.sh
