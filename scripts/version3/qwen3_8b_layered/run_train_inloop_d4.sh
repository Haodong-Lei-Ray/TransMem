#!/bin/bash
#SBATCH -J e09_v38l4_tr
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:6
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b_layered/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b_layered/%j.err

# Qwen3-8B D=4 ablation.  Reuse the D=8 mainline implementation while
# isolating its checkpoint prefix through the depth-aware default output path.
set -euo pipefail

export D=4
export GPUS=${GPUS:-6}
# Six ranks cannot form the D=8 run's exact global batch 32 with integer
# accumulation; 6x5=30 is the nearest configuration.
export ACCUM=${ACCUM:-5}

exec bash /mnt/petrelfs/leihaodong/Project4/scripts/version3/qwen3_8b_layered/run_train_inloop_d8.sh
