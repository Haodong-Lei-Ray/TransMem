#!/bin/bash
#SBATCH -J e09_v4_s26d4
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/%j_v4_s26d4.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/%j_v4_s26d4.err

set -euo pipefail
export S=26
export D=4
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/run_train_inloop.sh"
