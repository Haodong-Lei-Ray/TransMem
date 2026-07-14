#!/bin/bash
#SBATCH -J e09_v4g_lb8
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/%j_v4g_lb8.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/%j_v4g_lb8.err

set -euo pipefail
export INIT_SCHEME=scratch_joint
export S=36
export D=8
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/run_layered_inloop.sh"
