#!/bin/bash
#SBATCH -J e09_v4g_fb4
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/%j_v4g_fb4.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/%j_v4g_fb4.err

set -euo pipefail
export INIT_SCHEME=scratch_joint
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/run_final_offpolicy.sh"
