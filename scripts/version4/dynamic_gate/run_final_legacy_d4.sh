#!/bin/bash
#SBATCH -J e09_v4g_fa4
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/%j_v4g_fa4.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/%j_v4g_fa4.err

set -euo pipefail
export INIT_SCHEME=legacy_gate
# sbatch 把脚本拷到 /var/spool/slurmd 执行, BASH_SOURCE 定位会失效 — 用绝对路径
exec /mnt/petrelfs/leihaodong/Project4/scripts/version4/dynamic_gate/run_final_offpolicy.sh
