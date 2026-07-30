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
# sbatch 把脚本拷到 /var/spool/slurmd 执行, BASH_SOURCE 定位会失效 — 用绝对路径
exec /mnt/petrelfs/leihaodong/Project4/scripts/version4/dynamic_layer/run_train_inloop.sh
