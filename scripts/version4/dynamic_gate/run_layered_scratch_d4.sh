#!/bin/bash
#SBATCH -J e09_v4g_lb4
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/%j_v4g_lb4.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/%j_v4g_lb4.err

# 动态 gate, 从头联合训练 (方案 B / B1), D=4 末四层 (S=36, 层 32-35).
# 对照锚点 = v3.2 固定 g=1 的 D=4 (.445/.500). 严格 gate 有效性需再跑同 seed 的 B0.
set -euo pipefail
export INIT_SCHEME=scratch_joint
export S=36
export D=4
# sbatch 把脚本拷到 /var/spool/slurmd 执行, BASH_SOURCE 定位会失效 — 用绝对路径
exec /mnt/petrelfs/leihaodong/Project4/scripts/version4/dynamic_gate/run_layered_inloop.sh
