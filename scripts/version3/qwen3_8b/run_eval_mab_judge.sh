#!/bin/bash
#SBATCH -J e09_v38_judge
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --quotatype=reserved
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b/%j.err
set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
exec bash "$PROJ/scripts/eval/Qwen3-4B-Instruct-2507/run_eval_memory_agent_bench_judge.sh"
