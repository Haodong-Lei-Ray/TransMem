#!/bin/bash
#SBATCH -J e09_v38_locomo
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b/%j.err
set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
MODE=${MODE:-student}
export ModelName=Qwen/Qwen3-8B
export DATA_FILE=/mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo10.json
export MODES=$MODE
export OUT_ROOT=${OUT_ROOT:?Set OUT_ROOT}
if [[ "$MODE" == transmem ]]; then
  export CKPT=${CKPT:?Set CKPT for transmem}
  export N=${N:?Set N for transmem}
fi
exec bash "$PROJ/scripts/eval/Qwen3-4B-Instruct-2507/run_eval_locomo.sh"
