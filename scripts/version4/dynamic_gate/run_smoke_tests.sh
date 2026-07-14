#!/bin/bash
#SBATCH -J e09_v4g_test
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH -t 00:15:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/%j_v4g_test.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/%j_v4g_test.err

set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
PY=$PROJ/.venv-transmem/bin/python
cd "$PROJ"

"$PY" -m compileall -q transmem scripts/eval
"$PY" -m pytest -q transmem scripts/eval \
  --ignore=transmem/test_teacher_gen.py

for script in scripts/version4/dynamic_gate/*.sh; do
  bash -n "$script"
done

"$PY" -m transmem.train_offpolicy --help >/dev/null
"$PY" -m transmem.train_onpolicy --help >/dev/null
"$PY" -m transmem.train_inloop --help >/dev/null
"$PY" -m transmem.evaluate --help >/dev/null
"$PY" -m transmem.migrate_checkpoint --help >/dev/null
