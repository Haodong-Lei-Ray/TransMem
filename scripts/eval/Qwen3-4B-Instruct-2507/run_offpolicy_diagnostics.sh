#!/bin/bash
#SBATCH -J e09_posdiag
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH -t 02:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/offpolicy_diagnostics/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/offpolicy_diagnostics/%j.err

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"
cd "$PROJ"

DATA_DIR=${DATA_DIR:?set DATA_DIR to one Stage0 dev directory}
NAME=${NAME:-$(basename "$(dirname "$DATA_DIR")")_$(basename "$DATA_DIR")}
CKPT=${CKPT:-$PROJ/checkpoints/offpolicy_v3_p1_lmehqa_d4e60_forward_kl/best.pt}
OUT_ROOT=${OUT_ROOT:-$PROJ/eval_outputs/diagnostics/p1_offpolicy}
BATCH_SIZE=${BATCH_SIZE:-4}
SEED=${SEED:-20260713}
DTYPE=${DTYPE:-bfloat16}

mkdir -p "$OUT_ROOT"
test -f "$CKPT"
test -f "$DATA_DIR/meta.json"
test -f "$DATA_DIR/lm_head.pt"

$PY scripts/eval/eval_offpolicy_diagnostics.py \
  --data_dir "$DATA_DIR" --ckpt "$CKPT" \
  --batch_size "$BATCH_SIZE" --shuffle_seed "$SEED" --dtype "$DTYPE" \
  --output_json "$OUT_ROOT/${NAME}.json"

echo "Offline diagnostic complete: $OUT_ROOT/${NAME}.json"
