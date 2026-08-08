#!/bin/bash
#SBATCH -J transmem_eval
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH -t 12:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/logs/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/logs/%j.err

# ── 推理 + 评测 (Qasper dev) ──
# 第一步必做 (plan §9.6): 先证明 teacher >> student, 再看 transmem 能否拉近.
# Qasper 无长度梯度 (那是 hotpotqa/MemAgent 的东西), 直接在 dev 上评三种模式.
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

PY=/mnt/petrelfs/leihaodong/Project1/delta-Mem/.venv-eval/bin/python
PROJ=/mnt/petrelfs/leihaodong/Project4
DATA=/mnt/petrelfs/leihaodong/Project4/data/qasper
MODEL_PATH=${MODEL_PATH:-/mnt/petrelfs/leihaodong/models/Qwen3-4B-Instruct-2507}
cd $PROJ

EVAL_FILE=${EVAL_FILE:-$DATA/qasper_dev.json}
CKPT=${CKPT:-$PROJ/checkpoints/offpolicy_forward_kl/latest.pt}
N=${N:-4}
MAXS=${MAXS:-128}
ATTN=${ATTN:-sdpa}     # flash_attention_2 在本 venv import 失败, 默认 sdpa

echo "############ eval on $(basename $EVAL_FILE) (max_samples=$MAXS) ############"
# sanity check: 教师(上界, 看 evidence) 与 学生(基线, 看全文)
$PY -m transmem.evaluate --eval_file $EVAL_FILE --data_format qasper --model_path $MODEL_PATH \
    --mode teacher  --N $N --max_samples $MAXS --attn_impl $ATTN
$PY -m transmem.evaluate --eval_file $EVAL_FILE --data_format qasper --model_path $MODEL_PATH \
    --mode student  --N $N --max_samples $MAXS --attn_impl $ATTN
# 本方法 (需已有 ckpt)
[ -f "$CKPT" ] && $PY -m transmem.evaluate --eval_file $EVAL_FILE --data_format qasper \
    --model_path $MODEL_PATH --mode transmem --ckpt $CKPT --N $N --max_samples $MAXS --attn_impl $ATTN \
    || echo "(跳过 transmem: 缺 ckpt $CKPT)"

echo "✅ 评测完成"
