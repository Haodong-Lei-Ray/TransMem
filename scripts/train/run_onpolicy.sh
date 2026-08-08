#!/bin/bash
#SBATCH -J transmem_onpolicy
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/logs/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/logs/%j.err

# ── Stage 1 (on-policy / OPD): 学生在线 rollout (TransMem 在环) + 教师对齐 ──
# 需加载冻结 LLM (rollout + 教师 forward), 单卡需放得下 4B (bf16 ~8GB + 长文 KV).
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

PY=/mnt/petrelfs/leihaodong/Project1/delta-Mem/.venv-eval/bin/python
PROJ=/mnt/petrelfs/leihaodong/Project4
DATA=/mnt/petrelfs/leihaodong/Project4/data/qasper
MODEL_PATH=${MODEL_PATH:-/mnt/petrelfs/leihaodong/models/Qwen3-4B-Instruct-2507}
cd $PROJ

DIV=${DIV:-jsd}        # on-policy 蒸馏建议 jsd / reverse_kl
N=${N:-4}
ATTN=${ATTN:-sdpa}     # flash_attention_2 在本 venv import 失败, 默认 sdpa

$PY -m transmem.train_onpolicy \
  --data_path $DATA/qasper_train.json --data_format qasper \
  --model_path $MODEL_PATH --config $PROJ/transmem/config.json \
  --output_dir $PROJ/checkpoints/onpolicy_${DIV} \
  --divergence $DIV --temperature 1.0 --N $N \
  --max_answer_tokens 50 --sample --rollout_temperature 1.0 \
  --accum_steps 8 --lr 1e-4 --warmup_steps 200 --max_steps 5000 \
  --dtype bfloat16 --attn_impl $ATTN \
  --log_interval 20 --save_interval 1000

echo "✅ on-policy (OPD) 训练完成"
