#!/bin/bash
#SBATCH -J transmem_offpolicy
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/logs/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/logs/%j.err

# ── Stage 1 (off-policy): 训练 TransMem, 穿冻结 lm_head 逐位置蒸馏 ──
# 只用 Stage0 特征 + lm_head, 不加载 LLM, 很轻 (单卡足够).
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

PY=/mnt/petrelfs/leihaodong/Project1/delta-Mem/.venv-eval/bin/python
PROJ=/mnt/petrelfs/leihaodong/Project4
cd $PROJ

# 损失解耦: divergence ∈ {forward_kl, reverse_kl, jsd}; reg_weight>0 开表征回归热身
DIV=${DIV:-forward_kl}
TEMP=${TEMP:-1.0}
REG=${REG:-0.0}

$PY -m transmem.train_offpolicy \
  --data_dir $PROJ/data/stage0_train \
  --val_data_dir $PROJ/data/stage0_dev \
  --config $PROJ/transmem/config.json \
  --output_dir $PROJ/checkpoints/offpolicy_${DIV} \
  --divergence $DIV --temperature $TEMP --reg_weight $REG \
  --batch_size 128 --lr 1e-4 --epochs 30 \
  --warmup_steps 500 --grad_clip 1.0 \
  --dtype float32 --num_workers 4 \
  --log_interval 50 --val_interval 1000 --save_interval 5000

echo "✅ off-policy 训练完成"
