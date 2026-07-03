#!/bin/bash
#SBATCH -J e09_offpolicy
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/run_offpolicy/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/run_offpolicy/%j.err

# ── Stage 1 (off-policy): 训练 TransMem, 穿冻结 lm_head 逐位置蒸馏 ──
# 只用 Stage0 特征 + lm_head, 不加载 LLM, 很轻 (单卡足够).
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"   # uv run 方式跑 transmem 环境
cd $PROJ

# Stage0 特征根目录 (含 meta.json + lm_head.pt + shard_*); TAG 切 short128/think1024
DATA_ROOT=${DATA_ROOT:-$PROJ/data/qasper_data/Qwen3-4B-Instruct-2507}
TAG=${TAG:-short128}

# 损失解耦: divergence ∈ {forward_kl, reverse_kl, jsd}; reg_weight>0 开表征回归热身
DIV=${DIV:-forward_kl}
TEMP=${TEMP:-1.0}
REG=${REG:-0.0}
STEP=${STEP:-}             # 可选: 硬截 backward 步数 (冒烟用); 空=按 epochs 跑全量
SaveInterval=${SaveInterval:-5000}             # 可选: 硬截 backward 步数 (冒烟用); 空=按 epochs 跑全量
LogInterval=${LogInterval:-50}
EpochNum=${EpochNum:-30}

$PY -m transmem.train_offpolicy \
  --data_dir $DATA_ROOT/stage0_train_${TAG} \
  --val_data_dir $DATA_ROOT/stage0_dev_${TAG} \
  --config $PROJ/transmem/config.json \
  --output_dir $PROJ/checkpoints/offpolicy_${TAG}_${DIV} \
  --divergence $DIV --temperature $TEMP --reg_weight $REG \
  --batch_size 128 --lr 1e-4 --epochs $EpochNum \
  ${STEP:+--max_steps $STEP} \
  --warmup_steps 500 --grad_clip 1.0 \
  --dtype float32 --num_workers 4 \
  --log_interval $LogInterval --val_interval 1000 --save_interval $SaveInterval \
  --resume $PROJ/checkpoints/offpolicy_${TAG}_${DIV}/latest.pt

echo "✅ off-policy 训练完成"
