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
SaveInterval=${SaveInterval:-2000}
LogInterval=${LogInterval:-50}
Val_interval=${LogInterval:-1000}

# v2 = 序列级训练 (transmem正常化修改意见.md): batch 数的是序列条数 (每条 avg ~25 位置),
# 16 序列/批 ≈ 旧版 128 位置/批 的 3 倍位置吞吐; 30 epoch ≈ 4000 steps.
BS=${BS:-16}

OUTPUT_DIR=$PROJ/checkpoints/offpolicy_v2_${TAG}_${DIV}

# 断点重续: RESUME=1 强制续; RESUME=0 强制从头; 默认(空)自动检测 latest.pt.
# checkpoint(latest.pt) 存于 OUTPUT_DIR, --resume 会同时恢复 model+optimizer+step+epoch.
RESUME=${RESUME:-}
RESUME_CKPT=${RESUME_CKPT:-$OUTPUT_DIR/latest.pt}

RESUME_ARG=()
if [ "$RESUME" = "0" ]; then
  echo "从头训练 (RESUME=0)"
elif [ "$RESUME" = "1" ] || { [ -z "$RESUME" ] && [ -f "$RESUME_CKPT" ]; }; then
  if [ -f "$RESUME_CKPT" ]; then
    echo "断点重续: $RESUME_CKPT (含优化器状态)"
    RESUME_ARG=(--resume "$RESUME_CKPT")
  elif [ "$RESUME" = "1" ]; then
    echo "错误: RESUME=1 但找不到 checkpoint: $RESUME_CKPT" >&2
    exit 1
  fi
else
  echo "无 checkpoint, 从头训练"
fi

$PY -m transmem.train_offpolicy \
  --data_dir $DATA_ROOT/stage0_train_${TAG} \
  --val_data_dir $DATA_ROOT/stage0_dev_${TAG} \
  --config $PROJ/transmem/config.json \
  --output_dir $OUTPUT_DIR \
  --divergence $DIV --temperature $TEMP --reg_weight $REG \
  --batch_size $BS --lr 1e-4 --epochs 30 \
  ${STEP:+--max_steps $STEP} \
  --warmup_steps 150 --grad_clip 1.0 \
  --dtype float32 --num_workers 4 \
  --log_interval $LogInterval --val_interval $Val_interval --save_interval $SaveInterval \
  "${RESUME_ARG[@]}"

echo "✅ off-policy 训练完成"
