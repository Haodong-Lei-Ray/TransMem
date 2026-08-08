#!/bin/bash
#SBATCH -J e09_offpolicy
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/run_offpolicy/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/run_offpolicy/%j.err

# ── Stage 1 (off-policy): 训练 TransMem, 穿冻结 lm_head 逐位置蒸馏 ──
# 只用 Stage0 特征 + lm_head, 不加载 LLM. 默认 torchrun 8 卡 DDP (整机独占,
# 也避开单卡与他人进程共卡被挤 OOM 的问题, 见 10196150/10196617).
# 单卡跑法: GPUS=1 sbatch --gres=gpu:1 scripts/train/run_offpolicy.sh
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"   # uv run 方式跑 transmem 环境
cd $PROJ

GPUS=${GPUS:-8}   # DDP 进程数 (=卡数); 需与 --gres=gpu:N 一致

# Stage0 特征根目录 (含 meta.json + lm_head.pt + shard_*); TAG 切 short128/think1024
DATA_ROOT=${DATA_ROOT:-$PROJ/data/qasper_data/Qwen3-4B-Instruct-2507}
TAG=${TAG:-short128}
CONFIG=${CONFIG:-$PROJ/transmem/config.json}   # 换记忆池大小(n_mem)时传不同 config
# 多域训练: TRAIN_DIRS 逗号分隔多个 stage0 目录 (须同 N/dim), 默认单域 Qasper.
# 例: TRAIN_DIRS="$Q/stage0_train_short128,$H/stage0_train_short200" (Qasper+HotpotQA)
TRAIN_DIRS=${TRAIN_DIRS:-$DATA_ROOT/stage0_train_${TAG}}
VAL_DIRS=${VAL_DIRS:-$DATA_ROOT/stage0_dev_${TAG}}   # val 通常保持单域以对齐基线

# 损失解耦: divergence ∈ {forward_kl, reverse_kl, jsd}; reg_weight>0 开表征回归热身
DIV=${DIV:-forward_kl}
TEMP=${TEMP:-1.0}
REG=${REG:-0.0}
LOSS=${LOSS:-kd}          # kd=蒸馏教师软分布(默认); ce=对 golden token 交叉熵(需 stage0 --trajectory golden 特征)
STEP=${STEP:-}             # 可选: 硬截 backward 步数 (冒烟用); 空=按 epochs 跑全量
# 间隔默认按卡数折算 (多卡时 steps/epoch 除以 GPUS, 不折算就一次中途 val/save 都没有)
SaveInterval=${SaveInterval:-$((2000 / GPUS))}
LogInterval=${LogInterval:-$((50 / GPUS > 0 ? 50 / GPUS : 1))}
Val_interval=${Val_interval:-$((2000 / GPUS))}

# v2 = 序列级训练 (transmem正常化修改意见.md): batch 数的是序列条数 (每条 avg ~25 位置),
# 16 序列/批 ≈ 旧版 128 位置/批 的 3 倍位置吞吐; 30 epoch ≈ 4000 steps.
BS=${BS:-16}

OUTPUT_DIR=${OUTPUT_DIR:-$PROJ/checkpoints/offpolicy_v2_${TAG}_${DIV}}

# 断点重续: RESUME=1 强制续; RESUME=0 强制从头; 默认(空)自动检测 latest.pt.
# checkpoint(latest.pt) 存于 OUTPUT_DIR, --resume 会同时恢复 model+optimizer+step+epoch.
RESUME=${RESUME:-}
RESUME_CKPT=${RESUME_CKPT:-$OUTPUT_DIR/latest.pt}
EpochNum=${EpochNum:-30}

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

# 多卡: torchrun 起 $GPUS 个进程, 每进程一卡; 全局批 = BS x GPUS 序列/step,
# steps/epoch 相应除以 GPUS (总 epoch 数不变). LR/WARMUP 可按线性缩放规则调
# (LR=8e-4 对应 8 卡的 linear scaling), 默认保守不动 LR、只把 warmup 缩到 50.
LR=${LR:-1e-4}
WARMUP=${WARMUP:-50}
if [ "$GPUS" -gt 1 ]; then
  LAUNCH="$UV run --python $VENV/bin/python python -m torch.distributed.run --standalone --nproc_per_node=$GPUS -m transmem.train_offpolicy"
else
  LAUNCH="$PY -m transmem.train_offpolicy"
fi

$LAUNCH \
  --data_dir "$TRAIN_DIRS" \
  --val_data_dir "$VAL_DIRS" \
  --config $CONFIG \
  --output_dir $OUTPUT_DIR \
  --divergence $DIV --temperature $TEMP --reg_weight $REG --loss $LOSS \
  --batch_size $BS --lr $LR --epochs $EpochNum \
  ${STEP:+--max_steps $STEP} \
  --warmup_steps $WARMUP --grad_clip 1.0 \
  --dtype float32 --num_workers 2 \
  --log_interval $LogInterval --val_interval $Val_interval --save_interval $SaveInterval \
  "${RESUME_ARG[@]}"

echo "✅ off-policy 训练完成"
