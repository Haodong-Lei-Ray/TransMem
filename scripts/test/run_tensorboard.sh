#!/bin/bash
# 查看 TransMem 训练 loss 曲线 (TensorBoard). 直接在管理/登录节点前台跑, 很轻量, 不需要 sbatch/GPU.
#
# 用法:
#   bash scripts/test/run_tensorboard.sh                                          # 看 checkpoints/ 下所有实验 (多 run 对比)
#   bash scripts/test/run_tensorboard.sh checkpoints/offpolicy_short128_forward_kl # 只看一个实验
#   PORT=6007 bash scripts/test/run_tensorboard.sh                                 # 换端口
#
# 本机浏览器访问 (先在本机开一个终端):
#   ssh -L 6006:localhost:6006 leihaodong@<登录节点地址>
#   然后浏览器打开 http://localhost:6006

set -e
PROJ=/mnt/petrelfs/leihaodong/Project4
PY=$PROJ/.venv-transmem/bin/python
cd $PROJ

LOGDIR=${1:-$PROJ/checkpoints}
PORT=${PORT:-6006}

echo "TensorBoard logdir: $LOGDIR (port $PORT)"
echo "本地访问: ssh -L ${PORT}:localhost:${PORT} $(whoami)@<登录节点地址>, 然后打开 http://localhost:${PORT}"
exec $PY -m tensorboard.main --logdir "$LOGDIR" --port "$PORT" --bind_all
