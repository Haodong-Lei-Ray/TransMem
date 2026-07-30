#!/bin/bash
#SBATCH -J e09_pool_train
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH -t 12:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/run_offpolicy/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/run_offpolicy/%j.err

# ── N 消融训练包装 (双源池): S3 部分 sync 到 /nvme + 本地 pool128 拼接 ──
# 用法: NMEM=64 sbatch -J e09_pool_n64 scripts/run_offpolicy_pool.sh
# 数据: 前 6,935 样本 = S3 512 行全网格池 (桶满前写入, 读不受影响);
#       其余样本    = 本地 pool128 (128 行, {4..128} 网格).
#       两目录都带各自 meta/hm_maps, OffPolicyDataset 按 config.n_mem 切片后拼接.
# 结束: best.pt 尝试归档 S3 (桶满会失败, 容忍), latest/final 一律删 (省配额,
#       best.pt 1.6G 留本地).
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
export AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED

PROJ=/mnt/petrelfs/leihaodong/Project4
EP="--endpoint-url http://d-ceph-ssd-inside.pjlab.org.cn"
NMEM=${NMEM:?用法: NMEM=64 sbatch -J e09_pool_n64 scripts/run_offpolicy_pool.sh}

S3TRAIN=s3://datafrontier/leihaodong/Project4/data/hotpotqa_pool/stage0_train_short200_pool
NVME_A=/nvme/leihaodong/hotpotqa_pool_s3part
LOCAL_B=$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507/pool128

# S3 部分 -> /nvme (节点本地; 桶满只挡写, 读正常). sync 幂等, 同节点作业复用.
mkdir -p $NVME_A
synced=0
for i in 1 2 3; do
  if aws s3 sync $S3TRAIN $NVME_A $EP --only-show-errors; then synced=1; break; fi
  echo "sync 失败, 重试 $i/3"; sleep 30
done
[ $synced = 1 ] || { echo "FATAL: S3 池 sync 失败"; exit 1; }
# 注入分账后的 meta (S3 目录本身无顶层 meta.json — 合并发生在本地目录)
cp $PROJ/data/hotpotqa_data/meta_s3pool_train.json $NVME_A/meta.json
echo "S3 池就绪: $NVME_A ($(du -sh $NVME_A | cut -f1))"

export GPUS=${GPUS:-8}
export CONFIG=$PROJ/transmem/configs_nmem/config_n${NMEM}.json
export TRAIN_DIRS="$NVME_A,$LOCAL_B/stage0_train_short200_pool"
export VAL_DIRS="$LOCAL_B/stage0_dev_short200_pool"
export OUTPUT_DIR=${OUTPUT_DIR:-$PROJ/checkpoints/offpolicy_v2_hotpotqa_pool_n${NMEM}_forward_kl}
export TAG=pool_n${NMEM}
bash $PROJ/scripts/run_offpolicy.sh

# best 尝试归档 (桶满则容忍失败, 留本地); latest/final 删掉腾配额.
S3CK=s3://datafrontier/leihaodong/Project4/checkpoints/$(basename $OUTPUT_DIR)
for f in best.pt result.json; do
  if [ -f "$OUTPUT_DIR/$f" ]; then
    aws s3 cp "$OUTPUT_DIR/$f" "$S3CK/$f" $EP --only-show-errors || echo "⚠️ $f 归档失败 (桶满?), 留本地"
  fi
done
if [ -f "$OUTPUT_DIR/best.pt" ]; then
  rm -f "$OUTPUT_DIR"/latest.pt "$OUTPUT_DIR"/step_*.pt
  echo "已删 latest/final (best.pt 1.6G 留本地)"
fi
echo "✅ n_mem=$NMEM 训练完成"
