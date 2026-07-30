#!/bin/bash
#SBATCH -J e09_up_pool128
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/data/logs/layered/up_%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/data/logs/layered/up_%j.err

# pool128 (19G, N 消融训练特征, 实验已完结) 归档 S3 后删本地 — 给 D=8 训练腾配额.
# 字节级核对后才删; 不一致则保留并以非零码退出 (阻断 afterok 的 D=8).
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED
export AWS_RESPONSE_CHECKSUM_VALIDATION=WHEN_REQUIRED
EP=http://d-ceph-ssd-inside.pjlab.org.cn
LOCAL=/mnt/petrelfs/leihaodong/Project4/data/hotpotqa_data/Qwen3-4B-Instruct-2507/pool128
S3=s3://datafrontier/leihaodong/Project4/data/hotpotqa_pool128

aws s3 sync "$LOCAL" "$S3" --endpoint-url $EP --only-show-errors

# 逐文件求和 (du -sb 会把目录 inode 大小算进去, 与 S3 对象总和差出目录数×块大小)
LB=$(find "$LOCAL" -type f -printf '%s\n' | awk '{s+=$1}END{print s}')
RB=$(aws s3 ls "$S3/" --recursive --summarize --endpoint-url $EP | awk '/Total Size/{print $3}')
LC=$(find "$LOCAL" -type f | wc -l)
RC=$(aws s3 ls "$S3/" --recursive --summarize --endpoint-url $EP | awk '/Total Objects/{print $3}')
echo "local ${LB}B/${LC}f  s3 ${RB}B/${RC}f"
if [ "$LB" = "$RB" ] && [ "$LC" = "$RC" ]; then
  rm -rf "$LOCAL"
  echo "[$(date '+%F %T')] pool128 -> $S3 (verified ${RB}B/${RC}f), 本地已删" \
    >> /mnt/petrelfs/leihaodong/Project4/data/hotpotqa_data/Qwen3-4B-Instruct-2507/upload.txt
  echo "✅ pool128 归档+清理完成"
else
  echo "❌ 校验不一致, 保留本地"; exit 1
fi
