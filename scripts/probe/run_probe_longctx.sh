#!/bin/bash
#SBATCH -J e09_probe_ctx
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=spot
#SBATCH -t 0:40:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/data/logs/longmemeval/probe_%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/data/logs/longmemeval/probe_%j.err

set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY="$UV run --python $VENV/bin/python python"

mkdir -p /mnt/petrelfs/leihaodong/s3mount_logs
JOB_ID="${SLURM_JOB_ID:-$$}"
MOUNT_POINT="/mnt/petrelfs/leihaodong/tmp/s3_probe_${JOB_ID}"
CACHE_DIR="/nvme/leihaodong/s3cache_probe_${JOB_ID}"
mkdir -p "${MOUNT_POINT}" "${CACHE_DIR}"
/mnt/petrelfs/leihaodong/app/s3mount datafrontier "${MOUNT_POINT}" \
  --cache "${CACHE_DIR}" \
  --endpoint-url http://d-ceph-ssd-inside.pjlab.org.cn \
  --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &
S3PID=$!
sleep 20
cleanup() {
  fusermount -u "${MOUNT_POINT}" 2>/dev/null || true
  kill "${S3PID}" 2>/dev/null || true
  rm -rf "${MOUNT_POINT}" "${CACHE_DIR}" || true
}
trap cleanup EXIT

$PY $PROJ/scripts/probe/probe_longctx_mem.py \
  "${MOUNT_POINT}/leihaodong/Qwen/Qwen3-4B-Instruct-2507"
