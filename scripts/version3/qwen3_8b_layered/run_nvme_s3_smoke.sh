#!/bin/bash
#SBATCH -J e09_nvmes3_t
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH -t 00:10:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b_layered/%j_nvme_s3_smoke.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b_layered/%j_nvme_s3_smoke.err

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
export AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED
export AWS_RESPONSE_CHECKSUM_VALIDATION=WHEN_REQUIRED

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
ENDPOINT=http://d-ceph-ssd-inside.pjlab.org.cn
JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
NVME_DIR=/nvme/leihaodong/Project4/checkpoints/_smoke_nvme_s3_${JOB_ID}
S3_URI=s3://datafrontier/leihaodong/Project4/checkpoints/_smoke_nvme_s3/${JOB_ID}

cleanup() {
  rm -rf "$NVME_DIR"
  aws s3 rm "$S3_URI" --recursive --only-show-errors \
    --endpoint-url "$ENDPOINT" || true
}
trap cleanup EXIT

cd "$PROJ"
mkdir -p "$PROJ/logs/qwen3_8b_layered" "$NVME_DIR"
"$UV" run --python "$VENV/bin/python" python -m \
  scripts.version3.qwen3_8b_layered.smoke_nvme_s3 \
  --nvme-dir "$NVME_DIR" \
  --s3-uri "$S3_URI" \
  --endpoint-url "$ENDPOINT"
