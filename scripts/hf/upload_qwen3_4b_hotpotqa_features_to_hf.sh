#!/bin/bash
#SBATCH -J e09_hf_hqa4b
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH -t 48:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/hf_hotpotqa_upload/%j_qwen3_4b.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/hf_hotpotqa_upload/%j_qwen3_4b.err

set -euo pipefail

HF=/mnt/petrelfs/leihaodong/anaconda3/envs/claude/bin/hf
S3MOUNT=/mnt/petrelfs/leihaodong/app/s3mount
ENDPOINT=http://d-ceph-ssd-inside.pjlab.org.cn
HF_NAMESPACE=Rayleihaodong
PROJ=/mnt/petrelfs/leihaodong/Project4
STANDARD_SOURCE=$PROJ/data/hotpotqa_data/Qwen3-4B-Instruct-2507
LAYERED_REL=leihaodong/Project4/data/hotpotqa_layered/Qwen3-4B-Instruct-2507
JOB_TAG=${SLURM_JOB_ID:-manual_$$}
MOUNT_POINT=/nvme/leihaodong/s3_hf_hotpotqa4b_${JOB_TAG}
CACHE_DIR=/nvme/leihaodong/s3cache_hf_hotpotqa4b_${JOB_TAG}
STAGE_ROOT=/nvme/leihaodong/hf_hotpotqa4b_${JOB_TAG}
LOG_DIR=$PROJ/logs/hf_hotpotqa_upload
S3_LOG_DIR=/mnt/petrelfs/leihaodong/s3mount_logs

mkdir -p "$MOUNT_POINT" "$CACHE_DIR" "$STAGE_ROOT" "$LOG_DIR" "$S3_LOG_DIR"

saved_http_proxy=${http_proxy:-}
saved_https_proxy=${https_proxy:-}
saved_HTTP_PROXY=${HTTP_PROXY:-}
saved_HTTPS_PROXY=${HTTPS_PROXY:-}
saved_all_proxy=${all_proxy:-}
saved_ALL_PROXY=${ALL_PROXY:-}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

mount_pid=
cleanup() {
  if mountpoint -q "$MOUNT_POINT"; then
    fusermount -u "$MOUNT_POINT" 2>/dev/null \
      || umount "$MOUNT_POINT" 2>/dev/null \
      || echo "WARNING: could not unmount $MOUNT_POINT" >&2
  fi
  [[ -z "$mount_pid" ]] || kill "$mount_pid" 2>/dev/null || true
  [[ -z "$mount_pid" ]] || wait "$mount_pid" 2>/dev/null || true
  if ! mountpoint -q "$MOUNT_POINT"; then
    rmdir "$MOUNT_POINT" 2>/dev/null || true
  else
    echo "WARNING: mount is still active; preserving $MOUNT_POINT" >&2
  fi
  [[ "$CACHE_DIR" == /nvme/leihaodong/s3cache_hf_hotpotqa4b_* ]] && rm -rf "$CACHE_DIR"
  [[ "$STAGE_ROOT" == /nvme/leihaodong/hf_hotpotqa4b_* ]] && rm -rf "$STAGE_ROOT"
}
trap cleanup EXIT

"$S3MOUNT" datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" \
  --endpoint-url "$ENDPOINT" \
  --force-path-style \
  --log-directory "$S3_LOG_DIR" &
mount_pid=$!

LAYERED_SOURCE=$MOUNT_POINT/$LAYERED_REL
for _ in $(seq 1 60); do
  [[ -f "$LAYERED_SOURCE/stage0_train_short200_layered8/meta.json" ]] && break
  sleep 2
done
[[ -f "$LAYERED_SOURCE/stage0_train_short200_layered8/meta.json" ]] || {
  echo "FATAL: layered8 data is not visible through the S3 mount" >&2
  exit 1
}

# The s3mount child keeps its proxy-free environment. Restore the proxy in the
# upload shell for Hugging Face.
[[ -z "$saved_http_proxy" ]] || export http_proxy=$saved_http_proxy
[[ -z "$saved_https_proxy" ]] || export https_proxy=$saved_https_proxy
[[ -z "$saved_HTTP_PROXY" ]] || export HTTP_PROXY=$saved_HTTP_PROXY
[[ -z "$saved_HTTPS_PROXY" ]] || export HTTPS_PROXY=$saved_HTTPS_PROXY
[[ -z "$saved_all_proxy" ]] || export all_proxy=$saved_all_proxy
[[ -z "$saved_ALL_PROXY" ]] || export ALL_PROXY=$saved_ALL_PROXY

"$HF" auth whoami

upload_dataset() {
  local source_root=$1
  local train_name=$2
  local dev_name=$3
  local repo_name=$4
  local expected_train=$5
  local expected_dev=$6
  local repo_id=$HF_NAMESPACE/$repo_name
  local stage_dir=$STAGE_ROOT/$repo_name

  echo "=== $source_root -> $repo_id ==="
  for split in "$train_name" "$dev_name"; do
    [[ -f "$source_root/$split/meta.json" ]] || {
      echo "FATAL: missing $source_root/$split/meta.json" >&2
      return 1
    }
  done

  rm -rf "$stage_dir"
  mkdir -p "$stage_dir"
  cp -r "$source_root/$train_name" "$stage_dir/"
  cp -r "$source_root/$dev_name" "$stage_dir/"

  python - "$stage_dir" "$train_name" "$dev_name" \
    "$expected_train" "$expected_dev" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {sys.argv[2]: int(sys.argv[4]), sys.argv[3]: int(sys.argv[5])}
for split, count in expected.items():
    meta = json.loads((root / split / "meta.json").read_text())
    manifest_count = len(meta["samples"])
    file_count = sum(1 for _ in (root / split).glob("shard_*/sample_*.pt"))
    if manifest_count != count or file_count != count:
        raise SystemExit(
            f"{split}: expected={count}, manifest={manifest_count}, files={file_count}")
print(f"STAGE_OK train={expected[sys.argv[2]]} dev={expected[sys.argv[3]]}")
PY

  "$HF" upload-large-folder "$repo_id" "$stage_dir" \
    --repo-type dataset \
    --no-private \
    --num-workers 8 \
    --exclude ".cache/**" \
    --no-bars
  echo "UPLOAD_OK https://huggingface.co/datasets/$repo_id"
  rm -rf "$stage_dir"
}

upload_dataset \
  "$STANDARD_SOURCE" \
  stage0_train_short200 \
  stage0_dev_short200 \
  Transmem_ecsd_qwen3_4b_hotpotqa_n4 \
  31574 120

upload_dataset \
  "$LAYERED_SOURCE" \
  stage0_train_short200_layered8 \
  stage0_dev_short200_layered8 \
  Transmem_ecsd_qwen3_4b_hotpotqa_layered8_legacy \
  31574 120

echo "ALL_UPLOADS_OK count=2"
