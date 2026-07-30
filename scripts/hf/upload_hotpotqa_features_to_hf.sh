#!/bin/bash
#SBATCH -J e09_hf_hqa6
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH -t 48:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/hf_hotpotqa_upload/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/hf_hotpotqa_upload/%j.err

set -euo pipefail

HF=/mnt/petrelfs/leihaodong/anaconda3/envs/claude/bin/hf
S3MOUNT=/mnt/petrelfs/leihaodong/app/s3mount
ENDPOINT=http://d-ceph-ssd-inside.pjlab.org.cn
HF_NAMESPACE=Rayleihaodong
JOB_TAG=${SLURM_JOB_ID:-manual_$$}
MOUNT_POINT=/nvme/leihaodong/s3_hf_hotpotqa_${JOB_TAG}
CACHE_DIR=/nvme/leihaodong/s3cache_hf_hotpotqa_${JOB_TAG}
STAGE_ROOT=/nvme/leihaodong/hf_hotpotqa_${JOB_TAG}
LOG_DIR=/mnt/petrelfs/leihaodong/Project4/logs/hf_hotpotqa_upload
S3_LOG_DIR=/mnt/petrelfs/leihaodong/s3mount_logs

SOURCES=(
  "leihaodong/Project4/data/hotpotqa_data/Qwen3-8B-pool-n4-n8"
  "leihaodong/Project4/data/hotpotqa_pool_qwen3_14b_n4_n8"
  "leihaodong/Project4/data/hotpotqa_pool_qwen25_7b_n4_n8"
  "leihaodong/Project4/data/hotpotqa_pool_qwen25_14b_n4_n8"
  "leihaodong/Project4/data/hotpotqa_pool_llama31_8b_n4_n8"
  "leihaodong/Project4/data/hotpotqa_pool_minicpm5_1b_n4_n8"
)
REPOS=(
  "Transmem_ecsd_qwen3_8b_hotpotqa_n4_n8"
  "Transmem_ecsd_qwen3_14b_hotpotqa_n4_n8"
  "Transmem_ecsd_qwen2_5_7b_hotpotqa_n4_n8"
  "Transmem_ecsd_qwen2_5_14b_hotpotqa_n4_n8"
  "Transmem_ecsd_llama3_1_8b_hotpotqa_n4_n8"
  "Transmem_ecsd_minicpm5_1b_hotpotqa_n4_n8"
)

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
  [[ "$CACHE_DIR" == /nvme/leihaodong/s3cache_hf_hotpotqa_* ]] && rm -rf "$CACHE_DIR"
  [[ "$STAGE_ROOT" == /nvme/leihaodong/hf_hotpotqa_* ]] && rm -rf "$STAGE_ROOT"
}
trap cleanup EXIT

"$S3MOUNT" datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" \
  --endpoint-url "$ENDPOINT" \
  --force-path-style \
  --log-directory "$S3_LOG_DIR" &
mount_pid=$!

for _ in $(seq 1 60); do
  [[ -f "$MOUNT_POINT/${SOURCES[0]}/stage0_train_short200_pool/meta.json" ]] && break
  sleep 2
done
[[ -f "$MOUNT_POINT/${SOURCES[0]}/stage0_train_short200_pool/meta.json" ]] || {
  echo "FATAL: DataFrontier mount did not become ready" >&2
  exit 1
}

# s3mount keeps the proxy-free environment inherited above. Restore the proxy
# only in this shell so Hugging Face can use the external network.
[[ -z "$saved_http_proxy" ]] || export http_proxy=$saved_http_proxy
[[ -z "$saved_https_proxy" ]] || export https_proxy=$saved_https_proxy
[[ -z "$saved_HTTP_PROXY" ]] || export HTTP_PROXY=$saved_HTTP_PROXY
[[ -z "$saved_HTTPS_PROXY" ]] || export HTTPS_PROXY=$saved_HTTPS_PROXY
[[ -z "$saved_all_proxy" ]] || export all_proxy=$saved_all_proxy
[[ -z "$saved_ALL_PROXY" ]] || export ALL_PROXY=$saved_ALL_PROXY

"$HF" auth whoami

failures=()
for i in "${!SOURCES[@]}"; do
  source_dir="$MOUNT_POINT/${SOURCES[$i]}"
  repo_id="$HF_NAMESPACE/${REPOS[$i]}"
  stage_dir="$STAGE_ROOT/${REPOS[$i]}"

  echo "=== [$((i + 1))/${#SOURCES[@]}] $source_dir -> $repo_id ==="
  for required in \
    "$source_dir/stage0_train_short200_pool/meta.json" \
    "$source_dir/stage0_dev_short200_pool/meta.json"; do
    if [[ ! -f "$required" ]]; then
      echo "ERROR: missing $required" >&2
      failures+=("$repo_id:missing-source")
      continue 2
    fi
  done

  rm -rf "$stage_dir"
  mkdir -p "$stage_dir"
  if ! cp -r "$source_dir/." "$stage_dir/"; then
    echo "ERROR: staging failed for $repo_id" >&2
    failures+=("$repo_id:staging")
    continue
  fi

  if ! python - "$stage_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {}
for split in ("stage0_train_short200_pool", "stage0_dev_short200_pool"):
    meta = json.loads((root / split / "meta.json").read_text())
    expected[split] = len(meta["samples"])
    actual = sum(1 for _ in (root / split).glob("shard_*/sample_*.pt"))
    if actual != expected[split]:
        raise SystemExit(f"{split}: expected {expected[split]} sample files, got {actual}")
print(f"STAGE_OK train={expected['stage0_train_short200_pool']} "
      f"dev={expected['stage0_dev_short200_pool']}")
PY
  then
    failures+=("$repo_id:verification")
    continue
  fi

  if ! "$HF" upload-large-folder "$repo_id" "$stage_dir" \
      --repo-type dataset \
      --no-private \
      --num-workers 8 \
      --exclude ".cache/**" \
      --no-bars; then
    echo "ERROR: Hugging Face upload failed for $repo_id" >&2
    failures+=("$repo_id:upload")
    continue
  fi

  echo "UPLOAD_OK https://huggingface.co/datasets/$repo_id"
  rm -rf "$stage_dir"
done

if (( ${#failures[@]} )); then
  printf 'FAILED %s\n' "${failures[@]}" >&2
  exit 1
fi

echo "ALL_UPLOADS_OK count=${#REPOS[@]}"
