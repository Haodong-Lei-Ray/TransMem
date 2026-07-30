#!/bin/bash
#SBATCH -J e09_archive_v3
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/archive_obsolete_v3_%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/archive_obsolete_v3_%j.err

# Archive abandoned v3 checkpoint/stage0 directories, verify exact object
# count + byte sum, and only then remove the local copy.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
export AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED
export AWS_RESPONSE_CHECKSUM_VALIDATION=WHEN_REQUIRED

PROJ=/mnt/petrelfs/leihaodong/Project4
UPLOAD=$PROJ/checkpoints/upload_ckpt.sh
ENDPOINT=http://d-ceph-ssd-inside.pjlab.org.cn
ARCHIVE=s3://datafrontier/leihaodong/Project4/archive/obsolete_20260714
cd "$PROJ"

archive_one() {
  local src=$1
  local dst=$2
  local name remote existing local_count local_bytes remote_count remote_bytes
  if [[ ! -e "$src" ]]; then
    echo "SKIP missing: $src"
    return
  fi
  name=$(basename "$src")
  remote=${dst%/}/$name

  # This archive prefix is intentionally new. Refuse to merge with an old
  # partial upload because that would make count/byte verification ambiguous.
  # Ceph's `aws s3 ls` returns status 1 (without stderr) for a missing prefix.
  # This archive root was checked empty before submission, so empty output is
  # the expected case; upload/verification below still fail closed on errors.
  existing=$(aws s3 ls "$remote/" --recursive --endpoint-url "$ENDPOINT" || true)
  if [[ -n "$existing" ]]; then
    echo "FATAL: archive destination already contains objects: $remote" >&2
    exit 2
  fi

  local_count=$(find "$src" -type f | wc -l)
  local_bytes=$(find "$src" -type f -printf '%s\n' | awk '{s+=$1} END {printf "%.0f", s+0}')
  echo "ARCHIVE $src files=$local_count bytes=$local_bytes -> $remote"
  bash "$UPLOAD" "$src" "$dst"

  read -r remote_count remote_bytes < <(
    aws s3 ls "$remote/" --recursive --endpoint-url "$ENDPOINT" |
      awk '{n+=1; s+=$3} END {printf "%d %.0f\n", n+0, s+0}')
  if [[ "$local_count" != "$remote_count" || "$local_bytes" != "$remote_bytes" ]]; then
    echo "FATAL: verification mismatch for $src: local=$local_count/$local_bytes remote=$remote_count/$remote_bytes" >&2
    exit 3
  fi

  rm -rf -- "$src"
  echo "VERIFIED_AND_REMOVED $src files=$remote_count bytes=$remote_bytes"
}

# Abandoned Qwen3-8B EXP2 checkpoints and excluded offline-layered checkpoint.
archive_one "$PROJ/checkpoints/offpolicy_v3_qwen3_8b_exp2_lme_d2_e1000_n4_forward_kl" "$ARCHIVE/checkpoints"
archive_one "$PROJ/checkpoints/offpolicy_v3_qwen3_8b_exp2_lme_d2_e1000_n8_forward_kl" "$ARCHIVE/checkpoints"
archive_one "$PROJ/checkpoints/v3_p6_layered_d8_forward_kl" "$ARCHIVE/checkpoints"

# Completed older in-loop ablations; current D=8 training does not read them.
archive_one "$PROJ/checkpoints/v3_2_inloop_tf_d2" "$ARCHIVE/checkpoints"
archive_one "$PROJ/checkpoints/v3_2_inloop_tf_d4" "$ARCHIVE/checkpoints"

# Stage0 data from abandoned Qasper/LME/final-hidden diagnostic lines.
# Keep hotpotqa_data/Qwen3-4B-Instruct-2507 (active 4B D=8 training) and
# hotpotqa_data/Qwen3-8B-pool-n4-n8 (new 8B layered continuation) local.
archive_one "$PROJ/data/qasper_data/Qwen3-8B" "$ARCHIVE/stage0/qasper"
archive_one "$PROJ/data/qasper_data/Qwen3-4B-Instruct-2507" "$ARCHIVE/stage0/qasper"
archive_one "$PROJ/data/longmemeval_data/Qwen3-8B-pool-n4-n8" "$ARCHIVE/stage0/longmemeval"
archive_one "$PROJ/data/longmemeval_data/Qwen3-4B-Instruct-2507" "$ARCHIVE/stage0/longmemeval"
archive_one "$PROJ/data/hotpotqa_data/mdsub_4b" "$ARCHIVE/stage0/hotpotqa"
archive_one "$PROJ/data/stage0_dev" "$ARCHIVE/stage0/misc"

echo "ARCHIVE_COMPLETE"
petrelfs-ctl --getquota --uid leihaodong
