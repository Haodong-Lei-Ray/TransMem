#!/bin/bash
#SBATCH -J e09_ctx4bl
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH -t 12:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/benchmark/context_scaling_long_%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/benchmark/context_scaling_long_%j.err

set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
DELTA=/mnt/petrelfs/leihaodong/Project1/delta-Mem
PY=$DELTA/.venv-eval/bin/python
ENDPOINT=http://d-ceph-ssd-inside.pjlab.org.cn
OUT_DIR=${OUT_DIR:-$PROJ/eval_results/context_scaling_qwen3_4b_10k100k}
mkdir -p "$OUT_DIR" "$PROJ/logs/benchmark" \
  /mnt/petrelfs/leihaodong/s3mount_logs

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
export PYTHONPATH="$PROJ:$DELTA${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=${CUDA_HOME:-/mnt/petrelfs/share/cuda-12.4}
export LIBRARY_PATH="$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/mnt/petrelfs/leihaodong/.cache/gcc_libs:${LD_LIBRARY_PATH:-}"
export TRITON_CACHE_DIR=/mnt/petrelfs/leihaodong/.cache/triton
export OMP_NUM_THREADS=8

MOUNT_POINT=/mnt/petrelfs/leihaodong/tmp/s3_ctxbench_long_${SLURM_JOB_ID}
CACHE_DIR=/nvme/leihaodong/s3cache_ctxbench_long_${SLURM_JOB_ID}
LOCAL_DIR=/nvme/leihaodong/context_scaling_long_${SLURM_JOB_ID}
fusermount -u "$MOUNT_POINT" 2>/dev/null || true
rm -rf "$MOUNT_POINT" "$CACHE_DIR" "$LOCAL_DIR" 2>/dev/null || true
mkdir -p "$MOUNT_POINT" "$CACHE_DIR" "$LOCAL_DIR"

/mnt/petrelfs/leihaodong/app/s3mount datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" --endpoint-url "$ENDPOINT" --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &
S3PID=$!
cleanup() {
  fusermount -u "$MOUNT_POINT" 2>/dev/null || umount "$MOUNT_POINT" 2>/dev/null || true
  kill "$S3PID" 2>/dev/null || true
  rm -rf "$MOUNT_POINT" "$CACHE_DIR" "$LOCAL_DIR" || true
}
trap cleanup EXIT
sleep 20

MODEL_PATH=$MOUNT_POINT/leihaodong/Qwen/Qwen3-4B-Instruct-2507
DELTA_ADAPTER=$MOUNT_POINT/leihaodong/declare-lab/delta-mem_qwen3_4b-instruct
SOURCE_CKPT=$MOUNT_POINT/leihaodong/Project4/checkpoints/v4_gate_layered_scratch_joint_s36_d4/best.pt
[[ -f "$MODEL_PATH/config.json" ]] || {
  echo "FATAL: base model unavailable: $MODEL_PATH" >&2
  exit 1
}
[[ -f "$DELTA_ADAPTER/delta_mem_adapter.pt" ]] || {
  echo "FATAL: Delta-Mem adapter unavailable: $DELTA_ADAPTER" >&2
  exit 1
}
[[ -f "$SOURCE_CKPT" ]] || {
  echo "FATAL: TransMem checkpoint unavailable: $SOURCE_CKPT" >&2
  exit 1
}
TRANS_CKPT=$LOCAL_DIR/transmem_best.pt
cp "$SOURCE_CKPT" "$TRANS_CKPT"

echo "GPU benchmark protocol: batch=1 BF16 SDPA, contexts 10k..100k, fixed decode=32"
echo "FLOPs scope: memory components only (frozen backbone excluded)"
echo "Model: $MODEL_PATH"
echo "Delta: $DELTA_ADAPTER"
echo "TransMem: $SOURCE_CKPT"
echo "Output: $OUT_DIR"

cd "$PROJ"
"$PY" scripts/benchmark/benchmark_context_scaling.py \
  --model-path "$MODEL_PATH" \
  --delta-adapter-dir "$DELTA_ADAPTER" \
  --transmem-ckpt "$TRANS_CKPT" \
  --output-dir "$OUT_DIR" \
  --context-lengths \
    10000 20000 30000 40000 50000 \
    60000 70000 80000 90000 100000 \
  --decode-tokens 32 \
  --warmup-runs 2 \
  --measure-runs 5 \
  --flops-scope memory_only \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa
