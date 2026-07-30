#!/bin/bash
#SBATCH -J e09_gdiag_gd4
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_gate_diag.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_locomo/%j_gate_diag.err

# gate OOD 诊断: 4B gate-D4 best.pt 在域内 (hqa dev128) vs 跨域 (LoCoMo 2/8 分片,
# ~385 题) 的逐 token gate 分布对比 (mean/std/分位数/饱和率 frac_lt_025/frac_gt_175),
# 判断 gate_proj 是否在 OOD 输入上饱和/漂移 → 决定要不要加 gate 专属损失.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export all_proxy= ALL_PROXY=
export NLTK_DATA=/mnt/petrelfs/leihaodong/nltk_data
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

PROJ=/mnt/petrelfs/leihaodong/Project4
UV=/mnt/petrelfs/leihaodong/anaconda3/bin/uv
VENV=$PROJ/.venv-transmem
PY=("$UV" run --python "$VENV/bin/python" python)
ENDPOINT=http://d-ceph-ssd-inside.pjlab.org.cn
DATA_FILE=/mnt/petrelfs/leihaodong/OldProjectMaintain/locomo/data/locomo10.json
DEV_FILE=$PROJ/data/hotpotqa-benchmark/hotpotqa-agentmem/hotpotqa_dev.parquet
S3_CKPT_REL=leihaodong/Project4/checkpoints/v4_gate_layered_scratch_joint_s36_d4/best.pt
OUT_ROOT=$PROJ/eval_results/gate_diag_v4gd4
mkdir -p "$OUT_ROOT" /mnt/petrelfs/leihaodong/s3mount_logs
cd "$PROJ"

MOUNT_POINT=/mnt/petrelfs/leihaodong/tmp/s3_gdiag_${SLURM_JOB_ID}
CACHE_DIR=/nvme/leihaodong/s3cache_gdiag_${SLURM_JOB_ID}
LOCAL_CKPT_DIR=/nvme/leihaodong/gdiag_ckpt_${SLURM_JOB_ID}
fusermount -u "$MOUNT_POINT" 2>/dev/null || true
rm -rf "$MOUNT_POINT" "$CACHE_DIR" 2>/dev/null || true
mkdir -p "$MOUNT_POINT" "$CACHE_DIR"
/mnt/petrelfs/leihaodong/app/s3mount datafrontier "$MOUNT_POINT" \
  --cache "$CACHE_DIR" --endpoint-url "$ENDPOINT" --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &
S3PID=$!
PID_L1=""
cleanup() {
  [[ -n "$PID_L1" ]] && kill "$PID_L1" 2>/dev/null || true
  fusermount -u "$MOUNT_POINT" 2>/dev/null || umount "$MOUNT_POINT" 2>/dev/null || true
  kill "$S3PID" 2>/dev/null || true
  rm -rf "$MOUNT_POINT" "$CACHE_DIR" "$LOCAL_CKPT_DIR" || true
}
trap cleanup EXIT
sleep 20

MODEL_PATH=$MOUNT_POINT/leihaodong/Qwen/Qwen3-4B-Instruct-2507
[[ -f "$MODEL_PATH/config.json" ]] || { echo "FATAL: 模型不可见 $MODEL_PATH" >&2; exit 1; }
SOURCE_CKPT=$MOUNT_POINT/$S3_CKPT_REL
[[ -f "$SOURCE_CKPT" ]] || { echo "FATAL: ckpt 不可见 $SOURCE_CKPT" >&2; exit 1; }
mkdir -p "$LOCAL_CKPT_DIR"
CKPT=$LOCAL_CKPT_DIR/best.pt
cp "$SOURCE_CKPT" "$CKPT"
echo "gate diag: ckpt=$S3_CKPT_REL out=$OUT_ROOT"

# GPU1: LoCoMo shard 1/8 (后台立即起)
# 注意: 不要覆盖 CUDA_VISIBLE_DEVICES —— SLURM 已把它设成本作业分到的物理卡,
# 硬编码 0/1 会闯进同节点其他作业的卡 (10264914/10265545 两次 OOM 事故的根因).
# 用 --device cuda:N 的逻辑编号即可.
"${PY[@]}" scripts/eval/eval_locomo.py \
  --data_file "$DATA_FILE" --model_path "$MODEL_PATH" \
  --mode transmem --ckpt "$CKPT" --N 4 --max_answer_tokens 50 \
  --categories 1 2 3 4 --attn_impl sdpa --print_examples 1 --device cuda:1 \
  --num_shards 8 --shard_index 1 \
  --output_json "$OUT_ROOT/locomo_shard1.json" \
  --gate_diagnostics "$OUT_ROOT/gate_diag_locomo_s1.json" \
  >"$OUT_ROOT/locomo_shard1.log" 2>&1 &
PID_L1=$!

# GPU0: 先域内 hqa dev128 诊断
"${PY[@]}" -m transmem.evaluate \
  --eval_file "$DEV_FILE" --data_format hotpotqa-agentmem \
  --model_path "$MODEL_PATH" --mode transmem --ckpt "$CKPT" \
  --N 4 --max_answer_tokens 50 --max_samples 128 \
  --attn_impl sdpa --print_examples 3 --device cuda:0 \
  --gate_diagnostics "$OUT_ROOT/gate_diag_hqa_dev128.json" \
  >"$OUT_ROOT/hqa_dev128.log" 2>&1
echo "hqa dev128 域内诊断完成"

# GPU0: 再 LoCoMo shard 0/8
"${PY[@]}" scripts/eval/eval_locomo.py \
  --data_file "$DATA_FILE" --model_path "$MODEL_PATH" \
  --mode transmem --ckpt "$CKPT" --N 4 --max_answer_tokens 50 \
  --categories 1 2 3 4 --attn_impl sdpa --print_examples 1 --device cuda:0 \
  --num_shards 8 --shard_index 0 \
  --output_json "$OUT_ROOT/locomo_shard0.json" \
  --gate_diagnostics "$OUT_ROOT/gate_diag_locomo_s0.json" \
  >"$OUT_ROOT/locomo_shard0.log" 2>&1
echo "locomo shard0 诊断完成"

wait "$PID_L1"
PID_L1=""
echo "locomo shard1 诊断完成"
echo "✅ gate OOD 诊断完成: $OUT_ROOT"
