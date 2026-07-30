#!/bin/bash
#SBATCH -J e09_hp_l318
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --quotatype=reserved
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/eval_hotpot_official/%j_llama31_8b.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/eval_hotpot_official/%j_llama31_8b.err

set -euo pipefail
MODE=${MODE:-transmem}
case "$MODE" in
  student) SUFFIX=student ;;
  transmem) SUFFIX=transmem ;;
  *) echo "MODE must be student or transmem, got $MODE" >&2; exit 2 ;;
esac

export MODE
export GPUS=${GPUS:-${SLURM_GPUS_ON_NODE:-4}}
export WORKERS_PER_GPU=1
export MODEL_REL=leihaodong/meta-llama/Llama-3.1-8B-Instruct
export CKPT_REL=leihaodong/Project4/checkpoints/v5_gate_llama31_8b_scratch_joint_d4/best.pt
export OUTPUT_DIR=/mnt/petrelfs/leihaodong/Project4/eval_results/hotpot_official_llama31_8b_gate_d4_${SUFFIX}
exec /mnt/petrelfs/leihaodong/Project4/scripts/eval/run_hotpot_official_parallel.sh
