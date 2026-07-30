#!/bin/bash
#SBATCH -J e09_v5tb4d4
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/%j_v5tb4d4.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/%j_v5tb4d4.err

# Qwen3-4B / HotpotQA-agent / D=4 dynamic-gate transmem-before ablation.
# This copies the existing scratch-joint D=4 recipe and changes only the
# TransMem feature source.  Checkpoints stage on node-local NVMe and publish
# directly to an isolated S3 prefix.
set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
ENDPOINT=http://d-ceph-ssd-inside.pjlab.org.cn
RUN_NAME=v5_gate_qwen3_4b_hqa_d4_transmem_before

export INIT_SCHEME=scratch_joint
export GPUS=8
export S=36
export D=4
export ACCUM=4
export BASE_LR=1e-4
export GATE_LR=1e-4
export JOINT_STEPS=1250
export SEED=20260714
export TRANSMEM_BEFORE=1
export OUTPUT_DIR=$PROJ/checkpoints/$RUN_NAME
export S3_CKPT=s3://datafrontier/leihaodong/Project4/checkpoints/$RUN_NAME

NVME_CKPT_DIR=/nvme/leihaodong/Project4/checkpoints/${RUN_NAME}_${SLURM_JOB_ID}
export EXTRA="--transmem_before --save_nvme_s3 --nvme_checkpoint_dir $NVME_CKPT_DIR --s3_checkpoint_uri $S3_CKPT --s3_endpoint_url $ENDPOINT ${EXTRA:-}"

exec "$PROJ/scripts/version4/dynamic_gate/run_layered_inloop.sh"
