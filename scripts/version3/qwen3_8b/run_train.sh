#!/bin/bash
#SBATCH -J e09_v38_train
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/logs/qwen3_8b/%j.err
set -euo pipefail

PROJ=/mnt/petrelfs/leihaodong/Project4
EXP=${EXP:?Set EXP=exp1 or exp2}
NMEM=${NMEM:?Set NMEM=4 or 8}
[[ "$NMEM" == 4 || "$NMEM" == 8 ]] || { echo "NMEM must be 4 or 8" >&2; exit 2; }

LME=$PROJ/data/longmemeval_data/Qwen3-8B-pool-n4-n8
HQA=$PROJ/data/hotpotqa_data/Qwen3-8B-pool-n4-n8
export GPUS=8
export TAG=qwen3_8b_${EXP}_n${NMEM}
export RESUME=${RESUME:-}

case "$EXP" in
  exp1)
    export CONFIG=$PROJ/transmem/configs_8b/config_d4_n${NMEM}.json
    export TRAIN_DIRS=$LME/stage0_train_short200,$HQA/stage0_train_short200_pool
    export VAL_DIRS=$LME/stage0_dev_short200,$HQA/stage0_dev_short200_pool
    export OUTPUT_DIR=$PROJ/checkpoints/offpolicy_v3_qwen3_8b_exp1_mix_d4_e60_n${NMEM}_forward_kl
    export EpochNum=60
    export Val_interval=250
    export SaveInterval=250
    ;;
  exp2)
    export CONFIG=$PROJ/transmem/configs_8b/config_d2_n${NMEM}.json
    export TRAIN_DIRS=$LME/stage0_train_short200
    export VAL_DIRS=$LME/stage0_dev_short200
    export OUTPUT_DIR=$PROJ/checkpoints/offpolicy_v3_qwen3_8b_exp2_lme_d2_e1000_n${NMEM}_forward_kl
    export EpochNum=1000
    export Val_interval=25
    export SaveInterval=250
    ;;
  *)
    echo "EXP must be exp1 or exp2" >&2
    exit 2
    ;;
esac

exec bash "$PROJ/scripts/train/run_offpolicy.sh"
