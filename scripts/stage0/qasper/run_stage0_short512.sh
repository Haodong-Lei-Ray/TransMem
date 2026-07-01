#!/bin/bash
#SBATCH -J e09_transmem_stage0
#SBATCH -p DataFrontier_Explore
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved
#SBATCH -t 24:00:00
#SBATCH -o /mnt/petrelfs/leihaodong/Project4/data/logs/qasper/%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/Project4/data/logs/qasper/%j.err

# ── Stage 0: 离线特征抽取 (冻结 LLM forward -> HM_stu/HQ_stu/HQ_tea + lm_head) ──
# 数据集: Qasper (每条=一个有 evidence 的 QA; C_S=evidence 直供, C_L=全文, Q=question).
# 数据需先由 data/build_qasper_json.py 生成 qasper_{train,dev}.json.
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # 内网/本地模型, 关代理

PY=/mnt/petrelfs/leihaodong/Project1/delta-Mem/.venv-eval/bin/python
PROJ=/mnt/petrelfs/leihaodong/Project4
DATA=/mnt/petrelfs/leihaodong/Project4/data/qasper
MODEL_PATH=${MODEL_PATH:-/mnt/petrelfs/leihaodong/models/Qwen3-4B-Instruct-2507}

N=${N:-4}
MAX_ANS=${MAX_ANS:-512}
ATTN=${ATTN:-sdpa}          # flash_attention_2 在本 venv import 失败, 默认 sdpa
MAXN=${MAXN:-}              # 可选: 只抽前 MAXN 条 (先小跑); 空=全量

cd $PROJ

# 训练集 (2240 QA)
$PY -m transmem.extract_features \
  --data_path $DATA/qasper_train.json --data_format qasper \
  --model_path $MODEL_PATH \
  --output_dir $PROJ/data/qasper_data/stage0_train_short512 \
  --N $N --max_answer_tokens $MAX_ANS \
  --attn_impl $ATTN --save_dtype bfloat16 ${MAXN:+--max_samples $MAXN}

# 验证集 (927 QA)
$PY -m transmem.extract_features \
  --data_path $DATA/qasper_dev.json --data_format qasper \
  --model_path $MODEL_PATH \
  --output_dir $PROJ/data/qasper_data/stage0_dev_short512 \
  --N $N --max_answer_tokens $MAX_ANS \
  --attn_impl $ATTN --save_dtype bfloat16 ${MAXN:+--max_samples $MAXN}

echo "✅ Stage 0 完成: $PROJ/data/stage0_train , stage0_dev"
