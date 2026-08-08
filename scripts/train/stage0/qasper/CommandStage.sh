Qwen/Qwen3-8B
MAXN=16 ModelName=Qwen/Qwen3-8B sbatch --quotatype=spot run_stage0_think1024.sh
MAXN=16 MAX_ANS=200 ModelName=Qwen/Qwen3-8B sbatch --quotatype=spot run_stage0_short.sh


ModelName=Qwen/Qwen3-8B sbatch --quotatype=spot run_stage0_think1024.sh
MAX_ANS=200 ModelName=Qwen/Qwen3-8B sbatch --quotatype=spot run_stage0_short.sh