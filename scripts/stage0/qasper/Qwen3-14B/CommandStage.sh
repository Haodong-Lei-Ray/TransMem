MAXN=12 ModelName=Qwen/Qwen3-14B sbatch --quotatype=spot run_stage0_think1024.sh
MAXN=12 MAX_ANS=200 ModelName=Qwen/Qwen3-14B sbatch --quotatype=spot run_stage0_short.sh