MAXN=12 sbatch --quotatype=spot scripts/stage0/qasper/Qwen3-8B/run_stage0_think1024.sh
MAXN=12 MAX_ANS=200 sbatch --quotatype=spot scripts/stage0/qasper/Qwen3-8B/run_stage0_short.sh