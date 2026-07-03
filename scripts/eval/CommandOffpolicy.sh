# 冒烟 (20 题, 先确认跑通)
MAXQ=20 sbatch --quotatype=spot scripts/eval/Qwen3-4B-Instruct-2507/run_eval_locomo.sh

# 全量 (1540 题 × 3 模式; 不要 student 就 MODES="teacher transmem")
MODES=student OUT_ROOT=/mnt/petrelfs/leihaodong/Project4/eval_outputs/locomo_offpolicy_short128_forward_kl_student sbatch scripts/eval/Qwen3-4B-Instruct-2507/run_eval_locomo.sh
MODES=transmem OUT_ROOT=/mnt/petrelfs/leihaodong/Project4/eval_outputs/locomo_offpolicy_short128_forward_kl_transmem sbatch scripts/eval/Qwen3-4B-Instruct-2507/run_eval_locomo.sh