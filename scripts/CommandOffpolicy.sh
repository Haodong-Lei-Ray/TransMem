单元测试
STEP=2 SaveInterval=1 LogInterval=1 sbatch --quotatype=spot scripts/run_offpolicy.sh
正式
sbatch scripts/run_offpolicy.sh
断点重续
EpochNum=40 sbatch scripts/run_offpolicy-continue.sh