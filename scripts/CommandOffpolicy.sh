单元测试
STEP=2 SaveInterval=1 LogInterval=1 sbatch --quotatype=spot scripts/run_offpolicy.sh
正式
sbatch scripts/run_offpolicy.sh
断点重续
EpochNum=40 sbatch scripts/run_offpolicy-continue.sh

GPUS=1 EpochNum=40 RESUME=1 SaveInterval=5000 Val_interval=5000 \
  sbatch --gres=gpu:1 --quotatype=spot scripts/run_offpolicy.sh

GPUS=1 EpochNum=40 RESUME=1 SaveInterval=5000 Val_interval=5000 \
  sbatch --gres=gpu:1 --quotatype=reserved scripts/run_offpolicy.sh
  
EpochNum=40 RESUME=1 sbatch scripts/run_offpolicy.sh