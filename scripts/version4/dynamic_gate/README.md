# Dynamic-gate 实验脚本

本目录只提供脚本，不会自动提交作业。四个 `run_*_legacy/scratch_*.sh`
是 `sbatch` 入口；两个不带 `#SBATCH` 的脚本是共享实现，不能直接在登录节点运行。

## 主实验入口

- `run_layered_legacy_d8.sh`：layered D=8，A1 gate-only 后从 held-out 最优
  `gate_only_best.pt` 进入 A2 joint fine-tune。
- `run_layered_scratch_d8.sh`：layered D=8，TransMem 与 gate 从头联合训练。
- `run_final_legacy_d4.sh`：final-hidden D=4，A1+A2；提交时必须显式传
  `INIT_CHECKPOINT=/path/to/fixed-gate.pt`。
- `run_final_scratch_d4.sh`：final-hidden D=4，从头联合训练。

所有 RESERVED 作业名均以 `e09_` 开头。脚本检测到输出目录中的 `latest.pt` 时会
显式断点续跑；不会上传、删除 checkpoint，也不会自动提交后续评测。

## 可覆盖超参数

通过 `sbatch --export=ALL,KEY=VALUE ...` 覆盖：

- 通用：`OUTPUT_DIR`、`CONFIG`、`GPUS`、`SEED`、`BASE_LR`、`GATE_LR`、
  `PRIOR_WEIGHT`、`PRIOR_STEPS`、`CALIBRATION_STEPS`、`JOINT_STEPS`。
- layered：`S`、`D`、`ACCUM`、`TRAIN_DIR`、`VAL_DIR`、`DATA_PATH`、
  `VAL_DATA_PATH`、`MODEL_PATH`。
- final-hidden：`TRAIN_DIR`、`VAL_DIR`；legacy 方案还必须提供
  `INIT_CHECKPOINT`。

默认 `PRIOR_WEIGHT=0`，便于先验证任务损失能否自行学出非退化 gate。A1 边界会同时
保留 `gate_only.pt`（最后一步）和 `gate_only_best.pt`（held-out 最优），A2 只从后者
启动；若没有验证集候选，才回退到 `gate_only.pt`。A2 会重建 optimizer，避免把 A1
期间的 Adam 状态带入联合训练。

`run_smoke_tests.sh` 仅用于短时测试；它也不会被其他脚本自动提交。
