#!/usr/bin/env python3
"""
Stage 1 (off-policy) — 训练 TransMem: 逐位置蒸馏, 穿冻结 lm_head 反传到 TransMem.

数据来源 = Stage0 固定教师轨迹的特征 (off-policy: 轨迹由教师 rollout 给定, 与当前策略无关).

序列语义 (docs/version2/transmem正常化修改意见.md §2.3, 用户确认必做): 每条样本是
一整条有序序列, 不再摊平成 i.i.d. 位置池 — 位置 i 的 query 因果地看到同样本更早的
query, 与 token-by-token 推理一致.
  X             = [HM_stu ; HQ_stu_1..M]          [B, N+M_max, dim]  (批内右 padding)
  MS_1..M       = TransMem(X, return_all_queries) [B, M_max, dim]    (causal 并行前向)
  HQ'_i         = HQ_stu_i + a*MS_i
  按 q_mask 收集有效位 (padding 不进 loss):
  student_logits= lm_head(HQ'[q_mask])            (冻结 lm_head, 可微)
  teacher_logits= lm_head(HQ_tea[q_mask])         (固定软目标)
  L             = DistillLoss(P_tea, P_stu)       (forward_kl / reverse_kl / jsd; +可选 L_reg)

causal mask 下尾部 padding 严格不影响有效位输出 (test_shapes [8]), 故 attention 内
无需 padding mask, 只要 loss 端按 q_mask 过滤. batch_size 现在数的是**序列条数**
(每条贡献 M 个位置), 不再是位置数.

只反传 lm_head 这一个线性层, 不重跑整个 LLM -> 很轻. Stage0 已离线算好特征, 训练全程不碰 LLM.

用法:
  python -m transmem.train_offpolicy \
    --data_dir data/stage0_train --val_data_dir data/stage0_dev \
    --config transmem/config.json --output_dir checkpoints/offpolicy \
    --divergence forward_kl --temperature 1.0 --reg_weight 0.0 \
    --batch_size 16 --lr 1e-4 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transmem import TransMemConfig, TransMem, DistillLoss, FrozenLMHead
from transmem.diagnostics import parse_curve_steps

_DTYPE = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_TRAINING_RECIPE_VERSION = 1
_DEFERRED_RECIPE_FIELDS = {
    "training.resolved_total_steps",
    "training.resolved_schedule_total_steps",
}


def _normalize_path(value) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(Path(str(value).strip()).expanduser().resolve())


def _normalize_data_dirs(value) -> list[str]:
    if value is None:
        return []
    return [
        str(Path(part.strip()).expanduser().resolve())
        for part in str(value).split(",")
        if part.strip()
    ]


def build_training_recipe(
    args,
    config: dict,
    *,
    world_size: int,
    resolved_total_steps: int | None = None,
    resolved_schedule_total_steps: int | None = None,
) -> dict:
    """Build the canonical weight-affecting recipe persisted in checkpoints."""

    data_dirs = _normalize_data_dirs(getattr(args, "data_dir", None))
    val_data_dirs = _normalize_data_dirs(getattr(args, "val_data_dir", None))
    raw_max_steps = getattr(args, "max_steps", None)
    raw_schedule_steps = getattr(args, "schedule_total_steps", None)
    if resolved_total_steps is None and raw_max_steps is not None:
        resolved_total_steps = int(raw_max_steps)
    if resolved_schedule_total_steps is None:
        if raw_schedule_steps is not None:
            resolved_schedule_total_steps = int(raw_schedule_steps)
        elif resolved_total_steps is not None:
            resolved_schedule_total_steps = int(resolved_total_steps)

    lm_head_path = getattr(args, "lm_head_path", None)
    if not lm_head_path and data_dirs:
        lm_head_path = str(Path(data_dirs[0]) / "lm_head.pt")
    curve_steps = sorted({
        int(step) for step in (getattr(args, "curve_steps", None) or [])
    })
    batch_size = int(getattr(args, "batch_size", 0))
    world_size = int(world_size)

    return {
        "schema_version": _TRAINING_RECIPE_VERSION,
        "seed": getattr(args, "seed", None),
        "data": {
            "data_dirs": data_dirs,
            "val_data_dirs": val_data_dirs,
            "lm_head_path": _normalize_path(lm_head_path),
        },
        "model": {
            "config_path": _normalize_path(getattr(args, "config", None)),
            "config": json.loads(json.dumps(config, sort_keys=True)),
            "warm_start_model_path": _normalize_path(
                getattr(args, "model_path", None)),
        },
        "training": {
            "epochs": int(getattr(args, "epochs", 0)),
            "max_steps": None if raw_max_steps is None else int(raw_max_steps),
            "resolved_total_steps": (
                None if resolved_total_steps is None else int(resolved_total_steps)),
            "schedule_total_steps": (
                None if raw_schedule_steps is None else int(raw_schedule_steps)),
            "resolved_schedule_total_steps": (
                None if resolved_schedule_total_steps is None
                else int(resolved_schedule_total_steps)),
            "curve_steps": curve_steps,
            "overfit_batch": bool(getattr(args, "overfit_batch", False)),
        },
        "optimizer": {
            "name": "AdamW",
            "lr": float(getattr(args, "lr", 0.0)),
            "weight_decay": float(getattr(args, "weight_decay", 0.0)),
            "betas": [0.9, 0.999],
            "batch_size_per_rank": batch_size,
            "world_size": world_size,
            "global_batch_size": batch_size * world_size,
            "warmup_steps": int(getattr(args, "warmup_steps", 0)),
            "grad_clip": float(getattr(args, "grad_clip", 0.0)),
            "dtype": str(getattr(args, "dtype", "")),
        },
        "loss": {
            "name": str(getattr(args, "loss", "kd")),
            "divergence": str(getattr(args, "divergence", "forward_kl")),
            "temperature": float(getattr(args, "temperature", 1.0)),
            "reg_weight": float(getattr(args, "reg_weight", 0.0)),
            "jsd_beta": float(getattr(args, "jsd_beta", 0.5)),
        },
    }


def validate_resume_recipe(
    saved_recipe,
    current_recipe: dict,
    *,
    require_recipe: bool,
    allow_unresolved: bool = False,
) -> None:
    """Fail closed on recipe drift while retaining legacy ordinary resume."""

    if saved_recipe is None:
        if require_recipe:
            raise ValueError(
                "resume checkpoint is missing training_recipe; "
                "curve runs require a provenanced latest.pt")
        return
    if not isinstance(saved_recipe, dict):
        raise ValueError("checkpoint training_recipe must be a dict")

    mismatches: list[str] = []

    def compare(saved, current, path: str) -> None:
        if allow_unresolved and path in _DEFERRED_RECIPE_FIELDS and current is None:
            return
        if isinstance(saved, dict) and isinstance(current, dict):
            for key in sorted(set(saved) | set(current)):
                child = f"{path}.{key}" if path else key
                if key not in saved:
                    mismatches.append(f"{child}: missing from checkpoint")
                elif key not in current:
                    mismatches.append(f"{child}: missing from current recipe")
                else:
                    compare(saved[key], current[key], child)
            return
        if saved != current:
            mismatches.append(
                f"{path}: checkpoint={saved!r}, current={current!r}")

    compare(saved_recipe, current_recipe, "")
    if mismatches:
        details = "\n  - ".join(mismatches[:20])
        extra = "" if len(mismatches) <= 20 else (
            f"\n  - ... and {len(mismatches) - 20} more")
        raise ValueError(
            f"resume training_recipe mismatch:\n  - {details}{extra}")


def curve_checkpoint_path(output_dir, step: int) -> Path:
    return Path(output_dir) / f"curve_step_{int(step):07d}.pt"


def save_curve_checkpoint(path: str | Path, *, model, config: dict,
                          global_step: int, epoch: int, seed: int | None,
                          schedule_total_steps: int,
                          training_recipe: dict | None = None) -> None:
    """Atomically save a model-only checkpoint used by free-generation curves."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "kind": "curve",
        "seed": seed,
        "schedule_total_steps": int(schedule_total_steps),
    }
    if training_recipe is not None:
        checkpoint["training_recipe"] = training_recipe
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def seed_everything(seed: int | None) -> None:
    """Seed model initialization and data-order RNGs when explicitly requested."""
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_schedule_total_steps(total_steps: int,
                                 requested_steps: int | None) -> int:
    """Return a safe LR horizon that never ends before training stops."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    resolved = total_steps if requested_steps is None else int(requested_steps)
    if resolved < total_steps:
        raise ValueError(
            "schedule_total_steps must be >= actual total_steps, "
            f"got {resolved} < {total_steps}")
    return resolved


def setup_distributed():
    """torchrun 启动时 (WORLD_SIZE>1) 初始化 DDP; 普通单进程启动完全不受影响.

    返回 (rank, world_size, local_rank). backend: GPU 用 nccl, CPU 冒烟用 gloo.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return 0, 1, 0
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", "0"))


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 off-policy: TransMem 蒸馏训练")
    # 数据
    p.add_argument("--data_dir", required=True, help="Stage0 训练数据目录 (含 meta.json, lm_head.pt)")
    p.add_argument("--val_data_dir", default=None)
    p.add_argument("--lm_head_path", default=None,
                   help="lm_head.pt 路径; 默认取 data_dir/lm_head.pt")
    # 模型
    p.add_argument("--config", default="transmem/config.json")
    p.add_argument("--model_path", default=None,
                   help="warm_start=true 时用于热启动的 backbone 路径")
    # 损失 (解耦)
    p.add_argument("--divergence", default="forward_kl",
                   choices=["forward_kl", "reverse_kl", "jsd"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--reg_weight", type=float, default=0.0,
                   help="表征回归 ||HQ'-HQ_tea||^2 权重 (热身用)")
    p.add_argument("--jsd_beta", type=float, default=0.5)
    p.add_argument("--loss", default="kd", choices=["kd", "ce"],
                   help="kd=蒸馏教师软分布(散度,默认); ce=对 golden token 交叉熵(SFT-on-golden, "
                        "配 stage0 --trajectory golden 抽的特征)")
    # 训练
    p.add_argument("--output_dir", default="checkpoints/offpolicy")
    p.add_argument("--batch_size", type=int, default=16,
                   help="每批**序列条数** (每条含 M 个位置, avg M~25), 不是位置数")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--val_interval", type=int, default=1000)
    p.add_argument("--save_interval", type=int, default=5000)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--schedule_total_steps", type=int, default=None,
                   help="学习率调度总步数; 默认等于实际停止步数. 可让短跑沿用完整长跑 schedule")
    p.add_argument("--curve_steps", nargs="*", default=None, metavar="STEP",
                   help="保存 model-only 曲线快照(正整数步); step0 由 evaluator 的 student baseline 表示")
    p.add_argument("--seed", type=int, default=None,
                   help="显式固定模型初始化与数据顺序; 默认 None 保持历史随机行为")
    # 硬件
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--resume", default=None)
    p.add_argument("--overfit_batch", action="store_true")
    args = p.parse_args()
    try:
        spec = ",".join(args.curve_steps or [])
        args.curve_steps = list(parse_curve_steps(spec))
    except ValueError as exc:
        p.error(str(exc))
    return args


# ═══════════════════════════════════════════════════════════════════════════
# Dataset: 预载入 Stage0 特征, __getitem__ 按样本返回完整有序序列
# ═══════════════════════════════════════════════════════════════════════════

class OffPolicyDataset(Dataset):
    """全量预载入内存的**按样本序列**数据集 (一条 = 一个样本的完整有序轨迹).

    Stage0 每样本存 hm_stu [N,dim], hq_stu [M,dim], hq_tea [M,dim] (组内顺序即生成顺序).
    存储仍用扁平大张量 (省碎片): cond_mem [num_samples,N,dim],
    hq_stu_flat/hq_tea_flat [total_pairs,dim] + 每样本 (offset, M).
    __getitem__ 返回 (hm [N,dim], hq_stu [M,dim], hq_tea [M,dim]) — 保留组内顺序,
    TransMem 因果注意力才能让 query i 看到同样本更早的 query (与推理语义一致).
    变长 M 由 collate_sequences 右 padding + q_mask 处理.
    """

    def __init__(self, data_dir: str, load_dtype: torch.dtype = torch.float32,
                 n_mem: int | None = None):
        self.data_dir = Path(data_dir)
        self.load_dtype = load_dtype
        with open(self.data_dir / "meta.json") as f:
            self.meta = json.load(f)
        self.dim = self.meta["dim"]
        # 池化 stage0 (extract --pool_ns): hm_stu 是并集池 [P,dim], 载入时按 n_mem
        # 从每样本 hm_maps 切出 [N,dim] — 一份特征喂所有 N 的消融.
        pool_ns = self.meta.get("pool_ns")
        if pool_ns:
            if n_mem is None or int(n_mem) not in pool_ns:
                raise RuntimeError(
                    f"池化 stage0 数据需要显式 n_mem 且在 pool_ns={pool_ns} 内, 得到 {n_mem} "
                    f"({self.data_dir})")
            self.N = int(n_mem)
        else:
            self.N = self.meta["N"]
            if n_mem is not None and int(n_mem) != self.N:
                raise RuntimeError(
                    f"stage0 数据 N={self.N} != 请求 n_mem={n_mem} ({self.data_dir}); "
                    f"非池化特征换 N 必须重抽")
        self._pool = bool(pool_ns)
        store_dtype = _DTYPE.get(self.meta.get("save_dtype", "bfloat16"), torch.bfloat16)

        manifest = self.meta.get("samples")
        if not manifest:
            raise RuntimeError(f"Stage0 数据缺 manifest: {self.data_dir}")
        num_samples = len(manifest)
        total_pairs = sum(int(e["M"]) for e in manifest)
        if total_pairs == 0:
            raise RuntimeError(f"Stage0 数据 total_pairs=0: {self.data_dir}")

        print(f"OffPolicyDataset: 预载入 {num_samples} 样本 / {total_pairs} 位置 "
              f"({store_dtype}) from {self.data_dir} ...")
        self.cond_mem = torch.empty(num_samples, self.N, self.dim, dtype=store_dtype)
        self.hq_stu_flat = torch.empty(total_pairs, self.dim, dtype=store_dtype)
        self.hq_tea_flat = torch.empty(total_pairs, self.dim, dtype=store_dtype)
        self.ans_flat = torch.full((total_pairs,), -100, dtype=torch.long)  # golden token id (CE); -100=ignore
        self.sample_offset = torch.empty(num_samples, dtype=torch.long)
        self.sample_len = torch.empty(num_samples, dtype=torch.long)

        cursor = 0
        for s, entry in enumerate(manifest):
            data = torch.load(self.data_dir / entry["file"], map_location="cpu",
                              weights_only=False)
            M = data["hq_stu"].shape[0]
            hm = data["hm_stu"]
            if self._pool:
                hm = hm[data["hm_maps"][str(self.N)]]
            self.cond_mem[s] = hm.to(store_dtype)
            self.hq_stu_flat[cursor:cursor + M] = data["hq_stu"].to(store_dtype)
            self.hq_tea_flat[cursor:cursor + M] = data["hq_tea"].to(store_dtype)
            _av = data.get("answer_ids")
            if _av is not None:
                self.ans_flat[cursor:cursor + M] = _av[:M].long()
            self.sample_offset[s] = cursor
            self.sample_len[s] = M
            cursor += M
            if (s + 1) % 5000 == 0:
                print(f"  ...已载入 {s + 1}/{num_samples}")
        assert cursor == total_pairs, (cursor, total_pairs)
        self.total_pairs = total_pairs
        self.num_samples = num_samples
        print(f"OffPolicyDataset: 完成, avg M={total_pairs / num_samples:.1f}, "
              f"max M={int(self.sample_len.max())}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        o, M = int(self.sample_offset[idx]), int(self.sample_len[idx])
        hm_stu = self.cond_mem[idx].to(self.load_dtype)          # [N, dim]
        hq_stu = self.hq_stu_flat[o:o + M].to(self.load_dtype)   # [M, dim] 有序
        hq_tea = self.hq_tea_flat[o:o + M].to(self.load_dtype)   # [M, dim] 有序
        ans = self.ans_flat[o:o + M]                             # [M] golden token id (CE 用)
        return hm_stu, hq_stu, hq_tea, ans


def collate_sequences(batch):
    """右 padding 到批内最长 M -> (X [B,N+M_max,dim], hq_tea [B,M_max,dim], q_mask [B,M_max]).

    causal mask 下有效位看不到尾部 padding (test_shapes [8] 已验证), attention 内
    无需额外 mask; padding 位的输出是垃圾, 由 q_mask 挡在 loss 之外.
    """
    hm = torch.stack([b[0] for b in batch], dim=0)               # [B, N, dim]
    lens = [b[1].shape[0] for b in batch]
    B, m_max, dim = len(batch), max(lens), hm.shape[-1]
    hq_stu = hm.new_zeros(B, m_max, dim)
    hq_tea = hm.new_zeros(B, m_max, dim)
    ans = torch.full((B, m_max), -100, dtype=torch.long)         # golden token id, pad=-100 (CE ignore)
    q_mask = torch.zeros(B, m_max, dtype=torch.bool)
    for i, (_, s, t, a) in enumerate(batch):
        hq_stu[i, :lens[i]] = s
        hq_tea[i, :lens[i]] = t
        ans[i, :lens[i]] = a
        q_mask[i, :lens[i]] = True
    X = torch.cat([hm, hq_stu], dim=1)                           # [B, N+M_max, dim]
    return X, hq_tea, ans, q_mask


class _ConcatDatasets(torch.utils.data.ConcatDataset):
    """多域拼接 (Qasper + HotpotQA + ...): 各子集须同 N/dim (同一 stage0 N 与 backbone dim);
    暴露 .N/.dim 供 config 校验. __getitem__ 仍返回 (hm,hq_stu,hq_tea), collate 不变."""

    def __init__(self, datasets):
        super().__init__(datasets)
        Ns = {d.N for d in datasets}
        dims = {d.dim for d in datasets}
        assert len(Ns) == 1 and len(dims) == 1, (
            f"多域数据集 N/dim 不一致: N={Ns} dim={dims} "
            f"(须同一 stage0 --N 与同一 backbone 的 dim)")
        self.N = next(iter(Ns))
        self.dim = next(iter(dims))
        self.total_pairs = sum(d.total_pairs for d in datasets)   # 供 trainer 日志/统计
        self.num_samples = sum(d.num_samples for d in datasets)
        print(f"多域拼接完成: {len(datasets)} 域, 共 {self.num_samples} 样本 / "
              f"{self.total_pairs} 位置, N={self.N} dim={self.dim}")


def _build_dataset(data_dir, dtype, n_mem=None):
    """data_dir 可为逗号分隔的多个 stage0 目录 -> ConcatDataset (多域训练).
    n_mem: 池化 stage0 必填 (按 config.n_mem 切片); 非池化时用于一致性校验."""
    dirs = [d.strip() for d in str(data_dir).split(",") if d.strip()]
    if len(dirs) == 1:
        return OffPolicyDataset(dirs[0], load_dtype=dtype, n_mem=n_mem)
    return _ConcatDatasets([OffPolicyDataset(d, load_dtype=dtype, n_mem=n_mem)
                            for d in dirs])


def make_dataloader(data_dir, batch_size, num_workers, dtype, shuffle=True,
                    distributed=False, n_mem=None):
    ds = _build_dataset(data_dir, dtype, n_mem=n_mem)
    if distributed:
        # 各 rank 分到不重叠的样本子集; 每 epoch 需 sampler.set_epoch 重洗
        sampler = DistributedSampler(ds, shuffle=shuffle, drop_last=True)
    elif shuffle:
        sampler = torch.utils.data.RandomSampler(ds, num_samples=len(ds), replacement=False)
    else:
        sampler = torch.utils.data.SequentialSampler(ds)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                      pin_memory=True, drop_last=True, collate_fn=collate_sequences,
                      persistent_workers=(num_workers > 0))


# ═══════════════════════════════════════════════════════════════════════════
# 训练器
# ═══════════════════════════════════════════════════════════════════════════

class OffPolicyTrainer:
    def __init__(self, args):
        self.args = args
        # ── DDP: torchrun 启动时每 rank 一卡; rank0 独占 log/TB/val/save ──
        self.rank, self.world, local_rank = setup_distributed()
        self.is_main = (self.rank == 0)
        if self.world > 1 and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
        self.device = torch.device(args.device)
        self.train_dtype = _DTYPE[args.dtype]

        # Seed before TransMem construction: fixed-seed curve reruns must share
        # model initialization, not merely the data order.
        seed_everything(getattr(args, "seed", None))
        self.config = TransMemConfig.from_json(args.config)
        self.training_recipe = build_training_recipe(
            args, self.config.to_dict(), world_size=self.world)
        self._resume_recipe = None
        assert self.config.causal, (
            "序列级 off-policy 训练依赖 causal mask (尾部 padding 不可见 + "
            "query 只看历史); causal=false 与 token-by-token 推理语义不兼容")
        self.mem = TransMem(self.config).to(self.device, dtype=self.train_dtype)
        if self.config.warm_start:
            self._warm_start()
        # DDP wrap: 梯度 allreduce 用 self.net; state_dict/clip/correct 仍走裸 self.mem.
        # broadcast_buffers=False: 唯一 buffer 是常量 a, 且避免 rank0 单独 no_grad
        # 前向 (validate) 时其它 rank 卡在 buffer 广播上死锁.
        if self.world > 1:
            self.net = DDP(self.mem,
                           device_ids=[local_rank] if torch.cuda.is_available() else None,
                           broadcast_buffers=False)
        else:
            self.net = self.mem

        # 冻结 lm_head
        # 多域 data_dir 逗号分隔: 取第一个域的 lm_head.pt (同一 backbone, 各域相同)
        _first_dir = str(args.data_dir).split(",")[0].strip()
        lm_path = args.lm_head_path or str(Path(_first_dir) / "lm_head.pt")
        self.lm_head = FrozenLMHead.from_file(lm_path, device=self.device,
                                              dtype=self.train_dtype)
        self.loss_fn = DistillLoss(divergence=args.divergence, temperature=args.temperature,
                                   reg_weight=args.reg_weight, jsd_beta=args.jsd_beta)

        if self.is_main:
            print(f"TransMem: {self.mem.num_params():,} params "
                  f"({self.mem.num_params(True):,} trainable)"
                  + (f" | DDP x{self.world}" if self.world > 1 else ""))
            print(f"Config: depth={self.config.depth}, heads={self.config.num_heads}, "
                  f"kv={self.config.num_kv_heads}, dim={self.config.dim}, "
                  f"pos={self.config.pos_mode}, causal={self.config.causal}, "
                  f"a={self.config.a_init}, warm_start={self.config.warm_start}")
            print(f"Loss: {args.divergence}, T={args.temperature}, reg_weight={args.reg_weight}")
            print(f"lm_head: vocab={self.lm_head.proj.out_features}")

        self.optimizer = torch.optim.AdamW(
            (p for p in self.mem.parameters() if p.requires_grad),
            lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        self.global_step = 0
        self.epoch = 0
        self.best_val = float("inf")   # 最优 val_loss, 独立持久化到 result.json (resume 不重置)
        self.best_step = -1
        self.writer = (SummaryWriter(log_dir=str(Path(args.output_dir) / "tb"))
                       if self.is_main else None)

    def _warm_start(self):
        if not self.args.model_path:
            raise ValueError("warm_start=true 需要 --model_path 指向 backbone")
        from transformers import AutoModelForCausalLM
        print(f"热启动: 从 {self.args.model_path} 顶部 {self.config.depth} 层")
        backbone = AutoModelForCausalLM.from_pretrained(
            self.args.model_path, torch_dtype=self.train_dtype,
            local_files_only=True, trust_remote_code=True)
        self.mem.warm_start_from(backbone)
        del backbone
        torch.cuda.empty_cache()

    def _set_lr(self, step, total_steps):
        warmup = self.args.warmup_steps
        if step < warmup:
            lr = self.args.lr * step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(total_steps - warmup, 1)
            lr = self.args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def compute_loss(self, X, hq_tea, ans, q_mask, full: bool = False):
        """X [B,N+M,dim], hq_tea [B,M,dim], ans [B,M] golden ids, q_mask [B,M] -> (loss, metrics).

        一次并行前向算全部 query 位 MS (causal mask: 位置 i 只见 {HM, HQ_1..i}), 按 q_mask 收集有效位.
        loss='ce': 对 golden token 交叉熵 (SFT-on-golden, 规避 dirty teacher);
        loss='kd': 蒸馏教师软分布 (原路径). full=True 额外算 baseline(MS=0) 改善比.
        """
        N = self.mem.config.n_mem
        hq_stu = X[:, N:, :]                                     # [B, M, dim] 全部 query 位
        # 训练走 DDP wrapper (梯度 allreduce); no_grad (validate/诊断) 走裸模块,
        # 避免 rank0 单独 validate 时其它 rank 等不到集合通信而死锁
        net = self.net if torch.is_grad_enabled() else self.mem
        ms = net(X, return_all_queries=True)                     # [B, M, dim]
        hq_prime = self.mem.correct(ms, hq_stu)                  # [B, M, dim]

        valid = q_mask.to(X.device)                              # [B, M] bool
        hq_prime_v = hq_prime[valid]                             # [P, dim] 有效位
        student_logits = self.lm_head(hq_prime_v)                # [P, vocab]

        if self.args.loss == "ce":
            # SFT-on-golden: 直接对 golden token 交叉熵, 不需 teacher
            ans_v = ans.to(X.device)[valid]                      # [P] golden token ids
            loss = F.cross_entropy(student_logits, ans_v)
            metrics = {"loss": float(loss.detach()), "div": float(loss.detach()),
                       "positions": int(valid.sum())}
            with torch.no_grad():
                metrics["ms_norm"] = float(ms[valid].norm(dim=-1).mean())
                metrics["top1"] = float((student_logits.argmax(-1) == ans_v).float().mean())
                if full:
                    base_ce = float(F.cross_entropy(self.lm_head(hq_stu[valid]), ans_v))
                    metrics["base_div"] = base_ce                # MS=0 未纠正的 CE
                    metrics["improve"] = ((base_ce - metrics["div"]) / base_ce) if base_ce > 1e-8 else 0.0
            return loss, metrics

        # ── KD (原路径): 蒸馏教师软分布 ──────────────────────────────────
        hq_tea_v = hq_tea[valid]                                 # [P, dim]
        with torch.no_grad():
            teacher_logits = self.lm_head(hq_tea_v)
        loss, metrics = self.loss_fn(student_logits, teacher_logits, hq_prime_v, hq_tea_v)
        metrics["positions"] = int(valid.sum())
        with torch.no_grad():
            metrics["ms_norm"] = float(ms[valid].norm(dim=-1).mean())  # 记忆偏置强度: 0->涨->稳
            metrics["top1"] = float(                             # 纠正后学生与教师 argmax 一致率
                (student_logits.argmax(-1) == teacher_logits.argmax(-1)).float().mean())
            if full:
                base_logits = self.lm_head(hq_stu[valid])        # MS=0 未纠正的学生
                bd = float(self.loss_fn.divergence_only(base_logits, teacher_logits))
                metrics["base_div"] = bd                         # 初始 gap (student(C_L) vs teacher(C_S))
                metrics["improve"] = ((bd - metrics["div"]) / bd) if bd > 1e-8 else 0.0
        return loss, metrics

    def train_step(self, X, hq_tea, ans, q_mask, full: bool = False):
        self.net.train()
        loss, metrics = self.compute_loss(X, hq_tea, ans, q_mask, full=full)
        self.optimizer.zero_grad()
        loss.backward()
        if self.args.grad_clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.mem.parameters(), self.args.grad_clip)
            metrics["grad_norm"] = float(gn)                     # 裁剪前总范数: 看是否爆/削
        self.optimizer.step()
        return metrics

    @torch.no_grad()
    def validate(self, dataloader, max_batches=50):
        self.mem.eval()
        agg = {"val_loss": 0.0, "val_top1": 0.0, "val_improve": 0.0}
        n = 0
        for X, hq_tea, ans, q_mask in dataloader:
            X = X.to(self.device)
            hq_tea = hq_tea.to(self.device)
            _, m = self.compute_loss(X, hq_tea, ans, q_mask, full=True)
            agg["val_loss"] += m["loss"]
            agg["val_top1"] += m["top1"]
            agg["val_improve"] += m.get("improve", 0.0)
            n += 1
            if n >= max_batches:
                break
        return {k: v / max(n, 1) for k, v in agg.items()}

    def save(self, path, metrics=None, kind="latest"):
        """kind: latest=只覆写 latest.pt (周期性, 不累积占盘, 单 ckpt 4.6G);
        best=覆写 best.pt + latest.pt; final=额外落 step_XXXXXXX.pt 存档.
        长跑归档/腾空间用 checkpoints/upload_ckpt.sh 传对象存储."""
        if not self.is_main:
            return
        Path(path).mkdir(parents=True, exist_ok=True)
        base = {"model_state_dict": self.mem.state_dict(),
                "config": self.config.to_dict(), "global_step": self.global_step,
                "epoch": self.epoch, "training_recipe": self.training_recipe}
        if metrics:
            base["metrics"] = metrics
        # latest.pt 带 optimizer (断点续训需要); best/final 只存 model_state_dict —
        # 评测/推理够用, 省盘 (8B TransMem 含 optimizer 12.6G, 只存 model 4.2G).
        # 原子写 (tmp + os.replace): spot 抢占砍在写档中途不会留截断 zip
        # (10218607 实测半截 latest.pt 让 requeue 断点重续全体崩).
        def _asave(obj, p):
            tmp = p.with_suffix(p.suffix + ".tmp")
            torch.save(obj, tmp)
            os.replace(tmp, p)
        _asave(dict(base, optimizer_state_dict=self.optimizer.state_dict()),
               Path(path) / "latest.pt")
        names = ["latest.pt"]
        if kind == "best":
            _asave(base, Path(path) / "best.pt"); names.append("best.pt(model-only)")
        elif kind == "final":
            fn = f"step_{self.global_step:07d}.pt"
            _asave(base, Path(path) / fn); names.append(f"{fn}(model-only)")
        print(f"  Checkpoint saved: {', '.join(names)} (step {self.global_step})")

    def _maybe_save_curve_checkpoint(self, curve_steps: set[int],
                                     schedule_total_steps: int) -> None:
        """Save exactly requested curve points without touching ``latest.pt``."""
        if not self.is_main or self.global_step not in curve_steps:
            return
        path = curve_checkpoint_path(self.args.output_dir, self.global_step)
        save_curve_checkpoint(
            path,
            model=self.mem,
            config=self.config.to_dict(),
            global_step=self.global_step,
            epoch=self.epoch,
            seed=getattr(self.args, "seed", None),
            schedule_total_steps=schedule_total_steps,
            training_recipe=self.training_recipe,
        )
        print(f"  Curve checkpoint saved: {path.name} (model-only)")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        current_recipe = build_training_recipe(
            self.args, self.config.to_dict(), world_size=self.world)
        require_recipe = bool(getattr(self.args, "curve_steps", None))
        saved_recipe = ckpt.get("training_recipe")
        validate_resume_recipe(
            saved_recipe, current_recipe, require_recipe=require_recipe,
            allow_unresolved=True)
        if require_recipe and "optimizer_state_dict" not in ckpt:
            raise ValueError(
                "curve resume requires latest.pt with optimizer_state_dict")
        self._resume_recipe = saved_recipe
        self.training_recipe = current_recipe
        self.mem.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:          # best.pt 是 model-only, 无 optimizer
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.epoch = ckpt.get("epoch", 0)
        self._load_result()   # 恢复 best_val, 避免 resume 后覆盖历史最优
        if self.is_main:
            if saved_recipe is None:
                print("[WARN] legacy checkpoint has no training_recipe")
            print(f"Resumed: step={self.global_step}, best_val={self.best_val:.6f}")

    def _result_path(self):
        return Path(self.args.output_dir) / "result.json"

    def _load_result(self):
        """从 result.json 恢复 best_val/best_step (独立于 .pt checkpoint)."""
        p = self._result_path()
        if p.exists():
            try:
                r = json.loads(p.read_text())
                self.best_val = r.get("best_val", float("inf"))
                self.best_step = r.get("best_step", -1)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  警告: 读取 {p} 失败 ({e}), best_val 保持 {self.best_val}")

    def _save_result(self, extra=None):
        """把训练进度 + 最优结果落盘到 result.json (原子写). 仅 rank0."""
        if not self.is_main:
            return
        r = {"best_val": self.best_val, "best_step": self.best_step,
             "global_step": self.global_step, "epoch": self.epoch,
             "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        if extra:
            r.update(extra)
        p = self._result_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(r, indent=2, ensure_ascii=False))
        tmp.replace(p)   # 同目录 rename, 原子替换, 防止写一半的损坏文件

    def run(self):
        args = self.args
        train_dl = make_dataloader(args.data_dir, args.batch_size, args.num_workers,
                                   self.train_dtype, shuffle=True,
                                   distributed=(self.world > 1),
                                   n_mem=self.config.n_mem)
        # 验证只在 rank0 做 (裸模块前向, 无集合通信; 其余 rank 在下个 backward 处等它)
        val_dl = (make_dataloader(args.val_data_dir, args.batch_size, args.num_workers,
                                  self.train_dtype, shuffle=False,
                                  n_mem=self.config.n_mem)
                  if (args.val_data_dir and self.is_main) else None)
        assert train_dl.dataset.N == self.config.n_mem, (
            f"Stage0 数据 N={train_dl.dataset.N} != config n_mem={self.config.n_mem}: "
            f"forward 里 query 槽从第 n_mem 位切, 不一致会静默错位")
        total_steps = args.max_steps or (args.epochs * len(train_dl))
        schedule_total_steps = resolve_schedule_total_steps(
            total_steps, getattr(args, "schedule_total_steps", None))
        requested_curve_steps = set(getattr(args, "curve_steps", None) or [])
        curve_steps = {step for step in requested_curve_steps if step > 0}
        resolved_recipe = build_training_recipe(
            args, self.config.to_dict(), world_size=self.world,
            resolved_total_steps=total_steps,
            resolved_schedule_total_steps=schedule_total_steps)
        if self._resume_recipe is not None:
            validate_resume_recipe(
                self._resume_recipe, resolved_recipe, require_recipe=True)
        self.training_recipe = resolved_recipe
        if self.is_main:
            print(f"\n训练: steps/epoch≈{len(train_dl)}, total≈{total_steps}, "
                  f"bs={args.batch_size} 序列/rank x {self.world} rank "
                  f"(全局 {args.batch_size * self.world} 序列/step, avg "
                  f"{train_dl.dataset.total_pairs / train_dl.dataset.num_samples:.1f} 位置/序列), "
                  f"lr={args.lr}, device={self.device}")
            if schedule_total_steps != total_steps:
                print(f"LR schedule: {schedule_total_steps} steps (实际停止于 {total_steps})")
            if 0 in requested_curve_steps:
                print("[INFO] 忽略 curve step 0; evaluator 用 student baseline 作为起点")
            if curve_steps:
                print(f"自由生成曲线快照: {sorted(curve_steps)}")
                unreachable = sorted(step for step in curve_steps if step > total_steps)
                if unreachable:
                    print(f"[WARN] curve_steps 超过停止步数, 不会保存: {unreachable}")
            print("=" * 72)

        # On resume, preserve the requested current positive step before the
        # next optimizer update; fresh runs use the evaluator's student step 0.
        self._maybe_save_curve_checkpoint(curve_steps, schedule_total_steps)

        if args.overfit_batch:
            X, hq_tea, ans, q_mask = next(iter(train_dl))
            X, hq_tea = X.to(self.device), hq_tea.to(self.device)
            for step in range(500):
                m = self.train_step(X, hq_tea, ans, q_mask, full=(step % 50 == 0))
                if self.writer:
                    self.writer.add_scalar("train/loss", m["loss"], step)
                    self.writer.add_scalar("train/div", m["div"], step)
                    self.writer.add_scalar("train/ms_norm", m["ms_norm"], step)
                    self.writer.add_scalar("train/top1", m["top1"], step)
                    if "grad_norm" in m:
                        self.writer.add_scalar("train/grad_norm", m["grad_norm"], step)
                if self.is_main and step % 50 == 0:
                    print(f"  step {step:4d}: loss={m['loss']:.6f} div={m['div']:.6f} "
                          f"top1={m['top1']:.3f} ms_norm={m['ms_norm']:.3f} "
                          f"improve={m.get('improve', 0.0):.3f}")
            if self.writer:
                self.writer.close()
            return

        running = 0.0
        t0 = time.time()
        while self.global_step < total_steps:
            self.epoch += 1
            if isinstance(train_dl.sampler, DistributedSampler):
                train_dl.sampler.set_epoch(self.epoch)   # 每 epoch 换洗牌种子
            for X, hq_tea, ans, q_mask in train_dl:
                if self.global_step >= total_steps:
                    break
                self._set_lr(self.global_step, schedule_total_steps)
                X, hq_tea = X.to(self.device), hq_tea.to(self.device)
                # 该步过后若触发 log, 就算上 baseline 诊断 (base_div/improve)
                full = ((self.global_step + 1) % args.log_interval == 0)
                m = self.train_step(X, hq_tea, ans, q_mask, full=full)
                self.global_step += 1
                running += m["loss"]
                self._maybe_save_curve_checkpoint(curve_steps, schedule_total_steps)
                if self.is_main and self.global_step % args.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    sps = args.log_interval / max(time.time() - t0, 1e-3)
                    mem_gb = torch.cuda.max_memory_allocated() / 1024**3
                    print(f"  step {self.global_step:7d}/{total_steps} | "
                          f"loss {running/args.log_interval:.6f} | lr {lr:.2e} | "
                          f"top1 {m['top1']:.3f} | improve {m.get('improve', 0.0):.3f} | "
                          f"ms {m['ms_norm']:.2f} | gnorm {m.get('grad_norm', 0.0):.2f} | "
                          f"{sps:.1f} it/s | mem {mem_gb:.1f}GB")
                    self.writer.add_scalar("train/loss", running / args.log_interval, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    self.writer.add_scalar("train/it_per_s", sps, self.global_step)
                    self.writer.add_scalar("train/mem_gb", mem_gb, self.global_step)
                    self.writer.add_scalar("train/ms_norm", m["ms_norm"], self.global_step)
                    self.writer.add_scalar("train/top1", m["top1"], self.global_step)
                    if "grad_norm" in m:
                        self.writer.add_scalar("train/grad_norm", m["grad_norm"], self.global_step)
                    if "base_div" in m:
                        self.writer.add_scalar("train/base_div", m["base_div"], self.global_step)
                        self.writer.add_scalar("train/improve", m["improve"], self.global_step)
                    if "div" in m:
                        self.writer.add_scalar("train/div", m["div"], self.global_step)
                    if "reg" in m:
                        self.writer.add_scalar("train/reg", m["reg"], self.global_step)
                    running = 0.0
                    t0 = time.time()
                if val_dl and self.global_step % args.val_interval == 0:
                    vm = self.validate(val_dl)
                    print(f"  --- VAL step {self.global_step}: val_loss={vm['val_loss']:.6f} "
                          f"val_top1={vm['val_top1']:.3f} val_improve={vm['val_improve']:.3f} ---")
                    self.writer.add_scalar("val/loss", vm["val_loss"], self.global_step)
                    self.writer.add_scalar("val/top1", vm["val_top1"], self.global_step)
                    self.writer.add_scalar("val/improve", vm["val_improve"], self.global_step)
                    if vm["val_loss"] < self.best_val:
                        self.best_val = vm["val_loss"]
                        self.best_step = self.global_step
                        self.save(args.output_dir, {"val_loss": self.best_val, "best": True},
                                  kind="best")
                        self._save_result({"best_metrics": vm})
                if self.global_step % args.save_interval == 0:
                    self.save(args.output_dir)
        final = self.validate(val_dl, max_batches=200) if val_dl else {}
        self.save(args.output_dir, final, kind="final")
        self._save_result({"final_metrics": final, "done": True})
        if self.writer:
            self.writer.close()
        if self.world > 1:
            dist.barrier()                     # 等 rank0 存完 ckpt 再一起退出
            dist.destroy_process_group()
        if self.is_main:
            print("=" * 72)
            print(f"✅ off-policy 训练完成: {self.global_step} steps"
                  + (f", val_loss={final.get('val_loss', float('nan')):.6f}" if final else ""))
            print(f"TensorBoard: tensorboard --logdir {Path(args.output_dir) / 'tb'}")


def main():
    args = parse_args()
    trainer = OffPolicyTrainer(args)
    if args.resume:
        trainer.load(args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
