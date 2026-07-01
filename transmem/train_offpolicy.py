#!/usr/bin/env python3
"""
Stage 1 (off-policy) — 训练 TransMem: 逐位置蒸馏, 穿冻结 lm_head 反传到 TransMem.

数据来源 = Stage0 固定教师轨迹的特征 (off-policy: 轨迹由教师 rollout 给定, 与当前策略无关).
  X_i           = [HM_stu ; HQ_stu_i]            [B, N+1, dim]
  MS_i          = TransMem(X_i)                   [B, dim]
  HQ'_i         = HQ_stu_i + a*MS_i
  student_logits= lm_head(HQ'_i)                  (冻结 lm_head, 可微)
  teacher_logits= lm_head(HQ_tea_i)               (固定软目标)
  L             = DistillLoss(P_tea, P_stu)       (forward_kl / reverse_kl / jsd; +可选 L_reg)

只反传 lm_head 这一个线性层, 不重跑整个 LLM -> 很轻. Stage0 已离线算好特征, 训练全程不碰 LLM.

用法:
  python -m transmem.train_offpolicy \
    --data_dir data/stage0_train --val_data_dir data/stage0_dev \
    --config transmem/config.json --output_dir checkpoints/offpolicy \
    --divergence forward_kl --temperature 1.0 --reg_weight 0.0 \
    --batch_size 128 --lr 1e-4 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transmem import TransMemConfig, TransMem, DistillLoss, FrozenLMHead

_DTYPE = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


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
    # 训练
    p.add_argument("--output_dir", default="checkpoints/offpolicy")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--val_interval", type=int, default=1000)
    p.add_argument("--save_interval", type=int, default=5000)
    p.add_argument("--max_steps", type=int, default=None)
    # 硬件
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--resume", default=None)
    p.add_argument("--overfit_batch", action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Dataset: 预载入 Stage0 特征, __getitem__ 返回 (X=[hm;hq_stu], hq_tea)
# ═══════════════════════════════════════════════════════════════════════════

class OffPolicyDataset(Dataset):
    """全量预载入内存的逐位置数据集.

    Stage0 每样本存 hm_stu [N,dim], hq_stu [M,dim], hq_tea [M,dim].
    预载入摊平成: cond_mem [num_samples,N,dim], hq_stu_flat/hq_tea_flat [total_pairs,dim].
    __getitem__ 纯内存切片, 训练不再触盘 (复用 DiffusionMem Stage0Dataset 思路).
    """

    def __init__(self, data_dir: str, load_dtype: torch.dtype = torch.float32):
        self.data_dir = Path(data_dir)
        self.load_dtype = load_dtype
        with open(self.data_dir / "meta.json") as f:
            self.meta = json.load(f)
        self.dim = self.meta["dim"]
        self.N = self.meta["N"]
        store_dtype = _DTYPE.get(self.meta.get("save_dtype", "bfloat16"), torch.bfloat16)

        manifest = self.meta.get("samples")
        if not manifest:
            raise RuntimeError(f"Stage0 数据缺 manifest: {self.data_dir}")
        num_samples = len(manifest)
        total_pairs = sum(int(e["M"]) for e in manifest)
        if total_pairs == 0:
            raise RuntimeError(f"Stage0 数据 total_pairs=0: {self.data_dir}")

        print(f"OffPolicyDataset: 预载入 {num_samples} 样本 / {total_pairs} 对 "
              f"({store_dtype}) from {self.data_dir} ...")
        self.cond_mem = torch.empty(num_samples, self.N, self.dim, dtype=store_dtype)
        self.hq_stu_flat = torch.empty(total_pairs, self.dim, dtype=store_dtype)
        self.hq_tea_flat = torch.empty(total_pairs, self.dim, dtype=store_dtype)
        self.pair_to_sample = torch.empty(total_pairs, dtype=torch.long)

        cursor = 0
        for s, entry in enumerate(manifest):
            data = torch.load(self.data_dir / entry["file"], map_location="cpu",
                              weights_only=False)
            M = data["hq_stu"].shape[0]
            self.cond_mem[s] = data["hm_stu"].to(store_dtype)
            self.hq_stu_flat[cursor:cursor + M] = data["hq_stu"].to(store_dtype)
            self.hq_tea_flat[cursor:cursor + M] = data["hq_tea"].to(store_dtype)
            self.pair_to_sample[cursor:cursor + M] = s
            cursor += M
            if (s + 1) % 5000 == 0:
                print(f"  ...已载入 {s + 1}/{num_samples}")
        assert cursor == total_pairs, (cursor, total_pairs)
        self.total_pairs = total_pairs
        self.num_samples = num_samples
        print(f"OffPolicyDataset: 完成, avg M={total_pairs / num_samples:.1f}")

    def __len__(self):
        return self.total_pairs

    def __getitem__(self, flat_idx: int):
        s = int(self.pair_to_sample[flat_idx])
        hm_stu = self.cond_mem[s].to(self.load_dtype)            # [N, dim]
        hq_stu = self.hq_stu_flat[flat_idx].to(self.load_dtype)  # [dim]
        hq_tea = self.hq_tea_flat[flat_idx].to(self.load_dtype)  # [dim]
        X = torch.cat([hm_stu, hq_stu.unsqueeze(0)], dim=0)      # [N+1, dim] (查询槽末尾)
        return X, hq_tea


def make_dataloader(data_dir, batch_size, num_workers, dtype, shuffle=True):
    ds = OffPolicyDataset(data_dir, load_dtype=dtype)
    sampler = (torch.utils.data.RandomSampler(ds, num_samples=len(ds), replacement=False)
               if shuffle else torch.utils.data.SequentialSampler(ds))
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                      pin_memory=True, drop_last=True,
                      persistent_workers=(num_workers > 0))


# ═══════════════════════════════════════════════════════════════════════════
# 训练器
# ═══════════════════════════════════════════════════════════════════════════

class OffPolicyTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.train_dtype = _DTYPE[args.dtype]

        self.config = TransMemConfig.from_json(args.config)
        self.mem = TransMem(self.config).to(self.device, dtype=self.train_dtype)
        if self.config.warm_start:
            self._warm_start()

        # 冻结 lm_head
        lm_path = args.lm_head_path or str(Path(args.data_dir) / "lm_head.pt")
        self.lm_head = FrozenLMHead.from_file(lm_path, device=self.device,
                                              dtype=self.train_dtype)
        self.loss_fn = DistillLoss(divergence=args.divergence, temperature=args.temperature,
                                   reg_weight=args.reg_weight, jsd_beta=args.jsd_beta)

        print(f"TransMem: {self.mem.num_params():,} params "
              f"({self.mem.num_params(True):,} trainable)")
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

    def compute_loss(self, X, hq_tea):
        """X [B,N+1,dim], hq_tea [B,dim] -> (loss, metrics)."""
        hq_stu = X[:, -1, :]                       # 查询槽 = 末位
        ms = self.mem(X)
        hq_prime = self.mem.correct(ms, hq_stu)
        student_logits = self.lm_head(hq_prime)
        with torch.no_grad():
            teacher_logits = self.lm_head(hq_tea)
        return self.loss_fn(student_logits, teacher_logits, hq_prime, hq_tea)

    def train_step(self, X, hq_tea):
        self.mem.train()
        loss, metrics = self.compute_loss(X, hq_tea)
        self.optimizer.zero_grad()
        loss.backward()
        if self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.mem.parameters(), self.args.grad_clip)
        self.optimizer.step()
        return metrics

    @torch.no_grad()
    def validate(self, dataloader, max_batches=50):
        self.mem.eval()
        tot = 0.0
        n = 0
        for X, hq_tea in dataloader:
            X = X.to(self.device)
            hq_tea = hq_tea.to(self.device)
            _, m = self.compute_loss(X, hq_tea)
            tot += m["loss"]
            n += 1
            if n >= max_batches:
                break
        return {"val_loss": tot / max(n, 1)}

    def save(self, path, metrics=None):
        Path(path).mkdir(parents=True, exist_ok=True)
        ckpt = {"model_state_dict": self.mem.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.to_dict(), "global_step": self.global_step,
                "epoch": self.epoch}
        if metrics:
            ckpt["metrics"] = metrics
        torch.save(ckpt, Path(path) / f"step_{self.global_step:07d}.pt")
        torch.save(ckpt, Path(path) / "latest.pt")
        print(f"  Checkpoint saved: step_{self.global_step:07d}.pt")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.mem.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.epoch = ckpt.get("epoch", 0)
        print(f"Resumed: step={self.global_step}")

    def run(self):
        args = self.args
        train_dl = make_dataloader(args.data_dir, args.batch_size, args.num_workers,
                                   self.train_dtype, shuffle=True)
        val_dl = (make_dataloader(args.val_data_dir, args.batch_size, args.num_workers,
                                  self.train_dtype, shuffle=False)
                  if args.val_data_dir else None)
        total_steps = args.max_steps or (args.epochs * len(train_dl))
        print(f"\n训练: steps/epoch≈{len(train_dl)}, total≈{total_steps}, "
              f"bs={args.batch_size}, lr={args.lr}, device={self.device}")
        print("=" * 72)

        if args.overfit_batch:
            X, hq_tea = next(iter(train_dl))
            X, hq_tea = X.to(self.device), hq_tea.to(self.device)
            for step in range(500):
                m = self.train_step(X, hq_tea)
                if step % 50 == 0:
                    print(f"  step {step:4d}: loss={m['loss']:.6f} div={m['div']:.6f}")
            return

        best_val = float("inf")
        running = 0.0
        t0 = time.time()
        while self.global_step < total_steps:
            self.epoch += 1
            for X, hq_tea in train_dl:
                if self.global_step >= total_steps:
                    break
                self._set_lr(self.global_step, total_steps)
                X, hq_tea = X.to(self.device), hq_tea.to(self.device)
                m = self.train_step(X, hq_tea)
                self.global_step += 1
                running += m["loss"]
                if self.global_step % args.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    sps = args.log_interval / max(time.time() - t0, 1e-3)
                    print(f"  step {self.global_step:7d}/{total_steps} | "
                          f"loss {running/args.log_interval:.6f} | lr {lr:.2e} | "
                          f"{sps:.1f} it/s | "
                          f"mem {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
                    running = 0.0
                    t0 = time.time()
                if val_dl and self.global_step % args.val_interval == 0:
                    vm = self.validate(val_dl)
                    print(f"  --- VAL step {self.global_step}: val_loss={vm['val_loss']:.6f} ---")
                    if vm["val_loss"] < best_val:
                        best_val = vm["val_loss"]
                        self.save(args.output_dir, {"val_loss": best_val, "best": True})
                if self.global_step % args.save_interval == 0:
                    self.save(args.output_dir)
        final = self.validate(val_dl, max_batches=200) if val_dl else {}
        self.save(args.output_dir, final)
        print("=" * 72)
        print(f"✅ off-policy 训练完成: {self.global_step} steps"
              + (f", val_loss={final.get('val_loss', float('nan')):.6f}" if final else ""))


def main():
    args = parse_args()
    trainer = OffPolicyTrainer(args)
    if args.resume:
        trainer.load(args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
