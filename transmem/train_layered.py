#!/usr/bin/env python3
"""
训练 TransMem-Layer (v3 计划 6): 最后 D 层各一个 1 层 TransMem, 离线逐层目标.

数据 = stage0 layered 特征 (extract_features --dump_layers K): 每样本存
  hm_stu_layers  [K, N, dim]   各层 HM 槽 (上下文流, 不受查询位注入影响 → 离线严格有效)
  hq_stu_layers  [K, M, dim]   各层学生查询 hidden (只有最低插入层的在推理时严格成立)
  hq_tea_layers  [K, M, dim]   各层教师查询 hidden (逐层回归目标)
训练块只取 inject_layers ⊆ layer_ids (一次抽 K=8 超集, D∈{1,2,4,6,8} 全复用).

逐层损失 (块间无梯度耦合, 联合训练=独立训练, 一个 optimizer 顺手):
  最顶层 (LLM 末层): 冻结 final_norm + lm_head 的 forward-KL (对齐原 TransMem 配方;
      D=1 时整个损失退化为它 → 与原 final-hidden 设计的锚点, 差异仅 pre/post-norm).
  其余层: 相对残差回归 rel_l = ||h_in + g*MS − HQ_tea||² / ||h_in − HQ_tea||²
      (分母 detach; 初始 MS=0 → rel=1, 跨层可比, 自归一避免层间数值尺度差).
  α 插值增强 (训练时, 除最低插入层外): h_in = α·HQ_tea + (1−α)·HQ_stu, α~U[0,1]
      逐样本逐层独立采样 — 教上层"下层已把 hidden 推到 stu→tea 途中任意点时,
      继续推完剩下的路" (离线特征 train/inference 复合偏移的一阶对策).
      最低插入层 α≡0 (其输入离线严格成立). 验证一律 α=0.

用法:
  torchrun ... -m transmem.train_layered \
    --data_dir data/..._layered --val_data_dir ... --config transmem/config_layered.json \
    --D 4 --output_dir checkpoints/v3_p6_d4 --divergence forward_kl --epochs 30
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transmem import DistillLoss, FrozenLMHead
from transmem.gate_training import gate_metrics
from transmem.layers import Qwen3RMSNorm
from transmem.layered import LayeredConfig, TransMemLayered
from transmem.train_offpolicy import setup_distributed

_DTYPE = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_args():
    p = argparse.ArgumentParser(description="TransMem-Layer 训练 (v3 计划 6)")
    p.add_argument("--data_dir", required=True, help="stage0 layered 特征目录")
    p.add_argument("--val_data_dir", default=None)
    p.add_argument("--lm_head_path", default=None, help="默认 data_dir/lm_head.pt")
    p.add_argument("--final_norm_path", default=None, help="默认 data_dir/final_norm.pt")
    p.add_argument("--config", default="transmem/config_layered.json")
    p.add_argument("--D", type=int, default=None,
                   help="取 layer_ids 的最后 D 层注入; 与 --inject_layers 二选一")
    p.add_argument("--inject_layers", default=None, help="显式层号, 逗号分隔 (0-based)")
    # 损失
    p.add_argument("--divergence", default="forward_kl",
                   choices=["forward_kl", "reverse_kl", "jsd"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--jsd_beta", type=float, default=0.5)
    p.add_argument("--mse_weight", type=float, default=1.0,
                   help="下层相对残差损失的权重 (顶层 KL 权重恒 1)")
    p.add_argument("--alpha_mix", default="uniform", choices=["uniform", "off"],
                   help="上层输入 α 插值增强; off=全层 α=0 (消融)")
    # 训练
    p.add_argument("--output_dir", default="checkpoints/layered")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--val_interval", type=int, default=1000)
    p.add_argument("--save_interval", type=int, default=5000)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--resume", default=None)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class LayeredDataset(Dataset):
    """预载入 layered stage0 特征; __getitem__ 返回按注入层子集切好的序列.

    返回 (hm [D,N,dim], hq_stu [D,M,dim], hq_tea [D,M,dim]); M 变长由 collate 处理.
    """

    def __init__(self, data_dir: str, inject_layers: list[int],
                 load_dtype: torch.dtype = torch.float32):
        self.data_dir = Path(data_dir)
        self.load_dtype = load_dtype
        with open(self.data_dir / "meta.json") as f:
            self.meta = json.load(f)
        self.dim = self.meta["dim"]
        self.N = self.meta["N"]
        layer_ids = self.meta.get("layer_ids")
        if not layer_ids:
            raise RuntimeError(f"{self.data_dir} 不是 layered stage0 特征 "
                               f"(meta 缺 layer_ids; 需 extract --dump_layers)")
        missing = [l for l in inject_layers if l not in layer_ids]
        if missing:
            raise RuntimeError(f"inject_layers {missing} 不在特征 layer_ids {layer_ids} 内")
        self.inject_layers = list(inject_layers)
        sel = [layer_ids.index(l) for l in inject_layers]     # 特征内的层下标
        D = len(sel)
        store_dtype = _DTYPE.get(self.meta.get("save_dtype", "bfloat16"), torch.bfloat16)

        manifest = self.meta.get("samples")
        if not manifest:
            raise RuntimeError(f"stage0 数据缺 manifest: {self.data_dir}")
        num_samples = len(manifest)
        total_pairs = sum(int(e["M"]) for e in manifest)
        print(f"LayeredDataset: 预载入 {num_samples} 样本 / {total_pairs} 位置 × {D} 层 "
              f"({store_dtype}) from {self.data_dir} ...")

        sel_t = torch.tensor(sel, dtype=torch.long)
        self.cond_mem = torch.empty(num_samples, D, self.N, self.dim, dtype=store_dtype)
        self.hq_stu_flat = torch.empty(total_pairs, D, self.dim, dtype=store_dtype)
        self.hq_tea_flat = torch.empty(total_pairs, D, self.dim, dtype=store_dtype)
        self.sample_offset = torch.empty(num_samples, dtype=torch.long)
        self.sample_len = torch.empty(num_samples, dtype=torch.long)

        cursor = 0
        for s, entry in enumerate(manifest):
            data = torch.load(self.data_dir / entry["file"], map_location="cpu",
                              weights_only=False)
            hs = data["hq_stu_layers"]                        # [K, M, dim]
            ht = data["hq_tea_layers"]
            hm = data["hm_stu_layers"]                        # [K, N, dim]
            M = hs.shape[1]
            self.cond_mem[s] = hm[sel_t].to(store_dtype)
            self.hq_stu_flat[cursor:cursor + M] = hs[sel_t].transpose(0, 1).to(store_dtype)
            self.hq_tea_flat[cursor:cursor + M] = ht[sel_t].transpose(0, 1).to(store_dtype)
            self.sample_offset[s] = cursor
            self.sample_len[s] = M
            cursor += M
            if (s + 1) % 5000 == 0:
                print(f"  ...已载入 {s + 1}/{num_samples}")
        assert cursor == total_pairs, (cursor, total_pairs)
        self.total_pairs = total_pairs
        self.num_samples = num_samples
        print(f"LayeredDataset: 完成, avg M={total_pairs / num_samples:.1f}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        o, M = int(self.sample_offset[idx]), int(self.sample_len[idx])
        hm = self.cond_mem[idx].to(self.load_dtype)                       # [D, N, dim]
        hq_stu = self.hq_stu_flat[o:o + M].to(self.load_dtype).transpose(0, 1)  # [D, M, dim]
        hq_tea = self.hq_tea_flat[o:o + M].to(self.load_dtype).transpose(0, 1)
        return hm, hq_stu, hq_tea


def collate_layered(batch):
    """右 padding 到批内最长 M.
    -> hm [B,D,N,dim], hq_stu [B,D,M,dim], hq_tea [B,D,M,dim], q_mask [B,M]."""
    hm = torch.stack([b[0] for b in batch], dim=0)
    lens = [b[1].shape[1] for b in batch]
    B, m_max = len(batch), max(lens)
    D, _, dim = batch[0][0].shape
    hq_stu = hm.new_zeros(B, D, m_max, dim)
    hq_tea = hm.new_zeros(B, D, m_max, dim)
    q_mask = torch.zeros(B, m_max, dtype=torch.bool)
    for i, (_, s, t) in enumerate(batch):
        hq_stu[i, :, :lens[i]] = s
        hq_tea[i, :, :lens[i]] = t
        q_mask[i, :lens[i]] = True
    return hm, hq_stu, hq_tea, q_mask


def make_dataloader(data_dir, inject_layers, batch_size, num_workers, dtype,
                    shuffle=True, distributed=False):
    dirs = [d.strip() for d in str(data_dir).split(",") if d.strip()]
    dss = [LayeredDataset(d, inject_layers, load_dtype=dtype) for d in dirs]
    ds = dss[0] if len(dss) == 1 else torch.utils.data.ConcatDataset(dss)
    if len(dss) > 1:
        ds.N, ds.dim = dss[0].N, dss[0].dim
        ds.total_pairs = sum(d.total_pairs for d in dss)
        ds.num_samples = sum(d.num_samples for d in dss)
    if distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle, drop_last=True)
    elif shuffle:
        sampler = torch.utils.data.RandomSampler(ds)
    else:
        sampler = torch.utils.data.SequentialSampler(ds)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=True, drop_last=True,
                      collate_fn=collate_layered, persistent_workers=(num_workers > 0))


# ═══════════════════════════════════════════════════════════════════════════
# 冻结 final_norm (顶层 KL 需要: 层输出 → final_norm → lm_head)
# ═══════════════════════════════════════════════════════════════════════════

class FrozenFinalNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.norm = Qwen3RMSNorm(weight.shape[0], eps=eps)
        with torch.no_grad():
            self.norm.weight.copy_(weight)
        for p in self.parameters():
            p.requires_grad = False

    @classmethod
    def from_file(cls, path, device=None, dtype=None):
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(d["weight"].float(), float(d.get("eps", 1e-6)))
        return m.to(device=device, dtype=dtype)

    def forward(self, x):
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════

class LayeredTrainer:
    def __init__(self, args):
        self.args = args
        self.rank, self.world, local_rank = setup_distributed()
        self.is_main = (self.rank == 0)
        if self.world > 1 and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
        self.device = torch.device(args.device)
        self.train_dtype = _DTYPE[args.dtype]

        # inject_layers: --inject_layers 显式 > --D 取 meta layer_ids 尾部
        with open(Path(str(args.data_dir).split(",")[0].strip()) / "meta.json") as f:
            meta = json.load(f)
        layer_ids = meta.get("layer_ids") or []
        if args.inject_layers:
            inject = sorted(int(x) for x in args.inject_layers.split(","))
        elif args.D:
            assert len(layer_ids) >= args.D, f"--D {args.D} > 特征层数 {len(layer_ids)}"
            inject = sorted(layer_ids)[-args.D:]
        else:
            raise ValueError("需要 --D 或 --inject_layers")

        cfg = LayeredConfig.from_json(args.config)
        cfg.inject_layers = sorted(inject)
        cfg.__post_init__()
        assert cfg.n_mem == meta["N"], f"config n_mem={cfg.n_mem} != 特征 N={meta['N']}"
        self.config = cfg
        self.mem = TransMemLayered(cfg).to(self.device, dtype=self.train_dtype)
        if self.world > 1:
            self.net = DDP(self.mem,
                           device_ids=[local_rank] if torch.cuda.is_available() else None,
                           broadcast_buffers=False)
        else:
            self.net = self.mem

        _first = str(args.data_dir).split(",")[0].strip()
        self.lm_head = FrozenLMHead.from_file(
            args.lm_head_path or str(Path(_first) / "lm_head.pt"),
            device=self.device, dtype=self.train_dtype)
        self.final_norm = FrozenFinalNorm.from_file(
            args.final_norm_path or str(Path(_first) / "final_norm.pt"),
            device=self.device, dtype=self.train_dtype)
        self.loss_fn = DistillLoss(divergence=args.divergence,
                                   temperature=args.temperature,
                                   reg_weight=0.0, jsd_beta=args.jsd_beta)

        if self.is_main:
            print(f"TransMemLayered: {self.mem.num_params():,} params, "
                  f"inject_layers={cfg.inject_layers} (D={len(cfg.inject_layers)}), "
                  f"block_depth={cfg.block_depth}"
                  + (f" | DDP x{self.world}" if self.world > 1 else ""))
            print(f"Loss: 顶层 {args.divergence} (T={args.temperature}) + "
                  f"{args.mse_weight}×下层相对残差 | alpha_mix={args.alpha_mix}")

        self.optimizer = torch.optim.AdamW(
            (p for p in self.mem.parameters() if p.requires_grad),
            lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        self.global_step = 0
        self.epoch = 0
        self.best_val = float("inf")
        self.best_step = -1
        self.writer = (SummaryWriter(log_dir=str(Path(args.output_dir) / "tb"))
                       if self.is_main else None)

    # ── 损失 ─────────────────────────────────────────────────────────────
    def compute_loss(self, hm, hq_stu, hq_tea, q_mask, training: bool, full=False):
        """hm [B,D,N,dim], hq_stu/hq_tea [B,D,M,dim], q_mask [B,M].

        α 插值先构造全层 h_in, 再单次调用 net(hm, h_in) 跑全部块 —— DDP 梯度同步
        要求单一 forward 覆盖全部参数 (逐块调用 net.module.blocks[..] 会绕过 allreduce).
        """
        layers = self.config.inject_layers
        lowest, top = layers[0], layers[-1]
        valid = q_mask.to(hm.device)
        net = self.net if torch.is_grad_enabled() else self.mem

        if training and self.args.alpha_mix == "uniform" and len(layers) > 1:
            B, D = hq_stu.shape[0], hq_stu.shape[1]
            alpha = torch.rand(B, D, 1, 1, device=hq_stu.device, dtype=hq_stu.dtype)
            alpha[:, layers.index(lowest)] = 0.0    # 最低插入层输入离线严格成立
            h_in = hq_stu + alpha * (hq_tea - hq_stu)
        else:
            h_in = hq_stu
        proposals = net(hm, h_in)

        total = None
        metrics = {"positions": int(valid.sum())}
        metrics.update(gate_metrics(proposals.ms, proposals.gate, valid))
        rels = []
        for k, l in enumerate(layers):
            tea = hq_tea[:, k]                                  # [B, M, dim]
            proposal = proposals.layer(k)
            hq_prime = self.mem.block(l).correct(h_in[:, k], proposal)

            if l == top:
                s_logits = self.lm_head(self.final_norm(hq_prime[valid]))
                with torch.no_grad():
                    t_logits = self.lm_head(self.final_norm(tea[valid]))
                kl, _ = self.loss_fn(s_logits, t_logits)
                total = kl if total is None else total + kl
                metrics["kl_top"] = float(kl.detach())
                with torch.no_grad():
                    metrics["top1"] = float(
                        (s_logits.argmax(-1) == t_logits.argmax(-1)).float().mean())
                    metrics["ms_norm"] = float(proposal.ms[valid].norm(dim=-1).mean())
                if full:
                    with torch.no_grad():
                        base_logits = self.lm_head(self.final_norm(h_in[:, k][valid]))
                        bd = float(self.loss_fn.divergence_only(base_logits, t_logits))
                    metrics["base_div"] = bd
                    metrics["improve"] = ((bd - metrics["kl_top"]) / bd) if bd > 1e-8 else 0.0
            else:
                num = ((hq_prime - tea)[valid] ** 2).mean()
                den = ((h_in[:, k] - tea)[valid].detach() ** 2).mean().clamp_min(1e-8)
                rel = num / den
                rels.append(float(rel.detach()))
                term = self.args.mse_weight * rel
                total = term if total is None else total + term
        if rels:
            metrics["rel_mean"] = sum(rels) / len(rels)
        metrics["loss"] = float(total.detach())
        return total, metrics

    def train_step(self, batch, full=False):
        self.net.train()
        hm, hq_stu, hq_tea, q_mask = [t.to(self.device) if torch.is_tensor(t) else t
                                      for t in batch]
        loss, metrics = self.compute_loss(hm, hq_stu, hq_tea, q_mask,
                                          training=True, full=full)
        self.optimizer.zero_grad()
        loss.backward()
        if self.args.grad_clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.mem.parameters(), self.args.grad_clip)
            metrics["grad_norm"] = float(gn)
        self.optimizer.step()
        return metrics

    @torch.no_grad()
    def validate(self, dataloader, max_batches=50):
        self.mem.eval()
        agg = {}
        n = 0
        for batch in dataloader:
            hm, hq_stu, hq_tea, q_mask = [t.to(self.device) for t in batch]
            _, m = self.compute_loss(hm, hq_stu, hq_tea, q_mask,
                                     training=False, full=True)
            for k in ("loss", "kl_top", "rel_mean", "top1", "improve"):
                if k in m:
                    agg[f"val_{k}"] = agg.get(f"val_{k}", 0.0) + m[k]
            n += 1
            if n >= max_batches:
                break
        return {k: v / max(n, 1) for k, v in agg.items()}

    # ── 保存/恢复 (对齐 train_offpolicy 语义: latest/best/final + result.json) ──
    @staticmethod
    def _atomic_torch_save(obj, path: Path):
        """先写同目录 .tmp 再 os.replace: spot 抢占若砍在写档中途, 只留 .tmp 残骸,
        正式文件要么旧要么新, 不会出现截断 zip (10218607 实测: 半截 latest.pt
        让 requeue 断点重续全体崩)."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save(self, path, metrics=None, kind="latest"):
        if not self.is_main:
            return
        Path(path).mkdir(parents=True, exist_ok=True)
        base = {"model_state_dict": self.mem.state_dict(),
                "config": self.config.to_dict(), "global_step": self.global_step,
                "epoch": self.epoch}
        if metrics:
            base["metrics"] = metrics
        self._atomic_torch_save(
            dict(base, optimizer_state_dict=self.optimizer.state_dict()),
            Path(path) / "latest.pt")
        names = ["latest.pt"]
        if kind == "best":
            self._atomic_torch_save(base, Path(path) / "best.pt")
            names.append("best.pt(model-only)")
        elif kind == "final":
            fn = f"step_{self.global_step:07d}.pt"
            self._atomic_torch_save(base, Path(path) / fn)
            names.append(f"{fn}(model-only)")
        print(f"  Checkpoint saved: {', '.join(names)} (step {self.global_step})")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.mem.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.epoch = ckpt.get("epoch", 0)
        self._load_result()
        if self.is_main:
            print(f"Resumed: step={self.global_step}, best_val={self.best_val:.6f}")

    def _result_path(self):
        return Path(self.args.output_dir) / "result.json"

    def _load_result(self):
        p = self._result_path()
        if p.exists():
            try:
                r = json.loads(p.read_text())
                self.best_val = r.get("best_val", float("inf"))
                self.best_step = r.get("best_step", -1)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  警告: 读取 {p} 失败 ({e})")

    def _save_result(self, extra=None):
        if not self.is_main:
            return
        r = {"best_val": self.best_val, "best_step": self.best_step,
             "global_step": self.global_step, "epoch": self.epoch,
             "inject_layers": self.config.inject_layers,
             "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        if extra:
            r.update(extra)
        p = self._result_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(r, indent=2, ensure_ascii=False))
        tmp.replace(p)

    def _set_lr(self, step, total_steps):
        warmup = self.args.warmup_steps
        if step < warmup:
            lr = self.args.lr * step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(total_steps - warmup, 1)
            lr = self.args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def run(self):
        args = self.args
        train_dl = make_dataloader(args.data_dir, self.config.inject_layers,
                                   args.batch_size, args.num_workers, self.train_dtype,
                                   shuffle=True, distributed=(self.world > 1))
        val_dl = (make_dataloader(args.val_data_dir, self.config.inject_layers,
                                  args.batch_size, args.num_workers, self.train_dtype,
                                  shuffle=False)
                  if (args.val_data_dir and self.is_main) else None)
        total_steps = args.max_steps or (args.epochs * len(train_dl))
        if self.is_main:
            print(f"\n训练: steps/epoch≈{len(train_dl)}, total≈{total_steps}, "
                  f"bs={args.batch_size}×{self.world} rank, lr={args.lr}")
            print("=" * 72)

        running = 0.0
        t0 = time.time()
        while self.global_step < total_steps:
            self.epoch += 1
            if isinstance(train_dl.sampler, DistributedSampler):
                train_dl.sampler.set_epoch(self.epoch)
            for batch in train_dl:
                if self.global_step >= total_steps:
                    break
                self._set_lr(self.global_step, total_steps)
                full = ((self.global_step + 1) % args.log_interval == 0)
                m = self.train_step(batch, full=full)
                self.global_step += 1
                running += m["loss"]
                if self.is_main and self.global_step % args.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    sps = args.log_interval / max(time.time() - t0, 1e-3)
                    print(f"  step {self.global_step:7d}/{total_steps} | "
                          f"loss {running/args.log_interval:.6f} | "
                          f"kl_top {m.get('kl_top', 0.0):.4f} | "
                          f"rel {m.get('rel_mean', 0.0):.4f} | "
                          f"top1 {m.get('top1', 0.0):.3f} | "
                          f"improve {m.get('improve', 0.0):.3f} | lr {lr:.2e} | "
                          f"{sps:.1f} it/s")
                    for k in ("kl_top", "rel_mean", "top1", "ms_norm", "grad_norm",
                              "base_div", "improve"):
                        if k in m:
                            self.writer.add_scalar(f"train/{k}", m[k], self.global_step)
                    self.writer.add_scalar("train/loss", running / args.log_interval,
                                           self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    running = 0.0
                    t0 = time.time()
                if val_dl and self.global_step % args.val_interval == 0:
                    vm = self.validate(val_dl)
                    print(f"  --- VAL step {self.global_step}: " +
                          " ".join(f"{k}={v:.4f}" for k, v in vm.items()) + " ---")
                    for k, v in vm.items():
                        self.writer.add_scalar(f"val/{k.replace('val_', '')}", v,
                                               self.global_step)
                    if vm.get("val_loss", float("inf")) < self.best_val:
                        self.best_val = vm["val_loss"]
                        self.best_step = self.global_step
                        self.save(args.output_dir, {"val_loss": self.best_val},
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
            dist.barrier()
            dist.destroy_process_group()
        if self.is_main:
            print("=" * 72)
            print(f"✅ layered 训练完成: {self.global_step} steps, "
                  f"best_val={self.best_val:.6f}@{self.best_step}")


def main():
    args = parse_args()
    trainer = LayeredTrainer(args)
    resume = args.resume
    if resume is None:
        auto = Path(args.output_dir) / "latest.pt"
        if auto.exists():
            resume = str(auto)
    if resume:
        try:
            trainer.load(resume)
        except RuntimeError as e:
            # 抢占砍在写档中途会留截断 zip ("failed finding central directory") —
            # 挪走坏档从头训, 好过全体 rank 崩死 (老 save 的遗留伤; 新 save 已原子写)
            bad = str(Path(resume)) + ".corrupt"
            if trainer.is_main:
                print(f"⚠️ resume 档损坏 ({e}); 挪到 {bad}, 从头训练")
                try:
                    os.replace(resume, bad)
                except OSError:
                    pass
            trainer.global_step = 0
            trainer.epoch = 0
    trainer.run()


if __name__ == "__main__":
    main()
