#!/usr/bin/env python3
"""
v3.2: TransMem-Layer 在环训练 — LLM 层进训练环, 治 P6 离线方案的复合失配.

P6 判读: "离线逐层回归 + α 插值" 在 D≥6 连训练集都坍塌 (.375), 根因是运行时失配 —
低层注入后上层收到的 hidden 偏离离线特征分布, α 插值只保 stu→tea 线段的一阶近似.
本方案让学生前向全程走真实 LLM 层 (LayeredRollout.teacher_forced_forward), 上层
看到的就是下层注入后的真实分布, 梯度经冻结 LLM 上层流回各块 (深度信用分配),
失配从根上不存在. 唯一目标 = 顶端 forward-KL (P6 的逐层回归拐杖不再需要).

两种策略 (--policy):
  tf        教师强制在环: 轨迹 = stage0 教师答案 (answer_ids), 目标 = 离线教师
            logits (lm_head(hq_tea), post-norm). 与推理仅剩轨迹曝光偏差
            (final-hidden 系已证可控). 每微步 ≈ 一次 ~30k token 前向 + 后 D 层反传.
  onpolicy  DAgger 式: 当前块 rollout 学生自己的轨迹 (no-grad) → 教师 (C_S) 对该
            轨迹在环重打分 → 同轨迹 tf 梯度前向 → KL. 连轨迹偏差也消掉, 每微步
            贵 ~2.5× (prefill + M 步解码 + 教师短前向). 需要记录带 cs_text.

与 train_layered (离线) 的锚点关系: D=1 时 tf 目标与离线目标严格等价 (最低注入层
的离线输入本就成立), D>1 起两者分离 — 干净的对照设计.

数据: stage0 基础特征目录 (answer_ids + hq_tea; 不需要 layered8 逐层特征, HM/HQ
在环现取) + 原始数据文件 (context/question/cs_text 按 sample_idx 对齐重渲染).

DDP: 注入发生在 LLM hook 内逐块调用, 绕不开 DDP reducer 的单 forward 约束
→ 手动同步: 初始 broadcast 参数, 每优化步 all_reduce(AVG) 梯度. 坏梯度跳步的
判定放在 allreduce 之后 (各 rank 梯度一致 → 决策天然同步, 不会参数漂移).

用法:
  torchrun --standalone --nproc_per_node=8 -m transmem.train_inloop \
    --data_dir data/hotpotqa_data/<model>/stage0_train_short200 \
    --data_path data/hotpotqa-benchmark/hotpotqa-agentmem/hotpotqa_train_32k.parquet \
    --data_format hotpotqa-agentmem \
    --val_data_dir .../stage0_dev_short200 --val_data_path .../hotpotqa_dev.parquet \
    --model_path <Qwen3-4B> --config transmem/config_layered.json \
    --D 8 --policy tf --output_dir checkpoints/v3_2_inloop_tf_d8
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
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transmem import DistillLoss
from transmem.layered import LayeredConfig, TransMemLayered, LayeredRollout
from transmem.extract_features import load_records, build_chat_prompt_ids
from transmem.train_offpolicy import setup_distributed


def parse_args():
    p = argparse.ArgumentParser(description="TransMem-Layer 在环训练 (v3.2)")
    # 数据
    p.add_argument("--data_dir", required=True, help="stage0 特征目录 (answer_ids+hq_tea)")
    p.add_argument("--data_path", required=True, help="原始数据文件 (context/question)")
    p.add_argument("--data_format", default="hotpotqa-agentmem")
    p.add_argument("--val_data_dir", default=None)
    p.add_argument("--val_data_path", default=None)
    p.add_argument("--val_max", type=int, default=128, help="每次 val 的样本上限")
    p.add_argument("--max_samples", type=int, default=None, help="训练集截断 (调试)")
    # 模型/架构
    p.add_argument("--model_path", required=True)
    p.add_argument("--attn_impl", default="sdpa")
    p.add_argument("--config", default="transmem/config_layered.json")
    p.add_argument("--D", type=int, default=None, help="注入 LLM 最后 D 层")
    p.add_argument("--inject_layers", default=None, help="显式层号, 逗号分隔 (0-based)")
    # 目标
    p.add_argument("--policy", default="tf", choices=["tf", "onpolicy"])
    p.add_argument("--divergence", default="forward_kl",
                   choices=["forward_kl", "reverse_kl", "jsd"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--jsd_beta", type=float, default=0.5)
    p.add_argument("--sample_temp", type=float, default=0.0,
                   help="onpolicy rollout 采样温度; 0=贪心")
    p.add_argument("--max_answer_tokens", type=int, default=200,
                   help="onpolicy rollout 生成上限")
    # 训练
    p.add_argument("--output_dir", default="checkpoints/inloop")
    p.add_argument("--grad_accum", type=int, default=4, help="每 rank 微步数/优化步")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=25)
    p.add_argument("--val_interval", type=int, default=250)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--resume", default=None)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Dataset: stage0 manifest × 原始记录, 按 sample_idx 对齐
# ═══════════════════════════════════════════════════════════════════════════

class InLoopDataset(Dataset):
    """__getitem__ -> dict(context, question, cs_text, answer_ids [M], hq_tea [M,dim]).

    context/question 来自原始数据 (与 stage0/eval 同 loader 同渲染), 特征只取
    answer_ids (教师轨迹) 与 hq_tea (post-norm 教师目标). records 可注入 (测试)."""

    def __init__(self, data_dir: str, data_path: str, data_format: str,
                 policy: str = "tf", max_samples: int | None = None,
                 records: list | None = None):
        self.data_dir = Path(data_dir)
        with open(self.data_dir / "meta.json") as f:
            self.meta = json.load(f)
        self.N = self.meta["N"]
        if records is None:
            records = load_records(data_path, data_format, None)
        self.records = records

        manifest = self.meta.get("samples")
        if not manifest:
            raise RuntimeError(f"stage0 数据缺 manifest: {self.data_dir}")
        entries, drop_ctx, drop_cs = [], 0, 0
        for e in manifest:
            si = int(e["sample_idx"])
            if si >= len(records):
                drop_ctx += 1
                continue
            rec = records[si]
            if not rec.get("context") or not rec.get("question"):
                drop_ctx += 1
                continue
            if policy == "onpolicy" and not rec.get("cs_text"):
                drop_cs += 1
                continue
            entries.append((e["file"], si))
        if max_samples:
            entries = entries[:max_samples]
        self.entries = entries
        print(f"InLoopDataset: {len(entries)} 样本 from {self.data_dir} "
              f"(drop: 无context {drop_ctx}, 无cs_text {drop_cs})")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i: int):
        f, si = self.entries[i]
        d = torch.load(self.data_dir / f, map_location="cpu", weights_only=False)
        rec = self.records[si]
        return {"context": rec["context"], "question": rec["question"],
                "cs_text": rec.get("cs_text", ""),
                "answer_ids": d["answer_ids"], "hq_tea": d["hq_tea"]}


def collate_first(batch):
    """batch_size=1 (变长 ~30k token 序列, 不 padding); 直接取样本 dict."""
    return batch[0]


# ═══════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════

class InLoopTrainer:
    def __init__(self, args, model=None, tokenizer=None):
        self.args = args
        self.rank, self.world, local_rank = setup_distributed()
        self.is_main = (self.rank == 0)
        if self.world > 1 and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
        self.device = torch.device(args.device)

        # 冻结 LLM (测试可注入现成 model/tokenizer)
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16,
                attn_implementation=args.attn_impl).to(self.device).eval()
        self.model = model
        self.tok = tokenizer
        self.model.requires_grad_(False)
        n_layers = len(self.model.model.layers)

        if args.inject_layers:
            inject = sorted(int(x) for x in args.inject_layers.split(","))
        elif args.D:
            assert args.D <= n_layers
            inject = list(range(n_layers - args.D, n_layers))
        else:
            raise ValueError("需要 --D 或 --inject_layers")
        cfg = LayeredConfig.from_json(args.config)
        cfg.inject_layers = inject
        cfg.__post_init__()
        self.config = cfg
        self.mem = TransMemLayered(cfg).to(self.device, dtype=torch.float32).train()
        if self.world > 1:
            for p in self.mem.parameters():
                dist.broadcast(p.data, src=0)

        self.rollout = LayeredRollout(self.model, self.tok, self.device, self.mem)
        self.loss_fn = DistillLoss(divergence=args.divergence,
                                   temperature=args.temperature,
                                   reg_weight=0.0, jsd_beta=args.jsd_beta)
        self.optimizer = torch.optim.AdamW(
            (p for p in self.mem.parameters() if p.requires_grad),
            lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        self.global_step = 0            # 优化步 (非微步)
        self.epoch = 0
        self.best_val = float("inf")
        self.best_step = -1
        self.writer = (SummaryWriter(log_dir=str(Path(args.output_dir) / "tb"))
                       if self.is_main else None)
        if self.is_main:
            print(f"InLoop[{args.policy}]: {self.mem.num_params():,} params, "
                  f"inject={cfg.inject_layers} (D={len(cfg.inject_layers)}), "
                  f"LLM {n_layers} 层冻结"
                  + (f" | 手动DDP x{self.world}" if self.world > 1 else ""))

    # ── 单样本损失 (tf / onpolicy 共用梯度前向) ─────────────────────────
    def micro_loss(self, item: dict, policy: str):
        """-> (loss 标量[平均每位置 KL], M, metrics) 或 None (空轨迹)."""
        ctx, q = item["context"], item["question"]
        cq_ids = build_chat_prompt_ids(self.tok, ctx, q, self.device)
        len_cq = cq_ids.shape[1]
        len_cl = self.tok(ctx, return_tensors="pt",
                          add_special_tokens=False).input_ids.shape[1]

        if policy == "tf":
            ans = item["answer_ids"].tolist()
            M = len(ans)
            if M < 1:
                return None
            with torch.no_grad():
                hq_tea = item["hq_tea"].to(self.device, self.model.dtype)
                t_logits = self.model.lm_head(hq_tea)                # [M, vocab]
        else:                                                        # onpolicy
            with torch.no_grad():
                ans = self.rollout.generate_from_ids(
                    cq_ids, len_cl, max_new=self.args.max_answer_tokens,
                    sample=self.args.sample_temp > 0,
                    temperature=self.args.sample_temp)
            M = len(ans)
            if M < 1:
                return None
            # 教师 (C_S) 对学生轨迹在环重打分: hidden@len_ts-1+i 预测位置 i
            ts_ids = build_chat_prompt_ids(self.tok, item["cs_text"], q, self.device)
            len_ts = ts_ids.shape[1]
            t_full = (torch.cat([ts_ids, torch.tensor([ans[:-1]], device=self.device,
                                                      dtype=ts_ids.dtype)], dim=1)
                      if M > 1 else ts_ids)
            with torch.no_grad():
                t_h = self.model.model(
                    input_ids=t_full, attention_mask=torch.ones_like(t_full),
                    use_cache=False).last_hidden_state[0, len_ts - 1: len_ts + M - 1]
                t_logits = self.model.lm_head(t_h)

        full_ids = (torch.cat([cq_ids, torch.tensor([ans[:-1]], device=self.device,
                                                    dtype=cq_ids.dtype)], dim=1)
                    if M > 1 else cq_ids)
        h_q = self.rollout.teacher_forced_forward(full_ids, len_cl, len_cq, M)
        s_logits = self.model.lm_head(h_q)                           # [M, vocab]
        loss, _ = self.loss_fn(s_logits.float(), t_logits.float())
        with torch.no_grad():
            top1 = float((s_logits.argmax(-1) == t_logits.argmax(-1)).float().mean())
        return loss, M, {"top1": top1, "tokens": full_ids.shape[1]}

    # ── 手动 DDP: allreduce 梯度后统一步进 ──────────────────────────────
    def sync_and_step(self):
        if self.world > 1:
            for p in self.mem.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)     # 保证各 rank allreduce 同一集合
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        gn = torch.nn.utils.clip_grad_norm_(self.mem.parameters(), self.args.grad_clip)
        stepped = bool(torch.isfinite(gn))
        if stepped:                                  # allreduce 后判定 → 各 rank 同步
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(gn), stepped

    # ── 验证: 恒 tf (确定性, 与训练目标同构), 样本按 rank 分片 ──────────
    @torch.no_grad()
    def validate(self, val_ds):
        self.mem.eval()
        n = min(self.args.val_max, len(val_ds))
        kl_sum, pos, top1_sum = 0.0, 0, 0.0
        for i in range(self.rank, n, self.world):
            r = self.micro_loss(val_ds[i], policy="tf")
            if r is None:
                continue
            loss, M, m = r
            kl_sum += float(loss) * M
            pos += M
            top1_sum += m["top1"] * M
        t = torch.tensor([kl_sum, float(pos), top1_sum], device=self.device)
        if self.world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        self.mem.train()
        denom = max(float(t[1]), 1.0)
        return {"val_loss": float(t[0]) / denom, "val_top1": float(t[2]) / denom,
                "val_positions": int(t[1])}

    # ── 保存/恢复 (格式与 train_layered 一致 → evaluate.py 直接分发) ────
    @staticmethod
    def _atomic_torch_save(obj, path: Path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save(self, path, metrics=None, kind="latest"):
        if not self.is_main:
            return
        Path(path).mkdir(parents=True, exist_ok=True)
        base = {"model_state_dict": self.mem.state_dict(),
                "config": self.config.to_dict(),
                "train_mode": f"inloop_{self.args.policy}",
                "global_step": self.global_step, "epoch": self.epoch}
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
             "policy": self.args.policy,
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

    # ── 主循环 ──────────────────────────────────────────────────────────
    def run(self):
        args = self.args
        train_ds = InLoopDataset(args.data_dir, args.data_path, args.data_format,
                                 policy=args.policy, max_samples=args.max_samples)
        assert self.config.n_mem == train_ds.N, \
            f"config n_mem={self.config.n_mem} != 特征 N={train_ds.N}"
        val_ds = (InLoopDataset(args.val_data_dir, args.val_data_path,
                                args.data_format, policy="tf")
                  if args.val_data_dir else None)
        if self.world > 1:
            sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        else:
            sampler = torch.utils.data.RandomSampler(train_ds)
        dl = DataLoader(train_ds, batch_size=1, sampler=sampler,
                        num_workers=args.num_workers, collate_fn=collate_first,
                        pin_memory=False, drop_last=True,
                        persistent_workers=(args.num_workers > 0))
        steps_per_epoch = len(dl) // args.grad_accum
        total_steps = args.max_steps or (args.epochs * steps_per_epoch)
        if self.is_main:
            print(f"\n在环训练[{args.policy}]: {len(train_ds)} 样本, 微步/epoch={len(dl)}, "
                  f"优化步/epoch≈{steps_per_epoch}, total≈{total_steps}, "
                  f"全局批={self.world}x{args.grad_accum}, lr={args.lr}")
            print("=" * 72)

        # step0 基线 val: 零初始化恒等 → 这就是 student(C_L) vs teacher 的原始 KL
        if val_ds and self.global_step == 0:
            vm = self.validate(val_ds)
            if self.is_main:
                print(f"  --- VAL step 0 (零初始化基线): "
                      + " ".join(f"{k}={v:.4f}" for k, v in vm.items()) + " ---")
                for k, v in vm.items():
                    self.writer.add_scalar(f"val/{k.replace('val_', '')}", v, 0)

        micro_in_step = 0
        run_loss, run_top1, run_tok, n_micro = 0.0, 0.0, 0, 0
        t0 = time.time()
        while self.global_step < total_steps:
            self.epoch += 1
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(self.epoch)
            for item in dl:
                if self.global_step >= total_steps:
                    break
                r = self.micro_loss(item, policy=args.policy)
                if r is not None:
                    loss, M, m = r
                    (loss / args.grad_accum).backward()
                    run_loss += float(loss.detach())
                    run_top1 += m["top1"]
                    run_tok += m["tokens"]
                    n_micro += 1
                micro_in_step += 1
                if micro_in_step < args.grad_accum:
                    continue
                micro_in_step = 0
                self._set_lr(self.global_step, total_steps)
                gn, stepped = self.sync_and_step()
                self.global_step += 1
                if not stepped and self.is_main:
                    print(f"  ⚠️ step {self.global_step}: grad_norm={gn} 非有限, 跳过该步")

                if self.is_main and self.global_step % args.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    dt = max(time.time() - t0, 1e-3)
                    sps = n_micro / dt
                    avg = run_loss / max(n_micro, 1)
                    print(f"  step {self.global_step:6d}/{total_steps} | "
                          f"kl {avg:.4f} | top1 {run_top1/max(n_micro,1):.3f} | "
                          f"grad {gn:.3f} | lr {lr:.2e} | "
                          f"{sps:.2f} samp/s/rank | {run_tok/max(n_micro,1):.0f} tok/samp")
                    self.writer.add_scalar("train/kl", avg, self.global_step)
                    self.writer.add_scalar("train/top1", run_top1 / max(n_micro, 1),
                                           self.global_step)
                    self.writer.add_scalar("train/grad_norm", gn, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    run_loss, run_top1, run_tok, n_micro = 0.0, 0.0, 0, 0
                    t0 = time.time()
                if val_ds and self.global_step % args.val_interval == 0:
                    vm = self.validate(val_ds)
                    if self.is_main:
                        print(f"  --- VAL step {self.global_step}: "
                              + " ".join(f"{k}={v:.4f}" for k, v in vm.items()) + " ---")
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
        final = self.validate(val_ds) if val_ds else {}
        self.save(args.output_dir, final, kind="final")
        self._save_result({"final_metrics": final, "done": True})
        if self.writer:
            self.writer.close()
        if self.world > 1:
            dist.barrier()
            dist.destroy_process_group()
        if self.is_main:
            print("=" * 72)
            print(f"✅ 在环训练完成: {self.global_step} steps, "
                  f"best_val={self.best_val:.6f}@{self.best_step}")


def main():
    args = parse_args()
    trainer = InLoopTrainer(args)
    resume = args.resume
    if resume is None:
        auto = Path(args.output_dir) / "latest.pt"
        if auto.exists():
            resume = str(auto)
    if resume:
        try:
            trainer.load(resume)
        except RuntimeError as e:
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
