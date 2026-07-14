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
    --D 4 --S 32 --policy tf --output_dir checkpoints/v4_inloop_s32_d4
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
from transmem.checkpoints import (
    load_legacy_gate_state,
    materialize_migrated_gate_checkpoint,
)
from transmem.layered import (
    LayeredConfig,
    LayeredRollout,
    TransMemLayered,
    resolve_inject_layers,
)
from transmem.extract_features import load_records, build_chat_prompt_ids
from transmem.gate_training import (
    build_gate_optimizer,
    clear_base_grads_for_gate_only,
    gate_metrics,
    gate_prior_coefficient,
    gate_prior_loss,
    is_gate_only_phase,
    set_gate_optimizer_lrs,
    validate_dynamic_resume_checkpoint,
    validate_gate_training_options,
)
from transmem.train_offpolicy import seed_everything, setup_distributed


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
    p.add_argument("--D", type=int, default=None, help="注入窗口深度")
    p.add_argument(
        "--S",
        type=int,
        default=None,
        help="注入窗口的独占上界; S=32,D=4 表示层 28..31（默认取 LLM 总层数）",
    )
    p.add_argument("--inject_layers", default=None, help="显式层号, 逗号分隔 (0-based)")
    p.add_argument("--init_scheme", default="scratch_joint",
                   choices=["legacy_gate", "scratch_joint"])
    p.add_argument("--init_checkpoint", default=None,
                   help="legacy_gate 的 fixed-gate layered 父 checkpoint")
    p.add_argument("--gate_calibration_steps", type=int, default=0)
    p.add_argument("--joint_finetune_steps", type=int, default=None)
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
    p.add_argument("--gate_lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--gate_prior_weight", type=float, default=0.0)
    p.add_argument("--gate_prior_anneal_steps", type=int, default=0)
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
    p.add_argument("--seed", type=int, default=None,
                   help="显式固定 TransMem 初始化与数据顺序; 默认保持历史随机行为")
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
        seed_everything(getattr(args, "seed", None))

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

        inject = resolve_inject_layers(
            n_layers=n_layers,
            depth=args.D,
            stop=getattr(args, "S", None),
            explicit=args.inject_layers or None,
        )
        cfg = LayeredConfig.from_json(args.config)
        cfg.inject_layers = inject
        cfg.__post_init__()
        self.config = cfg
        init_scheme = getattr(args, "init_scheme", "scratch_joint")
        init_checkpoint = getattr(args, "init_checkpoint", None)
        gate_calibration_steps = getattr(args, "gate_calibration_steps", 0)
        joint_finetune_steps = getattr(args, "joint_finetune_steps", None)
        validate_gate_training_options(
            init_scheme=init_scheme,
            init_checkpoint=init_checkpoint,
            gate_mode=cfg.gate_mode,
            gate_calibration_steps=gate_calibration_steps,
            joint_finetune_steps=joint_finetune_steps,
        )
        self.mem = TransMemLayered(cfg).to(self.device, dtype=torch.float32).train()
        self.parent_checkpoint = None
        self._legacy_regression_pending = False
        if init_scheme == "legacy_gate":
            checkpoint = torch.load(
                init_checkpoint, map_location="cpu", weights_only=False)
            load_legacy_gate_state(self.mem, checkpoint)
            self.parent_checkpoint = str(Path(init_checkpoint).expanduser().resolve())
            del checkpoint
            migrated_path = Path(args.output_dir) / "migrated_init.pt"
            if self.is_main:
                materialize_migrated_gate_checkpoint(
                    self.mem,
                    migrated_path,
                    parent_checkpoint=self.parent_checkpoint,
                    gate_calibration_steps=gate_calibration_steps,
                    joint_finetune_steps=joint_finetune_steps,
                    seed=getattr(args, "seed", None),
                )
            if self.world > 1:
                dist.barrier()
            migrated = torch.load(
                migrated_path, map_location="cpu", weights_only=False)
            self.mem.load_state_dict(migrated["model_state_dict"], strict=True)
            del migrated
            self._legacy_regression_pending = True
        if self.world > 1:
            for p in self.mem.parameters():
                dist.broadcast(p.data, src=0)

        self.rollout = LayeredRollout(self.model, self.tok, self.device, self.mem)
        self.loss_fn = DistillLoss(divergence=args.divergence,
                                   temperature=args.temperature,
                                   reg_weight=0.0, jsd_beta=args.jsd_beta)
        self.optimizer = build_gate_optimizer(
            self.mem,
            base_lr=args.lr,
            gate_lr=getattr(args, "gate_lr", None),
            weight_decay=args.weight_decay,
        )
        self.global_step = 0            # 优化步 (非微步)
        self.epoch = 0
        self.best_val = float("inf")
        self.best_step = -1
        self.best_gate_val = float("inf")
        self.best_gate_step = -1
        self.joint_phase_initialized = not (
            init_scheme == "legacy_gate" and gate_calibration_steps > 0)
        self.writer = (SummaryWriter(log_dir=str(Path(args.output_dir) / "tb"))
                       if self.is_main else None)
        if self.is_main:
            print(f"InLoop[{args.policy}]: {self.mem.num_params():,} params, "
                  f"inject={cfg.inject_layers} (D={len(cfg.inject_layers)}), "
                  f"LLM {n_layers} 层冻结, gate={cfg.gate_mode}, init={init_scheme}"
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
        h_q, proposals = self.rollout.teacher_forced_forward(
            full_ids, len_cl, len_cq, M, return_proposals=True)
        s_logits = self.model.lm_head(h_q)                           # [M, vocab]
        if self._legacy_regression_pending:
            with torch.no_grad():
                if not torch.equal(proposals.gate, torch.ones_like(proposals.gate)):
                    raise RuntimeError("legacy_gate step-0 回归失败: gate 不严格等于 1")
                legacy_h_q = self.rollout.teacher_forced_forward(
                    full_ids, len_cl, len_cq, M, force_gate_one=True)
                legacy_logits = self.model.lm_head(legacy_h_q)
                tolerance = (1e-2 if s_logits.dtype == torch.bfloat16 else 1e-5)
                if not torch.allclose(
                        s_logits.detach(), legacy_logits,
                        atol=tolerance, rtol=tolerance):
                    error = float((s_logits.detach() - legacy_logits).abs().max())
                    raise RuntimeError(
                        f"legacy_gate step-0 logits 回归失败: max_error={error:.3e}")
            self._legacy_regression_pending = False
            if self.is_main:
                print("legacy_gate step-0 logits regression: PASS")
        task_loss, _ = self.loss_fn(s_logits.float(), t_logits.float())
        prior = gate_prior_loss(proposals.gate)
        prior_coef = gate_prior_coefficient(
            step=self.global_step,
            weight=getattr(self.args, "gate_prior_weight", 0.0),
            anneal_steps=getattr(self.args, "gate_prior_anneal_steps", 0),
        )
        loss = task_loss + prior_coef * prior
        with torch.no_grad():
            top1 = float((s_logits.argmax(-1) == t_logits.argmax(-1)).float().mean())
            metrics = gate_metrics(proposals.ms, proposals.gate)
        metrics.update({
            "top1": top1,
            "tokens": full_ids.shape[1],
            "task_loss": float(task_loss.detach()),
            "gate_prior": float(prior.detach()),
            "gate_prior_coef": prior_coef,
        })
        return loss, M, metrics

    # ── 手动 DDP: allreduce 梯度后统一步进 ──────────────────────────────
    def sync_and_step(self):
        if self.world > 1:
            for p in self.mem.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)     # 保证各 rank allreduce 同一集合
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        clear_base_grads_for_gate_only(
            self.optimizer,
            gate_only=is_gate_only_phase(
                getattr(self.args, "init_scheme", "scratch_joint"),
                self.global_step,
                getattr(self.args, "gate_calibration_steps", 0),
            ),
        )
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
        gate_sum = gate_std_sum = delta_sum = 0.0
        for i in range(self.rank, n, self.world):
            r = self.micro_loss(val_ds[i], policy="tf")
            if r is None:
                continue
            loss, M, m = r
            # Select A1/A2 by held-out task loss, never by the training-only
            # gate prior that intentionally favors g≈1 early in training.
            kl_sum += m.get("task_loss", float(loss)) * M
            pos += M
            top1_sum += m["top1"] * M
            gate_sum += m["gate_mean"] * M
            gate_std_sum += m["gate_std"] * M
            delta_sum += m["delta_norm"] * M
        t = torch.tensor(
            [kl_sum, float(pos), top1_sum, gate_sum, gate_std_sum, delta_sum],
            device=self.device)
        if self.world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        self.mem.train()
        denom = max(float(t[1]), 1.0)
        return {
            "val_loss": float(t[0]) / denom,
            "val_top1": float(t[2]) / denom,
            "val_gate_mean": float(t[3]) / denom,
            "val_gate_std": float(t[4]) / denom,
            "val_delta_norm": float(t[5]) / denom,
            "val_positions": int(t[1]),
        }

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
                "global_step": self.global_step, "epoch": self.epoch,
                "seed": getattr(self.args, "seed", None),
                "init_scheme": getattr(self.args, "init_scheme", "scratch_joint"),
                "parent_checkpoint": self.parent_checkpoint,
                "gate_calibration_steps": getattr(
                    self.args, "gate_calibration_steps", 0),
                "joint_finetune_steps": getattr(
                    self, "resolved_joint_finetune_steps",
                    getattr(self.args, "joint_finetune_steps", None)),
                "joint_phase_initialized": self.joint_phase_initialized}
        if metrics:
            base["metrics"] = metrics
        self._atomic_torch_save(
            dict(base, optimizer_state_dict=self.optimizer.state_dict()),
            Path(path) / "latest.pt")
        names = ["latest.pt"]
        if kind in {"best", "best_and_gate"}:
            self._atomic_torch_save(base, Path(path) / "best.pt")
            names.append("best.pt(model-only)")
        if kind in {"gate_best", "best_and_gate"}:
            self._atomic_torch_save(base, Path(path) / "gate_only_best.pt")
            names.append("gate_only_best.pt(model-only)")
        if kind == "calibrated":
            self._atomic_torch_save(base, Path(path) / "gate_only.pt")
            names.append("gate_only.pt(model-only)")
        elif kind == "final":
            fn = f"step_{self.global_step:07d}.pt"
            self._atomic_torch_save(base, Path(path) / fn)
            names.append(f"{fn}(model-only)")
        print(f"  Checkpoint saved: {', '.join(names)} (step {self.global_step})")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        validate_dynamic_resume_checkpoint(
            ckpt,
            config=self.config.to_dict(),
            init_scheme=getattr(self.args, "init_scheme", "scratch_joint"),
            parent_checkpoint=self.parent_checkpoint,
            gate_calibration_steps=getattr(
                self.args, "gate_calibration_steps", 0),
            joint_finetune_steps=getattr(
                self.args, "joint_finetune_steps", None),
            seed=getattr(self.args, "seed", None),
        )
        self.mem.load_state_dict(ckpt["model_state_dict"], strict=True)
        self._legacy_regression_pending = False
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.epoch = ckpt.get("epoch", 0)
        calibration_steps = getattr(self.args, "gate_calibration_steps", 0)
        self.joint_phase_initialized = bool(ckpt.get(
            "joint_phase_initialized",
            calibration_steps == 0 or self.global_step > calibration_steps,
        ))
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
                self.best_gate_val = r.get("best_gate_val", float("inf"))
                self.best_gate_step = r.get("best_gate_step", -1)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  警告: 读取 {p} 失败 ({e})")

    def _save_result(self, extra=None):
        if not self.is_main:
            return
        r = {"best_val": self.best_val, "best_step": self.best_step,
             "best_gate_val": self.best_gate_val,
             "best_gate_step": self.best_gate_step,
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

    def _start_joint_phase(self) -> None:
        """Reload A1's held-out best model and start A2 with fresh Adam state."""
        if self.joint_phase_initialized or self.resolved_joint_finetune_steps <= 0:
            return
        if self.world > 1:
            dist.barrier()
        source_name = None
        if self.is_main:
            output = Path(self.args.output_dir)
            source = output / "gate_only_best.pt"
            if not source.exists():
                source = output / "gate_only.pt"
            if not source.exists():
                raise RuntimeError(
                    "A2 启动失败: gate_only_best.pt 和 gate_only.pt 均不存在")
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            self.mem.load_state_dict(checkpoint["model_state_dict"], strict=True)
            source_name = source.name
        if self.world > 1:
            for parameter in self.mem.parameters():
                dist.broadcast(parameter.data, src=0)
        self.optimizer = build_gate_optimizer(
            self.mem,
            base_lr=self.args.lr,
            gate_lr=getattr(self.args, "gate_lr", None),
            weight_decay=self.args.weight_decay,
        )
        self.joint_phase_initialized = True
        if self.is_main:
            self.save(
                self.args.output_dir,
                {"phase": "joint_start", "a1_source": source_name},
            )
            print(f"  A2 joint fine-tune 从 {source_name} 启动 (optimizer 已重置)")
        if self.world > 1:
            dist.barrier()

    def _set_lr(self, step, total_steps):
        warmup = self.args.warmup_steps
        if step < warmup:
            factor = step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(total_steps - warmup, 1)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        set_gate_optimizer_lrs(
            self.optimizer,
            lr_factor=factor,
            gate_only=is_gate_only_phase(
                getattr(self.args, "init_scheme", "scratch_joint"),
                step,
                getattr(self.args, "gate_calibration_steps", 0),
            ),
        )

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
        default_total_steps = args.max_steps or (args.epochs * steps_per_epoch)
        calibration_steps = getattr(args, "gate_calibration_steps", 0)
        joint_steps = getattr(args, "joint_finetune_steps", None)
        if joint_steps is not None:
            requested_total = calibration_steps + joint_steps
            if args.max_steps is not None and args.max_steps != requested_total:
                raise ValueError(
                    f"--max_steps={args.max_steps} 必须等于 gate_calibration_steps + "
                    f"joint_finetune_steps={requested_total}")
            total_steps = requested_total
        else:
            total_steps = default_total_steps
        if calibration_steps > total_steps:
            raise ValueError("gate_calibration_steps 不能超过 total_steps")
        self.resolved_joint_finetune_steps = total_steps - calibration_steps
        if self.is_main:
            print(f"\n在环训练[{args.policy}]: {len(train_ds)} 样本, 微步/epoch={len(dl)}, "
                  f"优化步/epoch≈{steps_per_epoch}, total≈{total_steps}, "
                  f"全局批={self.world}x{args.grad_accum}, lr={args.lr}")
            print("=" * 72)

        # step0 held-out candidate: scratch 是裸 student；legacy_gate 是迁移前
        # fixed-gate checkpoint 的精确行为，并允许 A1 最终选择“完全不校准”。
        if val_ds and self.global_step == 0:
            vm = self.validate(val_ds)
            if self.is_main:
                baseline = ("迁移 fixed-gate 基线"
                            if getattr(args, "init_scheme", "scratch_joint") == "legacy_gate"
                            else "零初始化 student 基线")
                print(f"  --- VAL step 0 ({baseline}): "
                      + " ".join(f"{k}={v:.4f}" for k, v in vm.items()) + " ---")
                for k, v in vm.items():
                    self.writer.add_scalar(f"val/{k.replace('val_', '')}", v, 0)
            if (getattr(args, "init_scheme", "scratch_joint") == "legacy_gate"
                    and calibration_steps > 0):
                self.best_gate_val = vm["val_loss"]
                self.best_gate_step = 0
                self.save(
                    args.output_dir,
                    {"val_loss": self.best_gate_val, "phase": "gate_only"},
                    kind="gate_best",
                )
                self._save_result({"best_gate_metrics": vm})

        if (getattr(args, "init_scheme", "scratch_joint") == "legacy_gate"
                and calibration_steps > 0
                and self.global_step >= calibration_steps
                and not self.joint_phase_initialized):
            self._start_joint_phase()

        micro_in_step = 0
        run_loss, run_top1, run_tok, n_micro = 0.0, 0.0, 0, 0
        last_metrics = {
            "gate_mean": 1.0, "gate_std": 0.0,
            "gate_p10": 1.0, "gate_p50": 1.0, "gate_p90": 1.0,
            "gate_frac_lt_025": 0.0, "gate_frac_gt_175": 0.0,
            "ms_norm": 0.0, "delta_norm": 0.0,
            "gate_prior": 0.0, "gate_prior_coef": 0.0,
        }
        t0 = time.time()
        while self.global_step < total_steps:
            self.epoch += 1
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(self.epoch)
            for item in dl:
                if self.global_step >= total_steps:
                    break
                try:
                    r = self.micro_loss(item, policy=args.policy)
                    if r is not None:
                        loss, M, m = r
                        (loss / args.grad_accum).backward()
                        run_loss += float(loss.detach())
                        run_top1 += m["top1"]
                        run_tok += m["tokens"]
                        n_micro += 1
                        last_metrics = m
                except torch.OutOfMemoryError:
                    # 超长样本 (LME ~122k+梯度) OOM: 本 rank 本累积组梯度作废后跳过 —
                    # 微步计数照常推进, 各 rank allreduce 次数不变 (同步安全);
                    # 代价是该步梯度少一个 rank 的贡献 (可接受的偏差)
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    print(f"  ⚠️ rank{self.rank} OOM 跳样本 "
                          f"(~{item['answer_ids'].shape[0]} ans tok), 本组梯度作废")
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
                    lrs = {group.get("group_name", "base"): group["lr"]
                           for group in self.optimizer.param_groups}
                    lr = lrs.get("base", 0.0)
                    gate_lr = lrs.get("gate", lr)
                    dt = max(time.time() - t0, 1e-3)
                    sps = n_micro / dt
                    avg = run_loss / max(n_micro, 1)
                    print(f"  step {self.global_step:6d}/{total_steps} | "
                          f"kl {avg:.4f} | top1 {run_top1/max(n_micro,1):.3f} | "
                          f"grad {gn:.3f} | lr {lr:.2e}/gate {gate_lr:.2e} | "
                          f"gate {last_metrics['gate_mean']:.3f}±"
                          f"{last_metrics['gate_std']:.3f} | "
                          f"delta {last_metrics['delta_norm']:.2f} | "
                          f"{sps:.2f} samp/s/rank | {run_tok/max(n_micro,1):.0f} tok/samp")
                    self.writer.add_scalar("train/kl", avg, self.global_step)
                    self.writer.add_scalar("train/top1", run_top1 / max(n_micro, 1),
                                           self.global_step)
                    self.writer.add_scalar("train/grad_norm", gn, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    self.writer.add_scalar("train/gate_lr", gate_lr, self.global_step)
                    for key in ("gate_mean", "gate_std", "gate_p10", "gate_p50",
                                "gate_p90", "gate_frac_lt_025", "gate_frac_gt_175",
                                "ms_norm", "delta_norm", "gate_prior",
                                "gate_prior_coef"):
                        self.writer.add_scalar(
                            f"train/{key}", last_metrics[key], self.global_step)
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
                    improved_best = vm.get("val_loss", float("inf")) < self.best_val
                    improved_gate = (
                        getattr(args, "init_scheme", "scratch_joint") == "legacy_gate"
                        and self.global_step <= calibration_steps
                        and vm.get("val_loss", float("inf")) < self.best_gate_val)
                    if improved_best:
                        self.best_val = vm["val_loss"]
                        self.best_step = self.global_step
                    if improved_gate:
                        self.best_gate_val = vm["val_loss"]
                        self.best_gate_step = self.global_step
                    if improved_best or improved_gate:
                        kind = ("best_and_gate" if improved_best and improved_gate
                                else "best" if improved_best else "gate_best")
                        self.save(
                            args.output_dir,
                            {"val_loss": vm["val_loss"],
                             "phase": ("gate_only" if improved_gate else "joint")},
                            kind=kind,
                        )
                        payload = {}
                        if improved_best:
                            payload["best_metrics"] = vm
                        if improved_gate:
                            payload["best_gate_metrics"] = vm
                        self._save_result(payload)
                at_calibration_boundary = (
                    getattr(args, "init_scheme", "scratch_joint") == "legacy_gate"
                    and calibration_steps > 0
                    and self.global_step == calibration_steps)
                if at_calibration_boundary:
                    self.save(
                        args.output_dir,
                        {"phase": "gate_only_boundary"},
                        kind="calibrated",
                    )
                    self._start_joint_phase()
                elif self.global_step % args.save_interval == 0:
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
