#!/usr/bin/env python3
"""
Stage 1 (on-policy / OPD) — 训练 TransMem: 在学生自己 rollout 的轨迹上做逐位置蒸馏.

与 off-policy 的唯一区别是**数据来源**(train.png: off-policy 和 OPD 两种都做):
  off-policy: 轨迹 = 教师 rollout 的固定答案 (Stage0 离线特征);
  on-policy : 轨迹 = 当前策略 (冻结 LLM + TransMem) 在线采样的答案 A', 边训边采.
散度计算共用 DistillLoss (法则第 4 条).

每条样本 (序列语义, docs/version2/transmem正常化修改意见.md):
  1) no_grad 学生 rollout (TransMem 在环, 自己带 KV cache): prefill (C_L,Q) -> HM_stu;
     TransMem 先吃 [HM_stu; HQ_stu_1] prefill 自己的 cache, 之后每步只喂新 HQ_stu_i —
     位置 i 的查询因果地看到 {HM_1..N, HQ_1..i} 全部历史 (token-by-token, 与外层 LLM
     的 past_key_values 平行地各持一份状态). (MS_i,g_i) -> HQ'_i=HQ_stu_i+g_i*MS_i ->
     A'_i = sample(lm_head(HQ'_i)); 喂回 LLM. 缓存 HM_stu, HQ_stu_i [AN,dim], 轨迹 A'.
  2) no_grad 教师 teacher-forcing (C_S,Q,A'_[1:AN-1]) -> HQ_tea_i -> teacher_logits.
  3) WITH grad: 把缓存的 [HM_stu; HQ_stu_1..AN] 当一条 [1,N+AN,dim] 序列一次并行前向
     (return_all_queries=True; causal mask 与 1) 的逐步语义一致) -> MS_1..AN -> HQ'
     -> student_logits, loss = DistillLoss(P_tea, P_stu), 反传 (梯度只到 TransMem).

LLM 全程冻结且仅 forward (no_grad), 唯一反传的是 TransMem. on-policy 蒸馏散度建议 reverse_kl / jsd.

用法:
  python -m transmem.train_onpolicy \
    --data_path ../Project3/data/hotpotqa/hotpotqa_train_32k.parquet --data_format parquet \
    --model_path /path/to/Qwen3-4B-Instruct-2507 --config transmem/config.json \
    --output_dir checkpoints/onpolicy --divergence jsd --N 4 \
    --max_answer_tokens 50 --accum_steps 8 --lr 1e-4 --max_steps 5000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transmem import TransMemConfig, TransMem, DistillLoss
from transmem.checkpoints import load_legacy_gate_state
from transmem.extract_features import (
    load_records, extract_cs, build_chat_prompt_ids, resolve_eos_ids, hm_positions)
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
from transmem.train_offpolicy import seed_everything

_DTYPE = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _apply_hm_transform(hm: torch.Tensor, transform=None) -> torch.Tensor:
    """Apply an optional diagnostic HM override without changing its contract."""

    if transform is None:
        return hm
    transformed = transform(hm)
    if not isinstance(transformed, torch.Tensor):
        raise ValueError(
            f"HM transform must return torch.Tensor, got {type(transformed).__name__}")
    if transformed.shape != hm.shape:
        raise ValueError(
            f"HM transform changed shape from {tuple(hm.shape)} "
            f"to {tuple(transformed.shape)}")
    if transformed.dtype != hm.dtype:
        raise ValueError(
            f"HM transform changed dtype from {hm.dtype} to {transformed.dtype}")
    if transformed.device != hm.device:
        raise ValueError(
            f"HM transform changed device from {hm.device} to {transformed.device}")
    return transformed


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 on-policy (OPD): TransMem 蒸馏")
    p.add_argument("--data_path", required=True)
    p.add_argument("--data_format", default="parquet", choices=["parquet", "json"])
    p.add_argument("--model_path", required=True)
    p.add_argument("--config", default="transmem/config.json")
    p.add_argument("--output_dir", default="checkpoints/onpolicy")
    p.add_argument("--init_scheme", default="scratch_joint",
                   choices=["legacy_gate", "scratch_joint"])
    p.add_argument("--init_checkpoint", default=None)
    p.add_argument("--gate_calibration_steps", type=int, default=0)
    p.add_argument("--joint_finetune_steps", type=int, default=None)
    # 损失
    p.add_argument("--divergence", default="jsd",
                   choices=["forward_kl", "reverse_kl", "jsd"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--reg_weight", type=float, default=0.0)
    p.add_argument("--jsd_beta", type=float, default=0.5)
    # rollout
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--max_answer_tokens", type=int, default=50)
    p.add_argument("--sample", action="store_true", default=True,
                   help="rollout 用采样 (on-policy); 关则贪心")
    p.add_argument("--rollout_temperature", type=float, default=1.0)
    # 训练
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gate_lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--gate_prior_weight", type=float, default=0.0)
    p.add_argument("--gate_prior_anneal_steps", type=int, default=0)
    p.add_argument("--accum_steps", type=int, default=8, help="梯度累积 (= 有效 batch 的样本数)")
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--save_interval", type=int, default=1000)
    # 硬件
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--attn_impl", default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="显式固定 TransMem 初始化与 rollout 采样; 默认保持历史行为")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 在线 rollout (TransMem 在环) + 教师 teacher-forcing
# ═══════════════════════════════════════════════════════════════════════════

class OnPolicyRollout:
    """封装冻结 LLM + tokenizer, 提供学生 rollout 与教师 TF (全程 no_grad, LLM 仅 forward)."""

    def __init__(self, model, tokenizer, device, N: int, dtype, hm_mode: str = "floor"):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.N = N
        self.hm_mode = hm_mode      # HM 取位公式, 必须与该 ckpt 的 stage0 一致 (config.hm_mode)
        self.dtype = dtype
        self.dim = model.config.hidden_size
        self.eos_ids = resolve_eos_ids(model)
        self.last_gate_trace = None

    def _hook_norm(self, store: dict):
        def hook_fn(m, inp, out):
            store["h"] = out.detach()
        return self.model.model.norm.register_forward_hook(hook_fn)

    def _extract_hm(self, hidden_prefill: torch.Tensor, len_cl: int) -> torch.Tensor:
        """prefill hidden [L,dim] 中 C_L 前 len_cl 个位置取 N 个记忆槽 -> [N,dim].
        取位公式与 stage0 共用 hm_positions (floor=历史 ckpt, frac=池化消融 ckpt)."""
        idx = hm_positions(len_cl, self.N, self.hm_mode)
        return hidden_prefill[torch.tensor(idx, device=hidden_prefill.device)]

    @torch.no_grad()
    def student_rollout(self, mem: TransMem, context_long: str, question: str,
                        max_new: int, sample: bool, temperature: float,
                        hm_transform=None,
                        collect_gate_diagnostics: bool = False):
        """学生在线 rollout (TransMem 在环, token-by-token).

        TransMem 与外层 LLM 各持一份 KV cache: 第 1 步喂 [HM_stu; HQ_stu_1] prefill,
        之后每步只喂新 HQ_stu_i — 第 i 步查询因果地 attend {HM_1..N, HQ_1..i} 全部历史
        (transmem正常化修改意见.md §3.1/3.2), 而非固定记忆 + 孤立当前查询.
        返回 (HM_stu [N,dim], HQ_stu [AN,dim], A' ids).
        """
        cq_ids = build_chat_prompt_ids(self.tok, context_long, question, self.device)
        len_cl = self.tok(context_long, return_tensors="pt",
                          add_special_tokens=False).input_ids.shape[1]

        store = {}
        handle = self._hook_norm(store)
        try:
            # 只跑 base model: CausalLM 会算全长 logits [L,vocab] (~9GB@30k, ~37GB@122k
            # longmemeval 必 OOM), 而这里 logits 从不使用 (149 行手动过 lm_head).
            # attention_mask=ones 显式传 (transformers 4.57.6: None 不走 is_causal skip,
            # 物化 S×S mask, 125k 峰值 57.4GB vs 16.8GB — probe 10216593).
            out = self.model.model(input_ids=cq_ids,
                                   attention_mask=torch.ones_like(cq_ids),
                                   use_cache=True)
            past = out.past_key_values
            prefill_hidden = store["h"][0]                       # [L, dim]
            hm_stu = self._extract_hm(prefill_hidden, len_cl)    # [N, dim]
            hm_stu = _apply_hm_transform(hm_stu, hm_transform)

            hq_list, ans_ids = [], []
            trace = ({"gate": [], "ms_norm": [], "delta_norm": []}
                     if collect_gate_diagnostics else None)
            hq_cur = prefill_hidden[-1:, :]                      # HQ_stu_1 [1,dim]
            mem_past = DynamicCache()                            # TransMem 自己的 KV cache
            X = torch.cat([hm_stu, hq_cur], dim=0).unsqueeze(0).to(self.dtype)  # [1,N+1,dim]
            for _ in range(max_new):
                hq_list.append(hq_cur[0])
                proposal = mem(X, past_key_values=mem_past, use_cache=True)
                hq_prime = mem.correct(hq_cur, proposal)          # [1, dim]
                if trace is not None:
                    trace["gate"].append(float(proposal.gate.squeeze().float()))
                    trace["ms_norm"].append(
                        float(proposal.ms.float().norm(dim=-1).mean()))
                    trace["delta_norm"].append(
                        float(proposal.delta.float().norm(dim=-1).mean()))
                logits = self.model.lm_head(hq_prime)            # [1, vocab]
                if sample:
                    probs = torch.softmax(logits.float() / max(temperature, 1e-6), dim=-1)
                    nxt = torch.multinomial(probs, 1)[0]         # [1]
                else:
                    nxt = logits.argmax(dim=-1)                  # [1]
                tok_id = int(nxt.item())
                ans_ids.append(tok_id)
                if tok_id in self.eos_ids:
                    break
                step = self.model.model(input_ids=nxt.view(1, 1), past_key_values=past,
                                        use_cache=True)
                past = step.past_key_values
                hq_cur = store["h"][0][-1:, :]                   # HQ_stu_{i+1}
                X = hq_cur.unsqueeze(0).to(self.dtype)           # 增量: 只喂新查询 [1,1,dim]
        finally:
            handle.remove()

        hq_stu = torch.stack(hq_list, dim=0)                     # [AN, dim]
        self.last_gate_trace = ({
            "token_ids": list(ans_ids), "layers": {"final": trace}
        } if trace is not None else None)
        return hm_stu, hq_stu, ans_ids

    @torch.no_grad()
    def teacher_forward(self, cs_text: str, question: str, answer_ids: list[int],
                        lm_head):
        """教师 (C_S,Q,A'_[1:AN-1]) teacher-forcing -> (logits [AN,vocab], HQ_tea [AN,dim])."""
        AN = len(answer_ids)
        cq_ids = build_chat_prompt_ids(self.tok, cs_text, question, self.device)
        len_cq = cq_ids.shape[1]
        if AN <= 1:
            full = cq_ids
        else:
            prefix = torch.tensor([answer_ids[:-1]], device=self.device, dtype=cq_ids.dtype)
            full = torch.cat([cq_ids, prefix], dim=1)
        store = {}
        handle = self._hook_norm(store)
        try:
            # 同上: 不算全长 logits; ones mask 走 is_causal skip (避免 S×S mask 物化)
            self.model.model(input_ids=full, attention_mask=torch.ones_like(full),
                             use_cache=False)
        finally:
            handle.remove()
        hidden = store["h"][0]                                   # [total, dim]
        total = hidden.shape[0]
        positions = [len_cq - 1]
        for i in range(2, AN + 1):
            pos = len_cq + (i - 2)
            if pos < total:
                positions.append(pos)
            else:
                break
        hq_tea = hidden[torch.tensor(positions, device=hidden.device)]   # [AN', dim]
        return lm_head(hq_tea), hq_tea                           # logits [AN',vocab], [AN',dim]


# ═══════════════════════════════════════════════════════════════════════════
# 训练器
# ═══════════════════════════════════════════════════════════════════════════

class OnPolicyTrainer:
    def __init__(self, args):
        for name, default in {
            "init_scheme": "scratch_joint",
            "init_checkpoint": None,
            "gate_calibration_steps": 0,
            "joint_finetune_steps": None,
            "gate_lr": None,
            "gate_prior_weight": 0.0,
            "gate_prior_anneal_steps": 0,
        }.items():
            if not hasattr(args, name):
                setattr(args, name, default)
        self.args = args
        self.device = torch.device(args.device)
        self.dtype = _DTYPE[args.dtype]
        seed_everything(getattr(args, "seed", None))
        self._load_model()

        self.config = TransMemConfig.from_json(args.config)
        self.config.n_mem = args.N
        validate_gate_training_options(
            init_scheme=args.init_scheme,
            init_checkpoint=args.init_checkpoint,
            gate_mode=self.config.gate_mode,
            gate_calibration_steps=args.gate_calibration_steps,
            joint_finetune_steps=args.joint_finetune_steps,
            warm_start=self.config.warm_start,
        )
        self.mem = TransMem(self.config).to(self.device, dtype=self.dtype)
        self.parent_checkpoint = None
        if args.init_scheme == "legacy_gate":
            if self.config.warm_start:
                raise ValueError("legacy_gate 与 warm_start=true 冲突")
            checkpoint = torch.load(
                args.init_checkpoint, map_location="cpu", weights_only=False)
            load_legacy_gate_state(self.mem, checkpoint)
            self.parent_checkpoint = str(Path(args.init_checkpoint).expanduser().resolve())
            del checkpoint
        elif self.config.warm_start:
            self.mem.warm_start_from(self.model)
        self.rollout = OnPolicyRollout(self.model, self.tokenizer, self.device,
                                       args.N, self.dtype)
        self.loss_fn = DistillLoss(args.divergence, args.temperature,
                                   args.reg_weight, args.jsd_beta)
        self.optimizer = build_gate_optimizer(
            self.mem, base_lr=args.lr, gate_lr=args.gate_lr,
            weight_decay=args.weight_decay)
        self.global_step = 0
        self.writer = SummaryWriter(log_dir=str(Path(args.output_dir) / "tb"))
        print(f"TransMem: {self.mem.num_params(True):,} trainable | "
              f"loss={args.divergence} T={args.temperature} | "
              f"accum={args.accum_steps} sample={args.sample} | "
              f"gate={self.config.gate_mode} init={args.init_scheme}")

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        a = self.args
        print(f"加载 backbone (冻结): {a.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            a.model_path, local_files_only=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            a.model_path, torch_dtype=self.dtype, local_files_only=True,
            trust_remote_code=True, attn_implementation=a.attn_impl).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.lm_head = self.model.lm_head            # 冻结, 可微算子

    def _set_lr(self, step):
        warmup = self.args.warmup_steps
        if step < warmup:
            factor = step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(self.args.max_steps - warmup, 1)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        set_gate_optimizer_lrs(
            self.optimizer,
            lr_factor=factor,
            gate_only=is_gate_only_phase(
                self.args.init_scheme, step, self.args.gate_calibration_steps),
        )

    def sample_step(self, rec) -> dict | None:
        """处理一条样本 -> 一次 (累积) 反传. 返回 metrics 或 None (跳过)."""
        cs = extract_cs(rec["context"], rec["golden_index"]) if rec["golden_index"] is not None else ""
        if not cs or not rec["question"] or not rec["context"]:
            return None
        # 1) 学生在线 rollout (no_grad)
        hm_stu, hq_stu, ans_ids = self.rollout.student_rollout(
            self.mem, rec["context"], rec["question"], self.args.max_answer_tokens,
            self.args.sample, self.args.rollout_temperature)
        if len(ans_ids) == 0:
            return None
        # 2) 教师 TF (no_grad)
        teacher_logits, hq_tea = self.rollout.teacher_forward(
            cs, rec["question"], ans_ids, self.lm_head)
        AN = min(hq_stu.shape[0], teacher_logits.shape[0])
        if AN < 1:
            return None
        hm_stu = hm_stu.detach().to(self.dtype)
        hq_stu = hq_stu[:AN].detach().to(self.dtype)             # [AN, dim] (常量)
        teacher_logits = teacher_logits[:AN]
        hq_tea = hq_tea[:AN].detach().to(self.dtype)             # [AN, dim] (reg 目标)

        # 3) WITH grad: 整条 [HM; HQ_1..AN] 一次并行前向重算全部 MS (§3.3) —
        #    causal mask 保证第 i 位只看 {HM, HQ_1..i}, 与 rollout 的逐步语义一致
        self.mem.train()
        X = torch.cat([hm_stu, hq_stu], dim=0).unsqueeze(0)      # [1, N+AN, dim] 一条序列
        proposal = self.mem(X, return_all_queries=True)
        hq_prime = self.mem.correct(hq_stu.unsqueeze(0), proposal).squeeze(0)
        student_logits = self.lm_head(hq_prime)                 # [AN, vocab]
        task_loss, metrics = self.loss_fn(
            student_logits, teacher_logits, hq_prime, hq_tea)
        prior = gate_prior_loss(proposal.gate)
        prior_coef = gate_prior_coefficient(
            step=self.global_step,
            weight=self.args.gate_prior_weight,
            anneal_steps=self.args.gate_prior_anneal_steps,
        )
        loss = task_loss + prior_coef * prior
        metrics.update(gate_metrics(proposal.ms, proposal.gate))
        metrics["task_loss"] = float(task_loss.detach())
        metrics["gate_prior"] = float(prior.detach())
        metrics["gate_prior_coef"] = prior_coef
        metrics["loss"] = float(loss.detach())
        (loss / self.args.accum_steps).backward()
        metrics["AN"] = AN
        return metrics

    def run(self):
        args = self.args
        if args.joint_finetune_steps is not None:
            requested_total = args.gate_calibration_steps + args.joint_finetune_steps
            if args.max_steps != requested_total:
                raise ValueError(
                    f"--max_steps={args.max_steps} 必须等于 gate_calibration_steps + "
                    f"joint_finetune_steps={requested_total}")
        if args.gate_calibration_steps > args.max_steps:
            raise ValueError("gate_calibration_steps 不能超过 max_steps")
        self.resolved_joint_finetune_steps = (
            args.max_steps - args.gate_calibration_steps)
        records = load_records(args.data_path, args.data_format, args.max_samples)
        print(f"数据: {len(records)} 条; 目标 {args.max_steps} steps")
        print("=" * 72)

        ri = 0
        running = {"loss": 0.0, "div": 0.0}
        n_in_window = 0
        accum = 0
        t0 = time.time()
        self.optimizer.zero_grad()
        while self.global_step < args.max_steps:
            rec = records[ri % len(records)]
            ri += 1
            m = self.sample_step(rec)
            if m is None:
                continue
            running["loss"] += m["loss"]
            running["div"] += m["div"]
            n_in_window += 1
            accum += 1
            if accum >= args.accum_steps:
                self._set_lr(self.global_step)
                clear_base_grads_for_gate_only(
                    self.optimizer,
                    gate_only=is_gate_only_phase(
                        args.init_scheme,
                        self.global_step,
                        args.gate_calibration_steps,
                    ),
                )
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.mem.parameters(), args.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                accum = 0
                if (args.init_scheme == "legacy_gate"
                        and args.gate_calibration_steps > 0
                        and self.global_step == args.gate_calibration_steps):
                    self.save(args.output_dir, kind="calibrated")
                if self.global_step % args.log_interval == 0:
                    lrs = {group.get("group_name", "base"): group["lr"]
                           for group in self.optimizer.param_groups}
                    lr = lrs.get("base", 0.0)
                    gate_lr = lrs.get("gate", lr)
                    sps = args.log_interval / max(time.time() - t0, 1e-3)
                    mem_gb = torch.cuda.max_memory_allocated() / 1024**3
                    avg_loss = running["loss"] / max(n_in_window, 1)
                    avg_div = running["div"] / max(n_in_window, 1)
                    print(f"  step {self.global_step:6d}/{args.max_steps} | "
                          f"loss {avg_loss:.5f} | "
                          f"div {avg_div:.5f} | lr {lr:.2e}/gate {gate_lr:.2e} | "
                          f"{sps:.2f} it/s | mem {mem_gb:.1f}GB")
                    self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                    self.writer.add_scalar("train/div", avg_div, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    self.writer.add_scalar("train/gate_lr", gate_lr, self.global_step)
                    for key in ("gate_mean", "gate_std", "gate_p10", "gate_p50",
                                "gate_p90", "gate_frac_lt_025", "gate_frac_gt_175",
                                "ms_norm", "delta_norm", "gate_prior",
                                "gate_prior_coef"):
                        if key in m:
                            self.writer.add_scalar(f"train/{key}", m[key], self.global_step)
                    self.writer.add_scalar("train/it_per_s", sps, self.global_step)
                    self.writer.add_scalar("train/mem_gb", mem_gb, self.global_step)
                    running = {"loss": 0.0, "div": 0.0}
                    n_in_window = 0
                    t0 = time.time()
                if self.global_step % args.save_interval == 0:
                    self.save(args.output_dir)
        self.save(args.output_dir)
        self.writer.close()
        print("=" * 72)
        print(f"✅ on-policy (OPD) 训练完成: {self.global_step} steps")
        print(f"TensorBoard: tensorboard --logdir {Path(args.output_dir) / 'tb'}")

    def save(self, path, metrics=None, kind="latest"):
        Path(path).mkdir(parents=True, exist_ok=True)
        base = {"model_state_dict": self.mem.state_dict(), "config": self.config.to_dict(),
                "global_step": self.global_step,
                "train_mode": "onpolicy",
                "seed": getattr(self.args, "seed", None),
                "init_scheme": self.args.init_scheme,
                "parent_checkpoint": self.parent_checkpoint,
                "gate_calibration_steps": self.args.gate_calibration_steps,
                "joint_finetune_steps": getattr(
                    self, "resolved_joint_finetune_steps",
                    self.args.joint_finetune_steps)}
        torch.save(base, Path(path) / f"step_{self.global_step:07d}.pt")
        torch.save(dict(base, optimizer_state_dict=self.optimizer.state_dict()),
                   Path(path) / "latest.pt")
        names = [f"step_{self.global_step:07d}.pt", "latest.pt"]
        if kind == "calibrated":
            torch.save(base, Path(path) / "gate_only.pt")
            names.append("gate_only.pt")
        print(f"  Checkpoint saved: {', '.join(names)}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        validate_dynamic_resume_checkpoint(
            checkpoint,
            config=self.config.to_dict(),
            init_scheme=self.args.init_scheme,
            parent_checkpoint=self.parent_checkpoint,
            gate_calibration_steps=self.args.gate_calibration_steps,
            joint_finetune_steps=self.args.joint_finetune_steps,
            seed=getattr(self.args, "seed", None),
        )
        self.mem.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = int(checkpoint.get("global_step", 0))
        print(f"Resumed: step={self.global_step}")


def main():
    args = parse_args()
    trainer = OnPolicyTrainer(args)
    if args.resume:
        trainer.load(args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
