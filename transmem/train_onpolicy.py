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
     的 past_key_values 平行地各持一份状态). MS_i -> HQ'_i=HQ_stu_i+a*MS_i ->
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
from transmem.extract_features import (
    load_records, extract_cs, build_chat_prompt_ids, resolve_eos_ids)

_DTYPE = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 on-policy (OPD): TransMem 蒸馏")
    p.add_argument("--data_path", required=True)
    p.add_argument("--data_format", default="parquet", choices=["parquet", "json"])
    p.add_argument("--model_path", required=True)
    p.add_argument("--config", default="transmem/config.json")
    p.add_argument("--output_dir", default="checkpoints/onpolicy")
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
    p.add_argument("--weight_decay", type=float, default=0.0)
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
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 在线 rollout (TransMem 在环) + 教师 teacher-forcing
# ═══════════════════════════════════════════════════════════════════════════

class OnPolicyRollout:
    """封装冻结 LLM + tokenizer, 提供学生 rollout 与教师 TF (全程 no_grad, LLM 仅 forward)."""

    def __init__(self, model, tokenizer, device, N: int, dtype):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.N = N
        self.dtype = dtype
        self.dim = model.config.hidden_size
        self.eos_ids = resolve_eos_ids(model)

    def _hook_norm(self, store: dict):
        def hook_fn(m, inp, out):
            store["h"] = out.detach()
        return self.model.model.norm.register_forward_hook(hook_fn)

    def _extract_hm(self, hidden_prefill: torch.Tensor, len_cl: int) -> torch.Tensor:
        """prefill hidden [L,dim] 中 C_L 前 len_cl 个位置分 N 段取末位 -> [N,dim]."""
        N = self.N
        seg = max(len_cl // N, 1)
        idx = [max(min((i + 1) * seg, len_cl) - 1, 0) for i in range(N)]
        return hidden_prefill[torch.tensor(idx, device=hidden_prefill.device)]

    @torch.no_grad()
    def student_rollout(self, mem: TransMem, context_long: str, question: str,
                        max_new: int, sample: bool, temperature: float):
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
            out = self.model(input_ids=cq_ids, use_cache=True)
            past = out.past_key_values
            prefill_hidden = store["h"][0]                       # [L, dim]
            hm_stu = self._extract_hm(prefill_hidden, len_cl)    # [N, dim]

            hq_list, ans_ids = [], []
            hq_cur = prefill_hidden[-1:, :]                      # HQ_stu_1 [1,dim]
            mem_past = DynamicCache()                            # TransMem 自己的 KV cache
            X = torch.cat([hm_stu, hq_cur], dim=0).unsqueeze(0).to(self.dtype)  # [1,N+1,dim]
            for _ in range(max_new):
                hq_list.append(hq_cur[0])
                ms = mem(X, past_key_values=mem_past, use_cache=True)  # [1, dim] 读末位
                hq_prime = hq_cur.to(ms.dtype) + mem.a * ms      # [1, dim]
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
                step = self.model(input_ids=nxt.view(1, 1), past_key_values=past,
                                  use_cache=True)
                past = step.past_key_values
                hq_cur = store["h"][0][-1:, :]                   # HQ_stu_{i+1}
                X = hq_cur.unsqueeze(0).to(self.dtype)           # 增量: 只喂新查询 [1,1,dim]
        finally:
            handle.remove()

        hq_stu = torch.stack(hq_list, dim=0)                     # [AN, dim]
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
            self.model(input_ids=full, use_cache=False)
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
        self.args = args
        self.device = torch.device(args.device)
        self.dtype = _DTYPE[args.dtype]
        self._load_model()

        self.config = TransMemConfig.from_json(args.config)
        self.config.n_mem = args.N
        self.mem = TransMem(self.config).to(self.device, dtype=self.dtype)
        if self.config.warm_start:
            self.mem.warm_start_from(self.model)
        self.rollout = OnPolicyRollout(self.model, self.tokenizer, self.device,
                                       args.N, self.dtype)
        self.loss_fn = DistillLoss(args.divergence, args.temperature,
                                   args.reg_weight, args.jsd_beta)
        self.optimizer = torch.optim.AdamW(
            (p for p in self.mem.parameters() if p.requires_grad),
            lr=args.lr, weight_decay=args.weight_decay)
        self.global_step = 0
        self.writer = SummaryWriter(log_dir=str(Path(args.output_dir) / "tb"))
        print(f"TransMem: {self.mem.num_params(True):,} trainable | "
              f"loss={args.divergence} T={args.temperature} | "
              f"accum={args.accum_steps} sample={args.sample}")

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
            lr = self.args.lr * step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(self.args.max_steps - warmup, 1)
            lr = self.args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for g in self.optimizer.param_groups:
            g["lr"] = lr

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
        ms = self.mem(X, return_all_queries=True).squeeze(0)     # [AN, dim]
        hq_prime = hq_stu + self.mem.a * ms
        student_logits = self.lm_head(hq_prime)                 # [AN, vocab]
        loss, metrics = self.loss_fn(student_logits, teacher_logits, hq_prime, hq_tea)
        (loss / self.args.accum_steps).backward()
        metrics["AN"] = AN
        return metrics

    def run(self):
        args = self.args
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
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.mem.parameters(), args.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                accum = 0
                if self.global_step % args.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    sps = args.log_interval / max(time.time() - t0, 1e-3)
                    mem_gb = torch.cuda.max_memory_allocated() / 1024**3
                    avg_loss = running["loss"] / max(n_in_window, 1)
                    avg_div = running["div"] / max(n_in_window, 1)
                    print(f"  step {self.global_step:6d}/{args.max_steps} | "
                          f"loss {avg_loss:.5f} | "
                          f"div {avg_div:.5f} | lr {lr:.2e} | "
                          f"{sps:.2f} it/s | mem {mem_gb:.1f}GB")
                    self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                    self.writer.add_scalar("train/div", avg_div, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
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

    def save(self, path, metrics=None):
        Path(path).mkdir(parents=True, exist_ok=True)
        ckpt = {"model_state_dict": self.mem.state_dict(), "config": self.config.to_dict(),
                "global_step": self.global_step}
        torch.save(ckpt, Path(path) / f"step_{self.global_step:07d}.pt")
        torch.save(ckpt, Path(path) / "latest.pt")
        print(f"  Checkpoint saved: step_{self.global_step:07d}.pt")


def main():
    args = parse_args()
    OnPolicyTrainer(args).run()


if __name__ == "__main__":
    main()
