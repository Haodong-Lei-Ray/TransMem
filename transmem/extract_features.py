#!/usr/bin/env python3
"""
Stage 0 — 离线特征抽取: 用冻结 LLM forward, 产出 TransMem 训练数据.

与 Project3/diffusionmem/diffusion_mem/train_stage0.py 同一套已验证逻辑
(切 C_S、教师 rollout+hook HQ_tea、学生 teacher-forcing HM/HQ_stu、sharded .pt + manifest),
仅两点不同:
  1) 落盘 (hm_stu, hq_stu, hq_tea) —— TransMem off-policy 不需要 λ, 教师分布在 Stage1 现算;
  2) 额外 dump 一次冻结 lm_head 权重到 output_dir/lm_head.pt, 供 off-policy KD 穿它算 logits.

数据格式 (BytedTsinghua-SIA/hotpotqa):
  Train parquet 列: prompt(chat messages), context(长文 C_L), extra_info{index=golden 文档号},
                    reward_model{ground_truth}, ability.
  Eval  JSON 字段:  context, input(问题), index(golden 文档号), answers, num_docs.

C_S = context 中 "Document {index}:" 到下一个 "Document:" (或文末) 之间的文本.
Q   = prompt[0].content (train) 或 input (eval).  C_L = context 全文.

LLM 全程冻结. hidden 取 model.model.norm 输出 (最后 RMSNorm 之后、LM head 之前).

用法:
  python -m transmem.extract_features \
    --data_path ../Project3/data/hotpotqa/hotpotqa_train_32k.parquet \
    --data_format parquet \
    --model_path /path/to/Qwen3-4B-Instruct-2507 \
    --output_dir data/stage0_train \
    --N 4 --max_answer_tokens 50
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Stage 0: 离线抽取 LLM hidden states (TransMem)")
    p.add_argument("--data_path", required=True, help="数据路径 (.parquet 或 .json)")
    p.add_argument("--data_format", default="parquet",
                   choices=["parquet", "json", "qasper"])
    p.add_argument("--model_path", required=True, help="Qwen3-4B-Instruct-2507 路径")
    p.add_argument("--output_dir", required=True, help="输出目录, 写 sharded .pt + lm_head.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attn_impl", default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--N", type=int, default=4, help="记忆分段数 (HM_stu = [N, dim])")
    p.add_argument("--max_answer_tokens", type=int, default=50)
    p.add_argument("--samples_per_shard", type=int, default=1000)
    p.add_argument("--save_dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--dump_lm_head", action="store_true", default=True,
                   help="是否 dump 冻结 lm_head (off-policy KD 需要)")
    p.add_argument("--thinking", action="store_true",
                   help="chat prompt 是否用 thinking 系统提示 (build_chat_prompt_ids 的 thinking 参数)")
    return p.parse_args()


def get_dtype(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[s]


# ═══════════════════════════════════════════════════════════════════════════
# 数据适配: 统一 train parquet / eval JSON 字段访问
# ═══════════════════════════════════════════════════════════════════════════

def load_records(data_path: str, data_format: str, max_samples: Optional[int]):
    if data_format == "parquet":
        df = pd.read_parquet(data_path)
        if max_samples:
            df = df.head(max_samples)
        return [_parse_parquet_row(df.iloc[i], i) for i in range(len(df))]
    with open(data_path) as f:
        raw = json.load(f)
    if data_format == "qasper":
        # 单条记录 (含 selected_qa) 或记录列表
        items = raw if isinstance(raw, list) else [raw]
        if max_samples:
            items = items[:max_samples]
        return [_parse_qasper_record(item, i) for i, item in enumerate(items)]
    if max_samples:
        raw = raw[:max_samples]
    return [_parse_json_item(item, i) for i, item in enumerate(raw)]


def _parse_parquet_row(row, idx: int) -> dict:
    prompt_raw = row["prompt"]
    if isinstance(prompt_raw, str):
        question = prompt_raw.strip()
    elif hasattr(prompt_raw, "__getitem__") and len(prompt_raw) > 0:
        question = prompt_raw[0].get("content", str(prompt_raw[0])).strip()
    else:
        question = str(prompt_raw).strip()

    context = str(row["context"]).strip()
    extra_info = row.get("extra_info", {})
    golden_index = extra_info.get("index", None) if isinstance(extra_info, dict) else None

    reward = row.get("reward_model", {})
    ground_truth = ""
    if isinstance(reward, dict):
        gt_arr = reward.get("ground_truth", None)
        if gt_arr is not None and hasattr(gt_arr, "__iter__") and len(gt_arr) > 0:
            ground_truth = str(gt_arr[0]).strip()

    return {"question": question, "context": context, "golden_index": golden_index,
            "ground_truth": ground_truth, "sample_idx": idx}


def _parse_json_item(item: dict, idx: int) -> dict:
    return {
        "question": str(item.get("input", "")).strip(),
        "context": str(item.get("context", "")).strip(),
        "golden_index": item.get("index", None),
        "ground_truth": str(item.get("answers", [""])[0]).strip() if item.get("answers") else "",
        "sample_idx": idx,
    }


# ── Qasper: C_L=全文, C_S=evidence(直接给, 无需正则切), Q=question ──────────

def _build_qasper_context(item: dict) -> str:
    """把 Qasper 一篇论文拼成长上下文 C_L: Title + Abstract + 各 section 的 paragraphs."""
    parts = []
    if item.get("title"):
        parts.append(f"Title: {item['title']}")
    if item.get("abstract"):
        parts.append(f"Abstract:\n{item['abstract']}")
    ft = item.get("full_text", {}) or {}
    names = ft.get("section_name", []) or []
    paras = ft.get("paragraphs", []) or []
    for i, sec_paras in enumerate(paras):
        name = names[i] if i < len(names) else ""
        body = "\n".join(p for p in sec_paras if p)
        if name or body:
            parts.append((f"{name}\n{body}" if name else body).strip())
    return "\n\n".join(parts).strip()


def _qasper_ground_truth(qa: dict) -> str:
    """取参考答案 (仅供 eval/参考; Stage0 训练特征用教师 rollout)."""
    if qa.get("free_form_answer"):
        return str(qa["free_form_answer"]).strip()
    spans = qa.get("extractive_spans") or []
    if spans:
        return "; ".join(str(s).strip() for s in spans)
    if qa.get("yes_no") is not None:
        return "Yes" if qa["yes_no"] else "No"
    if qa.get("unanswerable"):
        return "Unanswerable"
    return ""


def _parse_qasper_record(item: dict, idx: int) -> dict:
    """Qasper 记录(含 selected_qa) -> 统一字段; cs_text 由 evidence 直接给出."""
    qa = item.get("selected_qa", {}) or {}
    evidence = qa.get("evidence", []) or []
    cs_text = "\n\n".join(e for e in evidence if e).strip()
    return {
        "question": str(qa.get("question", "")).strip(),
        "context": _build_qasper_context(item),
        "cs_text": cs_text,                  # C_S 直供, process_sample 优先用它
        "golden_index": None,
        "ground_truth": _qasper_ground_truth(qa),
        "sample_idx": idx,
        "paper_id": item.get("id", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════
# C_S 提取: 从 context 中按 golden_index 切出 golden 文档
# ═══════════════════════════════════════════════════════════════════════════

_DOC_HEADER_RE = re.compile(r"^Document (\d+):\s*$", re.MULTILINE)


def extract_cs(context: str, golden_index: int) -> str:
    """从 context 中提取 Document {golden_index} 的正文 (不含标题行)."""
    if golden_index is None:
        return ""
    matches = list(_DOC_HEADER_RE.finditer(context))
    if not matches:
        return ""
    target_match = next_match = None
    for i, m in enumerate(matches):
        if int(m.group(1)) == golden_index:
            target_match = m
            if i + 1 < len(matches):
                next_match = matches[i + 1]
            break
    if target_match is None:                      # fallback: positional
        idx = golden_index - 1
        if 0 <= idx < len(matches):
            target_match = matches[idx]
            if idx + 1 < len(matches):
                next_match = matches[idx + 1]
    if target_match is None:
        return ""
    start = target_match.end()
    end = next_match.start() if next_match else len(context)
    return context[start:end].strip()


# ═══════════════════════════════════════════════════════════════════════════
# 主类
# ═══════════════════════════════════════════════════════════════════════════

class Stage0Extractor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model_dtype = get_dtype(args.dtype)
        self.save_dtype = get_dtype(args.save_dtype)
        self.N = args.N

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        args = self.args
        print(f"加载 tokenizer: {args.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(torch_dtype=self.model_dtype, local_files_only=True,
                           trust_remote_code=True)
        if args.attn_impl:
            load_kwargs["attn_implementation"] = args.attn_impl
        print(f"加载模型: {args.model_path} (dtype={args.dtype}, attn={args.attn_impl})")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path, **load_kwargs).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.dim = self.model.config.hidden_size
        self.lm_head = self.model.lm_head
        self._eos_ids = resolve_eos_ids(self.model)
        print(f"  dim={self.dim}, 设备={self.device}, eos_ids={self._eos_ids}")

    def dump_lm_head(self, output_dir: Path):
        """dump 冻结 lm_head 权重 [vocab, dim] -> output_dir/lm_head.pt (off-policy KD 用)."""
        w = self.model.lm_head.weight.detach().to(self.save_dtype).cpu()
        tied = bool(getattr(self.model.config, "tie_word_embeddings", False))
        torch.save({"weight": w, "tied": tied,
                    "vocab_size": w.shape[0], "dim": w.shape[1]},
                   output_dir / "lm_head.pt")
        print(f"  dump lm_head.pt: weight{tuple(w.shape)} tied={tied}")

    # ── 教师生成答案 + 顺带捕获 HQ_tea_i (无需二次 forward) ──────────────
    @torch.inference_mode()
    def generate_answer(self, cs_text: str, question: str):
        """教师 (C_S, Q) greedy 自回归生成, 生成时 hook 末位隐状态拿 HQ_tea_i.
        返回 (answer_ids [M], answer_text, hq_tea [M, dim])."""
        cq_ids = build_chat_prompt_ids(self.tokenizer, cs_text, question, self.device,
                                        thinking=self.args.thinking)
        prompt_len = cq_ids.shape[1]

        captured: list[torch.Tensor] = []
        def hook_fn(m, inp, out):
            captured.append(out[0, -1, :].detach())     # 末位 = 下一 token 的预测隐状态
        handle = self.model.model.norm.register_forward_hook(hook_fn)
        try:
            generated = self.model.generate(
                input_ids=cq_ids, attention_mask=torch.ones_like(cq_ids),
                max_new_tokens=self.args.max_answer_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self._eos_ids)          # chat 模式: 答完吐 <|im_end|> 自然停
        finally:
            handle.remove()

        answer_ids = generated[0, prompt_len:].tolist()
        answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        M = len(answer_ids)
        if M > 0 and len(captured) >= M:
            hq_tea = torch.stack(captured[:M], dim=0)
        else:
            hq_tea = (torch.stack(captured, dim=0) if captured
                      else torch.empty(0, self.dim, device=self.device))
        return answer_ids, answer_text, hq_tea

    # ── 学生 forward: HM_stu + HQ_stu_i ──────────────────────────────
    @torch.inference_mode()
    def student_forward(self, context_long: str, question: str, answer_ids: list[int]):
        """(C_L, Q, A_[1:M-1]) -> (HM_stu [N, dim], HQ_stu_i [M, dim]).
        全程 token-id 空间拼接, 避免 BPE 边界 merge / 特殊 token 约定导致的位置错位."""
        M = len(answer_ids)
        cq_ids = build_chat_prompt_ids(self.tokenizer, context_long, question, self.device,
                                        thinking=self.args.thinking)
        len_cq = cq_ids.shape[1]
        cl_ids = self.tokenizer(context_long, return_tensors="pt",
                                add_special_tokens=False).input_ids.to(self.device)
        len_cl = cl_ids.shape[1]

        if M <= 1:
            hidden = self._forward_ids_and_hook(cq_ids)
            return self._extract_hm(hidden, len_cl), hidden[-1:, :]

        prefix_ids = torch.tensor([answer_ids[:-1]], device=self.device, dtype=cq_ids.dtype)
        full_ids = torch.cat([cq_ids, prefix_ids], dim=1)
        hidden = self._forward_ids_and_hook(full_ids)
        total_len = hidden.shape[0]
        hm = self._extract_hm(hidden, len_cl)

        positions = [len_cq - 1]                         # 预测 A[0]=HQ_stu_1
        for i in range(2, M + 1):
            pos = len_cq + (i - 2)                        # 预测 A[i-1]=HQ_stu_i
            if pos < total_len:
                positions.append(pos)
            else:
                break
        hq = hidden[torch.tensor(positions, device=hidden.device, dtype=torch.long)]
        return hm, hq

    def _extract_hm(self, hidden: torch.Tensor, len_cl: int) -> torch.Tensor:
        """C_L 前 len_cl 个位置分 N 段, 取每段末位 hidden."""
        N = self.N
        seg = max(len_cl // N, 1)
        idx = [max(min((i + 1) * seg, len_cl) - 1, 0) for i in range(N)]
        return hidden[torch.tensor(idx, device=hidden.device, dtype=torch.long)]

    def _forward_ids_and_hook(self, ids: torch.Tensor) -> torch.Tensor:
        captured = {}
        def hook_fn(m, inp, out): captured["h"] = out.detach()
        handle = self.model.model.norm.register_forward_hook(hook_fn)
        try:
            self.model(input_ids=ids, use_cache=False)
        finally:
            handle.remove()
        return captured["h"][0]

    # ── 单样本处理 ────────────────────────────────────────────────────
    def process_sample(self, rec: dict) -> Optional[dict]:
        try:
            question = rec["question"]
            context_long = rec["context"]
            golden_index = rec.get("golden_index")
            sample_idx = rec["sample_idx"]
            if not question or not context_long:
                return None
            # C_S: qasper 等已直供 cs_text; 否则按 golden_index 正则切 (hotpot)
            cs_text = rec.get("cs_text") or (
                extract_cs(context_long, golden_index) if golden_index is not None else "")
            if not cs_text:
                print(f"  [WARN] sample {sample_idx}: C_S 为空 (golden_index={golden_index})")
                return None

            answer_ids, answer_text, hq_tea = self.generate_answer(cs_text, question)
            if len(answer_ids) == 0:
                return None
            hm_stu, hq_stu = self.student_forward(context_long, question, answer_ids)

            actual_M = min(hq_tea.shape[0], hq_stu.shape[0])
            if actual_M < 1:
                return None
            return {
                "hm_stu": hm_stu.to(dtype=self.save_dtype).cpu(),
                "hq_stu": hq_stu[:actual_M].to(dtype=self.save_dtype).cpu(),
                "hq_tea": hq_tea[:actual_M].to(dtype=self.save_dtype).cpu(),
                "answer_ids": torch.tensor(answer_ids[:actual_M], dtype=torch.long),
                "answer_text": answer_text,
                "sample_idx": sample_idx,
                "M": actual_M, "dim": self.dim, "N": self.N,
            }
        except Exception as e:
            print(f"  [ERROR] sample {rec['sample_idx']}: {e}")
            traceback.print_exc()
            return None

    # ── 主循环 ────────────────────────────────────────────────────────
    def run(self):
        args = self.args
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.dump_lm_head:
            self.dump_lm_head(output_dir)

        records = load_records(args.data_path, args.data_format, args.max_samples)
        n_total = len(records)
        shard_size = args.samples_per_shard
        num_shards = math.ceil(n_total / shard_size)
        print(f"\n开始抽取: {n_total} 条 -> {num_shards} shards (每 {shard_size})")
        print(f"输出: {output_dir.resolve()}  N={args.N} max_ans={args.max_answer_tokens}")
        print("=" * 72)

        total_pairs = failed = 0
        manifest: list[dict] = []
        for shard_idx in range(num_shards):
            start = shard_idx * shard_size
            end = min(start + shard_size, n_total)
            shard_dir = output_dir / f"shard_{shard_idx:04d}"
            shard_dir.mkdir(exist_ok=True)
            shard_pairs = 0
            shard_samples = []
            pbar = tqdm(range(start, end), desc=f"Shard {shard_idx:04d}", unit="sample")
            for local_i, global_i in enumerate(pbar):
                result = self.process_sample(records[global_i])
                if result is None:
                    failed += 1
                    pbar.set_postfix({"fail": failed})
                    continue
                M = result.pop("M")
                fname = f"sample_{global_i:05d}.pt"
                torch.save(result, shard_dir / fname)
                shard_samples.append(global_i)
                manifest.append({"sample_idx": global_i, "shard_idx": shard_idx,
                                 "file": f"shard_{shard_idx:04d}/{fname}", "M": M})
                shard_pairs += M
                total_pairs += M
                pbar.set_postfix({"pairs": shard_pairs, "fail": failed})
                if (local_i + 1) % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            with open(shard_dir / "meta.json", "w") as f:
                json.dump({"shard_idx": shard_idx, "num_samples": len(shard_samples),
                           "num_pairs": shard_pairs, "sample_indices": shard_samples}, f)
            print(f"  Shard {shard_idx:04d}: {len(shard_samples)} 样本, {shard_pairs} 对")

        meta = {
            "data_path": args.data_path, "data_format": args.data_format,
            "model_path": args.model_path, "N": args.N, "dim": self.dim,
            "save_dtype": args.save_dtype, "max_answer_tokens": args.max_answer_tokens,
            "total_records": n_total, "succeeded": n_total - failed, "failed": failed,
            "total_pairs": total_pairs, "num_shards": num_shards,
            "has_lm_head": bool(args.dump_lm_head), "samples": manifest,
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print("=" * 72)
        print(f"✅ Stage 0 完成: 成功 {meta['succeeded']}/{n_total}, 共 {total_pairs} (X, HQ_tea) 对")


def _make_prompt(context: str, question: str) -> str:
    """chat user 消息正文 (Context + Question); assistant 生成提示交给 chat template."""
    return f"Context:\n{context}\n\nQuestion:\n{question}"


def build_chat_prompt_ids(tokenizer, context: str, question: str, device=None, thinking=False):
    """按 Qwen3 chat template 构造 prompt token ids (add_generation_prompt=True).

    这样是"对话"而非"文本续写", 模型答完会正常吐 <|im_end|>(151645) 停下,
    AN 自适应真实答案长度, 不再一路跑到 max_answer_tokens (官方推荐用法).
    teacher / student / 推理三处统一用它, 保证 prompt 格式一致 (对齐前提).
    """
    if thinking:
        system_prompt_item={"role":"system","content":"You are a precise QA assistant. Answer with the short answer and thinking. If you encounter a math problem, you should calculate it step by step. Think it through several times and criticize yourself a few times."}
    else:
        system_prompt_item={"role":"system","content":"You are a precise QA assistant. Answer ONLY with the short answer. No explanation."}
    messages = [
        system_prompt_item,
        {"role": "user", "content": _make_prompt(context, question)}
        ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)     # 官方示例写法
    # 模板已含全部特殊 token, add_special_tokens=False 避免再加 bos/eos (与 student 侧一致)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    return ids.to(device) if device is not None else ids


def resolve_eos_ids(model) -> list[int]:
    """从 generation_config 取停止 token (Qwen3: [<|im_end|>=151645, <|endoftext|>=151643])."""
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = model.config.eos_token_id
    return list(eos) if isinstance(eos, (list, tuple)) else [eos]


def main():
    args = parse_args()
    extractor = Stage0Extractor(args)
    extractor.load_model()
    extractor.run()


if __name__ == "__main__":
    main()
