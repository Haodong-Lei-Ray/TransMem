#!/usr/bin/env python3
"""
推理 + 评测: 逐步解码 answer, 算长度外推 accuracy. 含三种模式对比.

模式 (--mode):
  teacher  : 冻结 LLM 看 golden 短文 (C_S, Q)  —— 上界, 仅 sanity check (测试时本无 C_S)
  student  : 冻结 LLM 看含干扰长文 (C_L, Q)     —— 基线 (无 TransMem)
  transmem : 冻结 LLM + TransMem 看 (C_L, Q)    —— 本方法 (TransMem 在环逐步解码, §6)

plan §9.6: 先验证"教师确实比学生准"(teacher >> student), 再谈 TransMem 能否把 student 拉向 teacher.

用法:
  # sanity check: 教师 vs 学生 (不需 ckpt)
  python -m transmem.evaluate --eval_file ../Project3/data/hotpotqa/eval_50.json \
    --model_path /path/to/Qwen3-4B-Instruct-2507 --mode teacher --max_samples 128
  python -m transmem.evaluate --eval_file ... --model_path ... --mode student --max_samples 128
  # TransMem
  python -m transmem.evaluate --eval_file ... --model_path ... --mode transmem \
    --ckpt checkpoints/offpolicy/latest.pt --N 4 --max_samples 128
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transmem import TransMemConfig, TransMem
from transmem.extract_features import (
    load_records, extract_cs, build_chat_prompt_ids, resolve_eos_ids)
from transmem.train_onpolicy import OnPolicyRollout

_DTYPE = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_args():
    p = argparse.ArgumentParser(description="TransMem 推理 + 长度外推评测")
    p.add_argument("--eval_file", required=True, help="评测数据 (json / qasper / parquet)")
    p.add_argument("--data_format", default="json",
                   choices=["json", "qasper", "parquet", "hotpotqa-agentmem", "longmemeval"])
    p.add_argument("--model_path", required=True)
    p.add_argument("--mode", required=True, choices=["teacher", "student", "transmem"])
    p.add_argument("--ckpt", default=None, help="transmem 模式的 TransMem checkpoint")
    p.add_argument("--config", default="transmem/config.json")
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--max_answer_tokens", type=int, default=50)
    p.add_argument("--max_samples", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--attn_impl", default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--print_examples", type=int, default=3)
    p.add_argument("--gate_diagnostics", default=None,
                   help="可选 JSON 路径: 保存逐样本/逐 token/逐层 gate 轨迹")
    return p.parse_args()


# ── 评分: HotpotQA 风格 normalize + exact / substring ──────────────────

def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def score(pred: str, gold: str) -> tuple[int, int]:
    """返回 (exact, contains). exact=normalize 相等; contains=gold 是 pred 子串."""
    np_, ng = _normalize(pred), _normalize(gold)
    exact = int(np_ == ng)
    contains = int(ng in np_) if ng else 0
    return exact, contains


# ── 评测器 ─────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, args):
        if getattr(args, "gate_diagnostics", None) is None:
            args.gate_diagnostics = None    # SimpleNamespace 复用入口 (eval_locomo 等) 兜底
        if args.gate_diagnostics and args.mode != "transmem":
            raise ValueError("--gate_diagnostics 只适用于 --mode transmem")
        self.args = args
        self.device = torch.device(args.device)
        self.dtype = _DTYPE[args.dtype]
        self._load_model()
        self.mem = None
        self.rollout = None
        self._last_gate_trace = None
        if args.mode == "transmem":
            self._load_transmem()

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        a = self.args
        print(f"加载 backbone: {a.model_path}")
        self.tok = AutoTokenizer.from_pretrained(
            a.model_path, local_files_only=True, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            a.model_path, torch_dtype=self.dtype, local_files_only=True,
            trust_remote_code=True, attn_implementation=a.attn_impl).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._eos_ids = resolve_eos_ids(self.model)

    def _load_transmem(self):
        a = self.args
        if a.ckpt:
            ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
            cfg_dict = ckpt["config"]
            if isinstance(cfg_dict, dict) and cfg_dict.get("layered"):
                # v3 计划 6: TransMem-Layer ckpt (config 带 layered=true) 自动分发
                from transmem.layered import (LayeredConfig, TransMemLayered,
                                              LayeredRollout)
                lcfg = LayeredConfig.from_dict(cfg_dict)
                self.mem = TransMemLayered(lcfg).to(self.device, dtype=self.dtype)
                self.mem.load_state_dict(ckpt["model_state_dict"], strict=True)
                self.mem.eval()
                self.rollout = LayeredRollout(self.model, self.tok, self.device,
                                              self.mem, self.dtype)
                print(f"加载 TransMemLayered: {a.ckpt} (step={ckpt.get('global_step')}, "
                      f"inject_layers={lcfg.inject_layers})")
                return
            cfg = TransMemConfig(**cfg_dict)
            self.mem = TransMem(cfg).to(self.device, dtype=self.dtype)
            self.mem.load_state_dict(ckpt["model_state_dict"], strict=True)
            print(f"加载 TransMem: {a.ckpt} (step={ckpt.get('global_step')})")
        else:
            cfg = TransMemConfig.from_json(a.config)
            cfg.n_mem = a.N
            self.mem = TransMem(cfg).to(self.device, dtype=self.dtype)
            print("[WARN] transmem 模式未给 --ckpt, 用随机初始化 (仅调试)")
        self.mem.eval()
        # rollout 的 N/取位公式跟随 ckpt config (老 ckpt 无 hm_mode 字段 -> floor),
        # 与训练特征严格一致; --N 仅在无 ckpt 调试时生效.
        hm_mode = getattr(cfg, "hm_mode", "floor")
        if a.N != cfg.n_mem:
            print(f"[INFO] rollout 用 ckpt 的 n_mem={cfg.n_mem}/hm_mode={hm_mode} (忽略 --N {a.N})")
        self.rollout = OnPolicyRollout(self.model, self.tok, self.device, cfg.n_mem,
                                       self.dtype, hm_mode=hm_mode)

    @torch.no_grad()
    def _greedy_plain(self, context: str, question: str) -> str:
        """冻结 LLM 贪心解码 (无 TransMem), 用于 teacher / student 基线."""
        cq_ids = build_chat_prompt_ids(self.tok, context, question, self.device)
        gen = self.model.generate(
            input_ids=cq_ids, attention_mask=torch.ones_like(cq_ids),
            max_new_tokens=self.args.max_answer_tokens, do_sample=False,
            pad_token_id=self.tok.pad_token_id, eos_token_id=self._eos_ids)
        ids = gen[0, cq_ids.shape[1]:].tolist()
        return self.tok.decode(ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def _greedy_transmem(self, context_long: str, question: str) -> str:
        """冻结 LLM + TransMem 贪心逐步解码 (§6): TransMem 在环."""
        _, _, ans_ids = self.rollout.student_rollout(
            self.mem, context_long, question, self.args.max_answer_tokens,
            sample=False, temperature=1.0,
            collect_gate_diagnostics=bool(self.args.gate_diagnostics))
        self._last_gate_trace = self.rollout.last_gate_trace
        return self.tok.decode(ans_ids, skip_special_tokens=True).strip()

    def predict(self, rec) -> str:
        mode = self.args.mode
        if mode == "teacher":
            gi = rec.get("golden_index")
            cs = rec.get("cs_text") or (extract_cs(rec["context"], gi) if gi is not None else "")
            if not cs:
                return ""
            return self._greedy_plain(cs, rec["question"])
        if mode == "student":
            return self._greedy_plain(rec["context"], rec["question"])
        return self._greedy_transmem(rec["context"], rec["question"])

    def run(self):
        a = self.args
        records = load_records(a.eval_file, a.data_format, a.max_samples)
        n_exact = n_contain = n = 0
        examples = []
        gate_traces = []
        for rec in tqdm(records, desc=f"eval[{a.mode}]", unit="q"):
            gold = rec["ground_truth"]
            if not gold:
                continue
            pred = self.predict(rec)
            if a.mode == "transmem" and a.gate_diagnostics and self._last_gate_trace:
                gate_traces.append({
                    "sample_idx": rec.get("sample_idx", n),
                    "question": rec.get("question", ""),
                    **self._last_gate_trace,
                })
            e, c = score(pred, gold)
            n_exact += e
            n_contain += c
            n += 1
            if len(examples) < a.print_examples:
                examples.append((rec["question"][:80], gold, pred[:80], e, c))

        print("=" * 72)
        print(f"文件: {Path(a.eval_file).name}  模式: {a.mode}  样本: {n}")
        print(f"  Exact   : {n_exact}/{n} = {n_exact/max(n,1):.3f}")
        print(f"  Contains: {n_contain}/{n} = {n_contain/max(n,1):.3f}")
        print("-" * 72)
        for q, g, p, e, c in examples:
            print(f"  Q: {q}\n    gold={g!r}  pred={p!r}  exact={e} contains={c}")
        print("=" * 72)
        if a.gate_diagnostics:
            path = Path(a.gate_diagnostics)
            path.parent.mkdir(parents=True, exist_ok=True)
            summary = summarize_gate_traces(gate_traces)
            path.write_text(json.dumps(
                {"samples": gate_traces, "summary": summary},
                indent=2, ensure_ascii=False))
            for layer, values in summary["layers"].items():
                print(f"  gate[{layer}]: mean={values['mean']:.3f} "
                      f"std={values['std']:.3f} p10/50/90="
                      f"{values['p10']:.3f}/{values['p50']:.3f}/{values['p90']:.3f} "
                      f"<.25={values['frac_lt_025']:.3f} >1.75={values['frac_gt_175']:.3f}")
            print(f"Gate diagnostics: {path}")
        return {"exact": n_exact / max(n, 1), "contains": n_contain / max(n, 1), "n": n}


def summarize_gate_traces(traces: list[dict]) -> dict:
    """Aggregate per-layer distributions and answer-position gate curves."""
    by_layer: dict[str, dict[str, list[float]]] = {}
    by_position: dict[str, list[list[float]]] = {}
    for sample in traces:
        for layer, values in sample.get("layers", {}).items():
            target = by_layer.setdefault(
                str(layer), {"gate": [], "ms_norm": [], "delta_norm": []})
            for name in target:
                target[name].extend(float(value) for value in values.get(name, []))
            positions = by_position.setdefault(str(layer), [])
            for index, value in enumerate(values.get("gate", [])):
                while len(positions) <= index:
                    positions.append([])
                positions[index].append(float(value))

    layers = {}
    curves = {}
    for layer, values in by_layer.items():
        gate = torch.tensor(values["gate"], dtype=torch.float32)
        if gate.numel() == 0:
            continue
        quantiles = torch.quantile(gate, torch.tensor([0.1, 0.5, 0.9]))
        layers[layer] = {
            "count": int(gate.numel()),
            "mean": float(gate.mean()),
            "std": float(gate.std(unbiased=False)),
            "p10": float(quantiles[0]),
            "p50": float(quantiles[1]),
            "p90": float(quantiles[2]),
            "frac_lt_025": float((gate < 0.25).float().mean()),
            "frac_gt_175": float((gate > 1.75).float().mean()),
            "ms_norm_mean": float(torch.tensor(values["ms_norm"]).mean()),
            "delta_norm_mean": float(torch.tensor(values["delta_norm"]).mean()),
        }
        curves[layer] = [
            {"token_index": index, "mean": sum(items) / len(items), "count": len(items)}
            for index, items in enumerate(by_position[layer]) if items
        ]
    return {"layers": layers, "token_index_curves": curves}


def main():
    Evaluator(parse_args()).run()


if __name__ == "__main__":
    main()
