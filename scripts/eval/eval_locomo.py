#!/usr/bin/env python3
"""LoCoMo 评测驱动: teacher / student / transmem 三模式 (复用, 不重造).

复用两边现成代码 (coding-rule #4):
  - 预测端: transmem.evaluate.Evaluator —— prompt 走 build_chat_prompt_ids,
    与 Stage0 特征抽取 / off-policy 训练格式完全一致 (对齐前提).
  - 协议端: Project1/delta-Mem deltamem.eval.locomo_protocol —— 官方对话拼接
    (DATE + CONVERSATION)、类别处理 (cat2 加日期提示 / cat3 取第一答案 /
    cat1 multi-answer F1)、canonicalize、逐题 F1 评分、类别名.

与 delta-Mem 官方基线的差异 (对比数字时注意):
  - 解码: 贪心 (transmem 逐步解码只支持确定性对比), 非官方 temp=0.4 采样;
  - prompt: TransMem 训练格式 "Context:...\n\nQuestion:..." + 短答 system prompt,
    非 LoCoMo 官方 QA 包装. 三模式内部可比, 与论文表数字不严格可比.

三模式:
  teacher  : C_S = evidence 轮次所在会话片段 (dia_id 定位)  —— 上界
  student  : C_L = 完整多天对话                             —— 基线
  transmem : C_L + TransMem 在环逐步解码                    —— 本方法

断点续跑: 逐题追加 {output_json}.{mode}.progress.jsonl, 重跑自动跳过已完成题.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

_PROJ4 = "/mnt/petrelfs/leihaodong/Project4"
_DELTA = "/mnt/petrelfs/leihaodong/Project1/delta-Mem"
for p in (_PROJ4, _DELTA):
    if p not in sys.path:
        sys.path.insert(0, p)

from transmem.evaluate import Evaluator  # noqa: E402
from transmem.rl import split_thinking_answer  # noqa: E402
from deltamem.eval.locomo_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    OFFICIAL_CONV_START_PROMPT,
    build_official_context_text,
    canonicalize_locomo_prediction,
    prepare_locomo_question,
    render_locomo_turn,
    score_locomo_prediction,
)

# 短答指令 (裁自 OFFICIAL_QA_PROMPT), 三模式统一拼进 question, 否则 teacher 面对
# 短 evidence 爱写整句, 被 token-F1 重罚 (啰嗦伪影). 注意刻意去掉了官方的
# "Answer with exact words from the conversation": 实测它让模型照抄 "yesterday" /
# "1:56 pm on ..." 原文, 与 cat2 需要的日期换算 (Use DATE ...) 直接冲突, cat2 全线掉分.
SHORT_ANSWER_HINT = (
    "Write a short answer in a few words. "
    "Do not write complete and lengthy sentences.")

from tqdm import tqdm  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="TransMem LoCoMo 评测")
    p.add_argument("--data_file", required=True, help="locomo10.json")
    p.add_argument("--model_path", required=True)
    p.add_argument("--mode", required=True, choices=["teacher", "student", "transmem"])
    p.add_argument("--ckpt", default=None, help="transmem 模式的 checkpoint")
    p.add_argument("--config", default="transmem/config.json")
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--max_answer_tokens", type=int, default=50)  # LoCoMo 官方 50
    p.add_argument("--max_prompt_tokens", type=int, default=None)
    p.add_argument("--categories", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--max_questions", type=int, default=None, help="总题数上限 (冒烟)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--attn_impl", default="sdpa",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--output_json", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_examples", type=int, default=3)
    p.add_argument("--num_shards", type=int, default=1,
                   help="将完整问题列表轮转切成多少份，默认 1 保持原行为")
    p.add_argument("--shard_index", type=int, default=0,
                   help="当前进程负责的分片编号，范围 [0, num_shards)")
    p.add_argument("--gate_diagnostics", default=None,
                   help="transmem 模式: 逐题收集 gate trace, 汇总写入该 json 路径")
    p.add_argument("--thinking", action="store_true",
                   help="思考模式 (经 SimpleNamespace 透传给 Evaluator): hybrid 模型走 "
                        "enable_thinking, 纯 instruct 走 prompt hack; 输出按 </think> "
                        "或 'Answer:' 切分, thinking 与 answer 分开保存")
    return p.parse_args()


# ── 数据构造: locomo10.json -> transmem.evaluate 的 rec 字段 ─────────────

def _evidence_cs_text(conversation: dict, evidence: list) -> str:
    """按 dia_id (如 'D1:3') 定位 evidence 轮次拼 C_S: 说话人开场白 + 按会话分组
    带 DATE 头 (时间信息必须附上, cat2 全靠它)."""
    wanted = {str(e).strip() for e in (evidence or []) if str(e).strip()}
    if not wanted:
        return ""
    by_session: dict[int, list[dict]] = defaultdict(list)
    session_nums = sorted(
        int(k.split("_")[-1]) for k in conversation
        if k.startswith("session_") and not k.endswith("date_time"))
    for sn in session_nums:
        for dialog in conversation.get(f"session_{sn}", []):
            if dialog.get("dia_id") in wanted:
                by_session[sn].append(dialog)
    parts = []
    for sn in sorted(by_session):
        date = conversation.get(f"session_{sn}_date_time", "")
        turns = "".join(render_locomo_turn(d) for d in by_session[sn]).rstrip()
        parts.append(f"DATE: {date}\nCONVERSATION:\n{turns}")
    start = OFFICIAL_CONV_START_PROMPT.format(
        conversation["speaker_a"], conversation["speaker_b"])
    return start + "\n\n".join(parts)


def build_records(data_file: str, tokenizer, categories, max_questions, seed):
    """展平成逐题 record; context 每段对话只拼一次 (官方全量拼接, 预算给足)."""
    with open(data_file) as f:
        samples = json.load(f)
    cat_set = {int(c) for c in categories}
    records = []
    for sample in samples:
        conv = sample["conversation"]
        # 复用官方拼接 (question_prompt 传空, 预算给到 1e9 = 不截断的全历史)
        context = build_official_context_text(
            sample, tokenizer, "", max_context_tokens=10**9)
        for qi, qa in enumerate(sample["qa"]):
            cat = int(qa["category"])
            if cat not in cat_set:
                continue
            compat = dict(qa)
            if cat == 5 and "answer" not in compat:
                compat["answer"] = "No information available"
            spec = prepare_locomo_question(
                compat, sample_id=str(sample["sample_id"]),
                question_index=qi, seed=seed)
            records.append({
                "key": f"{sample['sample_id']}:{qi}",
                # spec.prompt_text 已含 cat2 日期提示 / cat5 选项; 再拼官方短答指令
                "question": f"{spec.prompt_text}\n{SHORT_ANSWER_HINT}",
                "context": context,
                "cs_text": _evidence_cs_text(conv, qa.get("evidence")),
                "ground_truth": str(compat.get("answer", "")),
                "qa": compat,
                "spec": spec,
                "category": cat,
            })
    if max_questions:
        records = records[:max_questions]
    return records


# ── 断点续跑 ────────────────────────────────────────────────────────────

def load_progress(path: Path) -> dict:
    done = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    done[row["key"]] = row
    return done


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards 必须 >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index 必须位于 [0, num_shards)")
    ev_args = types.SimpleNamespace(**vars(args))  # Evaluator 只取其中同名字段
    evaluator = Evaluator(ev_args)

    records = build_records(
        args.data_file, evaluator.tok, args.categories, args.max_questions, args.seed)
    if args.num_shards > 1:
        records = records[args.shard_index::args.num_shards]
    n_no_ev = sum(1 for r in records if not r["cs_text"])
    print(f"题数: {len(records)} (categories={args.categories}); "
          f"无 evidence: {n_no_ev}" + (" [teacher 模式下这些题预测为空]"
                                       if args.mode == "teacher" else ""))
    if args.num_shards > 1:
        print(f"分片: {args.shard_index + 1}/{args.num_shards}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prog_path = out_path.with_suffix(f".{args.mode}.progress.jsonl")
    done = load_progress(prog_path)
    if done:
        print(f"断点续跑: 已有 {len(done)} 题, 跳过")

    examples = []
    gate_traces = []
    with open(prog_path, "a") as prog_f:
        for rec in tqdm(records, desc=f"locomo[{args.mode}]", unit="q"):
            if rec["key"] in done:
                continue
            # teacher 且无 evidence 时 Evaluator.predict 自己返回 ""
            raw = evaluator.predict(rec) or ""
            if args.gate_diagnostics and getattr(evaluator, "_last_gate_trace", None):
                gate_traces.append({"key": rec["key"], "category": rec["category"],
                                    **evaluator._last_gate_trace})
            parsed = split_thinking_answer(raw)
            canonical = canonicalize_locomo_prediction(parsed.answer, rec["spec"])
            f1 = score_locomo_prediction(rec["qa"], canonical)
            row = {"key": rec["key"], "category": rec["category"],
                   "question": rec["qa"]["question"],
                   "answer": rec["ground_truth"],
                   "raw_prediction": raw, "thinking": parsed.thinking,
                   "has_answer_marker": parsed.has_answer_marker,
                   "format_valid": bool(
                       parsed.has_answer_marker and parsed.thinking.strip()),
                   "prediction": canonical,
                   "score": round(float(f1), 6)}
            prog_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            prog_f.flush()
            done[rec["key"]] = row
            if len(examples) < args.print_examples:
                examples.append(row)

    # ── 汇总 (只统计本次请求范围内的题) ──────────────────────────────────
    rows = [done[r["key"]] for r in records if r["key"] in done]
    cat_scores, cat_counts = defaultdict(float), defaultdict(int)
    for row in rows:
        cat_scores[row["category"]] += row["score"]
        cat_counts[row["category"]] += 1
    total = sum(cat_counts.values())
    overall = sum(cat_scores.values()) / max(total, 1)

    summary = {
        "mode": args.mode, "ckpt": args.ckpt, "model_path": args.model_path,
        "decode": "greedy", "prompt_format": "transmem_chat",
        "thinking": bool(getattr(args, "thinking", False)),
        "categories": args.categories, "num_questions": total,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "overall_f1": round(overall, 4),
        "format_valid_rate": (
            sum(bool(row.get("format_valid")) for row in rows)
            / max(total, 1)),
        "category_f1": {
            str(c): {"name": CATEGORY_NAMES.get(c, "unknown"),
                     "score": round(cat_scores[c] / cat_counts[c], 4),
                     "count": cat_counts[c]}
            for c in sorted(cat_counts)},
    }
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": rows}, f,
                  ensure_ascii=False, indent=2)

    print("=" * 72)
    print(f"LoCoMo  模式: {args.mode}  题数: {total}")
    print(f"  Overall F1: {overall:.4f}")
    for c in sorted(cat_counts):
        print(f"  cat{c} {CATEGORY_NAMES.get(c, '?'):12s}: "
              f"{cat_scores[c] / cat_counts[c]:.4f}  (n={cat_counts[c]})")
    print("-" * 72)
    for row in examples:
        print(f"  Q: {row['question'][:80]}\n    gold={row['answer']!r}  "
              f"pred={row['prediction'][:80]!r}  f1={row['score']:.3f}")
    print("=" * 72)
    print(f"结果: {out_path}")

    if args.gate_diagnostics:
        from transmem.evaluate import summarize_gate_traces
        diag_path = Path(args.gate_diagnostics)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diag_path, "w") as f:
            json.dump({"num_traces": len(gate_traces),
                       "summary": summarize_gate_traces(gate_traces),
                       "samples": gate_traces}, f, ensure_ascii=False)
        print(f"gate diagnostics -> {diag_path} ({len(gate_traces)} traces)")


if __name__ == "__main__":
    main()
