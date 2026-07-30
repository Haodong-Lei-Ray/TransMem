#!/usr/bin/env python3
"""
单样本 Stage0 dump: 对一条 QA 跑教师生成 + 学生 forward, 落盘所有中间产物供眼看.

复用 transmem.extract_features.Stage0Extractor 的已验证逻辑(教师 rollout+hook HQ_tea、
学生 teacher-forcing HM/HQ_stu), 只额外写一份人读 json(每个 token、每个 hidden 的范数/前几维).

用法 (GPU):
  python scripts/dump_one_sample.py \
    --record data/qasper/sample_qa.json --data_format qasper \
    --model_path /mnt/petrelfs/leihaodong/models/Qwen3-4B-Instruct-2507 \
    --output_dir scripts/smoke --N 4 --max_answer_tokens 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transmem.extract_features import Stage0Extractor, load_records, _make_prompt


def parse_args():
    p = argparse.ArgumentParser(description="单样本 Stage0 dump")
    p.add_argument("--record", default="data/qasper/sample_qa.json")
    p.add_argument("--data_format", default="qasper", choices=["qasper", "json", "parquet"])
    p.add_argument("--model_path",
                   default="/mnt/petrelfs/leihaodong/models/Qwen3-4B-Instruct-2507")
    p.add_argument("--output_dir", default="scripts/smoke")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--save_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--max_answer_tokens", type=int, default=50)
    p.add_argument("--which", type=int, default=0, help="数据是列表时取第几条")
    return p.parse_args()


def _tok_stats(t: torch.Tensor, k: int = 5):
    """一行 hidden -> {norm, head}. t: [dim]."""
    tf = t.float()
    return {"l2": round(float(tf.norm()), 4),
            "mean": round(float(tf.mean()), 5),
            "head": [round(float(x), 4) for x in tf[:k].tolist()]}


def main():
    a = parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    recs = load_records(a.record, a.data_format, None)
    rec = recs[a.which]
    print(f"样本: paper_id={rec.get('paper_id','')} | Q={rec['question'][:80]!r}")

    ext = Stage0Extractor(SimpleNamespace(
        model_path=a.model_path, device=a.device, dtype=a.dtype, save_dtype=a.save_dtype,
        attn_impl=a.attn_impl, N=a.N, max_answer_tokens=a.max_answer_tokens))
    ext.load_model()

    # 复算 C_S / token 数 (供人读); process_sample 内部也会算一遍
    cs_text = rec.get("cs_text") or ""
    tok = ext.tokenizer
    len_cl_tok = tok(rec["context"], add_special_tokens=False, return_tensors="pt").input_ids.shape[1]
    len_cs_tok = tok(cs_text, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]

    result = ext.process_sample(rec)
    if result is None:
        print("[FATAL] process_sample 返回 None (C_S 空 / 生成空答案?)")
        sys.exit(1)

    # ── 落盘原始张量 (.pt, 即真正进 Stage1 的东西) ──────────────────────
    torch.save(result, out / "sample.pt")

    # ── 人读版 ─────────────────────────────────────────────────────────
    hm, hq_stu, hq_tea = result["hm_stu"], result["hq_stu"], result["hq_tea"]
    ans_ids = result["answer_ids"].tolist()
    AN = result["M"]
    per_token = []
    for i, tid in enumerate(ans_ids):
        per_token.append({
            "pos": i + 1,
            "token_id": int(tid),
            "token": tok.decode([tid]),
            "hq_tea": _tok_stats(hq_tea[i]),     # 教师第 i 步查询 hidden
            "hq_stu": _tok_stats(hq_stu[i]),     # 学生第 i 步查询 hidden
        })

    readable = {
        "paper_id": rec.get("paper_id", ""),
        "question": rec["question"],
        "ground_truth_ref": rec.get("ground_truth", ""),
        "C_S(evidence)": {"chars": len(cs_text), "tokens": int(len_cs_tok),
                          "preview": cs_text[:300]},
        "C_L(full_paper)": {"chars": len(rec["context"]), "tokens": int(len_cl_tok),
                            "preview": rec["context"][:200]},
        "needle_in_haystack": bool(cs_text[:60] in rec["context"]) if cs_text else None,
        "teacher_answer": {"AN": AN, "text": result["answer_text"], "ids": ans_ids},
        "shapes": {"hm_stu": list(hm.shape), "hq_stu": list(hq_stu.shape),
                   "hq_tea": list(hq_tea.shape), "dim": result["dim"], "N": result["N"],
                   "save_dtype": str(hm.dtype)},
        "HM_stu(N段记忆,每段末位hidden)": [_tok_stats(hm[j]) for j in range(hm.shape[0])],
        "per_token(AN个)": per_token,
    }
    with open(out / "sample_readable.json", "w", encoding="utf-8") as f:
        json.dump(readable, f, ensure_ascii=False, indent=2)

    # ── 终端摘要 ───────────────────────────────────────────────────────
    print("=" * 72)
    print(f"教师答案 (AN={AN}): {result['answer_text']!r}")
    print(f"  C_L {len_cl_tok} tok / C_S {len_cs_tok} tok  | needle_in_haystack="
          f"{readable['needle_in_haystack']}")
    print(f"  HM_stu {list(hm.shape)}  HQ_stu {list(hq_stu.shape)}  "
          f"HQ_tea {list(hq_tea.shape)}  dtype={hm.dtype}")
    print(f"  逐 token: " + " ".join(f"[{p['pos']}]{p['token']!r}" for p in per_token[:12])
          + (" ..." if AN > 12 else ""))
    print("=" * 72)
    print(f"✅ 已落盘:\n  {out/'sample.pt'}  (原始张量, Stage1 真正吃的)\n"
          f"  {out/'sample_readable.json'}  (人读版: 每 token + 每 hidden 的范数/前5维)")


if __name__ == "__main__":
    main()
