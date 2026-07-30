#!/usr/bin/env python3
"""挑出 Stage0 抽取结果里评分系统判 ✗ (exact=0 且 contains=0) 的样本, 逐条落盘人读 json.

复用 dump_one_sample.py 的人读字段格式 (paper_id/Q/gold/C_S/C_L/needle/teacher_answer/
HM_stu/per_token), 但直接读已缓存的 shard .pt (hm_stu/hq_stu/hq_tea/answer_ids/answer_text),
不重跑模型 forward, 只用 tokenizer (CPU) 做 token 解码 + 长度统计, 跑得很快.

用法:
  python scripts/dump_wrong_samples.py \
    --stage0_dir data/qasper_data/stage0_train_short128 \
    --model_path /mnt/petrelfs/leihaodong/models/Qwen3-4B-Instruct-2507 \
    --output_dir scripts/smoke/wrong_train_short128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transmem.extract_features import load_records
from transmem.evaluate import score


def parse_args():
    p = argparse.ArgumentParser(description="挑出评分系统判 ✗ 的 Stage0 样本, 逐条落盘人读 json")
    p.add_argument("--stage0_dir", required=True, help="已抽取的 Stage0 输出目录 (含 meta.json)")
    p.add_argument("--model_path",
                   default="/mnt/petrelfs/leihaodong/models/Qwen3-4B-Instruct-2507")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def _tok_stats(t: torch.Tensor, k: int = 5):
    """一行 hidden -> {norm, head}. t: [dim]."""
    tf = t.float()
    return {"l2": round(float(tf.norm()), 4),
            "mean": round(float(tf.mean()), 5),
            "head": [round(float(x), 4) for x in tf[:k].tolist()]}


def main():
    a = parse_args()
    stage0_dir = Path(a.stage0_dir)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.load(open(stage0_dir / "meta.json"))
    records = load_records(meta["data_path"], meta["data_format"], None)
    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True,
                                        trust_remote_code=True)

    n_wrong = 0
    for s in meta["samples"]:
        obj = torch.load(stage0_dir / s["file"], map_location="cpu")
        rec = records[s["sample_idx"]]
        pred, gold = obj["answer_text"], rec["ground_truth"]
        exact, contains = score(pred, gold)
        if exact or contains:
            continue
        n_wrong += 1

        hm, hq_stu, hq_tea = obj["hm_stu"], obj["hq_stu"], obj["hq_tea"]
        ans_ids = obj["answer_ids"].tolist()
        AN = len(ans_ids)
        cs_text = rec.get("cs_text") or ""
        len_cl_tok = tok(rec["context"], add_special_tokens=False,
                          return_tensors="pt").input_ids.shape[1]
        len_cs_tok = tok(cs_text, add_special_tokens=False,
                          return_tensors="pt").input_ids.shape[1]

        per_token = []
        for i, tid in enumerate(ans_ids):
            per_token.append({
                "pos": i + 1,
                "token_id": int(tid),
                "token": tok.decode([tid]),
                "hq_tea": _tok_stats(hq_tea[i]),
                "hq_stu": _tok_stats(hq_stu[i]),
            })

        readable = {
            "sample_idx": s["sample_idx"],
            "score(exact/contains, 都为0才落这里)": {"exact": exact, "contains": contains},
            "paper_id": rec.get("paper_id", ""),
            "question": rec["question"],
            "ground_truth_ref": gold,
            "C_S(evidence)": {"chars": len(cs_text), "tokens": int(len_cs_tok),
                              "preview": cs_text[:300]},
            "C_L(full_paper)": {"chars": len(rec["context"]), "tokens": int(len_cl_tok),
                                "preview": rec["context"][:200]},
            "needle_in_haystack": bool(cs_text[:60] in rec["context"]) if cs_text else None,
            "teacher_answer": {"AN": AN, "text": obj["answer_text"], "ids": ans_ids},
            "shapes": {"hm_stu": list(hm.shape), "hq_stu": list(hq_stu.shape),
                       "hq_tea": list(hq_tea.shape), "dim": obj["dim"], "N": obj["N"],
                       "save_dtype": str(hm.dtype)},
            "HM_stu(N段记忆,每段末位hidden)": [_tok_stats(hm[j]) for j in range(hm.shape[0])],
            "per_token(AN个)": per_token,
        }
        fname = out / f"wrong_{s['sample_idx']:05d}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(readable, f, ensure_ascii=False, indent=2)
        print(f"[{s['sample_idx']:>3}] M={AN:<4} teacher={pred[:50]!r:<55} "
              f"gold={gold[:40]!r:<45} -> {fname.name}")

    print("=" * 72)
    print(f"✅ 共 {n_wrong}/{len(meta['samples'])} 条判 ✗, 已落盘到 {out}/")


if __name__ == "__main__":
    main()
