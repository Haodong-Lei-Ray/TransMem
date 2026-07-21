#!/usr/bin/env python3
"""Sharded LongMemEval generation for teacher/student/TransMem evaluation.

The prediction path is the same ``transmem.evaluate.Evaluator`` used by the
legacy single-GPU entrypoint.  This driver only adds deterministic sharding,
JSONL hypotheses, and per-question progress so several GPU workers can share
one LongMemEval split safely.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from transmem.evaluate import Evaluator, score
from transmem.extract_features import load_records
from transmem.rl import split_thinking_answer


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel TransMem LongMemEval worker")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--mode", default="transmem",
                        choices=["teacher", "student", "transmem"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--config", default="transmem/config.json")
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--max_answer_tokens", type=int, default=50)
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "bfloat16"])
    parser.add_argument("--attn_impl", default="sdpa",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--print_examples", type=int, default=1)
    return parser.parse_args()


def load_progress(path: Path) -> dict[str, dict]:
    done = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A preemption may leave one partial final line.  All earlier
                # complete rows remain reusable on the next Slurm attempt.
                print(f"WARN: ignore incomplete progress line {line_number} in {path}")
                break
            done[str(row["question_id"])] = row
    return done


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    if args.mode == "transmem" and not args.ckpt:
        raise ValueError("--ckpt is required in transmem mode")

    # Evaluator expects these attributes even though this driver handles its
    # own record loop and diagnostics.
    ev_args = types.SimpleNamespace(**vars(args), gate_diagnostics=None)
    evaluator = Evaluator(ev_args)
    records = load_records(args.data_file, "longmemeval", args.max_samples)
    records = records[args.shard_index::args.num_shards]

    output = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_progress(output)
    if done:
        print(f"Resume shard {args.shard_index}: {len(done)} completed questions")

    examples = []
    with output.open("a", encoding="utf-8") as progress:
        for rec in tqdm(records, desc=f"lme[{args.mode}] shard {args.shard_index}", unit="q"):
            question_id = str(rec["question_id"])
            if question_id in done:
                continue
            raw_prediction = evaluator.predict(rec) or ""
            parsed = split_thinking_answer(raw_prediction)
            prediction = parsed.answer
            exact, contains = score(prediction, rec["ground_truth"])
            row = {
                "question_id": question_id,
                "question_type": rec.get("question_type", ""),
                "question": rec["question"],
                "answer": rec["ground_truth"],
                # This is the field consumed by the official LongMemEval judge.
                "hypothesis": prediction,
                "raw_prediction": raw_prediction,
                "thinking": parsed.thinking,
                "has_answer_marker": parsed.has_answer_marker,
                "format_valid": bool(
                    parsed.has_answer_marker and parsed.thinking.strip()),
                "exact": exact,
                "contains": contains,
                "sample_idx": int(rec.get("sample_idx", -1)),
            }
            progress.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.flush()
            done[question_id] = row
            if len(examples) < args.print_examples:
                examples.append(row)

    requested_ids = {str(record["question_id"]) for record in records}
    rows = [row for qid, row in done.items() if qid in requested_ids]
    rows.sort(key=lambda row: row["sample_idx"])
    if len(rows) != len(records):
        raise RuntimeError(
            f"Shard incomplete: expected {len(records)} rows, found {len(rows)}")

    by_type = defaultdict(lambda: {"n": 0, "exact": 0, "contains": 0})
    for row in rows:
        bucket = by_type[row["question_type"]]
        bucket["n"] += 1
        bucket["exact"] += int(row["exact"])
        bucket["contains"] += int(row["contains"])
    n = len(rows)
    summary = {
        "mode": args.mode,
        "model_path": args.model_path,
        "ckpt": args.ckpt,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_questions": n,
        "thinking": bool(args.thinking),
        "max_prompt_tokens": args.max_prompt_tokens,
        "format_valid_rate": (
            sum(bool(row.get("format_valid")) for row in rows) / max(n, 1)),
        "exact": sum(row["exact"] for row in rows) / max(n, 1),
        "contains": sum(row["contains"] for row in rows) / max(n, 1),
        "by_type": {
            key: {
                "n": value["n"],
                "exact": value["exact"] / value["n"],
                "contains": value["contains"] / value["n"],
            }
            for key, value in sorted(by_type.items())
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in examples:
        print(f"Q: {row['question'][:100]}\n"
              f"gold={row['answer']!r}\npred={row['hypothesis'][:160]!r}")


if __name__ == "__main__":
    main()
