#!/usr/bin/env python3
"""Evaluate one resumable shard on official HotpotQA answer EM/F1."""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from transmem.evaluate import Evaluator, score


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    if prediction in {"yes", "no", "noanswer"} and prediction != ground_truth:
        return 0.0
    if ground_truth in {"yes", "no", "noanswer"} and prediction != ground_truth:
        return 0.0
    pred_tokens = prediction.split()
    gold_tokens = ground_truth.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def load_progress(path: Path) -> dict[str, dict]:
    records = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            records[str(record["question_id"])] = record
        except (json.JSONDecodeError, KeyError):
            continue
    return records


def summarize(records: list[dict]) -> dict:
    count = len(records)
    return {
        "num_questions": count,
        "answer_em": sum(float(record["em"]) for record in records) / max(count, 1),
        "answer_f1": sum(float(record["f1"]) for record in records) / max(count, 1),
        "contains": sum(float(record["contains"]) for record in records) / max(count, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=("student", "transmem"),
                        default="transmem")
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--max-answer-tokens", type=int, default=50)
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard ID")
    if args.mode == "transmem" and args.ckpt is None:
        raise ValueError("--ckpt is required in transmem mode")

    all_items = json.loads(args.data_file.read_text(encoding="utf-8"))
    items = all_items[args.shard_id :: args.num_shards]
    progress_path = args.output_json.with_suffix(".progress.jsonl")
    completed = load_progress(progress_path)

    evaluator = Evaluator(SimpleNamespace(
        model_path=args.model_path,
        mode=args.mode,
        ckpt=str(args.ckpt) if args.ckpt is not None else None,
        config="transmem/config.json",
        N=4,
        max_answer_tokens=args.max_answer_tokens,
        max_prompt_tokens=None,
        device="cuda:0",
        dtype="bfloat16",
        attn_impl=args.attn_impl,
        gate_diagnostics=None,
        thinking=args.thinking,
    ))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as progress:
        for item in tqdm(items, desc=f"hotpot[{args.shard_id}/{args.num_shards}]", unit="q"):
            question_id = str(item["question_id"])
            if question_id in completed:
                continue
            if args.mode == "transmem":
                prediction = evaluator._greedy_transmem(
                    item["context"], item["question"])
            else:
                prediction = evaluator._greedy_plain(
                    item["context"], item["question"])
            em, contains = score(prediction, item["answer"])
            record = {
                "question_id": question_id,
                "question": item["question"],
                "ground_truth": item["answer"],
                "prediction": prediction,
                "em": em,
                "f1": f1_score(prediction, item["answer"]),
                "contains": contains,
                "type": item.get("type"),
                "level": item.get("level"),
            }
            progress.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.flush()
            completed[question_id] = record

    expected_ids = {str(item["question_id"]) for item in items}
    records = [completed[question_id] for question_id in sorted(expected_ids)]
    if len(records) != len(items):
        raise RuntimeError(f"incomplete shard: {len(records)}/{len(items)}")
    payload = {
        "model_path": args.model_path,
        "mode": args.mode,
        "ckpt": str(args.ckpt) if args.ckpt is not None else None,
        "decode": "greedy_thinking" if args.thinking else "greedy_nonthinking",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "summary": summarize(records),
        "records": records,
    }
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
