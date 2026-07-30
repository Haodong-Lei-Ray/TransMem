#!/usr/bin/env python3
"""Compare LoCoMo outputs after excluding questions used by locomo-train.json."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_data", type=Path, required=True)
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--new_result", type=Path, required=True)
    parser.add_argument("--student_result", type=Path, required=True)
    parser.add_argument("--prior_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def identity(qa: dict) -> tuple[str, str, int, tuple[str, ...]]:
    return (
        str(qa.get("question", "")),
        str(qa.get("answer", "")),
        int(qa.get("category", 0)),
        tuple(map(str, qa.get("evidence", []))),
    )


def train_keys(full_data: list[dict], train_data: list[dict]) -> set[str]:
    full_by_id = {str(item["sample_id"]): item for item in full_data}
    keys: set[str] = set()
    for train_item in train_data:
        sample_id = str(train_item["sample_id"])
        full_item = full_by_id[sample_id]
        positions: defaultdict[tuple, list[int]] = defaultdict(list)
        for index, qa in enumerate(full_item.get("qa", [])):
            positions[identity(qa)].append(index)
        for qa in train_item.get("qa", []):
            matches = positions[identity(qa)]
            if not matches:
                raise RuntimeError(
                    f"train QA cannot map to full data: sample={sample_id} "
                    f"question={qa.get('question')!r} matches={matches}"
                )
            # The source contains a few fully duplicated QA records and the
            # sampled train JSON does not retain original QA indices. Exclude
            # every indistinguishable duplicate to prevent train/test leakage.
            keys.update(f"{sample_id}:{index}" for index in matches)
    return keys


def summarize(records: list[dict]) -> dict:
    categories: defaultdict[int, list[float]] = defaultdict(list)
    for record in records:
        categories[int(record["category"])].append(float(record["score"]))
    scores = [float(record["score"]) for record in records]
    return {
        "num_questions": len(scores),
        "overall_f1": round(statistics.fmean(scores), 6),
        "category_f1": {
            str(category): {
                "score": round(statistics.fmean(values), 6),
                "count": len(values),
            }
            for category, values in sorted(categories.items())
        },
    }


def paired_delta(new_records: list[dict], baseline_records: list[dict]) -> dict:
    new = {str(record["key"]): float(record["score"]) for record in new_records}
    baseline = {str(record["key"]): float(record["score"]) for record in baseline_records}
    if new.keys() != baseline.keys():
        raise RuntimeError(
            f"paired result keys differ: new={len(new)} baseline={len(baseline)}"
        )
    diffs = [new[key] - baseline[key] for key in sorted(new)]
    mean = statistics.fmean(diffs)
    stdev = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    t_stat = mean / (stdev / math.sqrt(len(diffs))) if stdev else 0.0
    return {
        "mean_f1_delta": round(mean, 6),
        "mean_f1_delta_points": round(100 * mean, 4),
        "paired_t": round(t_stat, 4),
        "num_questions": len(diffs),
    }


def load_records(path: Path, excluded: set[str]) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text())
    records = [
        record for record in payload["records"]
        if str(record["key"]) not in excluded
    ]
    return payload["summary"], records


def main() -> None:
    args = parse_args()
    full_data = json.loads(args.full_data.read_text())
    train_data = json.loads(args.train_data.read_text())
    excluded = train_keys(full_data, train_data)
    num_train_questions = sum(len(item.get("qa", [])) for item in train_data)
    if num_train_questions != 84:
        raise RuntimeError(f"expected 84 train questions, found {num_train_questions}")

    new_full, new_records = load_records(args.new_result, excluded)
    student_full, student_records = load_records(args.student_result, excluded)
    result = {
        "protocol": "LoCoMo categories 1-4; exclude every QA in locomo-train.json",
        "train_questions": num_train_questions,
        "excluded_eval_records": len(excluded),
        "new_full_eval": new_full,
        "new_heldout": summarize(new_records),
        "student_full_eval": student_full,
        "student_heldout": summarize(student_records),
        "new_vs_student_heldout": paired_delta(new_records, student_records),
    }
    if args.prior_result:
        prior_full, prior_records = load_records(args.prior_result, excluded)
        result["prior_d4_full_eval"] = prior_full
        result["prior_d4_heldout"] = summarize(prior_records)
        result["new_vs_prior_d4_heldout"] = paired_delta(new_records, prior_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
