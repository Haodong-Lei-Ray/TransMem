#!/usr/bin/env python3
"""Merge disjoint LongMemEval JSONL shards and report cheap local metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--metrics_json", required=True)
    parser.add_argument("shards", nargs="+")
    return parser.parse_args()


def main():
    args = parse_args()
    references = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    order = {str(row["question_id"]): index for index, row in enumerate(references)}
    rows = {}
    for shard in args.shards:
        with Path(shard).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row["question_id"])
                if qid in rows:
                    raise ValueError(f"Duplicate question_id across shards: {qid}")
                if qid not in order:
                    raise ValueError(f"Question is absent from reference split: {qid}")
                rows[qid] = row
    if len(rows) != len(references):
        missing = [qid for qid in order if qid not in rows]
        raise RuntimeError(
            f"Merged output is incomplete: {len(rows)}/{len(references)}, missing={missing[:5]}")

    merged = sorted(rows.values(), key=lambda row: order[str(row["question_id"])])
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_type = defaultdict(lambda: {"n": 0, "exact": 0, "contains": 0})
    for row in merged:
        bucket = by_type[row["question_type"]]
        bucket["n"] += 1
        bucket["exact"] += int(row["exact"])
        bucket["contains"] += int(row["contains"])
    n = len(merged)
    metrics = {
        "num_questions": n,
        "exact": sum(row["exact"] for row in merged) / max(n, 1),
        "contains": sum(row["contains"] for row in merged) / max(n, 1),
        "format_valid_rate": (
            sum(bool(row.get("format_valid")) for row in merged) / max(n, 1)),
        "by_type": {
            key: {
                "n": value["n"],
                "exact": value["exact"] / value["n"],
                "contains": value["contains"] / value["n"],
            }
            for key, value in sorted(by_type.items())
        },
    }
    Path(args.metrics_json).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
