#!/usr/bin/env python3
"""Merge disjoint eval_locomo.py shard outputs into the normal result schema."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("shards", nargs="+")
    return parser.parse_args()


def natural_key(row):
    sample_id, question_index = row["key"].rsplit(":", 1)
    return sample_id, int(question_index)


def main():
    args = parse_args()
    payloads = [json.loads(Path(path).read_text()) for path in args.shards]
    if not payloads:
        raise ValueError("至少需要一个分片")

    rows = []
    seen = set()
    for payload in payloads:
        for row in payload["records"]:
            if row["key"] in seen:
                raise ValueError(f"分片之间存在重复问题: {row['key']}")
            seen.add(row["key"])
            rows.append(row)
    rows.sort(key=natural_key)

    cat_scores = defaultdict(float)
    cat_counts = defaultdict(int)
    for row in rows:
        category = int(row["category"])
        cat_scores[category] += float(row["score"])
        cat_counts[category] += 1

    summary = dict(payloads[0]["summary"])
    summary.pop("shard_index", None)
    summary["num_shards"] = len(payloads)
    summary["num_questions"] = len(rows)
    summary["overall_f1"] = round(
        sum(cat_scores.values()) / max(len(rows), 1), 4)
    summary["format_valid_rate"] = (
        sum(bool(row.get("format_valid")) for row in rows) / max(len(rows), 1))
    source_categories = payloads[0]["summary"].get("category_f1", {})
    summary["category_f1"] = {
        str(category): {
            "name": source_categories.get(str(category), {}).get("name", "unknown"),
            "score": round(cat_scores[category] / cat_counts[category], 4),
            "count": cat_counts[category],
        }
        for category in sorted(cat_counts)
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        {"summary": summary, "records": rows}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
