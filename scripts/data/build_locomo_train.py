#!/usr/bin/env python3
"""Build a deterministic conversation/category-stratified LoCoMo train subset."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def retained_count(size: int, fraction: float) -> int:
    if size == 0:
        return 0
    return max(1, math.floor(size * fraction + 0.5))


def main() -> None:
    args = parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1]")

    with args.input.open(encoding="utf-8") as handle:
        source = json.load(handle)
    if not isinstance(source, list):
        raise TypeError("LoCoMo root must be a list")

    rng = random.Random(args.seed)
    output = copy.deepcopy(source)
    summary: list[dict[str, int]] = []
    if len(source) != len(output):
        raise AssertionError("deep copy changed conversation count")

    for conversation_index, (source_item, output_item) in enumerate(
        zip(source, output)
    ):
        qa = source_item.get("qa")
        if not isinstance(qa, list):
            raise TypeError(f"conversation {conversation_index}: qa must be a list")

        selected_indices: set[int] = set()
        row = {"conversation": conversation_index}
        for category in range(1, 5):
            indices = [
                index
                for index, record in enumerate(qa)
                if int(record["category"]) == category
            ]
            keep = retained_count(len(indices), args.fraction)
            selected_indices.update(rng.sample(indices, keep))
            row[f"cat{category}_source"] = len(indices)
            row[f"cat{category}_kept"] = keep

        output_item["qa"] = [
            record for index, record in enumerate(qa) if index in selected_indices
        ]
        summary.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(args.output)

    source_categories = Counter(
        int(record["category"]) for item in source for record in item["qa"]
    )
    output_categories = Counter(
        int(record["category"]) for item in output for record in item["qa"]
    )
    print(json.dumps({
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "fraction": args.fraction,
        "source_categories": dict(sorted(source_categories.items())),
        "output_categories": dict(sorted(output_categories.items())),
        "source_cat1_4_total": sum(source_categories[c] for c in range(1, 5)),
        "output_total": sum(output_categories.values()),
        "strata": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
