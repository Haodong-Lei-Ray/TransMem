#!/usr/bin/env python3
"""Merge HotpotQA shards and verify exact full-dev coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    records_by_id = {}
    sources = []
    modes = set()
    for shard_id in range(args.num_shards):
        path = args.input_dir / f"shard_{shard_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append(str(path))
        modes.add(payload.get("mode", "transmem"))
        for record in payload["records"]:
            question_id = str(record["question_id"])
            if question_id in records_by_id:
                raise RuntimeError(f"duplicate question ID {question_id}")
            records_by_id[question_id] = record
    records = list(records_by_id.values())
    if len(records) != 7405:
        raise RuntimeError(f"expected 7405 HotpotQA dev predictions, got {len(records)}")
    if len(modes) != 1:
        raise RuntimeError(f"inconsistent shard modes: {sorted(modes)}")
    summary = {
        "num_questions": len(records),
        "answer_em": sum(float(item["em"]) for item in records) / len(records),
        "answer_f1": sum(float(item["f1"]) for item in records) / len(records),
        "contains": sum(float(item["contains"]) for item in records) / len(records),
    }
    output = {
        "dataset": "hotpot_dev_distractor_v1",
        "mode": modes.pop(),
        "decode": "greedy_nonthinking",
        "train_overlap_questions": 0,
        "summary": summary,
        "source_shards": sources,
        "records": records,
    }
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    prediction = {"answer": {item["question_id"]: item["prediction"] for item in records}, "sp": {}}
    args.output_json.with_name("hotpot_predictions.json").write_text(
        json.dumps(prediction, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"HOTPOT_EVAL_COMPLETE output={args.output_json}")


if __name__ == "__main__":
    main()
