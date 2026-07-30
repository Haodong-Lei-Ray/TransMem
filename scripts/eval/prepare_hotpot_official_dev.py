#!/usr/bin/env python3
"""Convert official HotpotQA distractor dev and prove train-question disjointness."""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

import pandas as pd


def normalize_question(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def train_question(row) -> str:
    prompt = row["prompt"]
    if isinstance(prompt, str):
        return prompt
    if len(prompt):
        first = prompt[0]
        return str(first.get("content", first)) if isinstance(first, dict) else str(first)
    return ""


def render_context(context) -> str:
    # The original JSON stores ``[[title, sentences], ...]`` while the
    # Hugging Face parquet mirror stores parallel title/sentences arrays.
    if isinstance(context, dict):
        context = zip(context["title"], context["sentences"])
    parts = []
    for index, (title, sentences) in enumerate(context, start=1):
        body = "".join(str(sentence) for sentence in sentences).strip()
        parts.append(f"Document {index}:\n{title}\n{body}")
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dev", type=Path, required=True)
    parser.add_argument("--agent-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.official_dev.suffix == ".parquet":
        official = pd.read_parquet(args.official_dev).to_dict("records")
    else:
        official = json.loads(args.official_dev.read_text(encoding="utf-8"))
    train_df = pd.read_parquet(args.agent_train, columns=["prompt"])
    train_questions = {
        normalize_question(train_question(row))
        for _, row in train_df.iterrows()
    }
    dev_questions = {normalize_question(item["question"]) for item in official}
    overlap = sorted((train_questions & dev_questions) - {""})
    if overlap:
        raise RuntimeError(
            f"official dev overlaps hotpotqa-agent train by {len(overlap)} questions; "
            f"examples={overlap[:5]}"
        )

    converted = [
        {
            "question_id": item.get("_id", item.get("id")),
            "question": item["question"],
            "answer": item["answer"],
            "context": render_context(item["context"]),
            "type": item.get("type"),
            "level": item.get("level"),
        }
        for item in official
    ]
    if len(converted) != 7405:
        raise RuntimeError(f"expected 7405 official dev questions, got {len(converted)}")
    if len({item["question_id"] for item in converted}) != len(converted):
        raise RuntimeError("duplicate official HotpotQA question IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, ensure_ascii=False), encoding="utf-8")
    print(
        f"HOTPOT_OFFICIAL_DEV_READY questions={len(converted)} "
        f"train_questions={len(train_questions)} overlap=0 output={args.output}"
    )


if __name__ == "__main__":
    main()
