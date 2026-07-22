#!/usr/bin/env python3
"""Deterministically balance whole MAB sources across evaluator workers."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT4_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT4_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT4_ROOT))

from scripts.eval.eval_memory_agent_bench import MAIN_SOURCE_SPECS


@dataclass(frozen=True)
class WorkerPlan:
    worker_index: int
    question_count: int
    sources: tuple[str, ...]


def partition_sources(
    sources: Sequence[str],
    worker_count: int,
    max_questions_per_source: int | None = None,
) -> list[WorkerPlan]:
    """Use stable longest-processing-time scheduling on official question counts."""

    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    unique_sources = list(dict.fromkeys(sources))
    unknown = [source for source in unique_sources if source not in MAIN_SOURCE_SPECS]
    if unknown:
        raise ValueError(f"unknown MemoryAgentBench sources: {unknown}")
    if not unique_sources:
        raise ValueError("at least one source is required")
    if max_questions_per_source is not None and max_questions_per_source < 1:
        raise ValueError("max_questions_per_source must be positive")

    active_workers = min(worker_count, len(unique_sources))
    bins: list[list[str]] = [[] for _ in range(active_workers)]
    loads = [0] * active_workers
    source_order = {source: index for index, source in enumerate(unique_sources)}

    def question_count(source: str) -> int:
        count = MAIN_SOURCE_SPECS[source].question_count
        return min(count, max_questions_per_source) if max_questions_per_source else count

    ordered = sorted(
        unique_sources,
        key=lambda source: (-question_count(source), source_order[source]),
    )
    for source in ordered:
        worker = min(range(active_workers), key=lambda index: (loads[index], index))
        bins[worker].append(source)
        loads[worker] += question_count(source)

    return [
        WorkerPlan(index, loads[index], tuple(bins[index]))
        for index in range(active_workers)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--sources", nargs="+", default=list(MAIN_SOURCE_SPECS))
    parser.add_argument("--max_questions_per_source", type=int, default=None)
    parser.add_argument("--format", choices=["tsv", "human"], default="human")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = partition_sources(
        args.sources,
        args.workers,
        max_questions_per_source=args.max_questions_per_source,
    )
    for worker in plan:
        if args.format == "tsv":
            print(
                f"{worker.worker_index}\t{worker.question_count}\t"
                + " ".join(worker.sources)
            )
        else:
            print(
                f"worker={worker.worker_index} questions={worker.question_count} "
                f"sources={','.join(worker.sources)}"
            )


if __name__ == "__main__":
    main()
