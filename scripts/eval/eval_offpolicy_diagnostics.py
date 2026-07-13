#!/usr/bin/env python3
"""Offline P1 diagnostic: does the learned correction use the matching HM?

For every Stage0 trajectory this evaluator keeps the student query sequence
fixed and compares four candidate distributions:

  student   no TransMem correction
  real      TransMem with the sample's own HM
  shuffled  TransMem with another sample's HM (seeded derangement)
  zero      TransMem with an all-zero HM

Metrics are position-weighted and reported separately for all, first, and
later answer tokens. The evaluator only loads Stage0 tensors, TransMem, and
the frozen LM head; it does not load the backbone LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

_PROJ = Path(__file__).resolve().parents[2]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from transmem import FrozenLMHead, TransMem, TransMemConfig  # noqa: E402
from transmem.diagnostics import (  # noqa: E402
    MetricAccumulator,
    make_derangement,
    position_groups,
    token_metrics,
)
from transmem.train_offpolicy import (  # noqa: E402
    OffPolicyDataset,
    collate_sequences,
)

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class _IndexedPrefix(Dataset):
    """Expose stable local indices for the first count Stage0 samples."""

    def __init__(self, dataset: OffPolicyDataset, count: int):
        self.dataset = dataset
        self.count = int(count)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        return index, self.dataset[index]


def _collate_indexed(batch):
    indices = torch.tensor([row[0] for row in batch], dtype=torch.long)
    X, hq_tea, answer_ids, valid = collate_sequences(
        [row[1] for row in batch])
    return indices, X, hq_tea, answer_ids, valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare student/real/shuffled/zero HM on Stage0 trajectories")
    parser.add_argument("--data_dir", required=True,
                        help="one Stage0 directory containing meta.json and shards")
    parser.add_argument("--ckpt", required=True,
                        help="off-policy checkpoint (for example P1 best.pt)")
    parser.add_argument(
        "--lm_head_path", default=None,
        help="frozen lm_head.pt; default: DATA_DIR/lm_head.pt")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if args.log_interval <= 0:
        raise ValueError("--log_interval must be positive")


def _add_grouped(
    accumulator: MetricAccumulator,
    mode: str,
    metrics: dict[str, torch.Tensor],
    valid_cpu: torch.Tensor,
) -> None:
    """Apply all/first/later masks to flattened valid-position metrics."""

    for group, group_mask in position_groups(valid_cpu).items():
        select = group_mask[valid_cpu].to(next(iter(metrics.values())).device)
        accumulator.update(
            mode, group, {name: value[select] for name, value in metrics.items()})


def _candidate_hidden(
    memory: TransMem,
    hm: torch.Tensor,
    hq_student: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.cat([hm, hq_student], dim=1)
    memory_state = memory(X, return_all_queries=True)
    corrected = memory.correct(memory_state, hq_student)
    return corrected, corrected - hq_student


def _print_summary(summary: dict[str, dict[str, dict[str, float | int]]]) -> None:
    columns = (
        ("forward_kl", "KL"),
        ("teacher_top1", "tea@1"),
        ("trajectory_nll", "trajNLL"),
        ("trajectory_accuracy", "trajAcc"),
        ("correction_norm", "|corr|"),
        ("same_as_real", "=real"),
        ("changed_from_student", "chgStu"),
    )
    print("=" * 112)
    print(
        f"{'group':<7} {'mode':<9} {'positions':>9} "
        + " ".join(f"{label:>10}" for _, label in columns))
    print("-" * 112)
    for group in ("all", "first", "later"):
        for mode in ("student", "real", "shuffled", "zero"):
            row = summary[mode][group]
            values = []
            for key, _ in columns:
                value = row.get(key)
                values.append("       n/a" if value is None else f"{float(value):10.5f}")
            print(
                f"{group:<7} {mode:<9} {int(row['positions']):9d} "
                + " ".join(values))
    print("=" * 112)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    dtype = _DTYPES[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU diagnostics require --dtype float32")

    checkpoint_path = Path(args.ckpt)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_data = checkpoint.get("config")
    if not isinstance(config_data, dict):
        raise ValueError("checkpoint is missing a dict config")
    if config_data.get("layered"):
        raise ValueError("this evaluator supports final-hidden TransMem checkpoints only")
    config = TransMemConfig(**config_data)
    memory = TransMem(config)
    memory.load_state_dict(checkpoint["model_state_dict"], strict=True)
    memory = memory.to(device=device, dtype=dtype).eval()

    dataset = OffPolicyDataset(
        args.data_dir, load_dtype=dtype, n_mem=config.n_mem)
    if dataset.N != config.n_mem or dataset.dim != config.dim:
        raise ValueError(
            f"Stage0 N/dim={dataset.N}/{dataset.dim} but checkpoint expects "
            f"{config.n_mem}/{config.dim}")
    count = len(dataset)
    if args.max_samples is not None:
        count = min(count, args.max_samples)
    if count < 2:
        raise ValueError("real-vs-shuffled diagnostic needs at least two samples")

    lm_head_path = Path(args.lm_head_path or (Path(args.data_dir) / "lm_head.pt"))
    print(f"Loading LM head: {lm_head_path}")
    lm_head = FrozenLMHead.from_file(
        lm_head_path, device=device, dtype=dtype).eval()
    if lm_head.proj.in_features != config.dim:
        raise ValueError(
            f"LM head dim={lm_head.proj.in_features}, checkpoint dim={config.dim}")

    donors = make_derangement(count, args.shuffle_seed)
    loader = DataLoader(
        _IndexedPrefix(dataset, count),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=_collate_indexed,
        persistent_workers=(args.num_workers > 0),
    )

    accumulator = MetricAccumulator()
    started = time.time()
    completed = 0
    for batch_number, (indices, X_cpu, hq_tea_cpu, answer_cpu, valid_cpu) in enumerate(
            loader, start=1):
        X = X_cpu.to(device, non_blocking=True)
        hq_tea = hq_tea_cpu.to(device, non_blocking=True)
        answer_ids = answer_cpu.to(device, non_blocking=True)
        valid = valid_cpu.to(device, non_blocking=True)

        n_mem = config.n_mem
        hm_real = X[:, :n_mem]
        hq_student = X[:, n_mem:]
        hq_student_valid = hq_student[valid]
        hq_teacher_valid = hq_tea[valid]
        target_valid = answer_ids[valid]

        teacher_logits = lm_head(hq_teacher_valid)
        student_logits = lm_head(hq_student_valid)
        student_top1 = student_logits.argmax(dim=-1)

        real_hidden, real_correction = _candidate_hidden(
            memory, hm_real, hq_student)
        real_logits = lm_head(real_hidden[valid])
        real_top1 = real_logits.argmax(dim=-1)
        real_metrics = token_metrics(
            real_logits,
            teacher_logits,
            target_ids=target_valid,
            correction=real_correction[valid],
            real_top1=real_top1,
            student_top1=student_top1,
        )
        _add_grouped(accumulator, "real", real_metrics, valid_cpu)
        del real_logits, real_metrics, real_hidden, real_correction

        student_metrics = token_metrics(
            student_logits,
            teacher_logits,
            target_ids=target_valid,
            correction=torch.zeros_like(hq_student_valid),
            real_top1=real_top1,
            student_top1=student_top1,
        )
        _add_grouped(accumulator, "student", student_metrics, valid_cpu)
        del student_metrics, student_logits

        donor_hm_cpu = torch.stack([
            dataset[int(donors[int(index)])][0] for index in indices
        ])
        hm_modes = {
            "shuffled": donor_hm_cpu.to(device, non_blocking=True),
            "zero": torch.zeros_like(hm_real),
        }
        for mode, hm in hm_modes.items():
            candidate_hidden, correction = _candidate_hidden(
                memory, hm, hq_student)
            candidate_logits = lm_head(candidate_hidden[valid])
            metrics = token_metrics(
                candidate_logits,
                teacher_logits,
                target_ids=target_valid,
                correction=correction[valid],
                real_top1=real_top1,
                student_top1=student_top1,
            )
            _add_grouped(accumulator, mode, metrics, valid_cpu)
            del candidate_hidden, correction, candidate_logits, metrics

        completed += len(indices)
        if batch_number % args.log_interval == 0 or completed == count:
            elapsed = time.time() - started
            print(
                f"  evaluated {completed}/{count} samples "
                f"({completed / max(elapsed, 1e-6):.2f} samples/s)")

    metrics_summary = accumulator.summary()
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "offpolicy_hm_and_position_diagnostics",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": checkpoint.get("global_step"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "stage0_data_dir": str(Path(args.data_dir).resolve()),
        "lm_head_path": str(lm_head_path.resolve()),
        "samples": count,
        "shuffle_seed": args.shuffle_seed,
        "shuffle_donors": donors.tolist(),
        "dtype": args.dtype,
        "config": config.to_dict(),
        "metric_definitions": {
            "forward_kl": "KL(teacher distribution || candidate distribution)",
            "teacher_top1": "candidate argmax equals teacher argmax",
            "trajectory_nll": "negative log probability of Stage0 answer_ids token",
            "trajectory_accuracy": "candidate argmax equals Stage0 answer_ids token",
            "correction_norm": "L2 norm of HQ_prime - HQ_student",
            "same_as_real": "candidate argmax equals real-HM candidate argmax",
            "changed_from_student": "candidate argmax differs from raw student argmax",
        },
        "metrics": metrics_summary,
        "elapsed_seconds": round(time.time() - started, 3),
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "w") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, output)

    _print_summary(metrics_summary)
    print(f"JSON: {output}")
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
