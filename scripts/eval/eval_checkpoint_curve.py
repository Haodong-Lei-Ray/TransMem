#!/usr/bin/env python3
"""Evaluate a fixed free-generation sample across TransMem checkpoints.

The backbone/tokenizer and evaluation records are loaded once.  Only the
TransMem ``state_dict`` is replaced between points, so every point uses the
same model instance, prompts, record order, and greedy decoding protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from transmem import TransMem, TransMemConfig
from transmem.evaluate import Evaluator, score
from transmem.extract_features import load_records
from transmem.train_onpolicy import OnPolicyRollout


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    step: int


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"curve_step_(\d+)\.pt", path.name)
    if match:
        return int(match.group(1))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "global_step" not in checkpoint:
        raise ValueError(f"checkpoint has no global_step: {path}")
    return int(checkpoint["global_step"])


def discover_checkpoints(checkpoint_dir: str | Path | None,
                         explicit_paths: Iterable[str | Path] | None
                         ) -> list[CheckpointInfo]:
    """Discover curve snapshots and sort by checkpoint ``global_step``."""
    paths: list[Path] = []
    if checkpoint_dir is not None:
        paths.extend(Path(checkpoint_dir).glob("curve_step_*.pt"))
    if explicit_paths:
        paths.extend(Path(path) for path in explicit_paths)

    unique: dict[str, Path] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        unique[str(path.absolute())] = path
    if not unique:
        raise ValueError("no checkpoints found; pass --checkpoint_dir or --checkpoints")

    found = [CheckpointInfo(path=path, step=_checkpoint_step(path))
             for path in unique.values()]
    return sorted(found, key=lambda item: (item.step, str(item.path)))


def score_predictions(records: Sequence[dict], predictions: Sequence[str]) -> dict:
    """Score one prediction per fixed record and retain auditable row details."""
    if len(records) != len(predictions):
        raise ValueError(
            f"record/prediction length mismatch: {len(records)} != {len(predictions)}")
    exact_count = contains_count = n = 0
    rows = []
    for index, (record, prediction) in enumerate(zip(records, predictions)):
        gold = str(record.get("ground_truth", ""))
        prediction = str(prediction or "")
        if gold:
            exact, contains = score(prediction, gold)
            exact_count += exact
            contains_count += contains
            n += 1
        else:
            exact = contains = None
        rows.append({
            "index": index,
            "sample_idx": record.get("sample_idx"),
            "question": record.get("question", ""),
            "ground_truth": gold,
            "prediction": prediction,
            "exact": exact,
            "contains": contains,
        })
    return {
        "exact": exact_count / max(n, 1),
        "contains": contains_count / max(n, 1),
        "exact_count": exact_count,
        "contains_count": contains_count,
        "n": n,
        "records": rows,
    }


def _record_fingerprint(records: Sequence[dict]) -> str:
    fixed = [{
        "sample_idx": record.get("sample_idx"),
        "question": record.get("question", ""),
        "ground_truth": record.get("ground_truth", ""),
    } for record in records]
    payload = json.dumps(fixed, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


@torch.no_grad()
def _predict_fixed(evaluator: Evaluator, records: Sequence[dict], label: str) -> list[str]:
    return [str(evaluator.predict(record) or "")
            for record in tqdm(records, desc=f"curve[{label}]", unit="q")]


def _load_transmem_point(evaluator: Evaluator, checkpoint_path: Path,
                         expected_config: dict | None) -> tuple[dict, dict]:
    """Load one final-hidden checkpoint while retaining the same frozen LLM."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = checkpoint["config"]
    if config_dict.get("layered"):
        raise ValueError("checkpoint curves currently support final-hidden TransMem only")
    if expected_config is not None and config_dict != expected_config:
        raise ValueError(
            f"checkpoint config differs within one curve: {checkpoint_path}")

    if evaluator.mem is None:
        config = TransMemConfig(**config_dict)
        evaluator.mem = TransMem(config).to(evaluator.device, dtype=evaluator.dtype)
        evaluator.rollout = OnPolicyRollout(
            evaluator.model,
            evaluator.tok,
            evaluator.device,
            config.n_mem,
            evaluator.dtype,
            hm_mode=getattr(config, "hm_mode", "floor"),
        )
    evaluator.mem.load_state_dict(checkpoint["model_state_dict"])
    evaluator.mem.eval()
    evaluator.args.mode = "transmem"
    return checkpoint, config_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="固定小样本上的 TransMem checkpoint 自由生成曲线")
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--data_format", default="json",
                        choices=["json", "qasper", "parquet", "hotpotqa-agentmem",
                                 "longmemeval"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_dir", default=None,
                        help="自动发现 curve_step_*.pt")
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="额外/替代的 checkpoint 路径 (可包含 best.pt)")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_samples", type=int, default=32,
                        help="固定取前 N 条; 整个曲线只载入一次")
    parser.add_argument("--max_answer_tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_student_baseline", action="store_true",
                        help="默认先评无 TransMem student baseline")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--attn_impl", default="sdpa",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--print_examples", type=int, default=3)
    parser.add_argument("--N", type=int, default=4,
                        help="兼容 Evaluator 参数; TransMem 点始终使用 ckpt config")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoints = discover_checkpoints(args.checkpoint_dir, args.checkpoints)
    records = load_records(args.eval_file, args.data_format, args.max_samples)
    if not records:
        raise ValueError("evaluation set is empty")

    # Student mode initializes the backbone/tokenizer exactly once and defers
    # TransMem construction until the first checkpoint.
    evaluator_args = types.SimpleNamespace(**vars(args))
    evaluator_args.mode = "student"
    evaluator_args.ckpt = None
    evaluator_args.config = "transmem/config.json"
    evaluator = Evaluator(evaluator_args)

    output = {
        "metadata": {
            "eval_file": str(Path(args.eval_file).absolute()),
            "data_format": args.data_format,
            "model_path": args.model_path,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "num_records": len(records),
            "record_fingerprint": _record_fingerprint(records),
            "decode": "greedy",
        },
        "points": [],
    }
    output_path = Path(args.output_json)

    if not args.no_student_baseline:
        predictions = _predict_fixed(evaluator, records, "student")
        point = score_predictions(records, predictions)
        point.update({"series": "student_baseline", "label": "student",
                      "step": 0, "checkpoint": None})
        output["points"].append(point)
        _atomic_json_dump(output, output_path)
        print(f"student: Exact={point['exact']:.3f} Contains={point['contains']:.3f}")

    expected_config = None
    for info in checkpoints:
        checkpoint, expected_config = _load_transmem_point(
            evaluator, info.path, expected_config)
        label = f"transmem_step_{info.step:07d}"
        predictions = _predict_fixed(evaluator, records, label)
        point = score_predictions(records, predictions)
        point.update({
            "series": "transmem",
            "label": label,
            "step": int(checkpoint.get("global_step", info.step)),
            "epoch": checkpoint.get("epoch"),
            "checkpoint": str(info.path.absolute()),
            "checkpoint_seed": checkpoint.get("seed"),
            "schedule_total_steps": checkpoint.get("schedule_total_steps"),
        })
        output["points"].append(point)
        _atomic_json_dump(output, output_path)
        print(f"step {point['step']}: Exact={point['exact']:.3f} "
              f"Contains={point['contains']:.3f}")

    print(f"结果: {output_path}")


if __name__ == "__main__":
    main()
