#!/usr/bin/env python3
"""Free-generation HM ablation for a final-hidden TransMem checkpoint.

Each invocation evaluates one of four paired variants on the exact Stage0
sample manifest: plain student, real HM, globally shuffled HM, or zero HM.
The shuffled donor map is a seeded bijective derangement.  For TransMem modes
the online HM is checked against the cached Stage0 HM before an override is
applied, so a record/order/prompt mismatch fails loudly instead of producing a
plausible but invalid ablation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transmem.diagnostics import make_derangement  # noqa: E402
from transmem.evaluate import Evaluator, score  # noqa: E402
from transmem.extract_features import load_records  # noqa: E402


REFUSAL_MARKERS = (
    "i don't have access",
    "i do not have access",
    "not enough information",
    "cannot determine",
    "can't determine",
)


def parse_args():
    parser = argparse.ArgumentParser(description="real/shuffled/zero HM free-generation ablation")
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--data_format", required=True,
                        choices=["json", "qasper", "parquet", "hotpotqa-agentmem",
                                 "longmemeval"])
    parser.add_argument("--stage0_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--variant", required=True,
                        choices=["student", "real", "shuffled", "zero"])
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_answer_tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--hm_tolerance", type=float, default=0.05,
                        help="max absolute online-vs-cached HM difference")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--attn_impl", default="sdpa",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--print_examples", type=int, default=3)
    return parser.parse_args()


def load_manifest(stage0_dir: Path, max_samples: int | None):
    meta = json.loads((stage0_dir / "meta.json").read_text())
    manifest = list(meta.get("samples") or [])
    if not manifest:
        raise ValueError(f"Stage0 manifest is empty: {stage0_dir}")
    if max_samples is not None:
        manifest = manifest[:max_samples]
    return meta, manifest


def load_hm(stage0_dir: Path, entry: dict) -> torch.Tensor:
    sample = torch.load(stage0_dir / entry["file"], map_location="cpu", weights_only=False)
    hm = sample["hm_stu"]
    if "hm_maps" in sample:
        n_mem = int(sample.get("N") or len(hm))
        hm = hm[sample["hm_maps"][str(n_mem)]]
    return hm


def load_progress(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["sample_idx"])] = row
    return rows


def is_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def main() -> None:
    args = parse_args()
    stage0_dir = Path(args.stage0_dir)
    meta, manifest = load_manifest(stage0_dir, args.max_samples)
    all_records = load_records(args.eval_file, args.data_format, max_samples=None)
    selected = []
    for entry in manifest:
        sample_idx = int(entry["sample_idx"])
        if sample_idx >= len(all_records):
            raise IndexError(f"Stage0 sample_idx={sample_idx} exceeds {len(all_records)} records")
        selected.append((entry, all_records[sample_idx]))

    donors = make_derangement(len(selected), args.seed)
    cached_hm = [load_hm(stage0_dir, entry) for entry, _ in selected]
    expected_n = int(meta.get("N") or cached_hm[0].shape[0])
    if any(tuple(hm.shape) != (expected_n, int(meta["dim"])) for hm in cached_hm):
        raise ValueError("Stage0 HM shapes are inconsistent with meta.json")

    ev_args = types.SimpleNamespace(**vars(args))
    ev_args.mode = "student" if args.variant == "student" else "transmem"
    ev_args.N = expected_n
    ev_args.config = "transmem/config.json"
    evaluator = Evaluator(ev_args)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path.with_suffix(f".{args.variant}.progress.jsonl")
    done = load_progress(progress_path)
    examples = []

    with progress_path.open("a") as progress:
        for local_idx, (entry, record) in enumerate(
                tqdm(selected, desc=f"hm[{args.variant}]", unit="q")):
            sample_idx = int(entry["sample_idx"])
            if sample_idx in done:
                continue
            donor_local = int(donors[local_idx])
            donor_idx = int(selected[donor_local][0]["sample_idx"])
            comparison: dict[str, float] = {}

            if args.variant == "student":
                prediction = evaluator._greedy_plain(record["context"], record["question"])
            else:
                own = cached_hm[local_idx]
                donor = cached_hm[donor_local]

                def transform(online: torch.Tensor) -> torch.Tensor:
                    own_device = own.to(device=online.device, dtype=online.dtype)
                    max_abs = float((online - own_device).abs().max())
                    comparison["online_cached_max_abs"] = max_abs
                    if max_abs > args.hm_tolerance:
                        raise RuntimeError(
                            f"sample {sample_idx}: online/cached HM mismatch {max_abs:.6f} "
                            f"> tolerance {args.hm_tolerance}")
                    if args.variant == "real":
                        return online
                    if args.variant == "zero":
                        return torch.zeros_like(online)
                    return donor.to(device=online.device, dtype=online.dtype)

                _, _, answer_ids = evaluator.rollout.student_rollout(
                    evaluator.mem, record["context"], record["question"],
                    args.max_answer_tokens, sample=False, temperature=1.0,
                    hm_transform=transform)
                prediction = evaluator.tok.decode(
                    answer_ids, skip_special_tokens=True).strip()

            exact, contains = score(prediction, record["ground_truth"])
            row = {
                "sample_idx": sample_idx,
                "variant": args.variant,
                "donor_idx": donor_idx if args.variant == "shuffled" else None,
                "question": record["question"],
                "ground_truth": record["ground_truth"],
                "prediction": prediction,
                "exact": exact,
                "contains": contains,
                "refusal": is_refusal(prediction),
                **comparison,
            }
            progress.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.flush()
            done[sample_idx] = row
            if len(examples) < args.print_examples:
                examples.append(row)

    rows = [done[int(entry["sample_idx"])] for entry, _ in selected]
    n = len(rows)
    summary = {
        "schema_version": 1,
        "variant": args.variant,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "eval_file": str(Path(args.eval_file).resolve()),
        "stage0_dir": str(stage0_dir.resolve()),
        "stage0_source": meta.get("data_path"),
        "seed": args.seed,
        "shuffle_donors": donors.tolist(),
        "num_samples": n,
        "exact": sum(row["exact"] for row in rows) / max(n, 1),
        "contains": sum(row["contains"] for row in rows) / max(n, 1),
        "refusal_rate": sum(row["refusal"] for row in rows) / max(n, 1),
    }
    if args.variant != "student":
        diffs = [row["online_cached_max_abs"] for row in rows]
        summary["online_cached_hm_max_abs"] = max(diffs, default=0.0)
    out_path.write_text(json.dumps({"summary": summary, "records": rows},
                                   ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in examples:
        print(f"Q: {row['question'][:100]}\n  gold={row['ground_truth']!r}\n"
              f"  pred={row['prediction'][:120]!r}")


if __name__ == "__main__":
    main()
