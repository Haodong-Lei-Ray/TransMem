#!/usr/bin/env python3
"""Paired student/TransMem evaluation on MemoryAgentBench's main 13 sources.

This adapter deliberately reuses the benchmark's long-context query templates and
``post_process`` implementation, while using Project4's chat prompt and native
TransMem token-by-token correction path.  It reads the already-downloaded local
Hugging Face parquet files; it never contacts the Hub.

The official benchmark reports one score per source, not a cross-source overall.
Accordingly, this script writes one progress JSONL and one summary JSON per source,
plus a top-level index containing only the per-source summaries.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


PROJECT4_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAB_ROOT = Path(
    "/mnt/petrelfs/leihaodong/Project1/MemoryAgentBenchProject/MemoryAgentBench"
)
AGENT_INPUT_TOKENS = 128_000
AGENT_BUFFER_TOKENS = 4_000
MAX_PROMPT_TOKENS = AGENT_INPUT_TOKENS
LONG_CONTEXT_AGENT_NAME = "Long_context_agent_Qwen3-4B-Instruct-2507"


@dataclass(frozen=True)
class SourceSpec:
    """The source settings used by MemoryAgentBench's main experiment."""

    split: str
    question_count: int
    max_new_tokens: int
    primary_metric: str | None
    needs_judge: bool = False


# Order is the official ``bash_files/configs/long_context_agents.txt`` main block.
# Question totals are the totals in the local revision-main parquet snapshot.
MAIN_SOURCE_SPECS: dict[str, SourceSpec] = {
    "ruler_qa1_197K": SourceSpec(
        "Accurate_Retrieval", 100, 50, "substring_exact_match"),
    "ruler_qa2_421K": SourceSpec(
        "Accurate_Retrieval", 100, 50, "substring_exact_match"),
    "longmemeval_s*": SourceSpec(
        "Accurate_Retrieval", 300, 50, None, needs_judge=True),
    "eventqa_full": SourceSpec(
        "Accurate_Retrieval", 500, 40, "substring_exact_match"),
    "icl_banking77_5900shot_balance": SourceSpec(
        "Test_Time_Learning", 100, 20, "exact_match"),
    "icl_clinic150_7050shot_balance": SourceSpec(
        "Test_Time_Learning", 100, 20, "exact_match"),
    "icl_nlu_8296shot_balance": SourceSpec(
        "Test_Time_Learning", 100, 20, "exact_match"),
    "icl_trec_coarse_6600shot_balance": SourceSpec(
        "Test_Time_Learning", 100, 20, "exact_match"),
    "icl_trec_fine_6400shot_balance": SourceSpec(
        "Test_Time_Learning", 100, 20, "exact_match"),
    "recsys_redial_full": SourceSpec(
        "Test_Time_Learning", 200, 300, "recsys_recall@5"),
    "infbench_sum_eng_shots2": SourceSpec(
        "Long_Range_Understanding", 100, 1200, None, needs_judge=True),
    "factconsolidation_sh_262k": SourceSpec(
        "Conflict_Resolution", 100, 10, "substring_exact_match"),
    "factconsolidation_mh_262k": SourceSpec(
        "Conflict_Resolution", 100, 10, "substring_exact_match"),
}


def source_prompt_budget(
    source: str,
    agent_input_tokens: int = AGENT_INPUT_TOKENS,
) -> int:
    """Return the prompt budget for one MAB source and model context cap."""

    try:
        generation_tokens = MAIN_SOURCE_SPECS[source].max_new_tokens
    except KeyError as exc:
        raise ValueError(f"unknown MemoryAgentBench source: {source}") from exc
    budget = agent_input_tokens - AGENT_BUFFER_TOKENS - generation_tokens
    if budget <= 0:
        raise ValueError(
            f"{source} has no prompt budget after reserving "
            f"{AGENT_BUFFER_TOKENS} buffer and {generation_tokens} generation tokens")
    return budget


@dataclass(frozen=True)
class QueryRecord:
    key: str
    source: str
    context_index: int
    question_index: int
    question: str
    formatted_query: str
    answer: Any
    qa_pair_id: Any = None
    question_id: Any = None
    question_type: Any = None
    keypoints: Any = None


@dataclass(frozen=True)
class ContextWindow:
    """One shared context suffix and the resulting prompt lengths."""

    context: str
    original_context_tokens: int
    kept_context_tokens: int
    left_truncated_tokens: int
    prompt_lengths: tuple[int, ...]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _indexed_value(row: Mapping[str, Any], field: str, index: int) -> Any:
    metadata = row.get("metadata") or {}
    value = row.get(field)
    if value is None and isinstance(metadata, Mapping):
        value = metadata.get(field)
    values = _as_list(value)
    if not values:
        return None
    return values[index] if index < len(values) else value


def build_query_records(
    row: Mapping[str, Any],
    context_index: int,
    source: str,
    get_template: Callable[[str, str, str], str],
) -> list[QueryRecord]:
    """Render every question with the benchmark's long-context query template.

    ``answer=row['answers'][qi]`` is intentional.  It avoids the official
    ``ConversationCreator`` singleton branch that can accidentally pass the whole
    answer column when there is only one question.
    """

    questions = _as_list(row.get("questions"))
    answers = _as_list(row.get("answers"))
    if len(questions) != len(answers):
        raise ValueError(
            f"{source} context {context_index}: {len(questions)} questions but "
            f"{len(answers)} answers")
    template = get_template(source, "query", LONG_CONTEXT_AGENT_NAME)
    metadata = row.get("metadata") or {}
    keypoints = None
    if isinstance(metadata, Mapping):
        keypoints = metadata.get("keypoints")
        if keypoints is None:
            keypoints = metadata.get("summary/short_keypoints")
    if keypoints is None:
        keypoints = row.get("keypoints")
    records: list[QueryRecord] = []
    indexed_fields = (
        "question_dates", "question_types", "question_ids",
        "previous_events", "qa_pair_ids",
    )
    for qi, question in enumerate(questions):
        answer = answers[qi]
        values = dict(row)
        values.update({"question": question, "answer": answer, "source": source})
        for field in indexed_fields:
            indexed = _indexed_value(row, field, qi)
            if indexed is not None:
                values[field] = indexed
        try:
            formatted_query = template.format(**values)
        except KeyError as exc:
            raise KeyError(
                f"{source} context {context_index} question {qi}: missing template "
                f"field {exc}") from exc
        records.append(QueryRecord(
            key=f"{source}:{context_index}:{qi}",
            source=source,
            context_index=context_index,
            question_index=qi,
            question=str(question),
            formatted_query=formatted_query,
            answer=answer,
            qa_pair_id=_indexed_value(row, "qa_pair_ids", qi),
            question_id=_indexed_value(row, "question_ids", qi),
            question_type=_indexed_value(row, "question_types", qi),
            keypoints=keypoints,
        ))
    return records


def _encode(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        ids = tokenizer(text, add_special_tokens=False).input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(item) for item in ids]


def _flat_prompt_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"prompt builder returned batch size {len(value)}, expected 1")
        value = value[0]
    return [int(item) for item in value]


def plan_context_window(
    tokenizer: Any,
    context: str,
    questions: Sequence[str],
    prompt_builder: Callable[..., Any],
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
) -> ContextWindow:
    """Choose one left-truncated context suffix for all questions in a context.

    The longest rendered query determines the context budget.  The final prompts
    are rebuilt and checked because decode/re-tokenize boundary effects can differ
    by a token from the standalone context estimate.
    """

    if not questions:
        raise ValueError("at least one question is required")
    original_ids = _encode(tokenizer, context)
    empty_lengths = [
        len(_flat_prompt_ids(prompt_builder(tokenizer, "", question, device=None)))
        for question in questions
    ]
    keep = max_prompt_tokens - max(empty_lengths)
    if keep < 0:
        raise ValueError(
            f"longest query alone needs {max(empty_lengths)} tokens, above fixed "
            f"prompt limit {max_prompt_tokens}")
    candidate_ids = original_ids[-keep:] if keep else []

    # Usually one pass.  The loop handles BPE boundary re-tokenization exactly.
    while True:
        candidate = tokenizer.decode(candidate_ids, skip_special_tokens=False)
        prompt_lengths = tuple(
            len(_flat_prompt_ids(
                prompt_builder(tokenizer, candidate, question, device=None)))
            for question in questions
        )
        overflow = max(prompt_lengths) - max_prompt_tokens
        if overflow <= 0:
            break
        if not candidate_ids:
            raise ValueError(
                f"query prompt needs {max(prompt_lengths)} tokens even with empty context")
        drop = min(len(candidate_ids), max(1, overflow))
        candidate_ids = candidate_ids[drop:]

    kept_tokens = len(_encode(tokenizer, candidate))
    return ContextWindow(
        context=candidate,
        original_context_tokens=len(original_ids),
        kept_context_tokens=kept_tokens,
        left_truncated_tokens=max(len(original_ids) - kept_tokens, 0),
        prompt_lengths=prompt_lengths,
    )


def longest_common_prefix(sequences: Sequence[Sequence[int]]) -> list[int]:
    if not sequences:
        return []
    shortest = min(len(sequence) for sequence in sequences)
    end = 0
    while end < shortest:
        value = sequences[0][end]
        if any(sequence[end] != value for sequence in sequences[1:]):
            break
        end += 1
    return list(sequences[0][:end])


_LABEL_PATTERN = re.compile(r"label\s*:\s*\{?\s*(-?\d+)\s*\}?", re.IGNORECASE)
_INTEGER_PATTERN = re.compile(r"(?<!\d)-?\d+(?!\d)")


def parse_icl_label(text: str) -> str | None:
    """Parse a numeric ICL label while retaining the official strict EM too."""

    text = str(text).strip()
    match = _LABEL_PATTERN.search(text)
    if match:
        return match.group(1)
    match = _INTEGER_PATTERN.search(text)
    return match.group(0) if match else None


def _answer_labels(answer: Any) -> set[str]:
    labels: set[str] = set()
    stack = _as_list(answer)
    while stack:
        value = stack.pop()
        if isinstance(value, (list, tuple)):
            stack.extend(value)
            continue
        parsed = parse_icl_label(str(value))
        if parsed is not None:
            labels.add(parsed)
    return labels


def icl_parsed_label_exact(prediction: str, answer: Any) -> bool:
    """Flexible ICL label accuracy without mutating nested benchmark answers."""

    label = parse_icl_label(prediction)
    return bool(label is not None and label in _answer_labels(answer))


@contextmanager
def pushd(path: str | Path) -> Iterator[None]:
    """Temporarily anchor official relative resource paths (notably ReDial)."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def score_prediction(
    prediction: str,
    answer: Any,
    source: str,
    post_process: Callable[..., Any],
    mab_root: Path,
) -> tuple[dict[str, float | bool], dict[str, Any]]:
    """Run the official source post-processor with ReDial's cwd made explicit."""

    with pushd(mab_root):
        metrics, details = post_process(
            {"output": prediction}, answer, {"sub_dataset": source})
    metrics = _jsonable(metrics)
    details = _jsonable(details)
    if source.startswith("icl_"):
        label = parse_icl_label(prediction)
        metrics["icl_parsed_label_exact"] = icl_parsed_label_exact(
            prediction, answer)
        details["parsed_label"] = label
    return metrics, details


def _mean_metrics(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for name, value in (row.get(field) or {}).items():
            if isinstance(value, (bool, int, float)):
                totals[name] += float(value)
                counts[name] += 1
    return {
        name: round(totals[name] / counts[name], 6)
        for name in sorted(totals)
    }


def summarize_source_rows(
    source: str,
    rows: Sequence[Mapping[str, Any]],
    expected_questions: int | None = None,
) -> dict[str, Any]:
    """Aggregate a single source only; never invent a cross-source overall."""

    spec = MAIN_SOURCE_SPECS[source]
    requested = (
        spec.question_count if expected_questions is None else expected_questions)
    official = spec.question_count
    original = sum(int(row.get("context_tokens_original", 0)) for row in rows)
    kept = sum(int(row.get("context_tokens_kept", 0)) for row in rows)
    summary: dict[str, Any] = {
        "source": source,
        "num_questions": len(rows),
        "expected_questions": official,
        "requested_questions": requested,
        "capped": requested < official,
        "requested_complete": len(rows) == requested,
        "complete": len(rows) == official,
        "primary_metric": spec.primary_metric,
        "needs_judge": spec.needs_judge,
        "student": _mean_metrics(rows, "student_metrics"),
        "transmem": _mean_metrics(rows, "transmem_metrics"),
        "context_tokens": {
            "original_total": original,
            "kept_total": kept,
            "left_truncated_total": max(original - kept, 0),
        },
    }
    if spec.primary_metric is not None:
        summary["student_primary"] = summary["student"].get(spec.primary_metric)
        summary["transmem_primary"] = summary["transmem"].get(spec.primary_metric)
    else:
        summary["score_status"] = "proxy_metrics_only; official score requires judge"
    if source == "longmemeval_s*":
        summary.update({
            "evaluation_scope": "in_domain",
            "cross_domain_generalization_claim_allowed": False,
            "contamination_status": "overlap_not_measured_by_adapter",
            "interpretation_note": (
                "P1 was trained on LongMemEval-family data. Treat this as in-domain "
                "evaluation; an external manifest overlap audit is required before "
                "making any contamination-adjusted claim."),
        })
    return summary


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_mab(mab_root: Path):
    templates_path = mab_root / "utils" / "templates.py"
    metrics_path = mab_root / "utils" / "eval_other_utils.py"
    if not templates_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"not a MemoryAgentBench checkout: {mab_root}")
    templates = _load_module("_project4_mab_templates", templates_path)
    metrics = _load_module("_project4_mab_metrics", metrics_path)
    return templates.get_template, metrics.post_process


def _hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def find_split_parquets(
    split: str,
    mab_root: Path,
    data_dir: Path | None,
) -> list[Path]:
    """Find local revision-main parquet shards without any network fallback."""

    search_roots: list[Path] = []
    if data_dir is not None:
        search_roots.append(data_dir)
    else:
        search_roots.append(mab_root / "processed_data")
        for hub in _hf_cache_roots():
            search_roots.extend(sorted(
                (hub / "datasets--ai-hyz--MemoryAgentBench" / "snapshots").glob(
                    "*/data")))
    unique_roots: list[Path] = []
    seen_roots: set[Path] = set()
    for root in search_roots:
        resolved_root = root.resolve()
        if resolved_root not in seen_roots:
            seen_roots.add(resolved_root)
            unique_roots.append(root)

    groups: list[tuple[Path, list[Path]]] = []
    for root in unique_roots:
        matches: list[Path] = []
        if root.is_file() and root.name.startswith(f"{split}-"):
            matches.append(root)
        elif root.exists():
            matches.extend(root.rglob(f"{split}-*.parquet"))
        resolved_matches = [path.resolve() for path in matches]
        names: dict[str, list[Path]] = defaultdict(list)
        for path in resolved_matches:
            names[path.name].append(path)
        duplicate_names = sorted(
            name for name, paths in names.items() if len(set(paths)) > 1)
        if len(resolved_matches) != len(set(resolved_matches)) or duplicate_names:
            details = ", ".join(duplicate_names) or "same resolved file"
            raise ValueError(
                f"duplicate parquet shard for split {split} under {root.resolve()}: "
                f"{details}")
        unique = sorted(set(resolved_matches))
        if unique:
            groups.append((root.resolve(), unique))
    if not groups:
        roots = ", ".join(str(path) for path in unique_roots)
        raise FileNotFoundError(f"no local parquet for split {split}; searched: {roots}")
    if len(groups) > 1:
        roots = ", ".join(str(root) for root, _ in groups)
        raise ValueError(
            f"multiple local dataset roots contain split {split}: {roots}; "
            "pass one explicit --data_dir")
    return groups[0][1]


def load_rows_by_source(
    sources: Sequence[str],
    mab_root: Path,
    data_dir: Path | None,
) -> dict[str, list[dict[str, Any]]]:
    """Stream local parquet one row at a time and retain only requested sources."""

    import pyarrow.parquet as pq

    wanted = set(sources)
    grouped: dict[str, list[dict[str, Any]]] = {source: [] for source in sources}
    splits: dict[str, list[str]] = defaultdict(list)
    for source in sources:
        splits[MAIN_SOURCE_SPECS[source].split].append(source)
    for split, split_sources in splits.items():
        split_wanted = set(split_sources)
        row_index = 0
        for parquet in find_split_parquets(split, mab_root, data_dir):
            parquet_file = pq.ParquetFile(parquet)
            for batch in parquet_file.iter_batches(batch_size=1):
                for row in batch.to_pylist():
                    metadata = row.get("metadata") or {}
                    source = metadata.get("source", "")
                    if source in split_wanted:
                        row["_split_row_index"] = row_index
                        grouped[source].append(row)
                    row_index += 1
    missing = sorted(source for source in wanted if not grouped[source])
    if missing:
        raise ValueError(f"requested sources absent from local parquet: {missing}")
    return grouped


def _safe_source_name(source: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("_")


def _load_progress(path: Path, run_id: str) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") != run_id:
                raise ValueError(
                    f"{path}:{line_number} belongs to another model/checkpoint/config; "
                    "use a new output directory or --force")
            done[row["key"]] = row
    return done


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def write_summary_index(
    output_dir: Path,
    run_id: str,
    summaries: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge per-source summaries into the shared index under one file lock."""

    lock_path = output_dir / ".summary.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            indexed_summaries: dict[str, Any] = {}
            for summary_path in output_dir.glob("*.summary.json"):
                with summary_path.open(encoding="utf-8") as handle:
                    saved = json.load(handle)
                if saved.get("run_id") == run_id and saved.get("source"):
                    indexed_summaries[saved["source"]] = saved
            indexed_summaries.update(summaries)
            index = {
                "benchmark": "MemoryAgentBench main-13",
                "aggregation": (
                    "per_source_only (the official benchmark defines no overall)"),
                "run_id": run_id,
                "sources": indexed_summaries,
            }
            _write_json(output_dir / "summary.json", index)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return index


def resolve_checkpoint_step(
    *,
    runner_step: Any,
    rows: Sequence[Mapping[str, Any]],
    previous_summary: Mapping[str, Any] | None,
    run_id: str,
) -> int:
    """Resolve one non-null checkpoint step without loading model weights."""

    candidates: set[int] = set()
    if runner_step is not None:
        candidates.add(int(runner_step))
    for row in rows:
        step = row.get("checkpoint_step")
        if step is not None:
            candidates.add(int(step))
    if previous_summary and previous_summary.get("run_id") == run_id:
        step = previous_summary.get("checkpoint_step")
        if step is not None:
            candidates.add(int(step))
    if not candidates:
        raise ValueError(
            "checkpoint_step is unavailable from runner, progress, and prior summary")
    if len(candidates) != 1:
        raise ValueError(f"conflicting checkpoint_step values: {sorted(candidates)}")
    return next(iter(candidates))


class PairedTransMemGreedy:
    """Frozen Qwen + P1 TransMem with paired, cache-restored greedy decoding."""

    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if str(PROJECT4_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT4_ROOT))
        from transmem import TransMem, TransMemConfig
        from transmem.extract_features import build_chat_prompt_ids, resolve_eos_ids

        self.torch = torch
        self.mode = args.mode
        self.device = torch.device(args.device)
        self.dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
        self.build_chat_prompt_ids = build_chat_prompt_ids
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # The evaluation contract is deliberately 128k even if the local model
        # advertises a larger theoretical window.
        self.tokenizer.model_max_length = MAX_PROMPT_TOKENS
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=self.dtype,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation=args.attn_impl,
        ).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.eos_ids = set(resolve_eos_ids(self.model))
        self.config = None
        self.mem = None
        self.layered = False
        self.layered_rollout = None
        self.checkpoint_step = None
        if self.mode == "paired":
            checkpoint = torch.load(
                args.ckpt, map_location="cpu", weights_only=False)
            config_dict = checkpoint["config"]
            if isinstance(config_dict, dict) and config_dict.get("layered"):
                from transmem.layered import (
                    LayeredConfig, LayeredRollout, TransMemLayered)

                self.layered = True
                self.config = LayeredConfig.from_dict(config_dict)
                self.mem = TransMemLayered(self.config).to(
                    self.device, dtype=self.dtype).eval()
                self.layered_rollout = LayeredRollout(
                    self.model, self.tokenizer, self.device, self.mem, self.dtype)
            else:
                self.config = TransMemConfig(**config_dict)
                self.mem = TransMem(self.config).to(
                    self.device, dtype=self.dtype).eval()
            self.mem.load_state_dict(checkpoint["model_state_dict"])
            self.checkpoint_step = checkpoint.get("global_step")

    def _prompt(self, context: str, question: str):
        return self.build_chat_prompt_ids(
            self.tokenizer, context, question, self.device)

    def plan_context(
        self,
        context: str,
        questions: Sequence[str],
        max_prompt_tokens: int,
    ) -> ContextWindow:
        return plan_context_window(
            self.tokenizer,
            context,
            questions,
            self.build_chat_prompt_ids,
            max_prompt_tokens,
        )

    def _extract_hm(self, hidden, context_tokens: int):
        from transmem.extract_features import hm_positions

        positions = hm_positions(
            context_tokens, self.config.n_mem, getattr(self.config, "hm_mode", "floor"))
        if positions and positions[-1] >= hidden.shape[0]:
            raise AssertionError(
                f"native HM position {positions[-1]} outside cached prefix "
                f"length {hidden.shape[0]}")
        index = self.torch.tensor(positions, device=hidden.device)
        return hidden[index].to(self.dtype)

    def _crop(self, cache, length: int) -> None:
        if not hasattr(cache, "crop"):
            raise TypeError(
                f"expected transformers DynamicCache with crop(), got {type(cache)}")
        cache.crop(length)
        actual = int(cache.get_seq_length())
        if actual != length:
            raise RuntimeError(f"cache crop requested {length}, got {actual}")

    def _next_hidden(self, token_id: int, cache):
        token = self.torch.tensor([[token_id]], device=self.device, dtype=self.torch.long)
        output = self.model.model(input_ids=token, past_key_values=cache, use_cache=True)
        return output.last_hidden_state[0, -1:, :]

    def _student_greedy(self, cache, hq_first, max_new_tokens: int) -> list[int]:
        hq = hq_first
        answer: list[int] = []
        for _ in range(max_new_tokens):
            token_id = int(self.model.lm_head(hq).argmax(dim=-1).item())
            answer.append(token_id)
            if token_id in self.eos_ids:
                break
            hq = self._next_hidden(token_id, cache)
        return answer

    def _transmem_greedy(self, cache, hq_first, hm, max_new_tokens: int) -> list[int]:
        from transformers import DynamicCache

        hq = hq_first
        mem_cache = DynamicCache()
        memory_input = self.torch.cat([hm, hq], dim=0).unsqueeze(0).to(self.dtype)
        answer: list[int] = []
        for _ in range(max_new_tokens):
            proposal = self.mem(
                memory_input, past_key_values=mem_cache, use_cache=True)
            corrected = self.mem.correct(hq, proposal)
            token_id = int(self.model.lm_head(corrected).argmax(dim=-1).item())
            answer.append(token_id)
            if token_id in self.eos_ids:
                break
            hq = self._next_hidden(token_id, cache)
            memory_input = hq.unsqueeze(0).to(self.dtype)
        return answer

    def _paired_from_prompt(self, cache, hq_first, hm, prompt_length, max_new_tokens):
        try:
            student_ids = self._student_greedy(cache, hq_first, max_new_tokens)
        finally:
            self._crop(cache, prompt_length)
        try:
            transmem_ids = self._transmem_greedy(
                cache, hq_first, hm, max_new_tokens)
        finally:
            self._crop(cache, prompt_length)
        student = self.tokenizer.decode(student_ids, skip_special_tokens=True).strip()
        transmem = self.tokenizer.decode(transmem_ids, skip_special_tokens=True).strip()
        return student, transmem, len(student_ids), len(transmem_ids)

    def _decode_from_prompt(
        self, cache, hq_first, hm, prompt_length, max_new_tokens
    ) -> dict[str, Any]:
        if self.mode == "paired" and not self.layered:
            student, transmem, student_tokens, transmem_tokens = (
                self._paired_from_prompt(
                    cache, hq_first, hm, prompt_length, max_new_tokens))
            return {
                "student_prediction": student,
                "transmem_prediction": transmem,
                "student_output_tokens": student_tokens,
                "transmem_output_tokens": transmem_tokens,
            }
        try:
            student_ids = self._student_greedy(
                cache, hq_first, max_new_tokens)
        finally:
            self._crop(cache, prompt_length)
        return {
            "student_prediction": self.tokenizer.decode(
                student_ids, skip_special_tokens=True).strip(),
            "student_output_tokens": len(student_ids),
        }

    def _add_layered_predictions(
        self,
        context: str,
        questions: Sequence[str],
        predictions: list[dict[str, Any]],
        max_new_tokens: int,
    ) -> list[dict[str, Any]]:
        if not self.layered:
            return predictions
        if self.layered_rollout is None:
            raise AssertionError("layered checkpoint has no LayeredRollout")
        for question, prediction in zip(questions, predictions):
            _, _, answer_ids = self.layered_rollout.student_rollout(
                self.mem,
                context,
                question,
                max_new_tokens,
                sample=False,
                temperature=1.0,
            )
            prediction.update({
                "transmem_prediction": self.tokenizer.decode(
                    answer_ids, skip_special_tokens=True).strip(),
                "transmem_output_tokens": len(answer_ids),
            })
        return predictions

    def predict_context(
        self,
        context: str,
        questions: Sequence[str],
        max_new_tokens: int,
        no_prefix_cache: bool,
        window_questions: Sequence[str] | None = None,
        max_prompt_tokens: int | None = None,
    ) -> tuple[ContextWindow, list[dict[str, Any]]]:
        torch = self.torch
        if max_prompt_tokens is None:
            max_prompt_tokens = (
                AGENT_INPUT_TOKENS - AGENT_BUFFER_TOKENS - max_new_tokens)
        if max_prompt_tokens + max_new_tokens + AGENT_BUFFER_TOKENS > AGENT_INPUT_TOKENS:
            raise ValueError(
                "prompt, generation, and buffer budgets exceed the agent input limit")
        # Window planning must include already-completed questions too.  Otherwise
        # a resumed run could retain more context than its first invocation.
        window = self.plan_context(
            context, window_questions or questions, max_prompt_tokens)
        prompts = [self._prompt(window.context, question) for question in questions]
        context_tokens = len(_encode(self.tokenizer, window.context))
        predictions: list[dict[str, Any]] = []

        with torch.inference_mode():
            if no_prefix_cache:
                for prompt in prompts:
                    output = self.model.model(
                        input_ids=prompt,
                        attention_mask=torch.ones_like(prompt),
                        use_cache=True,
                    )
                    cache = output.past_key_values
                    hidden = output.last_hidden_state[0]
                    hm = (
                        self._extract_hm(hidden, context_tokens)
                        if self.mode == "paired" and not self.layered else None)
                    hq = hidden[-1:, :]
                    prompt_length = int(prompt.shape[1])
                    prediction = self._decode_from_prompt(
                        cache, hq, hm, prompt_length, max_new_tokens)
                    prediction["prompt_tokens"] = prompt_length
                    predictions.append(prediction)
                if self.layered:
                    del cache, output, hidden, hq, hm
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return window, self._add_layered_predictions(
                    window.context, questions, predictions, max_new_tokens)

            prompt_lists = [_flat_prompt_ids(prompt) for prompt in prompts]
            prefix_ids = longest_common_prefix(prompt_lists)
            if not prefix_ids:
                raise RuntimeError("chat prompts unexpectedly have no common token prefix")
            prefix = torch.tensor([prefix_ids], device=self.device, dtype=torch.long)
            prefix_output = self.model.model(
                input_ids=prefix,
                attention_mask=torch.ones_like(prefix),
                use_cache=True,
            )
            cache = prefix_output.past_key_values
            prefix_hidden = prefix_output.last_hidden_state[0]
            hm = (
                self._extract_hm(prefix_hidden, context_tokens)
                if self.mode == "paired" and not self.layered else None)
            prefix_last = prefix_hidden[-1:, :]
            prefix_length = len(prefix_ids)
            suffix_output = None

            for prompt_ids in prompt_lists:
                suffix_ids = prompt_ids[prefix_length:]
                prompt_length = len(prompt_ids)
                try:
                    if suffix_ids:
                        suffix = torch.tensor(
                            [suffix_ids], device=self.device, dtype=torch.long)
                        attention = torch.ones(
                            (1, prompt_length), device=self.device, dtype=torch.long)
                        suffix_output = self.model.model(
                            input_ids=suffix,
                            attention_mask=attention,
                            past_key_values=cache,
                            use_cache=True,
                        )
                        hq = suffix_output.last_hidden_state[0, -1:, :]
                    else:
                        hq = prefix_last
                    prediction = self._decode_from_prompt(
                        cache, hq, hm, prompt_length, max_new_tokens)
                    prediction.update({
                        "prompt_tokens": prompt_length,
                        "common_prefix_tokens": prefix_length,
                    })
                    predictions.append(prediction)
                finally:
                    self._crop(cache, prefix_length)
        if self.layered:
            del cache, prefix_output, prefix_hidden, prefix_last, hm, hq, suffix_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return window, self._add_layered_predictions(
            window.context, questions, predictions, max_new_tokens)


def _file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _checkpoint_fingerprint(
    path: Path,
    checkpoint_id: str | None,
) -> dict[str, Any]:
    """Identify copied checkpoints without depending on a job-local NVMe path."""

    if checkpoint_id:
        return {
            "identity": checkpoint_id,
            **_content_fingerprint(path),
        }
    return _file_fingerprint(path)


_MODEL_IDENTITY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _content_fingerprint(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"model identity file changed while reading: {path}")
    return {"size": after.st_size, "sha256": digest.hexdigest()}


def _model_fingerprint(model_root: Path) -> dict[str, Any]:
    """Identify a model without depending on an ephemeral S3 mount path."""

    model_root = model_root.resolve()
    config_path = model_root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config is missing: {config_path}")
    metadata = {
        name: _content_fingerprint(model_root / name)
        for name in _MODEL_IDENTITY_FILES
        if (model_root / name).is_file()
    }
    weight_paths = set(model_root.glob("*.safetensors"))
    weight_paths.update(model_root.glob("*.bin"))
    weights = [
        {"name": path.name, "size": path.stat().st_size}
        for path in sorted(weight_paths, key=lambda item: item.name)
        if path.is_file()
    ]
    if not weights:
        raise FileNotFoundError(f"model weights are missing under: {model_root}")
    return {
        "repository_name": model_root.name,
        "metadata": metadata,
        "weights": weights,
    }


def _run_id(args: argparse.Namespace) -> str:
    checkpoint = Path(args.ckpt).resolve() if args.ckpt else None
    model_root = Path(args.model_path).resolve()
    data_files = [
        _file_fingerprint(path)
        for split in sorted({spec.split for spec in MAIN_SOURCE_SPECS.values()})
        for path in find_split_parquets(split, args.mab_root, args.data_dir)
    ]
    payload = {
        "model": _model_fingerprint(model_root),
        "mode": args.mode,
        "checkpoint": (
            _checkpoint_fingerprint(checkpoint, args.checkpoint_id)
            if checkpoint is not None else None),
        "mab_templates": _file_fingerprint(args.mab_root / "utils" / "templates.py"),
        "mab_metrics": _file_fingerprint(
            args.mab_root / "utils" / "eval_other_utils.py"),
        "data_files": data_files,
        "source_specs": {
            source: spec.__dict__ for source, spec in MAIN_SOURCE_SPECS.items()
        },
        "agent_input_tokens": args.agent_input_tokens,
        "agent_buffer_tokens": AGENT_BUFFER_TOKENS,
        "prefix_cache": not args.no_prefix_cache,
        "max_questions_per_source": args.max_questions_per_source,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired P1 TransMem/student evaluation on MAB main-13 sources")
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--mode", choices=["paired", "student"], default="paired",
        help="paired runs student+TransMem; student skips checkpoint loading")
    parser.add_argument("--ckpt", default=None, help="native TransMem best.pt")
    parser.add_argument(
        "--checkpoint_id", default=None,
        help="Stable checkpoint identity for job-local copies, e.g. an S3 URI")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mab_root", type=Path, default=DEFAULT_MAB_ROOT)
    parser.add_argument(
        "--data_dir", type=Path, default=None,
        help="Local directory containing <split>-*.parquet; default searches MAB "
             "processed_data then the local Hugging Face cache")
    parser.add_argument(
        "--sources", nargs="+", default=list(MAIN_SOURCE_SPECS),
        help="Subset of official main-13 source names")
    parser.add_argument(
        "--max_questions_per_source", type=int, default=None,
        help="Smoke-test cap; default evaluates each source's full official question set")
    parser.add_argument(
        "--agent_input_tokens", type=int, default=AGENT_INPUT_TOKENS,
        help="Model-specific total input ceiling before buffer/generation reserves")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--attn_impl", choices=["flash_attention_2", "sdpa", "eager"],
        default="sdpa")
    parser.add_argument(
        "--no_prefix_cache", action="store_true",
        help="Control path: prefill the complete prompt independently for each query")
    parser.add_argument(
        "--force", action="store_true",
        help="Delete selected-source progress files before evaluating")
    parser.add_argument("--print_examples", type=int, default=2)
    args = parser.parse_args()
    unknown = [source for source in args.sources if source not in MAIN_SOURCE_SPECS]
    if unknown:
        parser.error(f"not official main sources: {unknown}")
    if args.max_questions_per_source is not None and args.max_questions_per_source < 1:
        parser.error("--max_questions_per_source must be positive")
    if args.agent_input_tokens <= AGENT_BUFFER_TOKENS:
        parser.error("--agent_input_tokens must exceed the agent buffer")
    if args.mode == "paired" and not args.ckpt:
        parser.error("--ckpt is required when --mode=paired")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = list(dict.fromkeys(args.sources))
    run_id = _run_id(args)
    get_template, post_process = load_official_mab(args.mab_root)
    rows_by_source = load_rows_by_source(sources, args.mab_root, args.data_dir)

    source_records: dict[str, list[tuple[dict[str, Any], list[QueryRecord]]]] = {}
    pending = 0
    progress_state: dict[str, tuple[Path, dict[str, dict[str, Any]]]] = {}
    for source in sources:
        spec = MAIN_SOURCE_SPECS[source]
        limit = spec.question_count
        if args.max_questions_per_source is not None:
            limit = min(limit, args.max_questions_per_source)
        groups: list[tuple[dict[str, Any], list[QueryRecord]]] = []
        count = 0
        for row in rows_by_source[source]:
            context_index = int(row["_split_row_index"])
            records = build_query_records(row, context_index, source, get_template)
            if count + len(records) > limit:
                records = records[:limit - count]
            if records:
                groups.append((row, records))
                count += len(records)
            if count >= limit:
                break
        if count != limit:
            raise ValueError(f"{source}: local parquet yielded {count}/{limit} questions")
        source_records[source] = groups

        stem = _safe_source_name(source)
        progress_path = output_dir / f"{stem}.progress.jsonl"
        if args.force and progress_path.exists():
            progress_path.unlink()
        done = _load_progress(progress_path, run_id)
        requested_keys = {record.key for _, records in groups for record in records}
        pending += len(requested_keys - done.keys())
        progress_state[source] = progress_path, done

    runner = PairedTransMemGreedy(args) if pending else None
    summaries: dict[str, Any] = {}
    for source in sources:
        spec = MAIN_SOURCE_SPECS[source]
        progress_path, done = progress_state[source]
        printed = 0
        with progress_path.open("a", encoding="utf-8") as progress:
            for row, records in source_records[source]:
                missing = [record for record in records if record.key not in done]
                if not missing:
                    continue
                if runner is None:
                    raise AssertionError("pending work without an initialized runner")
                context = str(row["context"])
                window, paired = runner.predict_context(
                    context,
                    [record.formatted_query for record in missing],
                    spec.max_new_tokens,
                    args.no_prefix_cache,
                    window_questions=[
                        record.formatted_query for record in records],
                    max_prompt_tokens=source_prompt_budget(
                        source, args.agent_input_tokens),
                )
                for record, prediction in zip(missing, paired):
                    student_metrics, student_details = score_prediction(
                        prediction["student_prediction"], record.answer, source,
                        post_process, args.mab_root)
                    transmem_metrics = None
                    transmem_details = None
                    if "transmem_prediction" in prediction:
                        transmem_metrics, transmem_details = score_prediction(
                            prediction["transmem_prediction"], record.answer, source,
                            post_process, args.mab_root)
                    result = {
                        "run_id": run_id,
                        "checkpoint_step": runner.checkpoint_step,
                        "key": record.key,
                        "source": source,
                        "context_index": record.context_index,
                        "question_index": record.question_index,
                        "qa_pair_id": record.qa_pair_id,
                        "question_id": record.question_id,
                        "question_type": record.question_type,
                        "keypoints": record.keypoints,
                        "question": record.question,
                        "formatted_query": record.formatted_query,
                        "answer": record.answer,
                        **prediction,
                        "student_metrics": student_metrics,
                        "student_postprocess": student_details,
                        "needs_judge": spec.needs_judge,
                        "context_tokens_original": window.original_context_tokens,
                        "context_tokens_kept": window.kept_context_tokens,
                        "context_tokens_left_truncated": window.left_truncated_tokens,
                        "agent_input_tokens": args.agent_input_tokens,
                        "buffer_token_budget": AGENT_BUFFER_TOKENS,
                        "generation_token_budget": spec.max_new_tokens,
                        "prompt_token_budget": source_prompt_budget(
                            source, args.agent_input_tokens),
                    }
                    if transmem_metrics is not None:
                        result["transmem_metrics"] = transmem_metrics
                        result["transmem_postprocess"] = transmem_details
                    progress.write(json.dumps(
                        _jsonable(result), ensure_ascii=False) + "\n")
                    progress.flush()
                    done[record.key] = result
                    if printed < args.print_examples:
                        message = (
                            f"[{source}] {record.key}\n"
                            f"  gold={record.answer!r}\n"
                            f"  student={prediction['student_prediction'][:160]!r}")
                        if "transmem_prediction" in prediction:
                            message += (
                                f"\n  transmem="
                                f"{prediction['transmem_prediction'][:160]!r}")
                        print(message, flush=True)
                        printed += 1

        ordered = [
            done[record.key]
            for _, records in source_records[source]
            for record in records
            if record.key in done
        ]
        expected = sum(len(records) for _, records in source_records[source])
        summary_path = output_dir / f"{_safe_source_name(source)}.summary.json"
        previous_summary = None
        if summary_path.is_file():
            with summary_path.open(encoding="utf-8") as handle:
                previous_summary = json.load(handle)
        checkpoint_step = None
        if args.mode == "paired":
            checkpoint_step = resolve_checkpoint_step(
                runner_step=getattr(runner, "checkpoint_step", None),
                rows=ordered,
                previous_summary=previous_summary,
                run_id=run_id,
            )
        summary = summarize_source_rows(source, ordered, expected_questions=expected)
        summary.update({
            "run_id": run_id,
            "model_path": str(Path(args.model_path).resolve()),
            "mode": args.mode,
            "checkpoint": (
                str(Path(args.ckpt).resolve()) if args.ckpt else None),
            "checkpoint_id": args.checkpoint_id,
            "checkpoint_step": checkpoint_step,
            "decode": (
                "paired_greedy" if args.mode == "paired" else "student_greedy"),
            "prompt_format": "Project4 build_chat_prompt_ids",
            "agent_input_tokens": args.agent_input_tokens,
            "buffer_token_budget": AGENT_BUFFER_TOKENS,
            "generation_token_budget": spec.max_new_tokens,
            "prompt_token_budget": source_prompt_budget(
                source, args.agent_input_tokens),
            "prefix_cache": not args.no_prefix_cache,
            "progress_jsonl": str(progress_path.resolve()),
        })
        _write_json(summary_path, summary)
        summaries[source] = summary
        primary = spec.primary_metric
        if primary is None:
            print(f"[{source}] complete={summary['complete']} needs_judge=True", flush=True)
        else:
            message = (
                f"[{source}] {primary}: student={summary['student_primary']}")
            if args.mode == "paired":
                message += f" transmem={summary['transmem_primary']}"
            print(f"{message} n={summary['num_questions']}", flush=True)

    # Disjoint source jobs share this output root.  The lock makes the rescan and
    # atomic index replacement one transaction, so a late writer cannot publish
    # a stale subset after another source has finished.
    write_summary_index(output_dir, run_id, summaries)
    print(f"Wrote per-source summaries to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
