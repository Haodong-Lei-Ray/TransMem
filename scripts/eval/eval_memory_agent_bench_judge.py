#!/usr/bin/env python3
"""Official-compatible LLM judges for MemoryAgentBench prediction JSONL.

The prediction adapter writes one JSON object per question.  This program judges
one prediction mode at a time and persists both API-call and completed-row events,
so an interrupted paid evaluation resumes without repeating successful calls.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


DEFAULT_MAB_ROOT = Path(
    "/mnt/petrelfs/leihaodong/Project1/MemoryAgentBenchProject/MemoryAgentBench"
)
LONGMEM_SOURCE = "longmemeval_s*"
INFBENCH_SOURCE = "infbench_sum_eng_shots2"
ADAPTER_VERSION = 5


class JudgeClient(Protocol):
    """Small external-API seam used by the evaluator and mock tests."""

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        ...


@dataclass(frozen=True)
class OfficialAssets:
    """Selected contracts loaded directly from the checked-out official files."""

    get_anscheck_prompt: Callable[[str, Any, Any, str, bool], str]
    longmem_sha256: str
    fluency_prompt_book: str
    recall_prompt_book: str
    precision_prompt_book: str
    parse_json: Callable[[str], dict[str, Any] | None]
    summarization_sha256: str

    @classmethod
    def load(cls, mab_root: Path) -> "OfficialAssets":
        path = mab_root / "llm_based_eval" / "longmem_qa_evaluate.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        function = next(
            (
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "get_anscheck_prompt"
            ),
            None,
        )
        if not isinstance(function, ast.FunctionDef):
            raise ValueError(f"get_anscheck_prompt is missing from {path}")
        namespace: dict[str, Any] = {}
        module = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(path), "exec"), namespace)

        summarization_path = (
            mab_root / "llm_based_eval" / "summarization_evaluate.py"
        )
        summarization_source = summarization_path.read_text(encoding="utf-8")
        summarization_tree = ast.parse(
            summarization_source, filename=str(summarization_path)
        )
        literals: dict[str, str] = {}
        wanted = {
            "fluency_prompt_book",
            "recall_prompt_book",
            "precision_prompt_book",
        }
        parse_function: ast.FunctionDef | None = None
        for node in summarization_tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "parse_json":
                parse_function = node
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in wanted:
                        value = ast.literal_eval(node.value)
                        if not isinstance(value, str):
                            raise ValueError(
                                f"{target.id} in {summarization_path} is not a string"
                            )
                        literals[target.id] = value
        missing = sorted(wanted - literals.keys())
        if missing or parse_function is None:
            raise ValueError(
                f"missing official summarization contracts in {summarization_path}: "
                f"{missing or ['parse_json']}"
            )
        parse_namespace: dict[str, Any] = {"json": json, "re": re}
        parse_module = ast.Module(body=[parse_function], type_ignores=[])
        ast.fix_missing_locations(parse_module)
        exec(compile(parse_module, str(summarization_path), "exec"), parse_namespace)
        return cls(
            get_anscheck_prompt=namespace["get_anscheck_prompt"],
            longmem_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            fluency_prompt_book=literals["fluency_prompt_book"],
            recall_prompt_book=literals["recall_prompt_book"],
            precision_prompt_book=literals["precision_prompt_book"],
            parse_json=parse_namespace["parse_json"],
            summarization_sha256=hashlib.sha256(
                summarization_source.encode("utf-8")
            ).hexdigest(),
        )


class OpenAICompatibleJudge:
    """OpenAI-compatible chat-completions client configured only from env vars."""

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set for real judge calls")
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
            "OPENAI_API_BASE"
        )
        from openai import OpenAI

        kwargs: dict[str, str] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        request: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "n": 1,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            request["top_p"] = top_p
        if seed is not None:
            request["seed"] = seed
        completion = self._client.chat.completions.create(**request)
        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("judge returned an empty message")
        return content.strip()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _read_predictions(path: Path, source: str, mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys: set[str] = set()
    prediction_field = f"{mode}_prediction"
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if row.get("source") != source:
                raise ValueError(
                    f"{path}:{line_number}: expected source {source!r}, "
                    f"got {row.get('source')!r}"
                )
            key = row.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path}:{line_number}: missing string key")
            if key in keys:
                raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
            keys.add(key)
            prediction = row.get(prediction_field)
            if not isinstance(prediction, str):
                raise ValueError(
                    f"{path}:{line_number}: missing string {prediction_field}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no prediction rows")
    return rows


def _load_events(path: Path, run_id: str) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return events
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: truncated or invalid progress; "
                    "refusing an unsafe resume"
                ) from exc
            if event.get("run_id") != run_id:
                raise ValueError(
                    f"{path}:{line_number}: progress belongs to another run; "
                    "use a new output directory"
                )
            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError(f"{path}:{line_number}: missing event_id")
            if event_id in events:
                raise ValueError(f"{path}:{line_number}: duplicate event_id {event_id!r}")
            events[event_id] = event
    return events


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _complete_with_retry(
    client: JudgeClient,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    max_attempts: int,
    initial_backoff: float,
    max_backoff: float,
    sleep: Callable[[float], None],
) -> str:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    delay = initial_backoff
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.complete(
                model=model, prompt=prompt, max_tokens=max_tokens, temperature=0
            )
            if not isinstance(response, str) or not response.strip():
                raise RuntimeError("judge returned an empty response")
            return response.strip()
        except Exception as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"judge call failed after {max_attempts} attempts"
                ) from exc
            sleep(min(delay, max_backoff))
            delay = min(max(delay * 2, initial_backoff), max_backoff)
    raise AssertionError("retry loop terminated unexpectedly")


def _complete_parsed_with_retry(
    client: JudgeClient,
    *,
    model: str,
    prompt: str,
    parser: Callable[[str], dict[str, Any] | None],
    validator: Callable[[Mapping[str, Any]], None],
    max_attempts: int,
    initial_backoff: float,
    max_backoff: float,
    sleep: Callable[[float], None],
) -> tuple[str, dict[str, Any]]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    delay = initial_backoff
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.complete(
                model=model,
                prompt=prompt,
                max_tokens=4096,
                temperature=0.1,
                top_p=0.9,
                seed=42,
            )
            if not isinstance(response, str) or not response.strip():
                raise RuntimeError("judge returned an empty response")
            parsed = parser(response.strip())
            if not isinstance(parsed, dict):
                raise ValueError("judge response has no parseable official JSON")
            validator(parsed)
            return response.strip(), parsed
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            sleep(min(delay, max_backoff))
            delay = min(max(delay * 2, initial_backoff), max_backoff)
    raise RuntimeError(
        f"judge call or parse failed after {max_attempts} attempts"
    ) from last_error


def _longmem_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[bool]] = defaultdict(list)
    labels: list[bool] = []
    for row in rows:
        label = bool(row["label"])
        labels.append(label)
        by_type[str(row["question_type"])].append(label)
    return {
        "accuracy": sum(labels) / len(labels),
        "by_question_type": {
            question_type: {
                "accuracy": sum(values) / len(values),
                "count": len(values),
            }
            for question_type, values in sorted(by_type.items())
        },
    }


def _infbench_inputs(row: Mapping[str, Any]) -> tuple[list[str], str]:
    keypoints = row.get("keypoints")
    if not isinstance(keypoints, list) or not keypoints:
        raise ValueError(f"{row.get('key')}: keypoints must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in keypoints):
        raise ValueError(f"{row.get('key')}: every keypoint must be a non-empty string")
    answer = row.get("answer")
    if isinstance(answer, str):
        expert_summary = answer
    elif (
        isinstance(answer, list)
        and len(answer) == 1
        and isinstance(answer[0], str)
    ):
        expert_summary = answer[0]
    else:
        raise ValueError(
            f"{row.get('key')}: answer must be a string or singleton string list"
        )
    if not expert_summary.strip():
        raise ValueError(f"{row.get('key')}: expert answer is empty")
    return list(keypoints), expert_summary


def _number(parsed: Mapping[str, Any], name: str, event_id: str) -> float:
    value = parsed.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{event_id}: parsed {name} is not numeric")
    return float(value)


def _validate_infbench_metric(
    metric: str,
    parsed: Mapping[str, Any],
    keypoint_count: int,
    event_id: str,
) -> None:
    if metric == "fluency":
        fluency = _number(parsed, "fluency", event_id)
        if fluency not in {0.0, 1.0}:
            raise ValueError(f"{event_id}: fluency must be 0 or 1")
        return
    if metric == "recall":
        recall = _number(parsed, "recall", event_id)
        if not 0 <= recall <= keypoint_count:
            raise ValueError(f"{event_id}: recall is outside keypoint bounds")
        return
    precision = _number(parsed, "precision", event_id)
    sentence_count = _number(parsed, "sentence_count", event_id)
    if sentence_count < 0 or precision < 0:
        raise ValueError(f"{event_id}: precision counts cannot be negative")
    if sentence_count == 0 and precision != 0:
        raise ValueError(f"{event_id}: nonzero precision with zero sentences")
    if sentence_count > 0 and precision > sentence_count:
        raise ValueError(f"{event_id}: precision exceeds sentence_count")


def _infbench_metrics(
    parsed: Mapping[str, Mapping[str, Any]], keypoint_count: int, event_id: str
) -> dict[str, float | int]:
    fluency = _number(parsed["fluency"], "fluency", event_id)
    recall_found = _number(parsed["recall"], "recall", event_id)
    precision_found = _number(parsed["precision"], "precision", event_id)
    precision_total = _number(parsed["precision"], "sentence_count", event_id)
    if fluency not in {0.0, 1.0}:
        raise ValueError(f"{event_id}: fluency must be 0 or 1")
    if not 0 <= recall_found <= keypoint_count:
        raise ValueError(f"{event_id}: recall is outside keypoint bounds")
    if precision_total < 0 or precision_found < 0:
        raise ValueError(f"{event_id}: precision counts cannot be negative")
    if precision_total == 0 and precision_found != 0:
        raise ValueError(f"{event_id}: nonzero precision with zero sentences")
    if precision_total > 0 and precision_found > precision_total:
        raise ValueError(f"{event_id}: precision exceeds sentence_count")
    recall = recall_found / keypoint_count if keypoint_count > 0 else 0.0
    precision = precision_found / precision_total if precision_total > 0 else 0.0
    f1 = (
        fluency * 2 * recall * precision / (recall + precision)
        if recall + precision > 0
        else 0.0
    )
    return {
        "fluency": fluency,
        "recall_total": keypoint_count,
        "recall_found": recall_found,
        "precision_total": precision_total,
        "precision_found": precision_found,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def _infbench_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        name: sum(float(row["metrics"][name]) for row in rows) / len(rows)
        for name in ("f1", "fluency", "precision", "recall")
    }


def evaluate_source(
    *,
    source: str,
    mode: str,
    input_path: Path,
    output_dir: Path,
    mab_root: Path = DEFAULT_MAB_ROOT,
    judge_model: str = "gpt-4o",
    client: JudgeClient | None = None,
    max_attempts: int = 6,
    initial_backoff: float = 2.0,
    max_backoff: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Judge one source/mode and return the atomically persisted summary."""

    if source not in {LONGMEM_SOURCE, INFBENCH_SOURCE}:
        raise ValueError(f"unsupported judge source: {source}")
    if mode not in {"student", "transmem"}:
        raise ValueError("mode must be student or transmem")
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    rows = _read_predictions(input_path, source, mode)
    assets = OfficialAssets.load(mab_root.resolve())
    input_sha256 = _sha256_bytes(input_path.read_bytes())
    run_config = {
        "adapter_version": ADAPTER_VERSION,
        "source": source,
        "mode": mode,
        "input_sha256": input_sha256,
        "judge_model": judge_model,
        "official_contract_sha256": (
            assets.longmem_sha256
            if source == LONGMEM_SOURCE
            else assets.summarization_sha256
        ),
    }
    run_id = _sha256_json(run_config)[:20]
    calls_path = output_dir / "calls.progress.jsonl"
    rows_path = output_dir / "rows.progress.jsonl"
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous_summary.get("run_id") != run_id:
            raise ValueError(
                f"{summary_path}: summary belongs to another run; "
                "use a new output directory"
            )
    calls = _load_events(calls_path, run_id)
    completed = _load_events(rows_path, run_id)
    for event in completed.values():
        call_event_ids = event.get("call_event_ids")
        if isinstance(call_event_ids, dict):
            referenced_calls = list(call_event_ids.values())
        else:
            referenced_calls = [event.get("call_event_id")]
        if not referenced_calls or any(
            call_event_id not in calls for call_event_id in referenced_calls
        ):
            raise ValueError(
                f"{rows_path}: completed row refers to a missing call event"
            )
    prediction_field = f"{mode}_prediction"
    for row in rows:
        row_event_id = f"row:{mode}:{row['key']}"
        if row_event_id in completed:
            continue
        if source == INFBENCH_SOURCE:
            keypoints, expert_summary = _infbench_inputs(row)
            prediction = row[prediction_field].strip()
            formatted_keypoints = "\n".join(
                f"{index + 1}. {keypoint}"
                for index, keypoint in enumerate(keypoints)
            )
            prompts = {
                "fluency": assets.fluency_prompt_book.format(text=prediction),
                "recall": assets.recall_prompt_book.format(
                    keypoints=formatted_keypoints, summary=prediction
                ),
                "precision": assets.precision_prompt_book.format(
                    expert_summary=expert_summary, summary=prediction
                ),
            }
            parsed: dict[str, Mapping[str, Any]] = {}
            call_event_ids: dict[str, str] = {}
            for metric in ("fluency", "recall", "precision"):
                prompt = prompts[metric]
                call_event_id = f"call:{mode}:{row['key']}:{metric}"
                call_event_ids[metric] = call_event_id
                call_event = calls.get(call_event_id)
                prompt_sha256 = _sha256_bytes(prompt.encode("utf-8"))
                if call_event is None:
                    if client is None:
                        client = OpenAICompatibleJudge()
                    raw_response, parsed_response = _complete_parsed_with_retry(
                        client,
                        model=judge_model,
                        prompt=prompt,
                        parser=assets.parse_json,
                        validator=lambda value, metric=metric: _validate_infbench_metric(
                            metric, value, len(keypoints), call_event_id
                        ),
                        max_attempts=max_attempts,
                        initial_backoff=initial_backoff,
                        max_backoff=max_backoff,
                        sleep=sleep,
                    )
                    call_event = {
                        "run_id": run_id,
                        "event_id": call_event_id,
                        "kind": "call",
                        "input_key": row["key"],
                        "metric": metric,
                        "judge_model": judge_model,
                        "prompt_sha256": prompt_sha256,
                        "raw_response": raw_response,
                        "parsed": parsed_response,
                    }
                    _append_event(calls_path, call_event)
                    calls[call_event_id] = call_event
                elif call_event.get("prompt_sha256") != prompt_sha256:
                    raise ValueError(
                        f"{call_event_id}: prompt hash changed; refusing resume"
                    )
                parsed_response = call_event.get("parsed")
                if not isinstance(parsed_response, dict):
                    raise ValueError(f"{call_event_id}: missing parsed judge JSON")
                parsed[metric] = parsed_response
            metrics = _infbench_metrics(parsed, len(keypoints), row_event_id)
            row_event = {
                "run_id": run_id,
                "event_id": row_event_id,
                "kind": "row",
                "key": row["key"],
                "source": source,
                "mode": mode,
                "qa_pair_id": row.get("qa_pair_id"),
                "keypoints": keypoints,
                "answer": row["answer"],
                "prediction": prediction,
                "metrics": metrics,
                "call_event_ids": call_event_ids,
            }
            _append_event(rows_path, row_event)
            completed[row_event_id] = row_event
            continue
        for required in ("question_id", "question_type", "question", "answer"):
            if row.get(required) is None:
                raise ValueError(f"{row['key']}: missing {required}")
        prediction = row[prediction_field]
        prompt = assets.get_anscheck_prompt(
            row["question_type"],
            row["question"],
            row["answer"],
            prediction,
            abstention="_abs" in str(row["question_id"]),
        )
        call_event_id = f"call:{mode}:{row['key']}:correctness"
        call_event = calls.get(call_event_id)
        prompt_sha256 = _sha256_bytes(prompt.encode("utf-8"))
        if call_event is None:
            if client is None:
                client = OpenAICompatibleJudge()
            raw_response = _complete_with_retry(
                client,
                model=judge_model,
                prompt=prompt,
                max_tokens=10,
                max_attempts=max_attempts,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                sleep=sleep,
            )
            label = "yes" in raw_response.lower()
            call_event = {
                "run_id": run_id,
                "event_id": call_event_id,
                "kind": "call",
                "input_key": row["key"],
                "metric": "correctness",
                "judge_model": judge_model,
                "prompt_sha256": prompt_sha256,
                "raw_response": raw_response,
                "parsed": {"label": label},
            }
            _append_event(calls_path, call_event)
            calls[call_event_id] = call_event
        elif call_event.get("prompt_sha256") != prompt_sha256:
            raise ValueError(f"{call_event_id}: prompt hash changed; refusing resume")
        label = call_event.get("parsed", {}).get("label")
        if not isinstance(label, bool):
            raise ValueError(f"{call_event_id}: missing boolean parsed label")
        row_event = {
            "run_id": run_id,
            "event_id": row_event_id,
            "kind": "row",
            "key": row["key"],
            "source": source,
            "mode": mode,
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "question": row["question"],
            "answer": row["answer"],
            "prediction": prediction,
            "label": label,
            "autoeval_label": {"model": judge_model, "label": label},
            "call_event_id": call_event_id,
        }
        _append_event(rows_path, row_event)
        completed[row_event_id] = row_event
    ordered = [completed[f"row:{mode}:{row['key']}"] for row in rows]
    metrics = (
        _longmem_summary(ordered)
        if source == LONGMEM_SOURCE
        else _infbench_summary(ordered)
    )
    summary = {
        "run_id": run_id,
        "run_config": run_config,
        "source": source,
        "mode": mode,
        "judge_model": judge_model,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "num_rows_expected": len(rows),
        "num_rows_scored": len(ordered),
        "complete": len(ordered) == len(rows),
        "metrics": metrics,
        "calls_progress_jsonl": str(calls_path),
        "rows_progress_jsonl": str(rows_path),
    }
    _write_json_atomic(summary_path, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official-compatible MemoryAgentBench LLM judge adapter"
    )
    parser.add_argument(
        "--source", required=True, choices=[LONGMEM_SOURCE, INFBENCH_SOURCE]
    )
    parser.add_argument("--mode", required=True, choices=["student", "transmem"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mab_root", type=Path, default=DEFAULT_MAB_ROOT)
    parser.add_argument("--judge_model", default="gpt-4o")
    parser.add_argument("--max_attempts", type=int, default=6)
    parser.add_argument("--initial_backoff", type=float, default=2.0)
    parser.add_argument("--max_backoff", type=float, default=30.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = evaluate_source(
        source=args.source,
        mode=args.mode,
        input_path=args.input,
        output_dir=args.output_dir,
        mab_root=args.mab_root,
        judge_model=args.judge_model,
        max_attempts=args.max_attempts,
        initial_backoff=args.initial_backoff,
        max_backoff=args.max_backoff,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
