#!/usr/bin/env python3
"""Behavior tests for the MemoryAgentBench judge adapter."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.eval.eval_memory_agent_bench_judge import evaluate_source


MAB_ROOT = Path(
    "/mnt/petrelfs/leihaodong/Project1/MemoryAgentBenchProject/MemoryAgentBench"
)


def official_prompt(name: str) -> str:
    path = MAB_ROOT / "llm_based_eval" / "summarization_evaluate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"official prompt {name} not found")


class FakeJudgeClient:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class MemoryAgentBenchJudgeTests(unittest.TestCase):
    def test_longmem_abstention_uses_official_prompt_and_aggregates(self):
        row = {
            "run_id": "prediction-run",
            "key": "longmemeval_s*:0:0",
            "source": "longmemeval_s*",
            "question_id": "sample_abs",
            "question_type": "temporal-reasoning",
            "question": "Q?",
            "answer": "Nothing says.",
            "student_prediction": "I cannot know.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            client = FakeJudgeClient(["YES"])

            summary = evaluate_source(
                source="longmemeval_s*",
                mode="student",
                input_path=input_path,
                output_dir=root / "out",
                mab_root=MAB_ROOT,
                judge_model="gpt-4o",
                client=client,
                sleep=lambda _: None,
            )

            expected = (
                "I will give you an unanswerable question, an explanation, and a "
                "response from a model. Please answer yes if the model correctly "
                "identifies the question as unanswerable. The model could say that "
                "the information is incomplete, or some other information is given "
                "but the asked information is not.\n\nQuestion: Q?\n\nExplanation: "
                "Nothing says.\n\nModel Response: I cannot know.\n\nDoes the model "
                "correctly identify the question as unanswerable? Answer yes or no "
                "only."
            )
            self.assertEqual(client.calls[0]["prompt"], expected)
            self.assertEqual(client.calls[0]["model"], "gpt-4o")
            self.assertEqual(summary["metrics"]["accuracy"], 1.0)
            self.assertEqual(
                summary["metrics"]["by_question_type"]["temporal-reasoning"],
                {"accuracy": 1.0, "count": 1},
            )
            self.assertTrue(summary["complete"])
            self.assertEqual(
                len((root / "out" / "calls.progress.jsonl").read_text().splitlines()),
                1,
            )
            self.assertEqual(
                len((root / "out" / "rows.progress.jsonl").read_text().splitlines()),
                1,
            )

    def test_infbench_uses_three_official_book_prompts_and_resumes(self):
        row = {
            "run_id": "prediction-run",
            "key": "infbench_sum_eng_shots2:0:0",
            "source": "infbench_sum_eng_shots2",
            "qa_pair_id": "book-0",
            "keypoints": ["Point one", "Point two"],
            "answer": ["Expert summary text."],
            "transmem_prediction": "Candidate summary.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            client = FakeJudgeClient([
                'Reasoning {"fluency": 1}',
                'Reasoning {"supported_key_points": [1], "recall": 1}',
                'Reasoning {"precision": 2, "sentence_count": 4}',
            ])

            summary = evaluate_source(
                source="infbench_sum_eng_shots2",
                mode="transmem",
                input_path=input_path,
                output_dir=root / "out",
                mab_root=MAB_ROOT,
                judge_model="gpt-4o",
                client=client,
                sleep=lambda _: None,
            )

            self.assertEqual(len(client.calls), 3)
            self.assertEqual(client.calls[0]["temperature"], 0.1)
            self.assertEqual(client.calls[0]["top_p"], 0.9)
            self.assertEqual(client.calls[0]["seed"], 42)
            self.assertEqual(
                client.calls[0]["prompt"],
                official_prompt("fluency_prompt_book").format(
                    text="Candidate summary."
                ),
            )
            self.assertEqual(
                client.calls[1]["prompt"],
                official_prompt("recall_prompt_book").format(
                    keypoints="1. Point one\n2. Point two",
                    summary="Candidate summary.",
                ),
            )
            self.assertEqual(
                client.calls[2]["prompt"],
                official_prompt("precision_prompt_book").format(
                    expert_summary="Expert summary text.",
                    summary="Candidate summary.",
                ),
            )
            self.assertEqual(
                summary["metrics"],
                {"f1": 0.5, "fluency": 1.0, "precision": 0.5, "recall": 0.5},
            )
            self.assertEqual(
                len((root / "out" / "calls.progress.jsonl").read_text().splitlines()),
                3,
            )

            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
                resumed = evaluate_source(
                    source="infbench_sum_eng_shots2",
                    mode="transmem",
                    input_path=input_path,
                    output_dir=root / "out",
                    mab_root=MAB_ROOT,
                    judge_model="gpt-4o",
                    client=None,
                    sleep=lambda _: None,
                )
            self.assertEqual(resumed, summary)

    def test_infbench_retries_unparseable_judge_response(self):
        row = {
            "run_id": "prediction-run",
            "key": "infbench_sum_eng_shots2:0:0",
            "source": "infbench_sum_eng_shots2",
            "qa_pair_id": "book-0",
            "keypoints": ["Only point"],
            "answer": "Expert summary.",
            "student_prediction": "Candidate summary.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            client = FakeJudgeClient([
                '{"wrong_schema": 1}',
                '{"fluency": 1}',
                '{"recall": 1}',
                '{"precision": 1, "sentence_count": 1}',
            ])
            sleeps: list[float] = []

            summary = evaluate_source(
                source="infbench_sum_eng_shots2",
                mode="student",
                input_path=input_path,
                output_dir=root / "out",
                mab_root=MAB_ROOT,
                client=client,
                max_attempts=2,
                initial_backoff=0.25,
                max_backoff=0.25,
                sleep=sleeps.append,
            )

            self.assertEqual(len(client.calls), 4)
            self.assertEqual(sleeps, [0.25])
            self.assertEqual(summary["metrics"]["f1"], 1.0)
            calls = (root / "out" / "calls.progress.jsonl").read_text().splitlines()
            self.assertEqual(len(calls), 3)
            self.assertNotIn("wrong_schema", "\n".join(calls))

    def test_sbatch_runner_is_no_gpu_and_maps_required_parameters(self):
        runner = (
            Path(__file__).resolve().parent
            / "Qwen3-4B-Instruct-2507"
            / "run_eval_memory_agent_bench_judge.sh"
        )
        lines = runner.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "#!/bin/bash")
        directive_end = next(
            index for index, line in enumerate(lines[1:], start=1)
            if not line.startswith("#SBATCH")
        )
        self.assertGreater(directive_end, 1)
        self.assertFalse(any("--gres" in line for line in lines))
        log_lines = [
            line for line in lines[1:directive_end]
            if line.startswith("#SBATCH -o") or line.startswith("#SBATCH -e")
        ]
        self.assertEqual(len(log_lines), 2)
        self.assertTrue(all("/mnt/petrelfs/leihaodong/Project4/logs/" in line for line in log_lines))

        environment = dict(os.environ)
        environment.update({
            "SOURCE": "longmemeval_s*",
            "MODE": "student",
            "INPUT": "/tmp/predictions.jsonl",
            "OUT": "/tmp/judge-output",
            "DRY_RUN": "1",
        })
        completed = subprocess.run(
            ["bash", str(runner)],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--source longmemeval_s\\*", completed.stdout)
        self.assertIn("--mode student", completed.stdout)
        self.assertIn("--input /tmp/predictions.jsonl", completed.stdout)
        self.assertIn("--output_dir /tmp/judge-output", completed.stdout)


if __name__ == "__main__":
    unittest.main()
