#!/usr/bin/env python3
"""Pure CPU tests for the MemoryAgentBench adapter.

Run with::

    python -m scripts.eval.test_mab_adapter
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from scripts.eval.eval_memory_agent_bench import (
    MAIN_SOURCE_SPECS,
    build_query_records,
    icl_parsed_label_exact,
    longest_common_prefix,
    parse_icl_label,
    plan_context_window,
    pushd,
    summarize_source_rows,
)


class FakeTokenizer:
    """Character tokenizer with a Hugging Face-like minimum surface."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(item) for item in ids)


def fake_prompt_builder(tokenizer, context, question, device=None):
    del device
    return [[900, *tokenizer.encode(context), 901,
             *tokenizer.encode(question), 902]]


def test_main_source_manifest_matches_official_main_experiment():
    assert list(MAIN_SOURCE_SPECS) == [
        "ruler_qa1_197K",
        "ruler_qa2_421K",
        "longmemeval_s*",
        "eventqa_full",
        "icl_banking77_5900shot_balance",
        "icl_clinic150_7050shot_balance",
        "icl_nlu_8296shot_balance",
        "icl_trec_coarse_6600shot_balance",
        "icl_trec_fine_6400shot_balance",
        "recsys_redial_full",
        "infbench_sum_eng_shots2",
        "factconsolidation_sh_262k",
        "factconsolidation_mh_262k",
    ]
    assert [spec.question_count for spec in MAIN_SOURCE_SPECS.values()] == [
        100, 100, 300, 500, 100, 100, 100, 100, 100, 200, 100, 100, 100,
    ]
    assert MAIN_SOURCE_SPECS["longmemeval_s*"].needs_judge
    assert MAIN_SOURCE_SPECS["infbench_sum_eng_shots2"].needs_judge
    assert MAIN_SOURCE_SPECS["eventqa_full"].primary_metric == "substring_exact_match"
    assert MAIN_SOURCE_SPECS["recsys_redial_full"].primary_metric == "recsys_recall@5"


def test_context_window_uses_longest_query_for_every_question():
    plan = plan_context_window(
        FakeTokenizer(),
        context="abcdefghij",
        questions=["x", "wxyz"],
        prompt_builder=fake_prompt_builder,
        max_prompt_tokens=12,
    )
    # Longest prompt overhead is [900], [901], four query chars, [902] = 7.
    # Both questions therefore receive the same five-token context suffix.
    assert plan.context == "fghij"
    assert plan.original_context_tokens == 10
    assert plan.kept_context_tokens == 5
    assert plan.left_truncated_tokens == 5
    assert plan.prompt_lengths == (9, 12)


def test_common_prefix_is_exact_and_empty_safe():
    assert longest_common_prefix([]) == []
    assert longest_common_prefix([[1, 2, 3]]) == [1, 2, 3]
    assert longest_common_prefix([[1, 2, 3], [1, 2, 4], [1, 2]]) == [1, 2]


def test_query_adapter_indexes_answers_not_the_whole_answer_column():
    row = {
        "context": "memory",
        "questions": ["first?", "second?"],
        "answers": [["alpha"], ["beta", "B"]],
        "metadata": {
            "source": "eventqa_full",
            "qa_pair_ids": ["qa-0", "qa-1"],
            "previous_events": [["e0"], ["e1"]],
        },
    }

    def template(source, template_name, agent_name):
        assert source == "eventqa_full"
        assert template_name == "query"
        assert "Long_context_agent" in agent_name
        return "Q={question}; previous={previous_events}"

    records = build_query_records(row, 7, "eventqa_full", template)
    assert [record.answer for record in records] == [["alpha"], ["beta", "B"]]
    assert records[1].formatted_query == "Q=second?; previous=['e1']"
    assert records[1].key == "eventqa_full:7:1"
    assert records[1].qa_pair_id == "qa-1"


def test_icl_label_parser_accepts_benchmark_and_model_forms():
    assert parse_icl_label("43") == "43"
    assert parse_icl_label("label: 43") == "43"
    assert parse_icl_label("label: {43}\n") == "43"
    assert parse_icl_label("The answer is label: 7.") == "7"
    assert parse_icl_label("no numerical label") is None

    answer = ["43"]
    assert icl_parsed_label_exact("label: {43}", answer)
    assert answer == ["43"], "scoring must not consume the answer list"


def test_source_summary_has_paired_metrics_and_no_fake_overall():
    rows = [
        {
            "student_metrics": {"exact_match": True, "f1": 0.5},
            "transmem_metrics": {"exact_match": False, "f1": 0.75},
            "context_tokens_original": 20,
            "context_tokens_kept": 12,
        },
        {
            "student_metrics": {"exact_match": False, "f1": 0.0},
            "transmem_metrics": {"exact_match": True, "f1": 0.25},
            "context_tokens_original": 10,
            "context_tokens_kept": 10,
        },
    ]
    summary = summarize_source_rows(
        "icl_banking77_5900shot_balance", rows, expected_questions=100)
    assert summary["num_questions"] == 2
    assert summary["expected_questions"] == 100
    assert summary["complete"] is False
    assert summary["student"]["exact_match"] == 0.5
    assert summary["transmem"]["f1"] == 0.5
    assert summary["context_tokens"]["left_truncated_total"] == 8
    assert "overall" not in summary


def test_pushd_resolves_redial_relative_path_and_restores_cwd():
    original = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        expected = target / "processed_data" / "Recsys_Redial"
        expected.mkdir(parents=True)
        with pushd(target):
            assert Path("./processed_data/Recsys_Redial").resolve() == expected
        assert Path.cwd() == original
    assert os.getcwd() == str(original)


def main():
    test_main_source_manifest_matches_official_main_experiment()
    test_context_window_uses_longest_query_for_every_question()
    test_common_prefix_is_exact_and_empty_safe()
    test_query_adapter_indexes_answers_not_the_whole_answer_column()
    test_icl_label_parser_accepts_benchmark_and_model_forms()
    test_source_summary_has_paired_metrics_and_no_fake_overall()
    test_pushd_resolves_redial_relative_path_and_restores_cwd()
    print("test_mab_adapter: PASS")


if __name__ == "__main__":
    main()
