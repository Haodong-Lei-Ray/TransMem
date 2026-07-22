#!/usr/bin/env python3
"""Pure CPU tests for the MemoryAgentBench adapter.

Run with::

    python -m scripts.eval.test_mab_adapter
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
import types
from pathlib import Path

import torch

from scripts.eval.eval_memory_agent_bench import (
    AGENT_BUFFER_TOKENS,
    AGENT_INPUT_TOKENS,
    MAIN_SOURCE_SPECS,
    PairedTransMemGreedy,
    _checkpoint_fingerprint,
    _model_fingerprint,
    build_query_records,
    find_split_parquets,
    icl_parsed_label_exact,
    longest_common_prefix,
    parse_icl_label,
    plan_context_window,
    pushd,
    resolve_checkpoint_step,
    source_prompt_budget,
    summarize_source_rows,
    write_summary_index,
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


class FakeCache:
    def __init__(self):
        self.tokens = []

    def crop(self, length):
        self.tokens = self.tokens[:length]

    def get_seq_length(self):
        return len(self.tokens)


class StatefulFakeBackbone:
    """Model boundary whose hidden state depends on the complete cache."""

    def __call__(self, input_ids, attention_mask=None,
                 past_key_values=None, use_cache=True):
        del attention_mask, use_cache
        cache = past_key_values if past_key_values is not None else FakeCache()
        hidden = []
        for token in input_ids[0].tolist():
            cache.tokens.append(int(token))
            state = sum(
                (index + 1) * value
                for index, value in enumerate(cache.tokens)
            ) % 997
            hidden.append([float(state)])
        return types.SimpleNamespace(
            past_key_values=cache,
            last_hidden_state=torch.tensor([hidden], dtype=torch.float32),
        )


class FakeLMHead:
    def __call__(self, hidden):
        token_ids = (hidden[..., 0].round().long() % 26) + 65
        logits = torch.full((*token_ids.shape, 128), -1000.0)
        return logits.scatter(-1, token_ids.unsqueeze(-1), 0.0)


class FakeModel:
    def __init__(self):
        self.model = StatefulFakeBackbone()
        self.lm_head = FakeLMHead()


class ZeroCorrectionMemory:
    def __call__(self, memory_input, past_key_values=None, use_cache=True):
        from transmem import TransMemOutput

        del past_key_values, use_cache
        ms = torch.zeros_like(memory_input[:, -1, :])
        return TransMemOutput(ms=ms, gate=torch.ones_like(ms[:, :1]))

    def correct(self, hq, proposal):
        return hq + proposal.delta


def tensor_prompt_builder(tokenizer, context, question, device=None):
    del device
    return torch.tensor(
        fake_prompt_builder(tokenizer, context, question), dtype=torch.long)


def make_fake_paired_runner():
    runner = object.__new__(PairedTransMemGreedy)
    runner.torch = torch
    runner.mode = "paired"
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    runner.tokenizer = FakeTokenizer()
    runner.build_chat_prompt_ids = tensor_prompt_builder
    runner.model = FakeModel()
    runner.mem = ZeroCorrectionMemory()
    runner.config = types.SimpleNamespace(n_mem=1, hm_mode="floor")
    runner.eos_ids = {127}
    return runner


def make_fake_student_runner():
    runner = make_fake_paired_runner()
    runner.mode = "student"
    runner.mem = None
    runner.config = None
    return runner


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


def test_source_prompt_budget_reserves_official_buffer_and_generation():
    # Official Long_context_agent_Qwen3-8B.yaml: input=128000, buffer=4000.
    # Dataset YAML generation maxima: Ruler QA1=50, InfBench summary=1200.
    assert AGENT_INPUT_TOKENS == 128_000
    assert AGENT_BUFFER_TOKENS == 4_000
    assert source_prompt_budget("ruler_qa1_197K") == 123_950
    assert source_prompt_budget("infbench_sum_eng_shots2") == 122_800
    for source, spec in MAIN_SOURCE_SPECS.items():
        assert (
            source_prompt_budget(source)
            + spec.max_new_tokens
            + AGENT_BUFFER_TOKENS
            == AGENT_INPUT_TOKENS
        )


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


def test_parquet_discovery_rejects_multiple_cached_revisions():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hub = root / "hub"
        for revision in ("revision-a", "revision-b"):
            data = (
                hub / "datasets--ai-hyz--MemoryAgentBench"
                / "snapshots" / revision / "data"
            )
            data.mkdir(parents=True)
            (data / "Accurate_Retrieval-00000-of-00001.parquet").touch()
        mab_root = root / "mab"
        (mab_root / "processed_data").mkdir(parents=True)

        saved = {name: os.environ.get(name)
                 for name in ("HF_HUB_CACHE", "HF_HOME", "HOME")}
        os.environ["HF_HUB_CACHE"] = str(hub)
        os.environ.pop("HF_HOME", None)
        os.environ["HOME"] = str(root / "home")
        try:
            try:
                find_split_parquets("Accurate_Retrieval", mab_root, None)
            except ValueError as exc:
                assert "multiple local dataset roots" in str(exc)
            else:
                raise AssertionError("multiple cached revisions must fail closed")
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def test_explicit_data_dir_rejects_duplicate_parquet_shards():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        shard_name = "Accurate_Retrieval-00000-of-00001.parquet"
        for copy in ("copy-a", "copy-b"):
            folder = data_dir / copy
            folder.mkdir()
            (folder / shard_name).touch()
        try:
            find_split_parquets(
                "Accurate_Retrieval", data_dir / "unused-mab-root", data_dir)
        except ValueError as exc:
            assert "duplicate parquet shard" in str(exc)
        else:
            raise AssertionError("duplicate explicit parquet shards must fail closed")


def test_common_prefix_is_exact_and_empty_safe():
    assert longest_common_prefix([]) == []
    assert longest_common_prefix([[1, 2, 3]]) == [1, 2, 3]
    assert longest_common_prefix([[1, 2, 3], [1, 2, 4], [1, 2]]) == [1, 2]


def test_prefix_cache_matches_full_prefill_without_cross_policy_or_question_state():
    runner = make_fake_paired_runner()
    questions = ["x", "yz"]
    cached_window, cached = runner.predict_context(
        "abcdef", questions, max_new_tokens=3, no_prefix_cache=False)
    plain_window, plain = runner.predict_context(
        "abcdef", questions, max_new_tokens=3, no_prefix_cache=True)

    assert cached_window.context == plain_window.context
    fields = (
        "student_prediction",
        "transmem_prediction",
        "student_output_tokens",
        "transmem_output_tokens",
    )
    assert [
        tuple(row[field] for field in fields) for row in cached
    ] == [
        tuple(row[field] for field in fields) for row in plain
    ]
    assert all(
        row["student_prediction"] == row["transmem_prediction"]
        for row in cached
    )


def test_student_only_prefix_cache_matches_full_prefill_without_transmem():
    runner = make_fake_student_runner()
    questions = ["x", "yz"]
    _, cached = runner.predict_context(
        "abcdef", questions, max_new_tokens=3, no_prefix_cache=False)
    _, plain = runner.predict_context(
        "abcdef", questions, max_new_tokens=3, no_prefix_cache=True)
    assert cached == plain or [
        row["student_prediction"] for row in cached
    ] == [
        row["student_prediction"] for row in plain
    ]
    assert all("transmem_prediction" not in row for row in cached + plain)


def test_query_adapter_indexes_answers_not_the_whole_answer_column():
    row = {
        "context": "memory",
        "questions": ["first?", "second?"],
        "answers": [["alpha"], ["beta", "B"]],
        "metadata": {
            "source": "eventqa_full",
            "qa_pair_ids": ["qa-0", "qa-1"],
            "question_ids": ["qid-0", "qid-1"],
            "question_types": ["single-session-user", "temporal-reasoning"],
            "keypoints": ["document point one", "document point two"],
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
    assert records[1].question_id == "qid-1"
    assert records[1].question_type == "temporal-reasoning"
    assert records[1].keypoints == [
        "document point one", "document point two"]


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


def test_longmemeval_summary_disallows_cross_domain_claim_without_overlap_audit():
    summary = summarize_source_rows(
        "longmemeval_s*", [], expected_questions=300)
    assert summary["evaluation_scope"] == "in_domain"
    assert summary["cross_domain_generalization_claim_allowed"] is False
    assert summary["contamination_status"] == "overlap_not_measured_by_adapter"
    assert "external manifest overlap audit" in summary["interpretation_note"]


def test_completed_resume_preserves_checkpoint_step_without_runner():
    assert resolve_checkpoint_step(
        runner_step=None,
        rows=[{"checkpoint_step": 4750}],
        previous_summary=None,
        run_id="run-a",
    ) == 4750


def test_model_fingerprint_is_stable_across_temporary_mount_paths():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def create_model(path: Path) -> None:
            path.mkdir(parents=True)
            (path / "config.json").write_text(
                '{"model_type":"qwen3"}', encoding="utf-8")
            (path / "tokenizer_config.json").write_text(
                '{"chat_template":"stable"}', encoding="utf-8")
            (path / "model.safetensors.index.json").write_text(
                '{"weight_map":{"layer":"model-00001-of-00001.safetensors"}}',
                encoding="utf-8",
            )
            (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")

        first = root / "s3_mab_111" / "Qwen3-4B-Instruct-2507"
        second = root / "s3_mab_222" / "Qwen3-4B-Instruct-2507"
        create_model(first)
        create_model(second)

        assert _model_fingerprint(first) == _model_fingerprint(second)
        (second / "config.json").write_text(
            '{"model_type":"other"}', encoding="utf-8")
        assert _model_fingerprint(first) != _model_fingerprint(second)


def test_explicit_checkpoint_id_is_stable_across_node_local_copies():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = root / "job-1" / "best.pt"
        second = root / "job-2" / "best.pt"
        first.parent.mkdir()
        second.parent.mkdir()
        first.write_bytes(b"same checkpoint")
        second.write_bytes(b"same checkpoint")
        checkpoint_id = (
            "s3://datafrontier/leihaodong/Project4/checkpoints/run/best.pt")

        assert _checkpoint_fingerprint(first, checkpoint_id) == (
            _checkpoint_fingerprint(second, checkpoint_id))

        second.write_bytes(b"different-sized checkpoint")
        assert _checkpoint_fingerprint(first, checkpoint_id) != (
            _checkpoint_fingerprint(second, checkpoint_id))
    assert resolve_checkpoint_step(
        runner_step=None,
        rows=[],
        previous_summary={"run_id": "run-a", "checkpoint_step": 4750},
        run_id="run-a",
    ) == 4750


def test_summary_index_waits_for_lock_and_merges_disjoint_source_jobs():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        run_id = "shared-run"
        source_a = {"run_id": run_id, "source": "source-a", "complete": True}
        source_b = {"run_id": run_id, "source": "source-b", "complete": True}
        (output_dir / "source-a.summary.json").write_text(
            json.dumps(source_a), encoding="utf-8")

        started = threading.Event()
        finished = threading.Event()
        errors = []

        def update_index():
            started.set()
            try:
                write_summary_index(output_dir, run_id, {"source-b": source_b})
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        lock_path = output_dir / ".summary.lock"
        with lock_path.open("a+", encoding="utf-8") as held_lock:
            fcntl.flock(held_lock.fileno(), fcntl.LOCK_EX)
            worker = threading.Thread(target=update_index)
            worker.start()
            assert started.wait(timeout=1)
            assert not finished.wait(timeout=0.1)
            fcntl.flock(held_lock.fileno(), fcntl.LOCK_UN)

        assert finished.wait(timeout=2)
        worker.join(timeout=1)
        assert not errors
        index = json.loads(
            (output_dir / "summary.json").read_text(encoding="utf-8"))
        assert set(index["sources"]) == {"source-a", "source-b"}


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
    test_source_prompt_budget_reserves_official_buffer_and_generation()
    test_context_window_uses_longest_query_for_every_question()
    test_parquet_discovery_rejects_multiple_cached_revisions()
    test_explicit_data_dir_rejects_duplicate_parquet_shards()
    test_common_prefix_is_exact_and_empty_safe()
    test_prefix_cache_matches_full_prefill_without_cross_policy_or_question_state()
    test_student_only_prefix_cache_matches_full_prefill_without_transmem()
    test_query_adapter_indexes_answers_not_the_whole_answer_column()
    test_icl_label_parser_accepts_benchmark_and_model_forms()
    test_source_summary_has_paired_metrics_and_no_fake_overall()
    test_longmemeval_summary_disallows_cross_domain_claim_without_overlap_audit()
    test_completed_resume_preserves_checkpoint_step_without_runner()
    test_model_fingerprint_is_stable_across_temporary_mount_paths()
    test_explicit_checkpoint_id_is_stable_across_node_local_copies()
    test_summary_index_waits_for_lock_and_merges_disjoint_source_jobs()
    test_pushd_resolves_redial_relative_path_and_restores_cwd()
    print("test_mab_adapter: PASS")


if __name__ == "__main__":
    main()
