#!/usr/bin/env python3
"""Tests for deterministic MemoryAgentBench source scheduling."""

from scripts.eval.plan_mab_parallel import partition_sources


def test_partition_covers_sources_once_and_balances_question_counts():
    sources = [
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
    plan = partition_sources(sources, worker_count=4)

    assigned = [source for worker in plan for source in worker.sources]
    assert sorted(assigned) == sorted(sources)
    assert len(assigned) == len(set(assigned))
    assert [worker.worker_index for worker in plan] == [0, 1, 2, 3]
    assert sum(worker.question_count for worker in plan) == 2000
    assert max(worker.question_count for worker in plan) <= 500
    assert max(worker.question_count for worker in plan) - min(
        worker.question_count for worker in plan) <= 100


def test_more_workers_than_sources_does_not_emit_empty_workers():
    plan = partition_sources(
        ["eventqa_full", "ruler_qa1_197K"], worker_count=8)
    assert len(plan) == 2
    assert [worker.question_count for worker in plan] == [500, 100]


def main():
    test_partition_covers_sources_once_and_balances_question_counts()
    test_more_workers_than_sources_does_not_emit_empty_workers()
    print("test_mab_parallel_plan: PASS")


if __name__ == "__main__":
    main()
