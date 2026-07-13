#!/usr/bin/env python3
"""CPU tests for the cheap v3 diagnostic experiment helpers.

Run with::

    python -m transmem.test_diagnostics
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from transmem.diagnostics import (
    MetricAccumulator,
    make_derangement,
    parse_curve_steps,
    position_groups,
    token_metrics,
)
from transmem.objectives import FrozenLMHead


def test_frozen_lm_head_from_file_honors_dtype_and_device() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "lm_head.pt"
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        torch.save({"weight": weight, "tied": True}, path)

        head = FrozenLMHead.from_file(
            path,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        assert head.proj.weight.device.type == "cpu"
        assert head.proj.weight.dtype == torch.bfloat16
        assert not head.proj.weight.requires_grad

        hidden = torch.ones(2, 4, dtype=torch.bfloat16)
        logits = head(hidden)
        expected = torch.nn.functional.linear(hidden, weight.to(torch.bfloat16))
        assert logits.dtype == torch.bfloat16
        assert torch.equal(logits, expected)


def test_derangement() -> None:
    first = make_derangement(17, seed=42)
    second = make_derangement(17, seed=42)
    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(17))
    assert torch.all(first != torch.arange(17))

    try:
        make_derangement(1, seed=42)
        raise AssertionError("one sample cannot be shuffled without a fixed point")
    except ValueError:
        pass


def test_position_groups() -> None:
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    groups = position_groups(mask)
    assert torch.equal(groups["all"], mask)
    assert groups["first"].tolist() == [[True, False, False],
                                        [True, False, False],
                                        [True, False, False]]
    assert groups["later"].sum().item() == 3
    assert not torch.any(groups["first"] & groups["later"])
    assert torch.equal(groups["first"] | groups["later"], mask)


def test_metric_accumulator_is_position_weighted() -> None:
    acc = MetricAccumulator()
    # Two positions in the first update and one in the second.  The answer
    # must be weighted by positions, not by batches.
    acc.update("real", "all", {"forward_kl": torch.tensor([1.0, 3.0]),
                                 "teacher_top1": torch.tensor([1.0, 0.0])})
    acc.update("real", "all", {"forward_kl": torch.tensor([8.0]),
                                 "teacher_top1": torch.tensor([1.0])})
    out = acc.summary()["real"]["all"]
    assert out["positions"] == 3
    assert abs(out["forward_kl"] - 4.0) < 1e-8
    assert abs(out["teacher_top1"] - 2 / 3) < 1e-8


def test_parse_curve_steps() -> None:
    assert parse_curve_steps("") == ()
    assert parse_curve_steps("0, 500,500, 4750") == (0, 500, 4750)
    for bad in ("-1", "abc", "1.5"):
        try:
            parse_curve_steps(bad)
            raise AssertionError(f"invalid curve steps accepted: {bad}")
        except ValueError:
            pass


def test_token_metrics() -> None:
    teacher = torch.tensor([
        [3.0, 1.0, 0.0],
        [0.0, 2.0, 1.0],
        [0.0, 1.0, 3.0],
    ])
    candidate = torch.tensor([
        [4.0, 0.0, 0.0],  # agrees with teacher and trajectory target
        [3.0, 0.0, 0.0],  # disagrees with teacher and trajectory target
        [0.0, 1.0, 3.0],  # target is unavailable
    ])
    target = torch.tensor([0, 1, -100])
    correction = torch.tensor([
        [3.0, 4.0],
        [0.0, 2.0],
        [0.0, 0.0],
    ])
    real_top1 = torch.tensor([0, 1, 1])
    student_top1 = torch.tensor([1, 0, 2])

    metrics = token_metrics(
        candidate,
        teacher,
        target_ids=target,
        correction=correction,
        real_top1=real_top1,
        student_top1=student_top1,
    )

    assert metrics["forward_kl"].shape == (3,)
    assert torch.all(metrics["forward_kl"] >= -1e-6)
    assert metrics["teacher_top1"].tolist() == [1.0, 0.0, 1.0]
    expected_nll = -torch.log_softmax(candidate, dim=-1)[torch.arange(2), target[:2]]
    assert torch.allclose(metrics["trajectory_nll"][:2], expected_nll)
    assert torch.isnan(metrics["trajectory_nll"][2])
    assert metrics["trajectory_accuracy"].tolist()[:2] == [1.0, 0.0]
    assert torch.isnan(metrics["trajectory_accuracy"][2])
    assert metrics["correction_norm"].tolist() == [5.0, 2.0, 0.0]
    assert metrics["same_as_real"].tolist() == [1.0, 0.0, 0.0]
    assert metrics["changed_from_student"].tolist() == [1.0, 0.0, 0.0]

    try:
        token_metrics(candidate[:2], teacher)
        raise AssertionError("mismatched logits accepted")
    except ValueError:
        pass


if __name__ == "__main__":
    test_frozen_lm_head_from_file_honors_dtype_and_device()
    test_derangement()
    test_position_groups()
    test_metric_accumulator_is_position_weighted()
    test_parse_curve_steps()
    test_token_metrics()
    print("diagnostic helper tests passed")
