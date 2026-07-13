#!/usr/bin/env python3
"""CPU tests for checkpoint-curve snapshotting and pure scoring helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from scripts.eval.eval_checkpoint_curve import discover_checkpoints, score_predictions
from transmem.train_offpolicy import (
    curve_checkpoint_path,
    resolve_schedule_total_steps,
    save_curve_checkpoint,
    seed_everything,
)


def test_schedule_horizon_covers_training():
    assert resolve_schedule_total_steps(10, None) == 10
    assert resolve_schedule_total_steps(10, 15) == 15
    try:
        resolve_schedule_total_steps(10, 9)
    except ValueError as exc:
        assert ">= actual total_steps" in str(exc)
    else:
        raise AssertionError("a shorter LR horizon must be rejected")


def test_seed_is_reproducible():
    seed_everything(17)
    first_model = torch.nn.Linear(4, 3)
    first_random = torch.rand(3)
    seed_everything(17)
    second_model = torch.nn.Linear(4, 3)
    second_random = torch.rand(3)
    assert torch.equal(first_model.weight, second_model.weight)
    assert torch.equal(first_random, second_random)


def test_model_only_curve_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        model = torch.nn.Linear(3, 2)
        path = curve_checkpoint_path(tmp, 7)
        save_curve_checkpoint(
            path,
            model=model,
            config={"dim": 3, "depth": 1},
            global_step=7,
            epoch=2,
            seed=123,
            schedule_total_steps=99,
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert path.name == "curve_step_0000007.pt"
        assert checkpoint["global_step"] == 7
        assert checkpoint["config"] == {"dim": 3, "depth": 1}
        assert checkpoint["seed"] == 123
        assert checkpoint["schedule_total_steps"] == 99
        assert "optimizer_state_dict" not in checkpoint
        assert set(checkpoint["model_state_dict"]) == {"weight", "bias"}


def test_discovery_and_scoring_are_stable():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Discovery order need not match training order.
        for step in (10, 2):
            torch.save(
                {"global_step": step, "config": {"dim": 3}, "model_state_dict": {}},
                root / f"curve_step_{step:07d}.pt",
            )
        found = discover_checkpoints(root, explicit_paths=None)
        assert [(item.step, item.path.name) for item in found] == [
            (2, "curve_step_0000002.pt"),
            (10, "curve_step_0000010.pt"),
        ]

    records = [
        {"question": "q1", "ground_truth": "The Alpha"},
        {"question": "q2", "ground_truth": "Beta"},
    ]
    result = score_predictions(records, ["alpha", "not beta-ish"])
    assert result["exact_count"] == 1
    assert result["contains_count"] == 2
    assert result["exact"] == 0.5
    assert result["contains"] == 1.0
    assert [row["index"] for row in result["records"]] == [0, 1]


def main():
    test_schedule_horizon_covers_training()
    test_seed_is_reproducible()
    test_model_only_curve_snapshot()
    test_discovery_and_scoring_are_stable()
    print("test_checkpoint_curve: PASS")


if __name__ == "__main__":
    main()
