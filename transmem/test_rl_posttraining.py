#!/usr/bin/env python3
"""CPU behavior tests for OPD warm-start and GRPO objectives."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import torch

from transmem.test_inloop import _fake_data, _mk, _trainer_args, _write_cfg
from transmem.test_layered import tiny_llm, _FakeTok


def test_hotpot_reward_matches_official_examples() -> None:
    from transmem.rl import hotpot_answer_reward

    exact = hotpot_answer_reward("The Eiffel Tower", "Eiffel Tower", 3)
    assert exact.em == 1.0 and exact.f1 == 1.0 and exact.total == 1.25

    partial = hotpot_answer_reward("New York City", "York City Hall", 3)
    assert partial.em == 0.0
    assert abs(partial.f1 - 2.0 / 3.0) < 1e-7, partial

    categorical = hotpot_answer_reward("maybe yes", "yes", 2)
    assert categorical.f1 == 0.0, categorical

    verbose = hotpot_answer_reward("correct", "correct", 48, verbosity_start=32)
    assert abs(verbose.verbosity_penalty - 0.025) < 1e-7, verbose
    assert abs(verbose.total - 1.225) < 1e-7, verbose


def test_thinking_is_excluded_from_answer_only_reward() -> None:
    from transmem.rl import split_thinking_answer, task_answer_reward

    hybrid = split_thinking_answer(
        "<think>Paris is the relevant city.</think>Paris")
    assert hybrid.thinking == "Paris is the relevant city."
    assert hybrid.answer == "Paris" and hybrid.has_answer_marker

    instruct = split_thinking_answer(
        "I should retrieve the city first.\nAnswer: Paris")
    assert instruct.thinking == "I should retrieve the city first."
    assert instruct.answer == "Paris" and instruct.has_answer_marker
    reward = task_answer_reward(
        instruct.answer, "Paris", 1, scorer="longmemeval",
        valid_format=instruct.has_answer_marker)
    assert reward.total == 1.25, reward

    malformed = split_thinking_answer("I think the answer is Paris")
    invalid = task_answer_reward(
        malformed.answer, "Paris", 6, scorer="longmemeval",
        valid_format=malformed.has_answer_marker)
    assert invalid.invalid_penalty == 1.0, invalid

    no_reasoning = split_thinking_answer("Answer: Paris")
    structured = bool(
        no_reasoning.has_answer_marker and no_reasoning.thinking.strip())
    invalid = task_answer_reward(
        no_reasoning.answer, "Paris", 1, scorer="longmemeval",
        valid_format=structured)
    assert invalid.invalid_penalty == 1.0, invalid


def test_locomo_reward_uses_category_specific_answer_f1() -> None:
    from transmem.rl import task_answer_reward

    multi = task_answer_reward(
        "hiking, painting", "painting, hiking", 3,
        scorer="locomo", category=1)
    assert multi.f1 == 1.0, multi
    open_domain = task_answer_reward(
        "New York", "New York; NYC", 2,
        scorer="locomo", category=3)
    assert open_domain.f1 == 1.0 and open_domain.em == 1.0, open_domain


def test_group_advantages_are_centered_and_constant_groups_are_zero() -> None:
    from transmem.rl import group_relative_advantages

    advantages, active = group_relative_advantages(
        torch.tensor([1.0, 2.0, 3.0]))
    expected = torch.tensor([-math.sqrt(1.5), 0.0, math.sqrt(1.5)])
    assert active
    assert torch.allclose(advantages, expected, atol=1e-6), advantages

    constant, active = group_relative_advantages(torch.tensor([0.0, 0.0, 0.0]))
    assert not active
    assert torch.equal(constant, torch.zeros(3))


def test_grpo_clipping_and_reference_kl() -> None:
    from transmem.rl import grpo_clipped_loss, sampled_reference_kl

    old = torch.zeros(2, 1)
    new = torch.tensor([[math.log(1.3)], [math.log(0.7)]])
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.ones_like(new, dtype=torch.bool)
    loss = grpo_clipped_loss(new, old, advantages, mask, clip_eps=0.2)
    assert abs(float(loss) + 0.2) < 1e-6, loss

    equal = sampled_reference_kl(torch.tensor([[-2.0]]), torch.tensor([[-2.0]]))
    shifted = sampled_reference_kl(torch.tensor([[-1.5]]), torch.tensor([[-2.0]]))
    assert float(equal) == 0.0
    assert float(shifted) > 0.0


def test_rollout_returns_behavior_policy_log_probs() -> None:
    model, _, rollout = _mk(inject=(3, 5), seed=17, noise=0.2)
    torch.manual_seed(19)
    prompt = torch.randint(0, 61, (1, 37))
    answer_ids, old_log_probs = rollout.generate_from_ids(
        prompt, len_cl=25, max_new=7, sample=False,
        temperature=1.0, return_log_probs=True)
    assert len(answer_ids) == old_log_probs.shape[0] > 0

    length = len(answer_ids)
    full_ids = (torch.cat([prompt, torch.tensor([answer_ids[:-1]])], dim=1)
                if length > 1 else prompt)
    with torch.no_grad():
        hidden = rollout.teacher_forced_forward(
            full_ids, len_cl=25, len_cq=prompt.shape[1], M=length)
        logits = model.lm_head(hidden)
        expected = torch.log_softmax(logits.float(), dim=-1).gather(
            -1, torch.tensor(answer_ids).view(-1, 1)).squeeze(-1)
    assert torch.allclose(old_log_probs, expected, atol=1e-5), (
        old_log_probs, expected)


def test_model_only_warm_start_resets_training_state() -> None:
    from transmem.train_inloop import InLoopTrainer

    with tempfile.TemporaryDirectory() as td:
        _write_cfg(td)
        args = _trainer_args(td, policy="onpolicy", epochs=1)
        args.warm_start_checkpoint = None
        trainer = InLoopTrainer(
            args, model=tiny_llm(seed=7), tokenizer=_FakeTok())

        expected = {}
        with torch.no_grad():
            for name, parameter in trainer.mem.named_parameters():
                parameter.fill_(0.125)
                expected[name] = parameter.detach().clone()
        checkpoint = Path(td) / "parent.pt"
        torch.save({
            "model_state_dict": trainer.mem.state_dict(),
            "config": trainer.config.to_dict(),
            "global_step": 1250,
            "epoch": 2,
            "best_val": 0.2,
            "optimizer_state_dict": trainer.optimizer.state_dict(),
        }, checkpoint)

        with torch.no_grad():
            for parameter in trainer.mem.parameters():
                parameter.zero_()
        trainer.global_step = 99
        trainer.epoch = 9
        trainer.best_val = 0.3
        trainer.warm_start(str(checkpoint))

        for name, parameter in trainer.mem.named_parameters():
            assert torch.equal(parameter, expected[name]), name
        assert trainer.global_step == 0 and trainer.epoch == 0
        assert math.isinf(trainer.best_val) and trainer.best_step == -1
        assert trainer.warm_start_checkpoint == str(checkpoint.resolve())


def test_resume_rejects_a_checkpoint_from_another_policy() -> None:
    from transmem.train_inloop import InLoopTrainer

    with tempfile.TemporaryDirectory() as td:
        _write_cfg(td)
        args = _trainer_args(td, policy="onpolicy", epochs=1)
        trainer = InLoopTrainer(
            args, model=tiny_llm(seed=13), tokenizer=_FakeTok())
        checkpoint = Path(td) / "wrong_policy.pt"
        torch.save({
            "model_state_dict": trainer.mem.state_dict(),
            "config": trainer.config.to_dict(),
            "train_mode": "inloop_tf",
            "global_step": 8,
        }, checkpoint)
        try:
            trainer.load(str(checkpoint))
        except ValueError as error:
            assert "train_mode" in str(error)
        else:
            raise AssertionError("resume 应拒绝另一种 policy 的 checkpoint")


def test_resume_rejects_a_different_warm_start_parent() -> None:
    from transmem.train_inloop import InLoopTrainer

    with tempfile.TemporaryDirectory() as td:
        _write_cfg(td)
        args = _trainer_args(td, policy="onpolicy", epochs=1)
        args.warm_start_id = "s3://parents/new.pt#etag=new"
        trainer = InLoopTrainer(
            args, model=tiny_llm(seed=29), tokenizer=_FakeTok())
        checkpoint = Path(td) / "wrong_parent.pt"
        torch.save({
            "model_state_dict": trainer.mem.state_dict(),
            "config": trainer.config.to_dict(),
            "train_mode": "inloop_onpolicy",
            "warm_start_checkpoint": "s3://parents/old.pt#etag=old",
        }, checkpoint)
        try:
            trainer.load(str(checkpoint))
        except ValueError as error:
            assert "warm-start provenance" in str(error)
        else:
            raise AssertionError("resume 应拒绝另一父模型的 checkpoint")


def test_grpo_group_backward_updates_only_transmem() -> None:
    from transmem.extract_features import build_chat_prompt_ids
    from transmem.train_grpo import GRPOTrainer, RewardDataset

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "feat"
        records = _fake_data(root, n=2)
        _write_cfg(td)
        args = _trainer_args(td, policy="grpo", epochs=1)
        args.group_size = 3
        args.clip_eps = 0.2
        args.reference_kl_beta = 0.02
        args.reward_em_weight = 0.25
        args.reward_verbosity_weight = 0.05
        args.reward_verbosity_start = 32
        args.reward_verbosity_cap = 64
        args.sample_temp = 0.8
        args.grpo_epochs = 2
        args.reference_id = "s3://test/reference.pt"
        args.warm_start_checkpoint = None
        trainer = GRPOTrainer(
            args, model=tiny_llm(seed=23), tokenizer=_FakeTok())
        reference = Path(td) / "reference.pt"
        torch.save({
            "model_state_dict": trainer.mem.state_dict(),
            "config": trainer.config.to_dict(),
        }, reference)
        trainer.set_reference(str(reference))

        dataset = RewardDataset(str(root), "", "json", records=records)
        item = dataset[0]
        prompt = build_chat_prompt_ids(
            trainer.tok, item["context"], item["question"], trainer.device)
        context_tokens = trainer.tok(
            item["context"], return_tensors="pt",
            add_special_tokens=False).input_ids.shape[1]
        torch.manual_seed(101)
        target = trainer.rollout.generate_from_ids(
            prompt, context_tokens, max_new=args.max_answer_tokens,
            sample=True, temperature=args.sample_temp)
        item["ground_truth"] = trainer.tok.decode(
            target, skip_special_tokens=True).strip()

        torch.manual_seed(101)
        group = trainer.collect_group(item)
        metrics = trainer.backward_group(group, loss_scale=1.0)
        assert metrics["reward_std"] > 0.0, metrics
        assert math.isfinite(metrics["loss"]), metrics
        assert abs(metrics["importance_ratio"] - 1.0) < 1e-5, metrics
        assert any(parameter.grad is not None
                   and float(parameter.grad.abs().max()) > 0
                   for parameter in trainer.mem.parameters())
        assert all(parameter.grad is None for parameter in trainer.model.parameters())

        trainer.sync_and_step()
        replay_metrics = trainer.backward_group(group, loss_scale=1.0)
        assert abs(replay_metrics["importance_ratio"] - 1.0) > 1e-7, replay_metrics

        checkpoint = Path(td) / "grpo_resume.pt"
        trainer.save(td)
        checkpoint = Path(td) / "latest.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["reference_checkpoint_id"] == "s3://test/reference.pt"
        assert payload["grpo_epochs"] == 2


if __name__ == "__main__":
    test_hotpot_reward_matches_official_examples()
    test_thinking_is_excluded_from_answer_only_reward()
    test_locomo_reward_uses_category_specific_answer_f1()
    test_group_advantages_are_centered_and_constant_groups_are_zero()
    test_grpo_clipping_and_reference_kl()
    test_rollout_returns_behavior_policy_log_probs()
    test_model_only_warm_start_resets_training_state()
    test_resume_rejects_a_checkpoint_from_another_policy()
    test_resume_rejects_a_different_warm_start_parent()
    test_grpo_group_backward_updates_only_transmem()
    print("RL/OPD post-training tests passed")
