#!/usr/bin/env python3
"""Regression tests for the optional pre-block TransMem feature source."""

from __future__ import annotations

import types

import torch

from transmem.extract_features import hm_positions
from transmem.layered import LayeredConfig, LayeredRollout, TransMemLayered
from transmem.test_layered import VOCAB, tiny_layered_cfg, tiny_llm
from transmem.transmem import TransMemOutput


def _capture_single_layer(*, transmem_before: bool, teacher_forced: bool):
    """Capture the raw LLM layer input/output and the two TransMem seams."""
    layer_idx = 4
    cfg = tiny_layered_cfg(inject=(layer_idx,))
    # Deliberately assign after construction so this test fails behaviorally on
    # code that has not implemented the new config field yet.
    cfg.transmem_before = transmem_before
    model = tiny_llm(seed=17)
    layered = TransMemLayered(cfg).eval()
    rollout = LayeredRollout(
        model, tokenizer=None, device="cpu", layered=layered,
        dtype=torch.float32,
    )
    block = layered.block(layer_idx)
    captured: dict[str, torch.Tensor] = {}

    def fake_forward(self, X, return_all_queries=False, **_kwargs):
        captured["transmem_input"] = X.detach().clone()
        query_hidden = X[:, cfg.n_mem:, :] if return_all_queries else X[:, -1, :]
        ms = torch.full_like(query_hidden, 0.125)
        gate = torch.ones(*query_hidden.shape[:-1], 1, dtype=X.dtype, device=X.device)
        return TransMemOutput(ms=ms, gate=gate)

    def fake_correct(self, base, proposal):
        captured["correction_base"] = base.detach().clone()
        return base + proposal.delta.to(base.dtype)

    block.forward = types.MethodType(fake_forward, block)
    block.correct = types.MethodType(fake_correct, block)

    raw_layer = model.model.layers[layer_idx]

    def capture_input(_module, args):
        captured["layer_input"] = args[0].detach().clone()

    def capture_output(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["layer_output"] = hidden.detach().clone()

    pre_handle = raw_layer.register_forward_pre_hook(capture_input)
    out_handle = raw_layer.register_forward_hook(capture_output)
    torch.manual_seed(29)
    len_cl = 24
    len_cq = 37
    try:
        cq = torch.randint(0, VOCAB - 1, (1, len_cq))
        if teacher_forced:
            M = 4
            answer_prefix = torch.randint(0, VOCAB - 1, (1, M - 1))
            full = torch.cat([cq, answer_prefix], dim=1)
            rollout.teacher_forced_forward(full, len_cl=len_cl, len_cq=len_cq, M=M)
            qpos = torch.arange(len_cq - 1, len_cq + M - 1)
        else:
            rollout.generate_from_ids(cq, len_cl=len_cl, max_new=1)
            qpos = torch.tensor([len_cq - 1])
    finally:
        pre_handle.remove()
        out_handle.remove()

    hm_idx = torch.tensor(hm_positions(len_cl, cfg.n_mem, cfg.hm_mode))
    source = captured["layer_input"] if transmem_before else captured["layer_output"]
    expected_input = torch.cat(
        [source[:, hm_idx, :], source[:, qpos, :]], dim=1,
    )
    captured["post_block_input"] = torch.cat(
        [captured["layer_output"][:, hm_idx, :],
         captured["layer_output"][:, qpos, :]],
        dim=1,
    )
    expected_base = captured["layer_output"][:, qpos, :]
    return captured, expected_input, expected_base


def test_teacher_forced_selects_source_but_keeps_post_block_target():
    for before in (False, True):
        captured, expected_input, expected_base = _capture_single_layer(
            transmem_before=before, teacher_forced=True,
        )
        assert torch.allclose(captured["transmem_input"], expected_input)
        assert torch.allclose(captured["correction_base"], expected_base)
        if before:
            assert not torch.allclose(
                captured["transmem_input"], captured["post_block_input"])


def test_generation_prefill_selects_source_but_keeps_post_block_target():
    for before in (False, True):
        captured, expected_input, expected_base = _capture_single_layer(
            transmem_before=before, teacher_forced=False,
        )
        assert torch.allclose(captured["transmem_input"], expected_input)
        assert torch.allclose(captured["correction_base"], expected_base)


def test_config_round_trip_and_legacy_default():
    cfg = tiny_layered_cfg(inject=(4,))
    cfg.transmem_before = True
    restored = LayeredConfig.from_dict(cfg.to_dict())
    assert restored.transmem_before is True

    legacy = cfg.to_dict()
    legacy.pop("transmem_before")
    assert LayeredConfig.from_dict(legacy).transmem_before is False
    assert "transmem_before" not in tiny_layered_cfg(inject=(4,)).to_dict()


def test_before_teacher_forcing_matches_incremental_decode():
    """The parallel training path must match every cached decode step."""
    cfg = tiny_layered_cfg(inject=(2, 4, 5))
    cfg.transmem_before = True
    model = tiny_llm(seed=41)
    layered = TransMemLayered(cfg).eval()
    torch.manual_seed(43)
    with torch.no_grad():
        for block in layered.blocks.values():
            block.out_proj.weight.normal_(0, 0.4)
    rollout = LayeredRollout(
        model, tokenizer=None, device="cpu", layered=layered,
        dtype=torch.float32,
    )
    cq = torch.randint(0, VOCAB - 1, (1, 43))
    trajectory = rollout.generate_from_ids(cq, len_cl=31, max_new=8)
    M = len(trajectory)
    full = (torch.cat([cq, torch.tensor([trajectory[:-1]])], dim=1)
            if M > 1 else cq)
    with torch.no_grad():
        hq = rollout.teacher_forced_forward(
            full, len_cl=31, len_cq=cq.shape[1], M=M,
        )
        teacher_forced_tokens = model.lm_head(hq).argmax(-1).tolist()
    assert teacher_forced_tokens == trajectory


def test_before_keeps_deep_credit_assignment():
    """The pre-block source remains differentiable into every TransMem block."""
    cfg = tiny_layered_cfg(inject=(2, 4, 5))
    cfg.transmem_before = True
    model = tiny_llm(seed=47)
    model.requires_grad_(False)
    layered = TransMemLayered(cfg).train()
    torch.manual_seed(53)
    with torch.no_grad():
        for block in layered.blocks.values():
            block.out_proj.weight.normal_(0, 0.1)
    rollout = LayeredRollout(
        model, tokenizer=None, device="cpu", layered=layered,
        dtype=torch.float32,
    )
    cq = torch.randint(0, VOCAB - 1, (1, 39))
    answer_prefix = torch.randint(0, VOCAB - 1, (1, 4))
    full = torch.cat([cq, answer_prefix], dim=1)
    hq = rollout.teacher_forced_forward(
        full, len_cl=27, len_cq=cq.shape[1], M=5,
    )
    model.lm_head(hq).float().logsumexp(-1).mean().backward()

    for layer, block in layered.blocks.items():
        grad = block.out_proj.weight.grad
        assert grad is not None, f"TransMem layer {layer} has no gradient"
        assert torch.isfinite(grad).all(), f"TransMem layer {layer} gradient is non-finite"
        assert float(grad.abs().max()) > 0, f"TransMem layer {layer} gradient is zero"
    assert all(parameter.grad is None for parameter in model.parameters())
