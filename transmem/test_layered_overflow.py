#!/usr/bin/env python3
"""Regression tests for layered TransMem overflow-memory inference."""

from __future__ import annotations

import types

import torch

from transmem.extract_features import hm_positions
from transmem.layered import LayeredRollout, TransMemLayered
from transmem.test_layered import DIM, VOCAB, tiny_layered_cfg, tiny_llm
from transmem.transmem import TransMemOutput


def test_capture_memory_from_ids_returns_each_injection_layers_slots():
    inject = (2, 4)
    cfg = tiny_layered_cfg(inject=inject)
    model = tiny_llm(seed=31)
    layered = TransMemLayered(cfg).eval()
    rollout = LayeredRollout(
        model, tokenizer=None, device="cpu", layered=layered,
        dtype=torch.float32,
    )
    torch.manual_seed(37)
    context_ids = torch.randint(0, VOCAB - 1, (1, 35))
    len_cl = 29
    raw: dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx in inject:
        def capture(_module, _args, output, layer=layer_idx):
            hidden = output[0] if isinstance(output, tuple) else output
            raw[layer] = hidden.detach().clone()
        handles.append(
            model.model.layers[layer_idx].register_forward_hook(capture))
    try:
        memory = rollout.capture_memory_from_ids(context_ids, len_cl=len_cl)
    finally:
        for handle in handles:
            handle.remove()

    indices = torch.tensor(hm_positions(len_cl, cfg.n_mem, "frac"))
    assert set(memory) == set(inject)
    for layer_idx in inject:
        assert torch.allclose(memory[layer_idx], raw[layer_idx][0, indices])
    assert int(indices[-1]) == len_cl - 1


def test_capture_memory_from_context_prefills_raw_context_only():
    cfg = tiny_layered_cfg(inject=(4,))
    model = tiny_llm(seed=39)
    layered = TransMemLayered(cfg).eval()

    class FakeTokenizer:
        def __call__(self, text, return_tensors, add_special_tokens):
            assert text == "raw overflow"
            assert return_tensors == "pt"
            assert add_special_tokens is False
            return types.SimpleNamespace(
                input_ids=torch.tensor([[11, 12, 13, 14]], dtype=torch.long))

    rollout = LayeredRollout(
        model, tokenizer=FakeTokenizer(), device="cpu", layered=layered,
        dtype=torch.float32,
    )
    captured = {}

    def fake_capture(self, context_ids, len_cl):
        captured["ids"] = context_ids.clone()
        captured["len_cl"] = len_cl
        return {4: torch.zeros(cfg.n_mem, DIM)}

    rollout.capture_memory_from_ids = types.MethodType(fake_capture, rollout)
    result = rollout.capture_memory_from_context("raw overflow")

    assert torch.equal(
        captured["ids"], torch.tensor([[11, 12, 13, 14]]))
    assert captured["len_cl"] == 4
    assert result[4].shape == (cfg.n_mem, DIM)


def test_generation_prepends_overflow_memory_before_limit_memory():
    layer_idx = 4
    cfg = tiny_layered_cfg(inject=(layer_idx,))
    model = tiny_llm(seed=41)
    layered = TransMemLayered(cfg).eval()
    rollout = LayeredRollout(
        model, tokenizer=None, device="cpu", layered=layered,
        dtype=torch.float32,
    )
    block = layered.block(layer_idx)
    captured: dict[str, torch.Tensor] = {}

    def fake_forward(self, X, **_kwargs):
        captured["transmem_input"] = X.detach().clone()
        hidden = X[:, -1, :]
        return TransMemOutput(
            ms=torch.zeros_like(hidden),
            gate=torch.ones(*hidden.shape[:-1], 1),
        )

    block.forward = types.MethodType(fake_forward, block)
    torch.manual_seed(43)
    cq = torch.randint(0, VOCAB - 1, (1, 37))
    len_cl = 29
    raw: dict[str, torch.Tensor] = {}

    def capture_layer(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        raw["hidden"] = hidden.detach().clone()

    handle = model.model.layers[layer_idx].register_forward_hook(capture_layer)
    overflow = torch.full((3, DIM), 7.0)
    try:
        rollout.generate_from_ids(
            cq,
            len_cl=len_cl,
            max_new=1,
            overflow_memory={layer_idx: overflow},
        )
    finally:
        handle.remove()

    indices = torch.tensor(hm_positions(len_cl, cfg.n_mem, cfg.hm_mode))
    expected_limit = raw["hidden"][0, indices]
    actual = captured["transmem_input"][0]
    assert torch.equal(actual[:3], overflow)
    assert torch.allclose(actual[3:3 + cfg.n_mem], expected_limit)
    assert torch.allclose(actual[-1], raw["hidden"][0, -1])
