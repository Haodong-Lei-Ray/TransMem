"""Behavioral tests for token-scalar TransMem gates."""

from __future__ import annotations

import torch

from transmem import TransMem, TransMemConfig, TransMemOutput


def _tiny_config(**overrides) -> TransMemConfig:
    values = {
        "dim": 64,
        "depth": 1,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 16,
        "intermediate_size": 128,
        "max_position_embeddings": 512,
        "attn_impl": "eager",
        "zero_init_out": False,
    }
    values.update(overrides)
    return TransMemConfig(**values)


def test_centered_gate_starts_as_legacy_correction() -> None:
    """A newly added dynamic gate must reproduce the old HQ + MS rule at step 0."""
    torch.manual_seed(7)
    memory = TransMem(_tiny_config(gate_mode="centered_sigmoid"))
    batch, queries = 2, 3
    inputs = torch.randn(batch, memory.config.n_mem + queries, memory.dim)
    hq = inputs[:, memory.config.n_mem :, :]

    proposal = memory(inputs, return_all_queries=True)

    assert isinstance(proposal, TransMemOutput)
    assert proposal.ms.shape == (batch, queries, memory.dim)
    assert proposal.gate.shape == (batch, queries, 1)
    assert torch.equal(proposal.gate, torch.ones_like(proposal.gate))
    assert torch.allclose(memory.correct(hq, proposal), hq + proposal.ms)


def test_constant_mode_keeps_legacy_parameters_and_scale() -> None:
    """A config without gate fields must remain a strict-loadable legacy model."""
    legacy_values = _tiny_config(a_init=0.25).to_dict()
    for name in (
        "gate_mode",
        "gate_granularity",
        "gate_max",
        "gate_temperature",
        "gate_init",
    ):
        legacy_values.pop(name, None)
    original = TransMem(TransMemConfig(**legacy_values))
    restored = TransMem(TransMemConfig(**legacy_values))
    restored.load_state_dict(original.state_dict(), strict=True)
    assert "gate_mode" not in restored.config.to_dict()
    assert not any(name.startswith("gate_proj.") for name, _ in restored.named_parameters())

    inputs = torch.randn(2, restored.config.n_mem + 2, restored.dim)
    hq = inputs[:, restored.config.n_mem :, :]
    proposal = restored(inputs, return_all_queries=True)
    assert torch.equal(proposal.gate, torch.ones_like(proposal.gate))
    assert torch.allclose(
        restored.correct(hq, proposal), hq + 0.25 * proposal.ms)


def test_gate_and_ms_cache_match_parallel_forward() -> None:
    """The dynamic proposal must preserve the existing causal cache equivalence."""
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(11)
    memory = TransMem(_tiny_config(gate_mode="centered_sigmoid")).eval()
    queries = 4
    inputs = torch.randn(1, memory.config.n_mem + queries, memory.dim)
    with torch.no_grad():
        full = memory(inputs, return_all_queries=True)
        cache = DynamicCache()
        steps = [memory(
            inputs[:, : memory.config.n_mem + 1],
            past_key_values=cache,
            use_cache=True,
        )]
        for index in range(1, queries):
            steps.append(memory(
                inputs[:, memory.config.n_mem + index : memory.config.n_mem + index + 1],
                past_key_values=cache,
                use_cache=True,
            ))
    incremental_ms = torch.stack([step.ms for step in steps], dim=1)
    incremental_gate = torch.stack([step.gate for step in steps], dim=1)
    assert torch.allclose(incremental_ms, full.ms, atol=1e-4, rtol=1e-4)
    assert torch.allclose(incremental_gate, full.gate, atol=1e-4, rtol=1e-4)


def test_task_loss_reaches_gate_projection() -> None:
    """With a non-zero MS, task loss must give the dynamic gate a useful gradient."""
    torch.manual_seed(13)
    memory = TransMem(_tiny_config(gate_mode="centered_sigmoid"))
    inputs = torch.randn(2, memory.config.n_mem + 3, memory.dim)
    hq = inputs[:, memory.config.n_mem :, :]
    proposal = memory(inputs, return_all_queries=True)
    memory.correct(hq, proposal).square().mean().backward()
    assert memory.gate_proj is not None
    assert memory.gate_proj.weight.grad is not None
    assert float(memory.gate_proj.weight.grad.abs().sum()) > 0.0


def test_sigmoid_gate_supports_suppress_only_ablation() -> None:
    """The optional sigmoid mode starts at g_init and remains strictly in (0, 1)."""
    torch.manual_seed(15)
    memory = TransMem(_tiny_config(
        gate_mode="sigmoid", gate_max=1.0, gate_init=0.9))
    inputs = torch.randn(2, memory.config.n_mem + 3, memory.dim)

    initial = memory(inputs, return_all_queries=True).gate
    assert torch.allclose(initial, torch.full_like(initial, 0.9), atol=1e-6)
    assert memory.gate_proj is not None
    with torch.no_grad():
        memory.gate_proj.weight.normal_(0.0, 0.2)
    changed = memory(inputs, return_all_queries=True).gate
    assert bool(((changed > 0.0) & (changed < 1.0)).all())


def test_explicit_legacy_migration_only_adds_gate_parameters() -> None:
    """Legacy migration must reject every missing/unexpected key except gate_proj."""
    from transmem.checkpoints import load_legacy_gate_state

    torch.manual_seed(17)
    legacy = TransMem(_tiny_config(gate_mode="constant"))
    dynamic = TransMem(_tiny_config(gate_mode="centered_sigmoid"))
    checkpoint = {
        "config": legacy.config.to_dict(),
        "model_state_dict": legacy.state_dict(),
    }

    result = load_legacy_gate_state(dynamic, checkpoint)

    assert set(result.missing_keys) == {"gate_proj.weight", "gate_proj.bias"}
    assert result.unexpected_keys == []
    inputs = torch.randn(2, legacy.config.n_mem + 2, legacy.dim)
    hq = inputs[:, legacy.config.n_mem :, :]
    with torch.no_grad():
        old = legacy.correct(hq, legacy(inputs, return_all_queries=True))
        new = dynamic.correct(hq, dynamic(inputs, return_all_queries=True))
    assert torch.equal(old, new)


def test_legacy_migration_rejects_a_missing_qwen_mlp_weight() -> None:
    """Qwen's own MLP gate_proj must never enter the scalar-gate whitelist."""
    import pytest

    from transmem.checkpoints import load_legacy_gate_state

    legacy = TransMem(_tiny_config(gate_mode="constant"))
    dynamic = TransMem(_tiny_config(gate_mode="centered_sigmoid"))
    state = dict(legacy.state_dict())
    qwen_gate_name = next(
        name for name in state if name.endswith("mlp.gate_proj.weight"))
    state.pop(qwen_gate_name)

    with pytest.raises(RuntimeError, match="白名单"):
        load_legacy_gate_state(dynamic, {
            "config": legacy.config.to_dict(),
            "model_state_dict": state,
        })


def test_layered_forward_exposes_one_independent_gate_per_layer() -> None:
    """Layered callers receive aligned MS/gate tensors and correct through each block."""
    from transmem.layered import LayeredConfig, LayeredOutput, TransMemLayered

    config = LayeredConfig(
        dim=64,
        block_depth=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=512,
        attn_impl="eager",
        inject_layers=[2, 4],
        gate_mode="centered_sigmoid",
    )
    layered = TransMemLayered(config)
    with torch.no_grad():
        for block in layered.blocks.values():
            block.out_proj.weight.normal_(0.0, 0.1)
    hm = torch.randn(2, 2, config.n_mem, config.dim)
    hq = torch.randn(2, 2, 3, config.dim)

    output = layered(hm, hq)

    assert isinstance(output, LayeredOutput)
    assert output.ms.shape == (2, 2, 3, config.dim)
    assert output.gate.shape == (2, 2, 3, 1)
    assert torch.equal(output.gate, torch.ones_like(output.gate))
    for index, layer in enumerate(config.inject_layers):
        corrected = layered.block(layer).correct(hq[:, index], output.layer(index))
        assert torch.allclose(corrected, hq[:, index] + output.ms[:, index])


def test_initialization_scheme_is_explicit_and_fail_closed() -> None:
    """A/B initialization must never be inferred from whether a path happens to exist."""
    import pytest

    from transmem.gate_training import validate_gate_training_options

    validate_gate_training_options(
        init_scheme="scratch_joint",
        init_checkpoint=None,
        gate_mode="centered_sigmoid",
        gate_calibration_steps=0,
        joint_finetune_steps=100,
    )
    validate_gate_training_options(
        init_scheme="legacy_gate",
        init_checkpoint="parent.pt",
        gate_mode="centered_sigmoid",
        gate_calibration_steps=20,
        joint_finetune_steps=80,
    )
    with pytest.raises(ValueError, match="init_checkpoint"):
        validate_gate_training_options(
            init_scheme="legacy_gate",
            init_checkpoint=None,
            gate_mode="centered_sigmoid",
            gate_calibration_steps=20,
            joint_finetune_steps=80,
        )
    with pytest.raises(ValueError, match="禁止"):
        validate_gate_training_options(
            init_scheme="scratch_joint",
            init_checkpoint="accidental.pt",
            gate_mode="centered_sigmoid",
            gate_calibration_steps=0,
            joint_finetune_steps=100,
        )
    with pytest.raises(ValueError, match="gate_calibration_steps"):
        validate_gate_training_options(
            init_scheme="scratch_joint",
            init_checkpoint=None,
            gate_mode="centered_sigmoid",
            gate_calibration_steps=1,
            joint_finetune_steps=99,
        )
    with pytest.raises(ValueError, match="warm_start=false"):
        validate_gate_training_options(
            init_scheme="scratch_joint",
            init_checkpoint=None,
            gate_mode="centered_sigmoid",
            gate_calibration_steps=0,
            joint_finetune_steps=100,
            warm_start=True,
        )


def test_dynamic_resume_rejects_config_or_provenance_drift() -> None:
    """State-shape compatibility must not hide changed gate semantics on resume."""
    import pytest

    from transmem.gate_training import validate_dynamic_resume_checkpoint

    config = _tiny_config(gate_mode="centered_sigmoid").to_dict()
    checkpoint = {
        "config": config,
        "init_scheme": "legacy_gate",
        "parent_checkpoint": "/tmp/parent.pt",
        "gate_calibration_steps": 20,
        "joint_finetune_steps": 80,
    }
    validate_dynamic_resume_checkpoint(
        checkpoint,
        config=config,
        init_scheme="legacy_gate",
        parent_checkpoint="/tmp/parent.pt",
        gate_calibration_steps=20,
        joint_finetune_steps=80,
    )

    changed = dict(config, gate_temperature=0.5)
    with pytest.raises(ValueError, match="config"):
        validate_dynamic_resume_checkpoint(
            checkpoint,
            config=changed,
            init_scheme="legacy_gate",
            parent_checkpoint="/tmp/parent.pt",
            gate_calibration_steps=20,
            joint_finetune_steps=80,
        )
    with pytest.raises(ValueError, match="init_scheme"):
        validate_dynamic_resume_checkpoint(
            checkpoint,
            config=config,
            init_scheme="scratch_joint",
            parent_checkpoint=None,
            gate_calibration_steps=0,
            joint_finetune_steps=100,
        )


def test_gate_only_phase_updates_no_legacy_parameter() -> None:
    """A1 calibration may compute through TransMem but must update only gate heads."""
    from transmem.gate_training import (
        build_gate_optimizer,
        clear_base_grads_for_gate_only,
        set_gate_optimizer_lrs,
    )

    torch.manual_seed(19)
    memory = TransMem(_tiny_config(gate_mode="centered_sigmoid"))
    optimizer = build_gate_optimizer(
        memory,
        base_lr=1e-2,
        gate_lr=2e-2,
        weight_decay=0.1,
    )
    before = {name: parameter.detach().clone()
              for name, parameter in memory.named_parameters()}
    inputs = torch.randn(2, memory.config.n_mem + 2, memory.dim)
    hq = inputs[:, memory.config.n_mem :, :]
    proposal = memory(inputs, return_all_queries=True)
    memory.correct(hq, proposal).square().mean().backward()

    set_gate_optimizer_lrs(optimizer, lr_factor=1.0, gate_only=True)
    clear_base_grads_for_gate_only(optimizer, gate_only=True)
    optimizer.step()

    changed = {
        name for name, parameter in memory.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    }
    assert changed
    assert changed <= {"gate_proj.weight", "gate_proj.bias"}
    base_parameters = next(
        group["params"] for group in optimizer.param_groups
        if group["group_name"] == "base")
    assert not any(parameter in optimizer.state for parameter in base_parameters)


def test_gate_prior_is_masked_and_linearly_annealed() -> None:
    """Padding never contributes to the prior, which reaches exactly zero on schedule."""
    from transmem.gate_training import gate_prior_coefficient, gate_prior_loss

    gate = torch.tensor([[[0.0], [2.0], [99.0]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.equal(gate_prior_loss(gate, mask), torch.tensor(1.0))
    assert gate_prior_coefficient(step=0, weight=0.2, anneal_steps=10) == 0.2
    assert gate_prior_coefficient(step=5, weight=0.2, anneal_steps=10) == 0.1
    assert gate_prior_coefficient(step=10, weight=0.2, anneal_steps=10) == 0.0
    assert gate_prior_coefficient(step=100, weight=0.2, anneal_steps=10) == 0.0


def test_gate_kl_correlation_accumulator_is_streaming_and_nan_safe() -> None:
    """Offline diagnostics report a true position-weighted Pearson value."""
    from scripts.eval.eval_offpolicy_diagnostics import _PearsonAccumulator

    accumulator = _PearsonAccumulator()
    accumulator.update(
        torch.tensor([0.0, 1.0, float("nan")]),
        torch.tensor([0.0, 2.0, 99.0]),
    )
    accumulator.update(torch.tensor([2.0]), torch.tensor([4.0]))

    assert accumulator.summary() == {"pearson": 1.0, "positions": 3}


def test_migration_writes_a_strict_dynamic_checkpoint(tmp_path) -> None:
    """The one-shot migration output is self-describing and strict-loadable."""
    from transmem.checkpoints import migrate_legacy_checkpoint

    legacy = TransMem(_tiny_config(gate_mode="constant"))
    source = tmp_path / "legacy.pt"
    destination = tmp_path / "dynamic.pt"
    torch.save({
        "config": legacy.config.to_dict(),
        "model_state_dict": legacy.state_dict(),
        "global_step": 123,
    }, source)

    migrate_legacy_checkpoint(
        source,
        destination,
        gate_config={
            "gate_mode": "centered_sigmoid",
            "gate_granularity": "token_scalar",
            "gate_max": 2.0,
            "gate_temperature": 1.0,
            "gate_init": 1.0,
        },
    )

    migrated = torch.load(destination, map_location="cpu", weights_only=False)
    config = TransMemConfig(**migrated["config"])
    restored = TransMem(config)
    restored.load_state_dict(migrated["model_state_dict"], strict=True)
    assert migrated["init_scheme"] == "legacy_gate"
    assert migrated["parent_checkpoint"] == str(source.resolve())
    assert migrated["global_step"] == 0


def test_layered_migration_whitelists_only_scalar_gate_heads(tmp_path) -> None:
    """Qwen MLP gate_proj weights remain required while layered scalar heads are added."""
    from transmem.checkpoints import migrate_legacy_checkpoint
    from transmem.layered import LayeredConfig, TransMemLayered

    legacy_config = LayeredConfig(
        dim=64,
        block_depth=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=512,
        attn_impl="eager",
        inject_layers=[2, 4],
    )
    legacy = TransMemLayered(legacy_config)
    source = tmp_path / "legacy_layered.pt"
    destination = tmp_path / "dynamic_layered.pt"
    torch.save({
        "config": legacy_config.to_dict(),
        "model_state_dict": legacy.state_dict(),
    }, source)
    migrate_legacy_checkpoint(source, destination, gate_config={
        "gate_mode": "centered_sigmoid",
        "gate_granularity": "token_scalar",
        "gate_max": 2.0,
        "gate_temperature": 1.0,
        "gate_init": 1.0,
    })
    checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
    restored = TransMemLayered(LayeredConfig.from_dict(checkpoint["config"]))
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    scalar_gate_names = {
        name for name, _ in restored.named_parameters()
        if name in {
            "blocks.2.gate_proj.weight", "blocks.2.gate_proj.bias",
            "blocks.4.gate_proj.weight", "blocks.4.gate_proj.bias",
        }
    }
    assert len(scalar_gate_names) == 4


def test_dynamic_checkpoint_roundtrip_preserves_gate_and_correction(tmp_path) -> None:
    torch.manual_seed(23)
    memory = TransMem(_tiny_config(gate_mode="centered_sigmoid"))
    assert memory.gate_proj is not None
    with torch.no_grad():
        memory.gate_proj.weight.normal_(0.0, 0.05)
        memory.gate_proj.bias.fill_(0.2)
    path = tmp_path / "dynamic.pt"
    torch.save({
        "config": memory.config.to_dict(),
        "model_state_dict": memory.state_dict(),
    }, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restored = TransMem(TransMemConfig(**checkpoint["config"]))
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    inputs = torch.randn(2, memory.config.n_mem + 3, memory.dim)
    hq = inputs[:, memory.config.n_mem :, :]
    with torch.no_grad():
        expected = memory(inputs, return_all_queries=True)
        actual = restored(inputs, return_all_queries=True)
    assert torch.equal(actual.gate, expected.gate)
    assert torch.equal(restored.correct(hq, actual), memory.correct(hq, expected))


def test_layered_dynamic_gate_matches_teacher_forcing_and_backpropagates() -> None:
    """D=1 and D>1 use identical per-token gates in rollout and teacher forcing."""
    from transmem.layered import TransMemLayered, LayeredRollout
    from transmem.test_layered import VOCAB, tiny_layered_cfg, tiny_llm

    for inject_layers in ((5,), (3, 5)):
        model = tiny_llm(seed=len(inject_layers)).eval().requires_grad_(False)
        config = tiny_layered_cfg(inject_layers)
        config.gate_mode = "centered_sigmoid"
        layered = TransMemLayered(config).train()
        with torch.no_grad():
            for block in layered.blocks.values():
                block.out_proj.weight.normal_(0.0, 0.2)
                assert block.gate_proj is not None
                block.gate_proj.weight.normal_(0.0, 0.03)
        rollout = LayeredRollout(
            model, tokenizer=None, device="cpu", layered=layered,
            dtype=torch.float32)
        torch.manual_seed(31 + len(inject_layers))
        prompt = torch.randint(0, VOCAB - 1, (1, 37))
        answer = rollout.generate_from_ids(
            prompt, len_cl=25, max_new=6, collect_gate_diagnostics=True)
        assert answer
        full = (torch.cat([prompt, torch.tensor([answer[:-1]])], dim=1)
                if len(answer) > 1 else prompt)
        hq, proposals = rollout.teacher_forced_forward(
            full, len_cl=25, len_cq=prompt.shape[1], M=len(answer),
            return_proposals=True)
        predicted = model.lm_head(hq).argmax(dim=-1).tolist()
        assert predicted == answer
        for index, layer in enumerate(inject_layers):
            trace = torch.tensor(
                rollout.last_gate_trace["layers"][str(layer)]["gate"])
            assert torch.allclose(
                trace, proposals.gate[0, index, :, 0], atol=1e-4, rtol=1e-4)

        model.lm_head(hq).float().square().mean().backward()
        for layer in inject_layers:
            gradient = layered.block(layer).gate_proj.weight.grad
            assert gradient is not None and float(gradient.abs().sum()) > 0.0
