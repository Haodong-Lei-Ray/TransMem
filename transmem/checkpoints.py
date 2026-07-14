"""Checkpoint compatibility helpers for explicit dynamic-gate migration."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from .gate_training import named_gate_parameters


_GATE_CONFIG_FIELDS = {
    "gate_mode",
    "gate_granularity",
    "gate_max",
    "gate_temperature",
    "gate_init",
}
_NON_STRUCTURAL_FIELDS = _GATE_CONFIG_FIELDS | {"warm_start"}


def _atomic_torch_save(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _normalized_config(module: nn.Module, values: Mapping[str, object]) -> dict:
    """Fill legacy defaults with the config class used by the target module."""
    config_type = type(module.config)  # type: ignore[attr-defined]
    if hasattr(config_type, "from_dict"):
        return config_type.from_dict(dict(values)).to_dict()
    return config_type(**dict(values)).to_dict()


def _assert_legacy_config_compatible(
    module: nn.Module,
    checkpoint_config: Mapping[str, object],
) -> None:
    parent = _normalized_config(module, checkpoint_config)
    target = module.config.to_dict()  # type: ignore[attr-defined]
    parent_mode = str(parent.get("gate_mode", "constant"))
    target_mode = str(target.get("gate_mode", "constant"))
    if parent_mode != "constant":
        raise ValueError(
            "legacy_gate 只能迁移 fixed-gate checkpoint; "
            f"父 checkpoint gate_mode={parent_mode!r}")
    if (target_mode != "centered_sigmoid"
            or target.get("gate_init") != 1.0
            or target.get("gate_max") != 2.0):
        raise ValueError(
            "legacy_gate 目标必须是 gate_mode='centered_sigmoid', gate_init=1.0, "
            "gate_max=2.0，才能用零初始化 gate_proj 严格复现旧 checkpoint")
    if float(parent.get("a_init", 1.0)) != 1.0 or bool(
            parent.get("learnable_a", False)):
        raise ValueError(
            "legacy_gate 只支持历史固定 a=1 checkpoint；可学习或非 1 的 a 无法在 "
            "g=1 初始化下严格迁移")

    mismatches = []
    for name in sorted(set(parent) | set(target)):
        if name in _NON_STRUCTURAL_FIELDS or name in {"a_init", "learnable_a"}:
            continue
        if parent.get(name) != target.get(name):
            mismatches.append(
                f"{name}: parent={parent.get(name)!r}, target={target.get(name)!r}")
    if mismatches:
        raise ValueError(
            "legacy checkpoint 与目标结构不兼容:\n  - " + "\n  - ".join(mismatches))


def load_legacy_gate_state(
    module: nn.Module,
    checkpoint: Mapping[str, object],
):
    """Load a fixed-gate checkpoint and whitelist only newly added gate heads.

    This is the sole ``strict=False`` seam used by trainers and migration tools.
    New dynamic checkpoints must continue to load with ``strict=True``.
    """
    config = checkpoint.get("config")
    state = checkpoint.get("model_state_dict")
    if not isinstance(config, Mapping):
        raise ValueError("legacy checkpoint 缺少 dict config")
    if not isinstance(state, Mapping):
        raise ValueError("legacy checkpoint 缺少 model_state_dict")
    _assert_legacy_config_compatible(module, config)

    expected_missing = {name for name, _ in named_gate_parameters(module)}
    if not expected_missing:
        raise ValueError("legacy_gate 目标没有 gate_proj 参数")
    result = module.load_state_dict(state, strict=False)
    missing = set(result.missing_keys)
    if missing != expected_missing or result.unexpected_keys:
        raise RuntimeError(
            "legacy checkpoint 迁移键不符合白名单: "
            f"missing={sorted(missing)}, expected={sorted(expected_missing)}, "
            f"unexpected={result.unexpected_keys}")

    for name, parameter in module.named_parameters():
        if name in expected_missing and torch.count_nonzero(parameter).item() != 0:
            raise RuntimeError(f"迁移后的 {name} 不是零初始化")
    return result


def materialize_migrated_gate_checkpoint(
    module: nn.Module,
    dst_ckpt: str | Path,
    *,
    parent_checkpoint: str,
    gate_calibration_steps: int,
    joint_finetune_steps: int | None,
    seed: int | None,
) -> Path:
    """Persist an in-memory migration, then prove it strict-loads before training."""
    destination = Path(dst_ckpt).expanduser().resolve()
    config = module.config.to_dict()  # type: ignore[attr-defined]
    metadata = {
        "config": config,
        "train_mode": "migrated_legacy_gate",
        "init_scheme": "legacy_gate",
        "parent_checkpoint": str(Path(parent_checkpoint).expanduser().resolve()),
        "gate_calibration_steps": int(gate_calibration_steps),
        "joint_finetune_steps": joint_finetune_steps,
        "seed": seed,
        "global_step": 0,
        "epoch": 0,
        "kind": "migrated_init",
    }
    if destination.exists():
        checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
        mismatches = [
            f"{name}: checkpoint={checkpoint.get(name)!r}, current={value!r}"
            for name, value in metadata.items() if checkpoint.get(name) != value
        ]
        if mismatches:
            raise ValueError(
                "已有 migrated_init.pt 与当前迁移请求不一致:\n  - "
                + "\n  - ".join(mismatches))
    else:
        checkpoint = dict(metadata, model_state_dict=module.state_dict())
        _atomic_torch_save(checkpoint, destination)
    module.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return destination


def migrate_legacy_checkpoint(
    src_ckpt: str | Path,
    dst_ckpt: str | Path,
    gate_config: Mapping[str, object] | str | Path,
) -> None:
    """Convert one fixed-gate checkpoint into a strict dynamic-gate checkpoint."""
    source = Path(src_ckpt).expanduser().resolve()
    destination = Path(dst_ckpt).expanduser().resolve()
    if source == destination:
        raise ValueError("迁移目标不能覆盖源 checkpoint")
    if isinstance(gate_config, (str, Path)):
        with open(gate_config) as handle:
            gate_values = json.load(handle)
    else:
        gate_values = dict(gate_config)
    allowed_gate_values = {
        name: value for name, value in gate_values.items()
        if name in _GATE_CONFIG_FIELDS
    }
    if allowed_gate_values.get("gate_mode") == "constant":
        raise ValueError("迁移目标 gate_mode 不能是 constant")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    parent_config = checkpoint.get("config")
    if not isinstance(parent_config, Mapping):
        raise ValueError("源 checkpoint 缺少 dict config")
    target_values = dict(parent_config)
    target_values.update(allowed_gate_values)

    if bool(parent_config.get("layered")):
        from .layered import LayeredConfig, TransMemLayered

        config = LayeredConfig.from_dict(target_values)
        module = TransMemLayered(config)
    else:
        from .transmem import TransMem, TransMemConfig

        config = TransMemConfig(**target_values)
        module = TransMem(config)
    load_legacy_gate_state(module, checkpoint)

    migrated = {
        "model_state_dict": module.state_dict(),
        "config": config.to_dict(),
        "train_mode": "migrated_legacy_gate",
        "init_scheme": "legacy_gate",
        "parent_checkpoint": str(source),
        "gate_calibration_steps": 0,
        "joint_finetune_steps": 0,
        "global_step": 0,
        "epoch": 0,
    }
    _atomic_torch_save(migrated, destination)
