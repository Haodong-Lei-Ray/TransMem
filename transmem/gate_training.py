"""Shared training policy for dynamic TransMem gates."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch
import torch.nn as nn


INIT_SCHEMES = ("legacy_gate", "scratch_joint")


def validate_dynamic_resume_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    config: Mapping[str, object],
    init_scheme: str,
    parent_checkpoint: str | None,
    gate_calibration_steps: int,
    joint_finetune_steps: int | None,
    seed: int | None = None,
) -> None:
    """Fail closed when state-compatible gate semantics or provenance drift."""
    if config.get("gate_mode", "constant") == "constant":
        return
    if checkpoint.get("config") != dict(config):
        raise ValueError("dynamic-gate resume checkpoint config 与当前 config 不一致")
    expected = {
        "init_scheme": init_scheme,
        "parent_checkpoint": parent_checkpoint,
        "gate_calibration_steps": gate_calibration_steps,
        "seed": seed,
    }
    if joint_finetune_steps is not None:
        expected["joint_finetune_steps"] = joint_finetune_steps
    mismatches = [
        f"{name}: checkpoint={checkpoint.get(name)!r}, current={value!r}"
        for name, value in expected.items() if checkpoint.get(name) != value
    ]
    if mismatches:
        raise ValueError(
            "dynamic-gate resume provenance 不一致:\n  - "
            + "\n  - ".join(mismatches))


def is_gate_only_phase(
    init_scheme: str,
    step: int,
    gate_calibration_steps: int,
) -> bool:
    return init_scheme == "legacy_gate" and step < gate_calibration_steps


def named_gate_parameters(module: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    """Yield only TransMem's scalar gate heads, excluding Qwen MLP gate_proj."""
    gate_parameter_ids: set[int] = set()
    for child in module.modules():
        config = getattr(child, "config", None)
        gate_head = getattr(child, "gate_proj", None)
        if (getattr(config, "gate_mode", "constant") != "constant"
                and isinstance(gate_head, nn.Linear)
                and gate_head.out_features == 1):
            gate_parameter_ids.update(id(parameter) for parameter in gate_head.parameters())
    for name, parameter in module.named_parameters():
        if id(parameter) in gate_parameter_ids:
            yield name, parameter


def build_gate_optimizer(
    module: nn.Module,
    *,
    base_lr: float,
    gate_lr: float | None,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build named parameter groups so A1 can freeze base weights with lr=0."""
    gate_named = list(named_gate_parameters(module))
    gate_ids = {id(parameter) for _, parameter in gate_named}
    base = [parameter for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) not in gate_ids]
    gate = [parameter for _, parameter in gate_named if parameter.requires_grad]
    groups = []
    if base:
        groups.append({
            "params": base,
            "group_name": "base",
            "base_lr": float(base_lr),
            "lr": float(base_lr),
            "weight_decay": float(weight_decay),
        })
    if gate:
        resolved_gate_lr = float(base_lr if gate_lr is None else gate_lr)
        groups.append({
            "params": gate,
            "group_name": "gate",
            "base_lr": resolved_gate_lr,
            "lr": resolved_gate_lr,
            "weight_decay": float(weight_decay),
        })
    if not groups:
        raise ValueError("没有可训练参数")
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def set_gate_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
    *,
    lr_factor: float,
    gate_only: bool,
) -> None:
    """Apply one schedule factor while keeping legacy parameters exact in A1."""
    if lr_factor < 0:
        raise ValueError("lr_factor 不能为负数")
    for group in optimizer.param_groups:
        base_lr = float(group.get("base_lr", group["lr"]))
        if gate_only and group.get("group_name") == "base":
            group["lr"] = 0.0
        else:
            group["lr"] = base_lr * lr_factor


def clear_base_grads_for_gate_only(
    optimizer: torch.optim.Optimizer,
    *,
    gate_only: bool,
) -> None:
    """Prevent frozen A1 parameters from accumulating Adam moments."""
    if not gate_only:
        return
    for group in optimizer.param_groups:
        if group.get("group_name") != "base":
            continue
        for parameter in group["params"]:
            parameter.grad = None


def gate_prior_coefficient(*, step: int, weight: float, anneal_steps: int) -> float:
    """Linearly anneal the early ``(g-1)^2`` stabilizer to exactly zero."""
    if step < 0:
        raise ValueError("step 不能为负数")
    if weight < 0:
        raise ValueError("gate_prior_weight 不能为负数")
    if anneal_steps < 0:
        raise ValueError("gate_prior_anneal_steps 不能为负数")
    if weight == 0.0 or anneal_steps == 0 or step >= anneal_steps:
        return 0.0
    return float(weight) * (1.0 - float(step) / float(anneal_steps))


def gate_prior_loss(gate: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return mean ``(g-1)^2`` over valid answer positions and all layers."""
    values = (gate.squeeze(-1) - 1.0).square()
    if mask is None:
        return values.mean()
    valid = mask.to(device=values.device, dtype=torch.bool)
    while valid.ndim < values.ndim:
        valid = valid.unsqueeze(1)
    try:
        valid = valid.expand_as(values)
    except RuntimeError as exc:
        raise ValueError(
            f"gate mask shape {tuple(mask.shape)} 不能广播到 {tuple(values.shape)}") from exc
    if not bool(valid.any()):
        raise ValueError("gate prior mask 没有有效答案位置")
    return values[valid].mean()


@torch.no_grad()
def gate_metrics(
    ms: torch.Tensor,
    gate: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize gate, raw proposal and applied delta over valid positions."""
    gate_values = gate.squeeze(-1)
    ms_norm = ms.float().norm(dim=-1)
    delta_norm = (gate * ms).float().norm(dim=-1)
    if mask is not None:
        valid = mask.to(device=gate.device, dtype=torch.bool)
        while valid.ndim < gate_values.ndim:
            valid = valid.unsqueeze(1)
        try:
            valid = valid.expand_as(gate_values)
        except RuntimeError as exc:
            raise ValueError(
                f"gate mask shape {tuple(mask.shape)} 不能广播到 "
                f"{tuple(gate_values.shape)}") from exc
        gate_values = gate_values[valid]
        ms_norm = ms_norm[valid]
        delta_norm = delta_norm[valid]
    else:
        gate_values = gate_values.reshape(-1)
        ms_norm = ms_norm.reshape(-1)
        delta_norm = delta_norm.reshape(-1)
    gate_values = gate_values.float()
    if gate_values.numel() == 0:
        raise ValueError("没有可汇总的 gate")
    quantiles = torch.quantile(
        gate_values, torch.tensor([0.1, 0.5, 0.9], device=gate_values.device))
    return {
        "gate_mean": float(gate_values.mean()),
        "gate_std": float(gate_values.std(unbiased=False)),
        "gate_p10": float(quantiles[0]),
        "gate_p50": float(quantiles[1]),
        "gate_p90": float(quantiles[2]),
        "gate_frac_lt_025": float((gate_values < 0.25).float().mean()),
        "gate_frac_gt_175": float((gate_values > 1.75).float().mean()),
        "ms_norm": float(ms_norm.mean()),
        "delta_norm": float(delta_norm.mean()),
    }


def validate_gate_training_options(
    *,
    init_scheme: str,
    init_checkpoint: str | None,
    gate_mode: str,
    gate_calibration_steps: int,
    joint_finetune_steps: int | None,
    warm_start: bool = False,
) -> None:
    """Validate A/B initialization without guessing intent from file presence."""
    if init_scheme not in INIT_SCHEMES:
        raise ValueError(f"init_scheme 必须是 {INIT_SCHEMES}, 得到 {init_scheme!r}")
    if gate_calibration_steps < 0:
        raise ValueError("gate_calibration_steps 不能为负数")
    if joint_finetune_steps is not None and joint_finetune_steps < 0:
        raise ValueError("joint_finetune_steps 不能为负数")
    if init_scheme == "legacy_gate":
        if not init_checkpoint:
            raise ValueError("legacy_gate 必须显式提供 --init_checkpoint")
        if gate_mode != "centered_sigmoid":
            raise ValueError(
                "legacy_gate 必须使用 centered_sigmoid，才能在 step 0 精确保持 g=1")
    else:
        if init_checkpoint:
            raise ValueError("scratch_joint 禁止传 --init_checkpoint")
        if gate_mode != "constant" and warm_start:
            raise ValueError(
                "dynamic scratch_joint 要求 warm_start=false；backbone 热启动须使用独立消融名")
        if gate_calibration_steps:
            raise ValueError(
                "scratch_joint 的 gate_calibration_steps 必须为 0；从第一个优化步联合训练")
