"""Shared helpers for the v3 memory-use and checkpoint diagnostics.

The helpers in this module are deliberately model-agnostic.  The expensive
evaluation entry points live under ``scripts/eval``; keeping permutation,
position grouping, and aggregation here makes their semantics unit-testable.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import torch


def make_derangement(size: int, seed: int) -> torch.Tensor:
    """Return a deterministic one-to-one donor map with no self donors.

    A seeded random ordering followed by a cyclic shift is both a permutation
    and a guaranteed derangement.  This is preferable to repeatedly sampling
    ``randperm`` until no fixed points happen: its runtime and random-number
    consumption are stable for every seed.
    """

    if size < 2:
        raise ValueError("HM shuffling needs at least two samples")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(size, generator=generator)
    donors = torch.empty(size, dtype=torch.long)
    donors[order] = order.roll(-1)
    return donors


def position_groups(valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split a padded ``[B, M]`` validity mask into all/first/later tokens."""

    if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be a boolean [batch, positions] tensor")
    first = torch.zeros_like(valid_mask)
    nonempty = valid_mask.any(dim=1)
    first[nonempty, 0] = True
    # Right-padded Stage0 sequences always have their first valid token at 0.
    if torch.any(valid_mask[nonempty, 0] == 0):
        raise ValueError("valid Stage0 sequences must start at position zero")
    later = valid_mask & ~first
    return {"all": valid_mask, "first": first, "later": later}


def parse_curve_steps(spec: str | None) -> tuple[int, ...]:
    """Parse a comma-separated, unique, sorted list of non-negative steps."""

    if spec is None or not str(spec).strip():
        return ()
    steps: set[int] = set()
    for raw in str(spec).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            step = int(raw)
        except ValueError as exc:
            raise ValueError(f"invalid curve step: {raw!r}") from exc
        if str(step) != raw and raw not in (f"+{step}",):
            raise ValueError(f"invalid curve step: {raw!r}")
        if step < 0:
            raise ValueError(f"curve steps must be non-negative, got {step}")
        steps.add(step)
    return tuple(sorted(steps))


class MetricAccumulator:
    """Position-weighted aggregation for nested ``mode/group/metric`` values.

    Values may contain NaNs for metrics unavailable at some positions (for
    example a legacy Stage0 sample without ``answer_ids``).  Such values are
    excluded only from that metric and its explicit ``*_positions`` count.
    """

    def __init__(self) -> None:
        self._sums: dict[tuple[str, str, str], float] = defaultdict(float)
        self._counts: dict[tuple[str, str, str], int] = defaultdict(int)
        self._positions: dict[tuple[str, str], int] = defaultdict(int)

    def update(self, mode: str, group: str,
               metrics: Mapping[str, torch.Tensor]) -> None:
        if not metrics:
            return
        lengths = {int(value.numel()) for value in metrics.values()}
        if len(lengths) != 1:
            raise ValueError(f"metric lengths differ for {mode}/{group}: {lengths}")
        self._positions[(mode, group)] += next(iter(lengths))
        for name, value in metrics.items():
            flat = value.detach().float().reshape(-1).cpu()
            finite = torch.isfinite(flat)
            key = (mode, group, name)
            self._sums[key] += float(flat[finite].double().sum())
            self._counts[key] += int(finite.sum())

    def summary(self) -> dict[str, dict[str, dict[str, float | int]]]:
        result: dict[str, dict[str, dict[str, float | int]]] = {}
        mode_groups = sorted(self._positions)
        for mode, group in mode_groups:
            row: dict[str, float | int] = {
                "positions": self._positions[(mode, group)],
            }
            keys = sorted(k for k in self._sums if k[:2] == (mode, group))
            for key in keys:
                name = key[2]
                count = self._counts[key]
                row[name] = self._sums[key] / max(count, 1)
                if count != row["positions"]:
                    row[f"{name}_positions"] = count
            result.setdefault(mode, {})[group] = row
        return result


@torch.no_grad()
def token_metrics(
    candidate_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    target_ids: torch.Tensor | None = None,
    correction: torch.Tensor | None = None,
    real_top1: torch.Tensor | None = None,
    student_top1: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return per-token diagnostics without reducing over positions.

    candidate_logits and teacher_logits have shape [..., vocab]. Every
    returned tensor has the leading shape [...] so callers can apply the exact
    same validity/position masks before aggregation.

    target_ids is the Stage0 trajectory's next-token id. Missing targets
    (-100 or any id outside the vocabulary) produce NaN for target-based
    metrics; MetricAccumulator then excludes only those entries. correction
    is the hidden-space delta HQ_prime - HQ_student.
    """

    if candidate_logits.shape != teacher_logits.shape:
        raise ValueError(
            "candidate_logits and teacher_logits must have identical shapes, "
            f"got {tuple(candidate_logits.shape)} and {tuple(teacher_logits.shape)}")
    if candidate_logits.ndim < 2:
        raise ValueError("logits must have shape [..., vocab]")
    leading = candidate_logits.shape[:-1]
    vocab = candidate_logits.shape[-1]

    candidate_logp = torch.log_softmax(candidate_logits.float(), dim=-1)
    teacher_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
    forward_kl = (
        teacher_logp.exp() * (teacher_logp - candidate_logp)
    ).sum(dim=-1)
    candidate_top1 = candidate_logits.argmax(dim=-1)
    teacher_top1 = teacher_logits.argmax(dim=-1)

    result = {
        "forward_kl": forward_kl,
        "teacher_top1": (candidate_top1 == teacher_top1).float(),
    }

    if target_ids is not None:
        if target_ids.shape != leading:
            raise ValueError(
                f"target_ids must have shape {tuple(leading)}, "
                f"got {tuple(target_ids.shape)}")
        target_ids = target_ids.to(candidate_logits.device)
        valid_target = (target_ids >= 0) & (target_ids < vocab)
        safe_target = target_ids.clamp(min=0, max=max(vocab - 1, 0))
        trajectory_nll = torch.full(
            leading, float("nan"), device=candidate_logits.device,
            dtype=torch.float32)
        trajectory_accuracy = torch.full_like(trajectory_nll, float("nan"))
        gathered = -candidate_logp.gather(
            dim=-1, index=safe_target.unsqueeze(-1)).squeeze(-1)
        trajectory_nll[valid_target] = gathered[valid_target]
        trajectory_accuracy[valid_target] = (
            candidate_top1[valid_target] == target_ids[valid_target]
        ).float()
        result["trajectory_nll"] = trajectory_nll
        result["trajectory_accuracy"] = trajectory_accuracy

    if correction is not None:
        if correction.shape[:-1] != leading:
            raise ValueError(
                f"correction must have leading shape {tuple(leading)}, "
                f"got {tuple(correction.shape)}")
        result["correction_norm"] = correction.float().norm(dim=-1)

    for name, reference in (
        ("same_as_real", real_top1),
        ("changed_from_student", student_top1),
    ):
        if reference is None:
            continue
        if reference.shape != leading:
            raise ValueError(
                f"{name} reference must have shape {tuple(leading)}, "
                f"got {tuple(reference.shape)}")
        reference = reference.to(candidate_top1.device)
        agreement = candidate_top1 == reference
        result[name] = agreement.float() if name == "same_as_real" else (~agreement).float()

    return result
