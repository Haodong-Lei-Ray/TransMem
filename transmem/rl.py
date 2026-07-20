"""Pure reward and objective functions for TransMem GRPO post-training."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

import torch


def normalize_hotpot_answer(text: str) -> str:
    """Match HotpotQA's official lowercase/punctuation/article normalization."""
    lowered = str(text).lower()
    no_punctuation = "".join(
        character for character in lowered if character not in set(string.punctuation))
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def hotpot_answer_metrics(prediction: str, ground_truth: str) -> tuple[float, float]:
    """Return official answer-only ``(F1, EM)`` for one HotpotQA response."""
    normalized_prediction = normalize_hotpot_answer(prediction)
    normalized_ground_truth = normalize_hotpot_answer(ground_truth)
    exact_match = float(normalized_prediction == normalized_ground_truth)

    categorical = {"yes", "no", "noanswer"}
    if ((normalized_prediction in categorical
         or normalized_ground_truth in categorical)
            and normalized_prediction != normalized_ground_truth):
        return 0.0, exact_match

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0, exact_match
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    matching = sum(common.values())
    if matching == 0:
        return 0.0, exact_match
    precision = matching / len(prediction_tokens)
    recall = matching / len(ground_truth_tokens)
    return 2.0 * precision * recall / (precision + recall), exact_match


@dataclass(frozen=True)
class AnswerReward:
    """Logged reward components; only ``total`` enters the policy objective."""

    total: float
    f1: float
    em: float
    verbosity_penalty: float
    invalid_penalty: float


def hotpot_answer_reward(
    prediction: str,
    ground_truth: str,
    answer_tokens: int,
    *,
    em_weight: float = 0.25,
    verbosity_weight: float = 0.05,
    verbosity_start: int = 32,
    verbosity_cap: int = 64,
    invalid_penalty: float = 1.0,
) -> AnswerReward:
    """Score correctness first, with bounded guards against empty/verbose output."""
    if verbosity_start < 0 or verbosity_cap <= verbosity_start:
        raise ValueError("verbosity_cap 必须严格大于非负的 verbosity_start")
    f1, em = hotpot_answer_metrics(prediction, ground_truth)
    overflow = max(0, min(int(answer_tokens), verbosity_cap) - verbosity_start)
    verbosity = verbosity_weight * overflow / (verbosity_cap - verbosity_start)
    invalid = float(invalid_penalty if not normalize_hotpot_answer(prediction) else 0.0)
    total = f1 + em_weight * em - verbosity - invalid
    return AnswerReward(
        total=float(total),
        f1=float(f1),
        em=float(em),
        verbosity_penalty=float(verbosity),
        invalid_penalty=invalid,
    )


def group_relative_advantages(
    rewards: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, bool]:
    """Normalize one prompt's rewards; constant groups intentionally yield zero."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("GRPO 每组至少需要两个一维 reward")
    values = rewards.float()
    std = values.std(unbiased=False)
    active = bool(torch.isfinite(std) and float(std) > eps)
    if not active:
        return torch.zeros_like(values), False
    return (values - values.mean()) / std, True


def grpo_clipped_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Token-mean clipped GRPO policy loss for a padded response group."""
    if new_log_probs.shape != old_log_probs.shape or new_log_probs.shape != mask.shape:
        raise ValueError("new/old log-prob 和 mask shape 必须相同")
    if advantages.ndim != 1 or advantages.shape[0] != new_log_probs.shape[0]:
        raise ValueError("advantages 必须是一条标量对应一条 response")
    if not 0.0 < clip_eps < 1.0:
        raise ValueError("clip_eps 必须位于 (0,1)")
    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    token_advantage = advantages.to(ratio).unsqueeze(-1)
    unclipped = ratio * token_advantage
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * token_advantage
    selected = torch.minimum(unclipped, clipped)
    valid = mask.to(dtype=torch.bool)
    if not bool(valid.any()):
        raise ValueError("GRPO response mask 没有有效 token")
    return -selected[valid].mean()


def sampled_reference_kl(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Non-negative sampled estimator of KL(policy || fixed reference)."""
    if policy_log_probs.shape != reference_log_probs.shape:
        raise ValueError("policy/reference log-prob shape 必须相同")
    log_ratio_inverse = reference_log_probs.detach() - policy_log_probs
    estimate = torch.exp(log_ratio_inverse) - log_ratio_inverse - 1.0
    if mask is None:
        return estimate.mean()
    valid = mask.to(dtype=torch.bool)
    if valid.shape != estimate.shape or not bool(valid.any()):
        raise ValueError("reference KL mask 无效")
    return estimate[valid].mean()
