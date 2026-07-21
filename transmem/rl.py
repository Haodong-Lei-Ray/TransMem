"""Pure reward and objective functions for TransMem GRPO post-training."""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass

import torch

try:
    from nltk.stem import PorterStemmer
except ImportError:  # pragma: no cover - reward remains usable in minimal envs
    PorterStemmer = None


_LOCOMO_STEMMER = PorterStemmer() if PorterStemmer is not None else None


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


@dataclass(frozen=True)
class StructuredAnswer:
    """A generated reasoning trace split from the answer used for reward."""

    thinking: str
    answer: str
    has_answer_marker: bool


def split_thinking_answer(text: str) -> StructuredAnswer:
    """Parse Qwen thinking output without leaking reasoning into answer reward.

    Hybrid Qwen models emit ``<think>...</think>``.  Pure instruct checkpoints
    are prompted to finish with an ``Answer:`` line.  The final marker wins so
    incidental marker text inside a trace cannot absorb the final answer.
    """
    raw = str(text).strip()
    thinking = ""
    answer = raw
    marked = False
    if "</think>" in answer:
        thinking, _, answer = answer.rpartition("</think>")
        thinking = thinking.replace("<think>", "").strip()
        answer = answer.strip()
        marked = True
    if "Answer:" in answer:
        head, _, tail = answer.rpartition("Answer:")
        head = head.strip()
        if head:
            thinking = f"{thinking}\n{head}".strip() if thinking else head
        answer = tail.strip()
        marked = True
    return StructuredAnswer(thinking=thinking, answer=answer,
                            has_answer_marker=marked)


def _normalize_locomo_answer(text: str) -> str:
    """Match LoCoMo's punctuation/article normalization for reward scoring."""
    text = unicodedata.normalize("NFD", str(text).replace(",", "")).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def _token_f1(prediction: str, ground_truth: str, *, locomo: bool) -> float:
    normalize = _normalize_locomo_answer if locomo else normalize_hotpot_answer
    prediction_tokens = normalize(prediction).split()
    ground_truth_tokens = normalize(ground_truth).split()
    if locomo and _LOCOMO_STEMMER is not None:
        prediction_tokens = [_LOCOMO_STEMMER.stem(token)
                             for token in prediction_tokens]
        ground_truth_tokens = [_LOCOMO_STEMMER.stem(token)
                               for token in ground_truth_tokens]
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    matching = sum((Counter(prediction_tokens) & Counter(ground_truth_tokens)).values())
    if matching == 0:
        return 0.0
    precision = matching / len(prediction_tokens)
    recall = matching / len(ground_truth_tokens)
    return 2.0 * precision * recall / (precision + recall)


def locomo_answer_metrics(
    prediction: str,
    ground_truth: str,
    category: int | None,
) -> tuple[float, float]:
    """Return LoCoMo-compatible answer-only ``(F1, EM)`` for categories 1-4."""
    answer = str(ground_truth)
    category = int(category or 0)
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category == 1:
        predictions = [part.strip() for part in str(prediction).split(",") if part.strip()]
        answers = [part.strip() for part in answer.split(",") if part.strip()]
        if predictions and answers:
            f1 = sum(
                max(_token_f1(candidate, reference, locomo=True)
                    for candidate in predictions)
                for reference in answers
            ) / len(answers)
        else:
            f1 = 0.0
    else:
        f1 = _token_f1(prediction, answer, locomo=True)
    exact = float(_normalize_locomo_answer(prediction)
                  == _normalize_locomo_answer(answer))
    return float(f1), exact


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


def task_answer_reward(
    prediction: str,
    ground_truth: str,
    answer_tokens: int,
    *,
    scorer: str = "hotpotqa",
    category: int | None = None,
    valid_format: bool = True,
    em_weight: float = 0.25,
    verbosity_weight: float = 0.05,
    verbosity_start: int = 32,
    verbosity_cap: int = 64,
    invalid_penalty: float = 1.0,
) -> AnswerReward:
    """Answer-only reward shared by HotpotQA, LongMemEval and LoCoMo GRPO."""
    if verbosity_start < 0 or verbosity_cap <= verbosity_start:
        raise ValueError("verbosity_cap 必须严格大于非负的 verbosity_start")
    if scorer == "locomo":
        f1, em = locomo_answer_metrics(prediction, ground_truth, category)
        normalized = _normalize_locomo_answer(prediction)
    elif scorer in {"hotpotqa", "longmemeval"}:
        f1, em = hotpot_answer_metrics(prediction, ground_truth)
        normalized = normalize_hotpot_answer(prediction)
    else:
        raise ValueError(f"未知 reward scorer: {scorer}")
    overflow = max(0, min(int(answer_tokens), verbosity_cap) - verbosity_start)
    verbosity = verbosity_weight * overflow / (verbosity_cap - verbosity_start)
    invalid = float(invalid_penalty if (not normalized or not valid_format) else 0.0)
    return AnswerReward(
        total=float(f1 + em_weight * em - verbosity - invalid),
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
