"""解耦的蒸馏目标: 把"学生分布 -> 教师分布"的对齐损失独立成模块.

train.png: "对齐, off-policy 和 OPD 俩种方式都做做".
两种方式的区别只在**轨迹/数据来源**(off-policy=Stage0 固定教师轨迹的特征; OPD=当前策略
在线 rollout), **散度计算是共享的** —— 都落到 (teacher_logits, student_logits) 上.
因此散度在这里实现一次, 供两种训练脚本复用 (法则第 4 条).

  teacher_logits = lm_head(HQ_tea_i)          (冻结, 作为稠密软目标)
  student_logits = lm_head(HQ'_i)             (HQ'_i = HQ_stu_i + a*MS_i, 可微回传到 TransMem)
  L = divergence(P_tea, P_stu)                (+ 可选表征回归 ||HQ'-HQ_tea||^2 热身)

散度可选 (config 驱动):
  forward_kl : KL(P_tea || P_stu)   —— OPSDL/标准 KD, mode-covering (plan 默认)
  reverse_kl : KL(P_stu || P_tea)   —— mode-seeking, on-policy 蒸馏常用
  jsd        : 广义 Jensen-Shannon (GKD), beta 插值

温度 T: 对 logits 软化 (/T), 损失乘 T^2 (标准 KD 缩放).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# 散度 (输入 logits, 内部转 log-prob, 数值稳定)
# ═══════════════════════════════════════════════════════════════════════

def _kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(P || Q), P=softmax(p_logits), Q=softmax(q_logits). 返回逐样本求和、batch 平均的标量.

    KL = sum_v P_v (logP_v - logQ_v).  用 log_softmax 保证数值稳定.
    """
    logp = F.log_softmax(p_logits, dim=-1)
    logq = F.log_softmax(q_logits, dim=-1)
    p = logp.exp()
    return (p * (logp - logq)).sum(dim=-1).mean()


def _jsd_from_logits(a_logits: torch.Tensor, b_logits: torch.Tensor,
                     beta: float = 0.5) -> torch.Tensor:
    """广义 Jensen-Shannon 散度 (GKD): JSD_beta(P||Q) = beta·KL(P||M) + (1-beta)·KL(Q||M),
    M = beta·P + (1-beta)·Q. 这里 P=teacher, Q=student."""
    logp = F.log_softmax(a_logits, dim=-1)
    logq = F.log_softmax(b_logits, dim=-1)
    p, q = logp.exp(), logq.exp()
    m = beta * p + (1.0 - beta) * q
    logm = m.clamp_min(1e-12).log()
    kl_pm = (p * (logp - logm)).sum(dim=-1)
    kl_qm = (q * (logq - logm)).sum(dim=-1)
    return (beta * kl_pm + (1.0 - beta) * kl_qm).mean()


# ═══════════════════════════════════════════════════════════════════════
# 蒸馏损失 (config 驱动, off-policy / OPD 共用)
# ═══════════════════════════════════════════════════════════════════════

class DistillLoss:
    """逐位置蒸馏损失 + 可选表征回归热身.

    Args:
      divergence: "forward_kl" | "reverse_kl" | "jsd"
      temperature: 软化温度 T (>0); 损失乘 T^2
      reg_weight: 表征回归 ||HQ'-HQ_tea||^2 的权重 beta (0=关). 完全不碰 lm_head, 适合热身.
      jsd_beta: jsd 的插值系数

    __call__(student_logits, teacher_logits, hq_prime=None, hq_tea=None) -> (loss, metrics).
    teacher 侧 (teacher_logits, hq_tea) 内部 detach, 作为固定目标.
    """

    def __init__(self, divergence: str = "forward_kl", temperature: float = 1.0,
                 reg_weight: float = 0.0, jsd_beta: float = 0.5):
        assert divergence in ("forward_kl", "reverse_kl", "jsd"), divergence
        assert temperature > 0
        self.divergence = divergence
        self.temperature = float(temperature)
        self.reg_weight = float(reg_weight)
        self.jsd_beta = float(jsd_beta)

    def __call__(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                 hq_prime: Optional[torch.Tensor] = None,
                 hq_tea: Optional[torch.Tensor] = None):
        teacher_logits = teacher_logits.detach()
        T = self.temperature
        # 散度在 fp32 下算, 避免 bf16 log_softmax 掉精度 (student 侧保留计算图)
        s = student_logits.float() / T
        t = teacher_logits.float() / T

        if self.divergence == "forward_kl":
            div = _kl_from_logits(t, s)          # KL(P_tea || P_stu)
        elif self.divergence == "reverse_kl":
            div = _kl_from_logits(s, t)          # KL(P_stu || P_tea)
        else:
            div = _jsd_from_logits(t, s, self.jsd_beta)
        div = div * (T * T)

        loss = div
        metrics = {"div": float(div.detach())}

        if self.reg_weight > 0.0 and hq_prime is not None and hq_tea is not None:
            reg = F.mse_loss(hq_prime, hq_tea.detach())
            loss = loss + self.reg_weight * reg
            metrics["reg"] = float(reg.detach())

        metrics["loss"] = float(loss.detach())
        return loss, metrics


# ═══════════════════════════════════════════════════════════════════════
# 冻结 LM head (穿它把 hidden -> logits, 梯度回传到 TransMem)
# ═══════════════════════════════════════════════════════════════════════

class FrozenLMHead(nn.Module):
    """冻结的 LM head: hidden [*, dim] -> logits [*, vocab].

    Qwen3-4B tie_word_embeddings=true, lm_head.weight 即 embed_tokens.weight.
    权重冻结 (requires_grad=False), 但作为线性算子可微, 梯度穿过它回传到 HQ' -> MS -> TransMem.
    """

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        vocab, dim = weight.shape
        self.proj = nn.Linear(dim, vocab, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(weight)
        self.proj.weight.requires_grad_(False)

    @classmethod
    def from_file(cls, path: str | Path, device=None, dtype=None) -> "FrozenLMHead":
        """从 dump_lm_head.py 落盘的 {'weight': [vocab, dim], 'tied': bool} 加载."""
        obj = torch.load(path, map_location="cpu", weights_only=False)
        w = obj["weight"] if isinstance(obj, dict) else obj
        if dtype is not None:
            w = w.to(dtype)
        head = cls(w)
        if device is not None:
            head = head.to(device)
        return head

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)
