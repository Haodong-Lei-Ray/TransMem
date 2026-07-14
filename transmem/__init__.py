"""TransMem: 给冻结 LLM 外挂的小 Transformer, 读 (记忆, 查询) 一次前向回归记忆偏置 MS.

LLM 全程冻结, TransMem 是唯一可训练模块. 详见 Project4/docs/plan.md.
"""

from .transmem import TransMemConfig, TransMemOutput, TransMem, build_transmem
from .objectives import DistillLoss, FrozenLMHead
from .layers import (
    to_qwen3_config,
    build_additive_causal_mask,
    copy_top_layers_from_backbone,
)

__all__ = [
    "TransMemConfig",
    "TransMemOutput",
    "TransMem",
    "build_transmem",
    "DistillLoss",
    "FrozenLMHead",
    "to_qwen3_config",
    "build_additive_causal_mask",
    "copy_top_layers_from_backbone",
]
