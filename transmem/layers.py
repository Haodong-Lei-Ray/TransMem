"""TransMem 的基础层: 直接复用 HF transformers 的 Qwen3 decoder block.

设计原则 (法则第 4 条: 复用现有, 不重造接口):
  TransMem 要求"与学生 LLM 同款 block"(GQA + RoPE + QK-Norm + SwiGLU + RMSNorm).
  transformers 里 `Qwen3DecoderLayer` 就是这个 block 的权威实现, 且我们本来就用
  transformers 加载 Qwen3-4B backbone, 所以:
    - 逐位与 backbone 等同, 数值行为一致;
    - 热启动只需把 backbone 顶部 L 层的 state_dict 拷进来 (同结构同维度);
    - 头数/kv 头/head_dim/rope_theta 等直接对齐, 不会写错.

本文件只提供围绕 Qwen3 层的薄封装: 构造 Qwen3Config、因果 mask、热启动拷贝,
并 re-export 复用到的 HF 类. 真正的 TransMem 模块在 transmem.py.

实测 (transformers 5.12.1):
  Qwen3DecoderLayer(config, layer_idx).forward(
      hidden_states, attention_mask=<4D 加性 mask 或 None>,
      position_ids, position_embeddings=(cos, sin), use_cache=False) -> Tensor
  Qwen3RotaryEmbedding(config).forward(x, position_ids) -> (cos, sin)
"""

from __future__ import annotations

from typing import Optional

import torch

from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

__all__ = [
    "Qwen3DecoderLayer",
    "Qwen3RMSNorm",
    "Qwen3RotaryEmbedding",
    "to_qwen3_config",
    "build_additive_causal_mask",
    "copy_top_layers_from_backbone",
]


# ═══════════════════════════════════════════════════════════════════════
# Qwen3Config 构造: 把 TransMemConfig 翻译成 backbone 同款 block 的配置
# ═══════════════════════════════════════════════════════════════════════

def to_qwen3_config(cfg) -> Qwen3Config:
    """从 TransMemConfig 造一个 transformers Qwen3Config, 使 block 与 backbone 同款.

    只设置与单个 decoder block 相关的字段; num_hidden_layers 设为 depth 仅为让
    layer_idx 合法 (TransMem 自己管理这 depth 层, 不走 Qwen3Model).
    """
    return Qwen3Config(
        hidden_size=cfg.dim,
        num_hidden_layers=cfg.depth,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_position_embeddings,
        initializer_range=cfg.initializer_range,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        attn_implementation=cfg.attn_impl,
    )


# ═══════════════════════════════════════════════════════════════════════
# 因果 mask: 查询槽放末尾, attend 前面所有记忆槽
# ═══════════════════════════════════════════════════════════════════════

def build_additive_causal_mask(seq_len: int, dtype: torch.dtype,
                               device: torch.device,
                               causal: bool = True,
                               past_len: int = 0) -> Optional[torch.Tensor]:
    """构造 [1, 1, S, past_len+S] 加性 attention mask (S=本次新喂入的长度).

    causal=True : 全局位置 past_len+i 的查询只能看 key 0..past_len+i -> 标准因果.
                  past_len=0 时退化为原来的 [1,1,S,S] 方阵; past_len>0 用于 KV cache
                  增量前向 (cache 里的历史 key 全可见, 新 token 间仍因果).
    causal=False: 返回 None -> Qwen3Attention 走全可见(记忆槽间双向).

    用 finfo(dtype).min 而非 -inf, 避免 bf16/half 下 sdpa/eager 的 NaN 风险
    (因果下每行至少对角线可见, 不会出现整行被屏蔽).
    """
    if not causal:
        return None
    total = past_len + seq_len
    min_val = torch.finfo(dtype).min
    mask = torch.full((seq_len, total), min_val, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1 + past_len)   # j > past_len+i 屏蔽(未来), 其余 0
    return mask.view(1, 1, seq_len, total)


# ═══════════════════════════════════════════════════════════════════════
# 热启动: 用 backbone 顶部 depth 层权重初始化 TransMem
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def copy_top_layers_from_backbone(layers: torch.nn.ModuleList,
                                  final_norm: Optional[torch.nn.Module],
                                  backbone) -> None:
    """把 backbone(Qwen3ForCausalLM)顶部 len(layers) 层拷进 TransMem 的 layers.

    backbone.model.layers[-L:] -> layers[0:L] (逐层 load_state_dict, 同结构同维度).
    若提供 final_norm, 再把 backbone.model.norm 拷进去.

    注意(表征空间一致性): backbone 的 decoder layer 工作在 pre-norm 残差流空间,
    若 TransMem 输入用的是 final-norm 之后的 hidden, 二者空间不一致, 热启动收益打折.
    用 warm_start 时建议把 Stage0 的 hidden 源改成 pre-norm 残差流(见 docs). 默认冷启动.
    """
    src_layers = backbone.model.layers
    L = len(layers)
    assert len(src_layers) >= L, (
        f"backbone 仅 {len(src_layers)} 层, 不足以热启动 {L} 层")
    for i in range(L):
        layers[i].load_state_dict(src_layers[len(src_layers) - L + i].state_dict())
    if final_norm is not None and hasattr(backbone.model, "norm"):
        final_norm.load_state_dict(backbone.model.norm.state_dict())
