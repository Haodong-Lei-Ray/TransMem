"""TransMem: L 层 Qwen3 同款 decoder block + 零初始化读出, 一次前向回归记忆偏置 MS.

  输入   X_i = [HM_stu ; HQ_stu_i]            [B, N+1, dim]   (查询槽放末尾)
         L 层 Qwen3 block (causal, RoPE)
         读末位查询槽 -> final_norm -> out_proj(零初始化)
  输出   MS_i                                  [B, dim]
  读出   HQ'_i = HQ_stu_i + a * MS_i           [B, dim]   (逐元素相加)

LLM 全程冻结, TransMem 是唯一可训练模块. out_proj 零初始化使初始 MS=0 -> HQ'=HQ_stu
恒等, 训练更稳. block 与 backbone 同款, 可选用 backbone 顶部 L 层热启动.

对应 inference.png: SLM 读 [H_L/N, ..., H_L, H_Q_QN] -> H_M(=MS), H'_Q = H_Q + H_M -> LM-HEAD.

参考: plan.md §4; 姊妹方案 Project3/diffusionmem/diffusion_mem(数据/教师-学生/OPSDL 一致,
      只把"扩散 DiT 多步采样 λ"换成"TransMem 一次前向回归 MS").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .layers import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    to_qwen3_config,
    build_additive_causal_mask,
    copy_top_layers_from_backbone,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer


# ═══════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TransMemConfig:
    """TransMem 全部超参数, 可从 json 加载. 默认对齐 Qwen3-4B-Instruct-2507."""

    # ── 架构 (对齐 backbone block) ────────────────────────────────────
    dim: int = 2560
    depth: int = 4
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    attn_impl: str = "sdpa"

    # ── TransMem 专属 (可解耦对比) ────────────────────────────────────
    causal: bool = True            # 因果 mask(查询末尾) vs 记忆槽双向
    pos_mode: str = "rope"         # none | rope | learned (位置注入方式)
    n_mem: int = 4                 # 记忆分段数 N (learned 位置 embedding 长度 N+1 用)
    final_norm: bool = True        # 读出前是否过 Qwen3RMSNorm
    zero_init_out: bool = True     # out_proj 零初始化 -> 初始 MS=0

    # ── 读出 scale ────────────────────────────────────────────────────
    a_init: float = 1.0            # HQ' = HQ_stu + a*MS
    learnable_a: bool = False
    warm_start: bool = False       # 是否热启动(权重在 build 后由训练脚本拷入)

    @classmethod
    def from_json(cls, path: str | Path) -> "TransMemConfig":
        with open(path, "r") as f:
            data = json.load(f)
        kwargs = {k: v for k, v in data.items() if not k.startswith("_")}
        return cls(**kwargs)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════
# TransMem 模块
# ═══════════════════════════════════════════════════════════════════════

class TransMem(nn.Module):
    """记忆偏置网络: L 层 Qwen3 block + 零初始化读出.

    forward(X) -> MS:  X [B, N+1, dim] -> MS [B, dim] (读末位查询槽).
    correct(MS, HQ_stu) -> HQ':  HQ'_i = HQ_stu_i + a*MS_i.
    唯一可训练模块 (LLM 冻结).
    """

    _POS_MODES = ("none", "rope", "learned")

    def __init__(self, config: TransMemConfig):
        super().__init__()
        assert config.pos_mode in self._POS_MODES, (
            f"pos_mode 必须是 {self._POS_MODES}, 得到 {config.pos_mode}")
        self.config = config
        dim = config.dim

        qcfg = to_qwen3_config(config)
        self.blocks = nn.ModuleList(
            [Qwen3DecoderLayer(qcfg, layer_idx=i) for i in range(config.depth)]
        )
        self.rotary = Qwen3RotaryEmbedding(qcfg)
        self.final_norm = (Qwen3RMSNorm(dim, eps=config.rms_norm_eps)
                           if config.final_norm else nn.Identity())

        # 读出: 零初始化 -> 初始 MS=0 -> HQ'=HQ_stu 恒等
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # learned 位置: 零初始化 -> 初始不扰动输入
        if config.pos_mode == "learned":
            self.pos_emb = nn.Parameter(torch.zeros(1, config.n_mem + 1, dim))
        else:
            self.pos_emb = None

        # scale a
        if config.learnable_a:
            self.a = nn.Parameter(torch.tensor(float(config.a_init)))
        else:
            self.register_buffer("a", torch.tensor(float(config.a_init)),
                                 persistent=False)

        self._init_weights()

    # ── 初始化 ────────────────────────────────────────────────────────
    def _init_weights(self):
        """冷启动: Qwen 风格 normal(0, initializer_range) 初始化 block 内 Linear,
        RMSNorm 置 1. 热启动(warm_start)时这些会被 backbone 权重覆盖, 故无害.
        out_proj 始终按 zero_init_out 处理(零初始化优先级最高, 保证恒等启动)."""
        std = self.config.initializer_range
        for m in self.blocks.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, Qwen3RMSNorm):
                nn.init.ones_(m.weight)
        if isinstance(self.final_norm, Qwen3RMSNorm):
            nn.init.ones_(self.final_norm.weight)
        if self.config.zero_init_out:
            nn.init.zeros_(self.out_proj.weight)
        else:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=std)

    # ── 热启动: 由训练脚本传入已加载的 backbone ──────────────────────
    def warm_start_from(self, backbone) -> None:
        """用 backbone 顶部 depth 层 + final norm 热启动(见 layers.copy_top_layers)."""
        fn = self.final_norm if isinstance(self.final_norm, Qwen3RMSNorm) else None
        copy_top_layers_from_backbone(self.blocks, fn, backbone)
        # 热启动后仍保证读出恒等
        if self.config.zero_init_out:
            nn.init.zeros_(self.out_proj.weight)

    # ── 前向: X -> MS ─────────────────────────────────────────────────
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: [B, N+1, dim] (末位为查询槽) -> MS: [B, dim]."""
        B, S, _ = X.shape
        h = X

        # 位置 id: rope=0..S-1; none/learned=全 0(RoPE 退化为恒等旋转)
        if self.config.pos_mode == "rope":
            pos_ids = torch.arange(S, device=X.device, dtype=torch.long)
        else:
            pos_ids = torch.zeros(S, device=X.device, dtype=torch.long)
        pos_ids = pos_ids.unsqueeze(0).expand(B, -1)            # [B, S]

        if self.config.pos_mode == "learned":
            assert S <= self.pos_emb.shape[1], (
                f"learned 位置 embedding 长度 {self.pos_emb.shape[1]} < 序列 {S}")
            h = h + self.pos_emb[:, :S, :].to(h.dtype)

        cos, sin = self.rotary(h, pos_ids)                      # [B, S, head_dim]
        mask = build_additive_causal_mask(S, h.dtype, h.device, self.config.causal)

        for block in self.blocks:
            out = block(
                hidden_states=h,
                attention_mask=mask,
                position_ids=pos_ids,
                position_embeddings=(cos, sin),
                use_cache=False,
            )
            h = out[0] if isinstance(out, tuple) else out

        h = self.final_norm(h)
        ms = self.out_proj(h[:, -1, :])                         # 读末位查询槽 [B, dim]
        return ms

    # ── 读出: HQ' = HQ_stu + a*MS ────────────────────────────────────
    def correct(self, ms: torch.Tensor, hq_stu: torch.Tensor) -> torch.Tensor:
        """HQ'_i = HQ_stu_i + a*MS_i.  ms, hq_stu: [B, dim]."""
        return hq_stu + self.a * ms

    @property
    def dim(self) -> int:
        return self.config.dim

    def num_params(self, trainable_only: bool = False) -> int:
        ps = (p for p in self.parameters() if (p.requires_grad or not trainable_only))
        return sum(p.numel() for p in ps)


# ═══════════════════════════════════════════════════════════════════════
# 工厂
# ═══════════════════════════════════════════════════════════════════════

def build_transmem(config_path: str | Path | None = None,
                   config: TransMemConfig | dict | None = None,
                   dim: int | None = None) -> TransMem:
    """构建 TransMem. 三选一: 传 config 对象 / 传 config_path / 用默认."""
    if config is not None:
        cfg = TransMemConfig(**config) if isinstance(config, dict) else config
    elif config_path is not None:
        cfg = TransMemConfig.from_json(config_path)
    else:
        cfg = TransMemConfig()
    if dim is not None:
        cfg.dim = dim
    return TransMem(cfg)
