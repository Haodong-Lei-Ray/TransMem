"""TransMem: L 层 Qwen3 同款 decoder block + 零初始化读出, 回归记忆偏置 MS.

TransMem 自己就是一个小自回归 decoder (docs/version2/transmem正常化修改意见.md):
  序列   X = [HM_stu ; HQ_stu_1 .. HQ_stu_M]   [B, N+M, dim]
         L 层 Qwen3 block (causal, RoPE) — 位置 i 的 query 因果地看到
         {HM_1..N, HQ_1..i}, 而不是只看固定记忆 + 孤立当前查询.
  读出   final_norm -> out_proj(零初始化):
         训练 (teacher-forcing 并行): return_all_queries=True -> MS_1..M [B, M, dim]
         推理 (token-by-token):       past_key_values=DynamicCache 增量前向,
                                      每步只喂新 HQ_i [B,1,dim], 读末位 -> MS_i [B, dim]
  纠正   HQ'_i = HQ_stu_i + a * MS_i            (逐元素相加)

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
    causal: bool = True            # 因果 mask(查询看历史) vs 全双向
    pos_mode: str = "rope"         # none | rope | learned (位置注入方式)
    n_mem: int = 4                 # 记忆分段数 N (query 槽从第 N 位开始)
    hm_mode: str = "floor"         # HM 取位公式: floor=历史分段 | frac=分数取位(池化消融).
                                   # 模型本身不用; 记录进 ckpt 供 rollout/evaluate 对齐 stage0
    max_queries: int = 256         # learned 位置 embedding 覆盖的最大 query 数 (长度 N+max_queries)
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
    """记忆偏置网络: L 层 Qwen3 block + 零初始化读出 (自身是小自回归 decoder).

    forward(X) -> MS: 三种用法 (见 forward docstring):
      默认            X [B, S, dim] -> 读末位 [B, dim]
      并行读全部 query return_all_queries=True -> [B, S-N, dim]  (训练)
      增量 KV cache    past_key_values + use_cache=True          (token-by-token 推理)
    correct(MS, HQ_stu) -> HQ':  HQ'_i = HQ_stu_i + a*MS_i (广播逐元素).
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

        # learned 位置: 零初始化 -> 初始不扰动输入 (覆盖 N 记忆槽 + max_queries 个 query 位)
        if config.pos_mode == "learned":
            self.pos_emb = nn.Parameter(
                torch.zeros(1, config.n_mem + config.max_queries, dim))
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
    def forward(self, X: torch.Tensor,
                past_key_values=None,
                use_cache: bool = False,
                return_all_queries: bool = False) -> torch.Tensor:
        """X: [B, S, dim] -> MS.

        三种用法:
          1) 默认: 一次前向, 读末位 -> [B, dim].
             X=[HM; HQ_1..i] 时末位恰是当前查询 (rollout 无 cache 的朴素版).
          2) return_all_queries=True: 读第 n_mem 位起的全部 query 位 -> [B, S-N, dim].
             X=[HM; HQ_1..M] 一次并行前向, causal mask 保证位置 i 只见 {HM, HQ_1..i}
             (训练侧 teacher-forcing 并行, 与逐步推理语义一致).
          3) past_key_values=transformers DynamicCache, use_cache=True: 增量前向.
             首次喂 [HM; HQ_1] prefill, 之后每步只喂当前 HQ_i [B,1,dim];
             K/V 就地累积在 cache 里, 读末位 -> [B, dim] (token-by-token 推理).
        """
        B, S, _ = X.shape
        past_len = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
        if past_len > 0:
            assert self.config.causal, "KV cache 增量前向要求 causal=True (双向注意力无法增量)"
        assert not (return_all_queries and past_len > 0), \
            "return_all_queries 是整段并行读出, 与增量 cache 不同时使用"
        h = X

        # 位置 id: rope=全局位置 past_len..past_len+S-1; none/learned=全 0(RoPE 恒等旋转)
        if self.config.pos_mode == "rope":
            pos_ids = torch.arange(past_len, past_len + S, device=X.device, dtype=torch.long)
        else:
            pos_ids = torch.zeros(S, device=X.device, dtype=torch.long)
        pos_ids = pos_ids.unsqueeze(0).expand(B, -1)            # [B, S]
        cache_position = torch.arange(past_len, past_len + S,
                                      device=X.device, dtype=torch.long)

        if self.config.pos_mode == "learned":
            end = past_len + S
            assert end <= self.pos_emb.shape[1], (
                f"learned 位置 embedding 长度 {self.pos_emb.shape[1]} < 序列 {end} "
                f"(调大 config.max_queries)")
            h = h + self.pos_emb[:, past_len:end, :].to(h.dtype)

        cos, sin = self.rotary(h, pos_ids)                      # [B, S, head_dim]
        mask = build_additive_causal_mask(S, h.dtype, h.device, self.config.causal,
                                          past_len=past_len)    # [1,1,S,past+S] | None

        for block in self.blocks:
            out = block(
                hidden_states=h,
                attention_mask=mask,
                position_ids=pos_ids,
                position_embeddings=(cos, sin),
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            h = out[0] if isinstance(out, tuple) else out

        h = self.final_norm(h)
        if return_all_queries:
            return self.out_proj(h[:, self.config.n_mem:, :])   # 全部 query 位 [B, S-N, dim]
        ms = self.out_proj(h[:, -1, :])                         # 读末位 = 当前查询 [B, dim]
        return ms

    # ── 读出: HQ' = HQ_stu + a*MS ────────────────────────────────────
    def correct(self, ms: torch.Tensor, hq_stu: torch.Tensor) -> torch.Tensor:
        """HQ'_i = HQ_stu_i + a*MS_i.  ms, hq_stu: [B, dim] 或 [B, M, dim] (逐元素)."""
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
