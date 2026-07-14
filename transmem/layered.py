"""TransMem-Layer: 冻结 LLM 的指定 D 个 decoder 层各插一个 1 层 TransMem.

与原 TransMem (final-hidden 单点偏置) 的区别: 偏置注入发生在 LLM 层栈**内部** —
第 l 层输出的查询位 hidden 被 `h + a*MS^l` 纠正后才进入第 l+1 层, 使记忆纠正参与
后续层的计算 (深注入). 每层一个独立的 1 层 Qwen3-block TransMem, 读该层的
[HM^l ; HQ^l_1..i] 因果序列, 回归该层的记忆偏置 MS^l. LLM 全程冻结.

训练两条路:
  train_inloop.py (v3.2, 在环): LLM 层进训练环, teacher_forced_forward 单次并行前向
    与逐步 rollout 因果等价, 上层输入零失配 — P6 离线方案 D≥6 坍塌后的正解尝试.
  train_layered.py (v3 P6, 离线特征, 已被排除 — 存档):
  - 逐层目标: 把该层查询 hidden 推向教师同层 hidden (残差回归到 HQ_tea^l);
  - 上层输入 α∈[0,1] 插值增强 `h_in = α·HQ_tea + (1−α)·HQ_stu` — 离线特征只对
    "最低插入层"严格有效 (下层一旦注入, 上层离线 HQ_stu 就与推理输入不符),
    插值教上层"无论下层把 hidden 推到 stu→tea 路径哪个位置, 都继续往教师推";
  - 最顶层 (LLM 末层) 走冻结 final_norm + lm_head 的 forward-KL (对齐原配方).

推理 (LayeredRollout): 在 model.model.layers[l] 上挂 forward hook, prefill 时从该层
输出取 HM^l 并纠正末位查询, 之后每步增量纠正新 token 位; 每层 TransMem 各持
KV cache (查询 i 因果地看 {HM^l, HQ^l_1..i}).

ckpt config 带 "layered": true, evaluate.py 以此自动分发.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn as nn

from .transmem import TransMemConfig, TransMem


# ═══════════════════════════════════════════════════════════════════════
# 注入窗口
# ═══════════════════════════════════════════════════════════════════════

def resolve_inject_layers(
    n_layers: int,
    depth: int | None = None,
    stop: int | None = None,
    explicit: str | list[int] | tuple[int, ...] | None = None,
) -> list[int]:
    """Resolve the 0-based LLM layers that receive TransMem blocks.

    ``stop`` is an exclusive upper bound.  For example, a 36-layer model with
    ``depth=4, stop=32`` injects layers ``[28, 29, 30, 31]``.  Omitting
    ``stop`` preserves the original behavior of injecting the final ``depth``
    layers.  ``explicit`` remains available for non-contiguous ablations.
    """
    if n_layers <= 0:
        raise ValueError(f"n_layers 必须为正数，收到 {n_layers}")

    if explicit is not None:
        if depth is not None or stop is not None:
            raise ValueError("explicit 与 depth/stop 二选一")
        if isinstance(explicit, str):
            values = [part.strip() for part in explicit.split(",") if part.strip()]
        else:
            values = list(explicit)
        if not values:
            raise ValueError("explicit 注入层不能为空")
        try:
            layers = [int(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"explicit 注入层必须为整数: {explicit!r}") from exc
        if len(set(layers)) != len(layers):
            raise ValueError(f"explicit 注入层不能重复: {layers}")
        if any(layer < 0 or layer >= n_layers for layer in layers):
            raise ValueError(
                f"explicit 注入层超出合法范围 [0, {n_layers}): {layers}"
            )
        return sorted(layers)

    if depth is None:
        raise ValueError("depth 必填（或改用 explicit）")
    if depth <= 0:
        raise ValueError(f"depth 必须为正数，收到 {depth}")
    resolved_stop = n_layers if stop is None else stop
    if resolved_stop <= 0 or resolved_stop > n_layers:
        raise ValueError(
            f"stop 必须在合法范围 [1, {n_layers}]，收到 {resolved_stop}"
        )
    if depth > resolved_stop:
        raise ValueError(
            f"depth={depth} 不能大于 stop={resolved_stop}，否则窗口越过第 0 层"
        )
    return list(range(resolved_stop - depth, resolved_stop))


# ═══════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LayeredConfig:
    """块架构字段与 TransMemConfig 对齐 (每块 depth=block_depth, 默认 1 层);
    inject_layers = 注入的 LLM decoder 层号 (0-based, 升序)."""

    dim: int = 2560
    block_depth: int = 1
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    attn_impl: str = "sdpa"

    causal: bool = True
    pos_mode: str = "rope"
    n_mem: int = 4
    hm_mode: str = "floor"
    max_queries: int = 256
    final_norm: bool = True
    zero_init_out: bool = True

    a_init: float = 1.0
    learnable_a: bool = False

    inject_layers: list[int] = field(default_factory=lambda: [35])
    layered: bool = True          # ckpt 识别标记 (evaluate.py 分发用)

    def __post_init__(self):
        self.inject_layers = sorted(int(l) for l in self.inject_layers)
        assert self.inject_layers, "inject_layers 不能为空"
        self.layered = True

    @classmethod
    def from_json(cls, path: str | Path) -> "LayeredConfig":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "LayeredConfig":
        names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items()
                  if not k.startswith("_") and k in names}
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    def block_config(self) -> TransMemConfig:
        return TransMemConfig(
            dim=self.dim, depth=self.block_depth, num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads, head_dim=self.head_dim,
            intermediate_size=self.intermediate_size, rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range, attn_impl=self.attn_impl,
            causal=self.causal, pos_mode=self.pos_mode, n_mem=self.n_mem,
            hm_mode=self.hm_mode, max_queries=self.max_queries,
            final_norm=self.final_norm, zero_init_out=self.zero_init_out,
            a_init=self.a_init, learnable_a=self.learnable_a, warm_start=False)


# ═══════════════════════════════════════════════════════════════════════
# 模块: D 个独立 1 层 TransMem
# ═══════════════════════════════════════════════════════════════════════

class TransMemLayered(nn.Module):
    """ModuleDict{layer_idx: TransMem(depth=block_depth)}. 各块零初始化读出 →
    初始全层恒等 (bias=0), 与冻结 LLM 完全等价."""

    def __init__(self, config: LayeredConfig):
        super().__init__()
        self.config = config
        bcfg = config.block_config()
        self.blocks = nn.ModuleDict(
            {str(l): TransMem(bcfg) for l in config.inject_layers})

    def block(self, layer_idx: int) -> TransMem:
        return self.blocks[str(layer_idx)]

    def forward(self, hm: torch.Tensor, h_in: torch.Tensor) -> torch.Tensor:
        """训练用并行前向: 全部块跑一遍 (DDP 梯度同步要求单一 forward 覆盖全部参数).

        hm   [B, D, N, dim]  各注入层的记忆槽
        h_in [B, D, M, dim]  各注入层的查询输入 (训练侧可为 α 插值)
        ->   ms  [B, D, M, dim]  各层记忆偏置 (causal 并行, 与 TransMem 训练语义一致)
        """
        out = []
        for k, l in enumerate(self.config.inject_layers):
            X = torch.cat([hm[:, k], h_in[:, k]], dim=1)        # [B, N+M, dim]
            out.append(self.blocks[str(l)](X, return_all_queries=True))
        return torch.stack(out, dim=1)

    @property
    def inject_layers(self) -> list[int]:
        return self.config.inject_layers

    @property
    def top_layer(self) -> int:
        return self.config.inject_layers[-1]

    @property
    def lowest_layer(self) -> int:
        return self.config.inject_layers[0]

    def num_params(self, trainable_only: bool = False) -> int:
        ps = (p for p in self.parameters() if (p.requires_grad or not trainable_only))
        return sum(p.numel() for p in ps)


# ═══════════════════════════════════════════════════════════════════════
# 推理: hook 在环逐步解码
# ═══════════════════════════════════════════════════════════════════════

class LayeredRollout:
    """冻结 LLM + TransMemLayered 贪心/采样逐步解码.

    每个注入层挂 forward hook:
      prefill: 该层输出取 HM^l (len_cl 内 N 个槽位) + 末位查询 HQ^l_1,
               TransMem prefill [HM;HQ_1] → MS^l_1, 末位 hidden += a*MS;
      decode : 每步新 token 位即当前查询 HQ^l_i, 增量喂 TransMem (KV cache),
               该位 hidden += a*MS^l_i.
    纠正后的 hidden 继续流入上层 → 上层的 HQ/KV 都基于纠正后的流 (深注入语义).
    LLM 侧 KV cache: 查询位的 K/V 由纠正后 hidden 计算 (与训练语义一致);
    上下文位不受影响 (bias 只加在查询位).
    """

    def __init__(self, model, tokenizer, device, layered: TransMemLayered,
                 dtype=torch.bfloat16):
        from .extract_features import resolve_eos_ids
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.layered = layered
        self.dtype = dtype
        self.n_mem = layered.config.n_mem
        self.hm_mode = layered.config.hm_mode
        self.eos_ids = resolve_eos_ids(model)
        n_layers = len(model.model.layers)
        assert layered.top_layer < n_layers, (
            f"inject_layers 最大 {layered.top_layer} 超出 LLM 层数 {n_layers}")

    # ── 可测试核心: 纯 token-id 入口 (CPU 测试不依赖 tokenizer) ─────────
    @torch.no_grad()
    def generate_from_ids(self, cq_ids: torch.Tensor, len_cl: int, max_new: int,
                          sample: bool = False, temperature: float = 1.0) -> list[int]:
        from transformers.cache_utils import DynamicCache
        from .extract_features import hm_positions

        hm_idx = hm_positions(len_cl, self.n_mem, self.hm_mode)
        hm_idx_t = torch.tensor(hm_idx, device=cq_ids.device, dtype=torch.long)
        mem_dtype = next(self.layered.parameters()).dtype

        state: dict[int, DynamicCache] = {}
        phase = {"mode": "prefill"}

        def mk_hook(layer_idx: int):
            block = self.layered.block(layer_idx)

            def hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out       # [1, S, dim]
                if phase["mode"] == "prefill":
                    hm = h[0, hm_idx_t, :]                          # [N, dim]
                    hq = h[:, -1:, :]                               # [1, 1, dim]
                    cache = state[layer_idx] = DynamicCache()
                    X = torch.cat([hm.unsqueeze(0), hq], dim=1).to(mem_dtype)
                    ms = block(X, past_key_values=cache, use_cache=True)   # [1, dim]
                else:
                    hq = h[:, -1:, :]
                    ms = block(hq.to(mem_dtype),
                               past_key_values=state[layer_idx], use_cache=True)
                h = h.clone()
                h[:, -1, :] = h[:, -1, :] + block.a.to(h.dtype) * ms.to(h.dtype)
                if isinstance(out, tuple):
                    return (h,) + tuple(out[1:])
                return h

            return hook

        handles = [self.model.model.layers[l].register_forward_hook(mk_hook(l))
                   for l in self.layered.inject_layers]
        ans_ids: list[int] = []
        try:
            # base-model 调用: 不算全长 logits (122k 上下文时 CausalLM 全长 logits ~37GB);
            # last_hidden_state 已过 final_norm 且含末层 hook 偏置 → 手动过 lm_head.
            # ones mask 显式传: transformers 4.57.6 mask=None 不走 is_causal skip,
            # 会物化 S×S mask (probe 10216593: 125k 峰值 57.4GB vs 16.8GB).
            out = self.model.model(input_ids=cq_ids,
                                   attention_mask=torch.ones_like(cq_ids),
                                   use_cache=True)
            past = out.past_key_values
            logits = self.model.lm_head(out.last_hidden_state[:, -1, :])   # [1, vocab]
            phase["mode"] = "decode"
            for _ in range(max_new):
                if sample:
                    probs = torch.softmax(logits.float() / max(temperature, 1e-6), dim=-1)
                    nxt = torch.multinomial(probs, 1)[0]
                else:
                    nxt = logits.argmax(dim=-1)
                tok_id = int(nxt.item())
                ans_ids.append(tok_id)
                if tok_id in self.eos_ids:
                    break
                step = self.model.model(input_ids=nxt.view(1, 1),
                                        past_key_values=past, use_cache=True)
                past = step.past_key_values
                logits = self.model.lm_head(step.last_hidden_state[:, -1, :])
        finally:
            for hd in handles:
                hd.remove()
        return ans_ids

    # ── 在环教师强制前向 (v3.2 训练用, 梯度可通; 无 no_grad) ─────────────
    def teacher_forced_forward(self, full_ids: torch.Tensor, len_cl: int,
                               len_cq: int, M: int) -> torch.Tensor:
        """full_ids = [CQ ; A_1..M-1] (与 stage0 student_forward 同构) 的单次并行前向,
        注入位 = M 个答案生成位 qpos = len_cq-1 .. len_cq+M-2 (prefill 末位 + 之后每个
        decode 位), 与 generate_from_ids 的注入集合逐位一致; LLM 因果注意力 + 块内
        因果序列 [HM; HQ_1..M] ⇒ 与逐步 rollout 数学等价 (test_inloop [2] 贪心一致性).

        梯度经真实 LLM 上层流回各块 (LLM 参数冻结, 只在注入位之后建图) — 这就是
        "LLM 层进训练环": 上层看到的是下层注入后的真实分布, 无离线近似.
        返回 h_q [M, dim]: 查询位 post-final_norm hidden (含全部注入效应)."""
        from .extract_features import hm_positions

        assert full_ids.shape[0] == 1 and full_ids.shape[1] == len_cq + M - 1, \
            (full_ids.shape, len_cq, M)
        dev = full_ids.device
        hm_idx = torch.tensor(hm_positions(len_cl, self.n_mem, self.hm_mode),
                              device=dev, dtype=torch.long)
        qpos = torch.arange(len_cq - 1, len_cq + M - 1, device=dev, dtype=torch.long)
        mem_dtype = next(self.layered.parameters()).dtype

        def mk_hook(layer_idx: int):
            block = self.layered.block(layer_idx)

            def hook(_mod, _inp, out):
                h0 = out[0] if isinstance(out, tuple) else out      # [1, T, dim]
                hm = h0[:, hm_idx, :]                               # [1, N, dim]
                hq = h0[:, qpos, :]                                 # [1, M, dim]
                X = torch.cat([hm, hq], dim=1).to(mem_dtype)
                ms = block(X, return_all_queries=True)              # [1, M, dim]
                # 读 h0、写 clone: 无原地读写混叠, autograd 干净
                h = h0.clone()
                h[:, qpos, :] = hq + block.a.to(h0.dtype) * ms.to(h0.dtype)
                if isinstance(out, tuple):
                    return (h,) + tuple(out[1:])
                return h

            return hook

        handles = [self.model.model.layers[l].register_forward_hook(mk_hook(l))
                   for l in self.layered.inject_layers]
        try:
            out = self.model.model(input_ids=full_ids,
                                   attention_mask=torch.ones_like(full_ids),
                                   use_cache=False)
        finally:
            for hd in handles:
                hd.remove()
        return out.last_hidden_state[0, qpos, :]                    # [M, dim] post-norm

    # ── evaluate.py 接口 (与 OnPolicyRollout.student_rollout 同签名) ────
    @torch.no_grad()
    def student_rollout(self, _mem_unused, context_long: str, question: str,
                        max_new: int, sample: bool = False, temperature: float = 1.0):
        from .extract_features import build_chat_prompt_ids
        cq_ids = build_chat_prompt_ids(self.tok, context_long, question, self.device)
        len_cl = self.tok(context_long, return_tensors="pt",
                          add_special_tokens=False).input_ids.shape[1]
        ans_ids = self.generate_from_ids(cq_ids, len_cl, max_new,
                                         sample=sample, temperature=temperature)
        return None, None, ans_ids
