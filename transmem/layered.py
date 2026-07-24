"""TransMem-Layer: 冻结 LLM 的指定 D 个 decoder 层各插一个 1 层 TransMem.

与原 TransMem (final-hidden 单点偏置) 的区别: 偏置注入发生在 LLM 层栈**内部** —
第 l 层输出的查询位 hidden 被 `h + g*MS^l` 纠正后才进入第 l+1 层, 使记忆纠正参与
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

推理 (LayeredRollout): 在 model.model.layers[l] 上挂 forward hook. 默认从该层输出
读取 HM/HQ; 可选 transmem_before 消融改为从该层输入读取 HM/HQ. 两种模式都把
纠正量加到该层输出, 且每层 TransMem 各持 KV cache.

ckpt config 带 "layered": true, evaluate.py 以此自动分发.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

from .transmem import TransMemConfig, TransMemOutput, TransMem


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

    gate_mode: str = "constant"
    gate_granularity: str = "token_scalar"
    gate_max: float = 2.0
    gate_scale: float | None = None
    gate_shift: float = 0.0
    gate_temperature: float = 1.0
    gate_init: float = 1.0

    inject_layers: list[int] = field(default_factory=lambda: [35])
    transmem_before: bool = False
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
        data = asdict(self)
        # Keep the serialized default byte-for-byte compatible with legacy
        # checkpoints.  A before-mode checkpoint records the opt-in explicitly.
        if not self.transmem_before:
            data.pop("transmem_before")
        if self.gate_mode == "shifted_sigmoid":
            data.pop("gate_max")
        else:
            data.pop("gate_scale")
            data.pop("gate_shift")
        if self.gate_mode == "constant":
            for name in (
                "gate_mode",
                "gate_granularity",
                "gate_max",
                "gate_temperature",
                "gate_init",
            ):
                data.pop(name)
        return data

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
            a_init=self.a_init, learnable_a=self.learnable_a, warm_start=False,
            gate_mode=self.gate_mode, gate_granularity=self.gate_granularity,
            gate_max=self.gate_max, gate_scale=self.gate_scale,
            gate_shift=self.gate_shift,
            gate_temperature=self.gate_temperature,
            gate_init=self.gate_init)


@dataclass
class LayeredOutput:
    """Aligned memory proposals from all configured injection layers."""

    ms: torch.Tensor
    gate: torch.Tensor

    @property
    def delta(self) -> torch.Tensor:
        return self.gate * self.ms

    def layer(self, index: int) -> TransMemOutput:
        return TransMemOutput(ms=self.ms[:, index], gate=self.gate[:, index])


def replace_query_positions(
    hidden: torch.Tensor,
    qpos: torch.Tensor,
    corrected: torch.Tensor,
) -> torch.Tensor:
    """Return a copy with only answer-query positions replaced.

    Keeping this operation in one small seam makes it testable that context,
    HM and question positions are never touched by layered injection.
    """
    if hidden.ndim != 3 or corrected.ndim != 3 or qpos.ndim != 1:
        raise ValueError(
            "hidden/corrected/qpos 必须分别是 [B,T,D]/[B,M,D]/[M]")
    if (hidden.shape[0] != corrected.shape[0]
            or hidden.shape[2] != corrected.shape[2]
            or corrected.shape[1] != qpos.numel()):
        raise ValueError(
            f"qpos 替换 shape 不匹配: hidden={tuple(hidden.shape)}, "
            f"corrected={tuple(corrected.shape)}, qpos={tuple(qpos.shape)}")
    if qpos.dtype != torch.long:
        raise ValueError("qpos 必须是 torch.long")
    result = hidden.clone()
    result.index_copy_(1, qpos, corrected)
    return result


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

    def forward(self, hm: torch.Tensor, h_in: torch.Tensor) -> LayeredOutput:
        """训练用并行前向: 全部块跑一遍 (DDP 梯度同步要求单一 forward 覆盖全部参数).

        hm   [B, D, N, dim]  各注入层的记忆槽
        h_in [B, D, M, dim]  各注入层的查询输入 (训练侧可为 α 插值)
        ->   ms  [B, D, M, dim]  各层记忆偏置 (causal 并行, 与 TransMem 训练语义一致)
        """
        proposals = []
        for k, l in enumerate(self.config.inject_layers):
            X = torch.cat([hm[:, k], h_in[:, k]], dim=1)        # [B, N+M, dim]
            proposals.append(self.blocks[str(l)](X, return_all_queries=True))
        return LayeredOutput(
            ms=torch.stack([proposal.ms for proposal in proposals], dim=1),
            gate=torch.stack([proposal.gate for proposal in proposals], dim=1),
        )

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
      prefill: 特征源取 HM (len_cl 内 N 个槽位) + 末位查询 HQ_1,
               TransMem prefill [HM;HQ_1] → (MS^l_1,g^l_1), 末位 hidden += g*MS;
      decode : 每步新 token 位即当前查询 HQ^l_i, 增量喂 TransMem (KV cache),
               该位 hidden += g^l_i*MS^l_i.
    默认特征源是本层输出 H^l; transmem_before=True 时源改为本层输入 H^(l-1),
    但纠正目标始终是本层输出 H^l. 纠正后的 hidden 继续流入上层 → 上层的
    HQ/KV 都基于纠正后的流 (深注入语义).
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
        self.last_gate_trace = None
        n_layers = len(model.model.layers)
        assert layered.top_layer < n_layers, (
            f"inject_layers 最大 {layered.top_layer} 超出 LLM 层数 {n_layers}")

    @torch.no_grad()
    def capture_memory_from_ids(
        self,
        context_ids: torch.Tensor,
        len_cl: int,
    ) -> dict[int, torch.Tensor]:
        """Prefill one independent context chunk and retain N HM per layer.

        This is the long-context overflow seam.  It deliberately runs the
        frozen student without TransMem injection and without an LLM KV cache,
        so different overflow chunks cannot leak cache state into one another.
        The returned tensors are ordered exactly like ``inject_layers`` and
        use the checkpoint's historical HM-position convention.
        """
        from .extract_features import hm_positions

        if context_ids.ndim != 2 or context_ids.shape[0] != 1:
            raise ValueError(
                f"context_ids 必须是 [1,T]，收到 {tuple(context_ids.shape)}")
        if len_cl < 1:
            raise ValueError(f"len_cl 必须为正数，收到 {len_cl}")
        model_limit = int(getattr(
            self.model.config,
            "max_position_embeddings",
            context_ids.shape[1],
        ))
        if context_ids.shape[1] > model_limit:
            raise ValueError(
                f"overflow prompt 长度 {context_ids.shape[1]} 超出模型上限 "
                f"{model_limit}")
        positions = hm_positions(len_cl, self.n_mem, self.hm_mode)
        if positions[-1] >= context_ids.shape[1]:
            raise ValueError(
                f"HM 位置 {positions[-1]} 超出输入长度 {context_ids.shape[1]} "
                f"(len_cl={len_cl})")
        index = torch.tensor(
            positions, device=context_ids.device, dtype=torch.long)
        mem_dtype = next(self.layered.parameters()).dtype
        captured: dict[int, torch.Tensor] = {}

        def mk_hook(layer_idx: int):
            def hook(_module, inputs, output):
                target = output[0] if isinstance(output, tuple) else output
                source = (
                    inputs[0]
                    if self.layered.config.transmem_before
                    else target
                )
                captured[layer_idx] = (
                    source[0].index_select(0, index).detach().to(mem_dtype))

            return hook

        handles = [
            self.model.model.layers[layer].register_forward_hook(mk_hook(layer))
            for layer in self.layered.inject_layers
        ]
        try:
            output = self.model.model(
                input_ids=context_ids,
                attention_mask=torch.ones_like(context_ids),
                use_cache=False,
            )
            missing = set(self.layered.inject_layers) - set(captured)
            if missing:
                raise RuntimeError(
                    f"overflow prefill hooks 未捕获层: {sorted(missing)}")
            return {
                layer: captured[layer]
                for layer in self.layered.inject_layers
            }
        finally:
            for handle in handles:
                handle.remove()
            if "output" in locals():
                del output

    @torch.no_grad()
    def capture_memory_from_context(
        self,
        context: str,
    ) -> dict[int, torch.Tensor]:
        """Render and prefill one overflow context, with no real question."""
        from .extract_features import build_chat_prompt_ids

        context_ids = build_chat_prompt_ids(
            self.tok, context, "", self.device, thinking=False)
        len_cl = self.tok(
            context,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.shape[1]
        return self.capture_memory_from_ids(context_ids, len_cl)

    def _normalize_overflow_memory(
        self,
        overflow_memory: Mapping[int, torch.Tensor] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[int, torch.Tensor]:
        """Validate a complete per-layer overflow-memory bundle."""
        if overflow_memory is None:
            return {}
        expected = set(self.layered.inject_layers)
        actual = set(overflow_memory)
        if actual != expected:
            raise ValueError(
                "overflow_memory 层集合必须与 inject_layers 完全一致: "
                f"expected={sorted(expected)}, actual={sorted(actual)}")
        normalized: dict[int, torch.Tensor] = {}
        for layer in self.layered.inject_layers:
            memory = overflow_memory[layer]
            if memory.ndim != 2 or memory.shape[1] != self.layered.config.dim:
                raise ValueError(
                    f"overflow_memory[{layer}] 必须是 [K*N,{self.layered.config.dim}]，"
                    f"收到 {tuple(memory.shape)}")
            normalized[layer] = memory.detach().to(device=device, dtype=dtype)
        slot_counts = {memory.shape[0] for memory in normalized.values()}
        if len(slot_counts) != 1:
            raise ValueError(
                "所有注入层的 overflow memory 槽数必须一致，收到 "
                f"{sorted(slot_counts)}")
        return normalized

    # ── 可测试核心: 纯 token-id 入口 (CPU 测试不依赖 tokenizer) ─────────
    @torch.no_grad()
    def generate_from_ids(self, cq_ids: torch.Tensor, len_cl: int, max_new: int,
                          sample: bool = False, temperature: float = 1.0,
                          collect_gate_diagnostics: bool = False,
                          return_log_probs: bool = False,
                          overflow_memory: Mapping[int, torch.Tensor] | None = None):
        """Generate answer ids, optionally returning behavior-policy log-probs.

        The default remains the historical ``list[int]`` return.  GRPO opts in
        to ``(ids, log_probs)`` so importance ratios use exactly the tempered
        distribution that sampled each token.
        """
        from transformers.cache_utils import DynamicCache
        from .extract_features import hm_positions

        hm_idx = hm_positions(len_cl, self.n_mem, self.hm_mode)
        hm_idx_t = torch.tensor(hm_idx, device=cq_ids.device, dtype=torch.long)
        mem_dtype = next(self.layered.parameters()).dtype
        retained_overflow = self._normalize_overflow_memory(
            overflow_memory, device=cq_ids.device, dtype=mem_dtype)

        state: dict[int, DynamicCache] = {}
        phase = {"mode": "prefill"}
        trace = {
            str(layer): {"gate": [], "ms_norm": [], "delta_norm": []}
            for layer in self.layered.inject_layers
        } if collect_gate_diagnostics else None

        def mk_hook(layer_idx: int):
            block = self.layered.block(layer_idx)

            def hook(_mod, _inp, out):
                target = out[0] if isinstance(out, tuple) else out  # H^l [1,S,D]
                source = (_inp[0] if self.layered.config.transmem_before
                          else target)                               # H^(l-1) or H^l
                if phase["mode"] == "prefill":
                    hm_limit = source[0, hm_idx_t, :]               # [N, dim]
                    hm = (
                        torch.cat(
                            [retained_overflow[layer_idx], hm_limit], dim=0)
                        if retained_overflow else hm_limit
                    )                                               # [(K+1)N, dim]
                    hq_source = source[:, -1, :]                    # [1, dim]
                    cache = state[layer_idx] = DynamicCache()
                    X = torch.cat(
                        [hm.unsqueeze(0), hq_source.unsqueeze(1)], dim=1,
                    ).to(mem_dtype)
                    proposal = block(X, past_key_values=cache, use_cache=True)
                else:
                    hq_source = source[:, -1, :]
                    proposal = block(hq_source.unsqueeze(1).to(mem_dtype),
                                     past_key_values=state[layer_idx], use_cache=True)
                if trace is not None:
                    layer_trace = trace[str(layer_idx)]
                    layer_trace["gate"].append(float(proposal.gate.squeeze().float()))
                    layer_trace["ms_norm"].append(
                        float(proposal.ms.float().norm(dim=-1).mean()))
                    layer_trace["delta_norm"].append(
                        float(proposal.delta.float().norm(dim=-1).mean()))
                h = target.clone()
                h[:, -1, :] = block.correct(target[:, -1, :], proposal)
                if isinstance(out, tuple):
                    return (h,) + tuple(out[1:])
                return h

            return hook

        handles = [self.model.model.layers[l].register_forward_hook(mk_hook(l))
                   for l in self.layered.inject_layers]
        ans_ids: list[int] = []
        chosen_log_probs: list[torch.Tensor] = []
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
            for token_index in range(max_new):
                if sample:
                    scaled_logits = logits.float() / max(temperature, 1e-6)
                    if return_log_probs:
                        log_distribution = torch.log_softmax(scaled_logits, dim=-1)
                        probs = log_distribution.exp()
                    else:
                        probs = torch.softmax(scaled_logits, dim=-1)
                    nxt = torch.multinomial(probs, 1)[0]
                else:
                    nxt = logits.argmax(dim=-1)
                    if return_log_probs:
                        scaled_logits = logits.float() / max(temperature, 1e-6)
                        log_distribution = torch.log_softmax(scaled_logits, dim=-1)
                if return_log_probs:
                    chosen_log_probs.append(
                        log_distribution.gather(-1, nxt.view(-1, 1)).squeeze())
                tok_id = int(nxt.item())
                ans_ids.append(tok_id)
                if tok_id in self.eos_ids or token_index + 1 >= max_new:
                    break
                step = self.model.model(input_ids=nxt.view(1, 1),
                                        past_key_values=past, use_cache=True)
                past = step.past_key_values
                logits = self.model.lm_head(step.last_hidden_state[:, -1, :])
        finally:
            for hd in handles:
                hd.remove()
        self.last_gate_trace = ({"token_ids": list(ans_ids), "layers": trace}
                                if trace is not None else None)
        if return_log_probs:
            values = (torch.stack(chosen_log_probs)
                      if chosen_log_probs else torch.empty(
                          0, device=cq_ids.device, dtype=torch.float32))
            return ans_ids, values
        return ans_ids

    # ── 在环教师强制前向 (v3.2 训练用, 梯度可通; 无 no_grad) ─────────────
    def teacher_forced_forward(self, full_ids: torch.Tensor, len_cl: int,
                               len_cq: int, M: int,
                               return_proposals: bool = False,
                               force_gate_one: bool = False):
        """Run teacher forcing with hooks scoped to the forward call.

        Callers using Hugging Face gradient checkpointing must instead use
        :meth:`teacher_forced_backward_context`, because decoder-layer
        recomputation happens later during ``backward``.
        """
        result, handles = self._teacher_forced_forward_with_handles(
            full_ids, len_cl, len_cq, M,
            return_proposals=return_proposals,
            force_gate_one=force_gate_one)
        try:
            return result
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def teacher_forced_backward_context(
        self, full_ids: torch.Tensor, len_cl: int, len_cq: int, M: int,
        return_proposals: bool = False, force_gate_one: bool = False,
    ):
        """Keep injection hooks alive through loss construction and backward."""
        result, handles = self._teacher_forced_forward_with_handles(
            full_ids, len_cl, len_cq, M,
            return_proposals=return_proposals,
            force_gate_one=force_gate_one)
        try:
            yield result
        finally:
            for handle in handles:
                handle.remove()

    def _teacher_forced_forward_with_handles(
        self, full_ids: torch.Tensor, len_cl: int, len_cq: int, M: int,
        return_proposals: bool = False, force_gate_one: bool = False,
    ):
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
        captured: dict[int, TransMemOutput] = {}

        def mk_hook(layer_idx: int):
            block = self.layered.block(layer_idx)

            def hook(_mod, _inp, out):
                target = out[0] if isinstance(out, tuple) else out  # H^l [1,T,D]
                source = (_inp[0] if self.layered.config.transmem_before
                          else target)                               # H^(l-1) or H^l
                hm = source[:, hm_idx, :]                           # [1, N, dim]
                hq_source = source[:, qpos, :]                      # [1, M, dim]
                X = torch.cat([hm, hq_source], dim=1).to(mem_dtype)
                proposal = block(X, return_all_queries=True)
                if return_proposals:
                    captured[layer_idx] = proposal
                applied = (TransMemOutput(
                    ms=proposal.ms, gate=torch.ones_like(proposal.gate))
                    if force_gate_one else proposal)
                # 提案可读 H^(l-1), 但残差基底始终是 H^l; 写 clone 避免混叠.
                hq_target = target[:, qpos, :]
                h = replace_query_positions(
                    target, qpos, block.correct(hq_target, applied))
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
            h_q = out.last_hidden_state[0, qpos, :]                 # [M, dim] post-norm
            if not return_proposals:
                result = h_q
            else:
                missing = set(self.layered.inject_layers) - set(captured)
                if missing:
                    raise RuntimeError(
                        f"teacher-forced hooks 未捕获层: {sorted(missing)}")
                ordered = [captured[layer] for layer in self.layered.inject_layers]
                result = h_q, LayeredOutput(
                    ms=torch.stack([proposal.ms for proposal in ordered], dim=1),
                    gate=torch.stack([proposal.gate for proposal in ordered], dim=1),
                )
            return result, handles
        except BaseException:
            for handle in handles:
                handle.remove()
            raise

    # ── evaluate.py 接口 (与 OnPolicyRollout.student_rollout 同签名) ────
    @torch.no_grad()
    def student_rollout(self, _mem_unused, context_long: str, question: str,
                        max_new: int, sample: bool = False, temperature: float = 1.0,
                        collect_gate_diagnostics: bool = False,
                        thinking: bool = False,
                        max_prompt_tokens: int | None = None,
                        overflow_memory: Mapping[int, torch.Tensor] | None = None):
        from .extract_features import (
            build_chat_prompt_ids, fit_context_to_prompt_budget)
        context_long = fit_context_to_prompt_budget(
            self.tok, context_long, question, max_prompt_tokens, thinking)
        cq_ids = build_chat_prompt_ids(
            self.tok, context_long, question, self.device, thinking=thinking)
        len_cl = self.tok(context_long, return_tensors="pt",
                          add_special_tokens=False).input_ids.shape[1]
        ans_ids = self.generate_from_ids(cq_ids, len_cl, max_new,
                                         sample=sample, temperature=temperature,
                                         collect_gate_diagnostics=collect_gate_diagnostics,
                                         overflow_memory=overflow_memory)
        return None, None, ans_ids
