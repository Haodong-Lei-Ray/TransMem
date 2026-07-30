#!/usr/bin/env python3
"""Compare Qwen3-4B inference scaling for base, Delta-Mem and TransMem.

The benchmark deliberately uses exact token counts and a fixed number of
greedy decode steps.  This avoids tokenizer-length drift and early-EOS timing
bias.  Each method is loaded and measured separately on the same CUDA device.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


PROJECT4 = Path("/mnt/petrelfs/leihaodong/Project4")
DELTA_ROOT = Path("/mnt/petrelfs/leihaodong/Project1/delta-Mem")
for path in (str(PROJECT4), str(DELTA_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from deltamem.core import reset_delta_mem_states  # noqa: E402
from deltamem.eval.common import load_delta_model_and_tokenizer  # noqa: E402
from transmem.layered import LayeredConfig, LayeredRollout, TransMemLayered  # noqa: E402


METHOD_LABELS = {
    "baseline": "Qwen3-4B baseline",
    "delta_mem": "Delta-Mem TSW (rank 8)",
    "transmem_d4_gate": "TransMem D=4 dynamic gate",
}
METHOD_COLORS = {
    "baseline": "#4C78A8",
    "delta_mem": "#F58518",
    "transmem_d4_gate": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--delta-adapter-dir", type=Path, required=True)
    parser.add_argument("--transmem-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=list(range(1000, 10001, 1000)),
    )
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measure-runs", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--flops-scope",
        choices=("total", "memory_only"),
        default="total",
        help=(
            "total: profile the complete forward passes; memory_only: use "
            "analytical FLOPs for Delta-Mem/TransMem only and report zero "
            "for the bare baseline"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def synchronize(device: str) -> None:
    torch.cuda.synchronize(device=device)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def choose_fill_tokens(tokenizer) -> list[int]:
    token_ids = tokenizer(
        "Memory benchmark sentence with stable tokenization. ",
        add_special_tokens=False,
    ).input_ids
    if not token_ids:
        raise RuntimeError("Tokenizer produced no fill tokens")
    return [int(token_id) for token_id in token_ids]


def make_input_ids(
    length: int,
    fill_tokens: list[int],
    device: str,
) -> torch.Tensor:
    fill = torch.tensor(fill_tokens, dtype=torch.long, device=device)
    repeats = (length + fill.numel() - 1) // fill.numel()
    return fill.repeat(repeats)[:length].unsqueeze(0)


@torch.inference_mode()
def fixed_decode_standard(
    model,
    input_ids: torch.Tensor,
    decode_tokens: int,
    *,
    before_run: Callable[[], None] | None = None,
    timing: bool = True,
) -> dict[str, float]:
    """Run one full prefill followed by exactly ``decode_tokens`` greedy tokens."""
    if before_run is not None:
        before_run()
    device = str(input_ids.device)
    attention_mask = torch.ones_like(input_ids)
    started = torch.cuda.Event(enable_timing=True) if timing else None
    prefill_ended = torch.cuda.Event(enable_timing=True) if timing else None
    ended = torch.cuda.Event(enable_timing=True) if timing else None
    if timing:
        synchronize(device)
        started.record()

    output = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past = output.past_key_values
    logits = model.lm_head(output.last_hidden_state[:, -1, :])
    if timing:
        prefill_ended.record()

    for token_index in range(decode_tokens):
        next_token = logits.argmax(dim=-1)
        if token_index + 1 == decode_tokens:
            break
        output = model.model(
            input_ids=next_token.view(1, 1),
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        logits = model.lm_head(output.last_hidden_state[:, -1, :])
    if timing:
        ended.record()
        synchronize(device)
        prefill_ms = started.elapsed_time(prefill_ended)
        total_ms = started.elapsed_time(ended)
        return {
            "prefill_seconds": prefill_ms / 1000.0,
            "decode_seconds": (total_ms - prefill_ms) / 1000.0,
            "total_seconds": total_ms / 1000.0,
        }
    return {}


@torch.inference_mode()
def fixed_decode_transmem(
    rollout: LayeredRollout,
    input_ids: torch.Tensor,
    decode_tokens: int,
    *,
    timing: bool = True,
) -> dict[str, float]:
    """Time LayeredRollout while marking the first backbone prefill boundary."""
    device = str(input_ids.device)
    started = torch.cuda.Event(enable_timing=True) if timing else None
    prefill_ended = torch.cuda.Event(enable_timing=True) if timing else None
    ended = torch.cuda.Event(enable_timing=True) if timing else None
    original_forward = rollout.model.model.forward
    first_backbone_call = True

    def timed_forward(*args, **kwargs):
        nonlocal first_backbone_call
        result = original_forward(*args, **kwargs)
        if timing and first_backbone_call:
            prefill_ended.record()
            first_backbone_call = False
        return result

    rollout.model.model.forward = timed_forward
    old_eos_ids = rollout.eos_ids
    rollout.eos_ids = set()  # force an identical decode length for every method
    try:
        if timing:
            synchronize(device)
            started.record()
        answer_ids = rollout.generate_from_ids(
            input_ids,
            len_cl=input_ids.shape[1],
            max_new=decode_tokens,
            sample=False,
        )
        if len(answer_ids) != decode_tokens:
            raise RuntimeError(
                f"TransMem generated {len(answer_ids)} tokens, expected {decode_tokens}")
        if timing:
            ended.record()
            synchronize(device)
            prefill_ms = started.elapsed_time(prefill_ended)
            total_ms = started.elapsed_time(ended)
            return {
                "prefill_seconds": prefill_ms / 1000.0,
                "decode_seconds": (total_ms - prefill_ms) / 1000.0,
                "total_seconds": total_ms / 1000.0,
            }
        return {}
    finally:
        rollout.model.model.forward = original_forward
        rollout.eos_ids = old_eos_ids


def aggregate_times(
    runs: list[dict[str, float]],
    decode_tokens: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in ("prefill_seconds", "decode_seconds", "total_seconds"):
        values = [run[field] for run in runs]
        result[f"{field}_median"] = statistics.median(values)
        result[f"{field}_p10"] = percentile(values, 0.10)
        result[f"{field}_p90"] = percentile(values, 0.90)
    result["decode_tokens_per_second"] = (
        decode_tokens / max(result["decode_seconds_median"], 1e-12)
    )
    return result


def gqa_sdpa_flops(
    query_shape,
    key_shape,
    value_shape,
    *_args,
    out_shape=None,
    **_kwargs,
) -> int:
    """Torch 2.6's stock SDPA formula rejects grouped-query attention.

    Qwen3 uses 32 query heads and 8 KV heads.  SDPA logically repeats each KV
    head across its query group, so both QK and AV work scale with the number
    of query heads.  Keep the stock counter's non-causal/full-matrix
    convention so all three methods remain directly comparable.
    """
    batch, query_heads, query_tokens, query_dim = query_shape
    key_batch, kv_heads, key_tokens, key_dim = key_shape
    value_batch, value_heads, value_tokens, value_dim = value_shape
    if batch != key_batch or batch != value_batch:
        raise ValueError("SDPA batch dimensions do not match")
    if kv_heads != value_heads or key_tokens != value_tokens:
        raise ValueError("SDPA key/value shapes do not match")
    if query_dim != key_dim or query_heads % kv_heads:
        raise ValueError(
            f"Unsupported GQA shapes: Q={query_shape}, K={key_shape}, V={value_shape}")
    return int(
        2 * batch * query_heads * query_tokens * key_tokens
        * (query_dim + value_dim)
    )


def count_flops(run: Callable[[], Any]) -> int:
    synchronize("cuda:0")
    sdpa_mapping = {
        torch.ops.aten._scaled_dot_product_efficient_attention: gqa_sdpa_flops,
        torch.ops.aten._scaled_dot_product_flash_attention: gqa_sdpa_flops,
        torch.ops.aten._scaled_dot_product_cudnn_attention: gqa_sdpa_flops,
    }
    with FlopCounterMode(
        display=False,
        custom_mapping=sdpa_mapping,
    ) as counter:
        run()
    synchronize("cuda:0")
    return int(counter.get_total_flops())


def delta_scan_flops(
    *,
    prompt_length: int,
    decode_tokens: int,
    num_layers: int,
    rank: int,
) -> int:
    """Arithmetic hidden inside the custom Triton affine-scan kernel.

    Per state token and layer:
      state@q + state@k                  4r²
      write/pred outer products          2r²
      three scaled state terms + 2 adds  5r²
    Total: 11r².  Projection matmuls are ordinary aten ops and are already
    included by FlopCounterMode, so they are intentionally excluded here.
    """
    processed_tokens = prompt_length + max(0, decode_tokens - 1)
    return int(11 * rank * rank * num_layers * processed_tokens)


def delta_memory_component_flops(
    *,
    prompt_length: int,
    decode_tokens: int,
    num_layers: int,
    rank: int,
    num_state_heads: int,
    hidden_size: int,
    query_out_features: int,
    key_out_features: int,
    value_out_features: int,
    output_out_features: int,
    delta_heads: set[str],
    rankwise_gates: bool,
    couple_lambda: bool,
) -> int:
    """Analytical FLOPs for Delta-Mem, excluding the frozen backbone.

    A multiply-add counts as two FLOPs.  The projections are evaluated once
    for every prompt token and every generated-input token.  The custom
    affine scan is added explicitly because it is opaque to PyTorch's FLOP
    counter.  Elementwise normalization, gates and nonlinearities are omitted.
    """
    processed_tokens = prompt_length + max(0, decode_tokens - 1)
    state_dim = rank * num_state_heads
    gate_dim = (rank if rankwise_gates else 1) * num_state_heads

    per_token_layer = 2 * hidden_size * (3 * state_dim)
    per_token_layer += 2 * hidden_size * gate_dim
    if not couple_lambda:
        per_token_layer += 2 * hidden_size * gate_dim
    head_outputs = {
        "q": query_out_features,
        "k": key_out_features,
        "v": value_out_features,
        "o": output_out_features,
    }
    per_token_layer += sum(
        2 * state_dim * head_outputs[head] for head in delta_heads
    )
    per_token_layer += 11 * rank * rank * num_state_heads
    return int(per_token_layer * num_layers * processed_tokens)


def transmem_memory_component_flops(
    *,
    config: LayeredConfig,
    decode_tokens: int,
) -> int:
    """Analytical FLOPs for all layered TransMem blocks only.

    Each injected block sees exactly N memory slots plus the first query at
    prefill, followed by one cached query per decode step.  Consequently this
    cost does not depend on the backbone context length.
    """
    if config.block_depth != 1:
        raise ValueError(
            "The memory-only formula currently requires block_depth=1, got "
            f"{config.block_depth}"
        )
    dim = int(config.dim)
    query_dim = int(config.num_heads * config.head_dim)
    kv_dim = int(config.num_kv_heads * config.head_dim)
    intermediate = int(config.intermediate_size)
    n_blocks = len(config.inject_layers)
    first_tokens = int(config.n_mem + 1)

    # Q/K/V/O plus Qwen3 gated-MLP projections (gate/up/down).
    linear_per_token = (
        2 * dim * query_dim
        + 4 * dim * kv_dim
        + 2 * query_dim * dim
        + 6 * dim * intermediate
    )
    # out_proj and the dynamic scalar gate; the latter is absent for a
    # constant gate, preserving compatibility with fixed-gate checkpoints.
    readout_per_query = 2 * dim * dim
    if config.gate_mode != "constant":
        readout_per_query += 2 * dim

    # Attention includes QK^T and AV: 4 * Hq * Q * K * head_dim.
    first = (
        linear_per_token * first_tokens
        + 4 * config.num_heads * first_tokens * first_tokens * config.head_dim
        + readout_per_query
    )
    later = 0
    for decode_index in range(1, decode_tokens):
        key_tokens = first_tokens + decode_index
        later += (
            linear_per_token
            + 4 * config.num_heads * key_tokens * config.head_dim
            + readout_per_query
        )
    return int(n_blocks * (first + later))


def load_base(args: argparse.Namespace):
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).to(args.device).eval()
    return model, tokenizer


def load_delta(args: argparse.Namespace):
    model, tokenizer, config = load_delta_model_and_tokenizer(
        model_path=args.model_path,
        adapter_dir=args.delta_adapter_dir,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    return model.eval(), tokenizer, config


def load_transmem(args: argparse.Namespace):
    model, tokenizer = load_base(args)
    checkpoint = torch.load(
        args.transmem_ckpt, map_location="cpu", weights_only=False)
    config = LayeredConfig.from_dict(checkpoint["config"])
    memory = TransMemLayered(config).to(
        device=args.device,
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
    )
    memory.load_state_dict(checkpoint["model_state_dict"], strict=True)
    memory.eval()
    rollout = LayeredRollout(
        model,
        tokenizer,
        torch.device(args.device),
        memory,
        torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
    )
    return model, tokenizer, memory, rollout, config


def release(*objects) -> None:
    del objects
    gc.collect()
    torch.cuda.empty_cache()
    synchronize("cuda:0")


def benchmark_method(
    *,
    method: str,
    tokenizer,
    context_lengths: list[int],
    warmup_runs: int,
    measure_runs: int,
    decode_tokens: int,
    device: str,
    timed_run: Callable[[torch.Tensor, bool], dict[str, float]],
    flop_run: Callable[[torch.Tensor], None] | None = None,
    extra_flops: Callable[[int], int] | None = None,
    analytical_flops: Callable[[int], int] | None = None,
) -> list[dict[str, Any]]:
    if (flop_run is None) == (analytical_flops is None):
        raise ValueError("Provide exactly one of flop_run or analytical_flops")
    fill_tokens = choose_fill_tokens(tokenizer)
    rows = []
    for context_length in context_lengths:
        input_ids = make_input_ids(context_length, fill_tokens, device)
        print(
            f"[bench] method={method} context={context_length} "
            f"warmup={warmup_runs} measure={measure_runs}",
            flush=True,
        )
        for _ in range(warmup_runs):
            timed_run(input_ids, True)
        measured = [timed_run(input_ids, True) for _ in range(measure_runs)]
        torch.cuda.reset_peak_memory_stats(device=device)
        if analytical_flops is not None:
            raw_flops = None
            corrected_flops = analytical_flops(context_length)
        else:
            assert flop_run is not None
            raw_flops = count_flops(lambda: flop_run(input_ids))
            corrected_flops = (
                raw_flops
                + (extra_flops(context_length) if extra_flops else 0)
            )
        row = {
            "method": method,
            "context_tokens": context_length,
            **aggregate_times(measured, decode_tokens),
            "flops_counter": raw_flops,
            "flops_total": corrected_flops,
            "tflops_total": corrected_flops / 1e12,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device=device) / (1024**3),
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
        del input_ids
    return rows


def save_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    flops_scope: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.2, 5.2))
    for method in METHOD_LABELS:
        selected = [row for row in rows if row["method"] == method]
        xs = [row["context_tokens"] / 1000 for row in selected]
        ys = [row["total_seconds_median"] for row in selected]
        low = [row["total_seconds_median"] - row["total_seconds_p10"] for row in selected]
        high = [row["total_seconds_p90"] - row["total_seconds_median"] for row in selected]
        plt.errorbar(
            xs, ys, yerr=[low, high], marker="o", linewidth=2, capsize=3,
            label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    plt.xlabel("Context length (K tokens)")
    plt.ylabel("End-to-end inference time (seconds)")
    plt.title("Qwen3-4B: 32-token generation latency vs. context length")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "latency_vs_context.png", dpi=180)
    plt.savefig(output_dir / "latency_vs_context.pdf")
    plt.close()

    baseline_times = {
        row["context_tokens"]: row["total_seconds_median"]
        for row in rows if row["method"] == "baseline"
    }
    plt.figure(figsize=(8.2, 5.2))
    for method in ("delta_mem", "transmem_d4_gate"):
        selected = [row for row in rows if row["method"] == method]
        plt.plot(
            [row["context_tokens"] / 1000 for row in selected],
            [
                row["total_seconds_median"]
                - baseline_times[row["context_tokens"]]
                for row in selected
            ],
            marker="o", linewidth=2, label=METHOD_LABELS[method],
            color=METHOD_COLORS[method])
    plt.axhline(0.0, color="#777777", linewidth=1, alpha=0.6)
    plt.xlabel("Context length (K tokens)")
    plt.ylabel("Additional latency over baseline (seconds)")
    plt.title("Qwen3-4B memory-method latency overhead")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "latency_overhead_vs_context.png", dpi=180)
    plt.savefig(output_dir / "latency_overhead_vs_context.pdf")
    plt.close()

    plt.figure(figsize=(8.2, 5.2))
    for method in ("delta_mem", "transmem_d4_gate"):
        selected = [row for row in rows if row["method"] == method]
        plt.plot(
            [row["context_tokens"] / 1000 for row in selected],
            [
                (
                    row["flops_total"] / 1e9
                    if flops_scope == "memory_only"
                    else row["tflops_total"]
                )
                for row in selected
            ],
            marker="o", linewidth=2, label=METHOD_LABELS[method],
            color=METHOD_COLORS[method])
    plt.xlabel("Context length (K tokens)")
    if flops_scope == "memory_only":
        plt.ylabel("Memory-component FLOPs (GFLOPs)")
        plt.title("Qwen3-4B memory-component FLOPs vs. context length")
    else:
        plt.ylabel("End-to-end forward FLOPs (TFLOPs)")
        plt.title("Qwen3-4B memory methods: FLOPs vs. context length")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "flops_vs_context.png", dpi=180)
    plt.savefig(output_dir / "flops_vs_context.pdf")
    plt.close()

    if flops_scope == "total":
        baseline_flops = {
            row["context_tokens"]: row["flops_total"]
            for row in rows if row["method"] == "baseline"
        }
        plt.figure(figsize=(8.2, 5.2))
        for method in ("delta_mem", "transmem_d4_gate"):
            selected = [row for row in rows if row["method"] == method]
            plt.plot(
                [row["context_tokens"] / 1000 for row in selected],
                [
                    (
                        row["flops_total"]
                        - baseline_flops[row["context_tokens"]]
                    ) / 1e9
                    for row in selected
                ],
                marker="o", linewidth=2, label=METHOD_LABELS[method],
                color=METHOD_COLORS[method])
        plt.xlabel("Context length (K tokens)")
        plt.ylabel("Additional FLOPs over baseline (GFLOPs)")
        plt.title("Memory-module FLOP overhead vs. context length")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "extra_flops_vs_context.png", dpi=180)
        plt.savefig(output_dir / "extra_flops_vs_context.pdf")
        plt.close()


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    fields = [
        "method", "context_tokens",
        "prefill_seconds_median", "decode_seconds_median", "total_seconds_median",
        "total_seconds_p10", "total_seconds_p90",
        "flops_counter", "flops_total", "tflops_total", "peak_allocated_gb",
    ]
    with output.open("w", encoding="utf-8") as handle:
        handle.write(",".join(fields) + "\n")
        for row in rows:
            handle.write(",".join(str(row[field]) for field in fields) + "\n")


def write_report(payload: dict[str, Any], output: Path) -> None:
    rows = payload["results"]
    time_header = (
        "| Context | Baseline (s) | Delta-Mem (s) | Δ latency (s) | "
        "TransMem D4 gate (s) | Δ latency (s) |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
    )
    time_lines = []
    for length in payload["context_lengths"]:
        values = {
            row["method"]: row["total_seconds_median"]
            for row in rows if row["context_tokens"] == length
        }
        time_lines.append(
            f"| {length // 1000}k | {values['baseline']:.4f} | "
            f"{values['delta_mem']:.4f} | "
            f"{values['delta_mem'] - values['baseline']:.4f} | "
            f"{values['transmem_d4_gate']:.4f} | "
            f"{values['transmem_d4_gate'] - values['baseline']:.4f} |")

    memory_only = payload["flops_scope"] == "memory_only"
    flop_header = (
        "| Context | Delta-Mem component (GFLOPs) | "
        "TransMem component (GFLOPs) |\n"
        "|---:|---:|---:|\n"
        if memory_only
        else (
            "| Context | Delta total (TFLOPs) | TransMem total (TFLOPs) | "
            "Delta extra (GFLOPs) | TransMem extra (GFLOPs) |\n"
            "|---:|---:|---:|---:|---:|\n"
        )
    )
    flop_lines = []
    for length in payload["context_lengths"]:
        selected = {
            row["method"]: row
            for row in rows if row["context_tokens"] == length
        }
        if memory_only:
            flop_lines.append(
                f"| {length // 1000}k | "
                f"{selected['delta_mem']['flops_total'] / 1e9:.3f} | "
                f"{selected['transmem_d4_gate']['flops_total'] / 1e9:.3f} |")
        else:
            flop_lines.append(
                f"| {length // 1000}k | "
                f"{selected['delta_mem']['tflops_total']:.4f} | "
                f"{selected['transmem_d4_gate']['tflops_total']:.4f} | "
                f"{(selected['delta_mem']['flops_total'] - selected['baseline']['flops_total']) / 1e9:.3f} | "
                f"{(selected['transmem_d4_gate']['flops_total'] - selected['baseline']['flops_total']) / 1e9:.3f} |")

    if memory_only:
        flop_protocol = (
            "- FLOPs scope: memory components only; frozen Qwen backbone is "
            "excluded and the bare baseline is 0. Delta-Mem includes its "
            "memory projections, active delta heads and affine scan. TransMem "
            "includes its four one-layer blocks, memory attention, output "
            "projection and dynamic gate. Elementwise nonlinearities, "
            "normalization and RoPE are not counted.\n"
        )
        flop_section = (
            "## Memory-component FLOPs\n\n"
            + flop_header + "\n".join(flop_lines)
            + "\n\n![Memory FLOPs](flops_vs_context.png)\n"
        )
    else:
        flop_protocol = (
            "- FLOPs: PyTorch FlopCounterMode plus the arithmetic hidden in "
            "Delta-Mem's custom Triton affine-scan kernel (11r² per processed "
            "token/layer). Elementwise nonlinearities are not counted.\n"
        )
        flop_section = (
            "## End-to-end FLOPs\n\n"
            + flop_header + "\n".join(flop_lines)
            + "\n\n![Total FLOPs](flops_vs_context.png)\n\n"
            "## Additional memory-method FLOPs\n\n"
            "The total curves overlap because the frozen 4B backbone dominates. "
            "Subtracting the baseline exposes the different scaling laws.\n\n"
            "![Additional FLOPs](extra_flops_vs_context.png)\n"
        )

    output.write_text(
        "# Qwen3-4B context-scaling benchmark\n\n"
        "## Protocol\n\n"
        f"- Batch size: 1; dtype: {payload['dtype']}; attention: "
        f"{payload['attn_implementation']}.\n"
        f"- Fixed greedy generation: {payload['decode_tokens']} tokens.\n"
        f"- Timing: {payload['warmup_runs']} warmups + "
        f"{payload['measure_runs']} measured runs; table uses CUDA-event median.\n"
        + flop_protocol + "\n"
        "## End-to-end latency\n\n"
        + time_header + "\n".join(time_lines)
        + "\n\n![Latency](latency_vs_context.png)\n\n"
        "## Additional latency over baseline\n\n"
        "This subtraction isolates the wall-clock overhead associated with "
        "each memory method from the shared frozen-backbone cost.\n\n"
        "![Latency overhead](latency_overhead_vs_context.png)\n\n"
        + flop_section,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise SystemExit("This benchmark requires CUDA")
    if args.decode_tokens < 1:
        raise ValueError("--decode-tokens must be positive")
    if sorted(args.context_lengths) != args.context_lengths:
        raise ValueError("--context-lengths must be sorted")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    print("[load] baseline", flush=True)
    base_model, base_tokenizer = load_base(args)
    baseline_flop_kwargs = (
        {"analytical_flops": lambda _length: 0}
        if args.flops_scope == "memory_only"
        else {
            "flop_run": lambda ids: fixed_decode_standard(
                base_model, ids, args.decode_tokens, timing=False
            )
        }
    )
    rows += benchmark_method(
        method="baseline",
        tokenizer=base_tokenizer,
        context_lengths=args.context_lengths,
        warmup_runs=args.warmup_runs,
        measure_runs=args.measure_runs,
        decode_tokens=args.decode_tokens,
        device=args.device,
        timed_run=lambda ids, timing: fixed_decode_standard(
            base_model, ids, args.decode_tokens, timing=timing),
        **baseline_flop_kwargs,
    )
    del base_model, base_tokenizer
    release()

    print("[load] Delta-Mem", flush=True)
    delta_model, delta_tokenizer, delta_config = load_delta(args)
    delta_layers = len(delta_model.model.layers)
    delta_rank = int(delta_config["rank"])
    delta_module = next(
        (
            module
            for module in delta_model.modules()
            if hasattr(module, "memory_q_proj")
            and hasattr(module, "active_delta_heads")
        ),
        None,
    )
    if delta_module is None:
        raise RuntimeError("Could not locate a Delta-Mem attention module")
    delta_memory_kwargs = {
        "num_layers": delta_layers,
        "rank": int(delta_module.rank),
        "num_state_heads": int(delta_module.num_state_heads),
        "hidden_size": int(delta_module.hidden_size),
        "query_out_features": int(delta_module.query_out_features),
        "key_out_features": int(delta_module.key_out_features),
        "value_out_features": int(delta_module.base_v_out_features),
        "output_out_features": int(delta_module.base.o_proj.out_features),
        "delta_heads": set(delta_module.active_delta_heads),
        "rankwise_gates": bool(delta_module.rankwise_gates),
        "couple_lambda": bool(delta_module.couple_lambda),
    }
    delta_flop_kwargs = (
        {
            "analytical_flops": lambda length: delta_memory_component_flops(
                prompt_length=length,
                decode_tokens=args.decode_tokens,
                **delta_memory_kwargs,
            )
        }
        if args.flops_scope == "memory_only"
        else {
            "flop_run": lambda ids: fixed_decode_standard(
                delta_model,
                ids,
                args.decode_tokens,
                before_run=lambda: reset_delta_mem_states(delta_model),
                timing=False,
            ),
            "extra_flops": lambda length: delta_scan_flops(
                prompt_length=length,
                decode_tokens=args.decode_tokens,
                num_layers=delta_layers,
                rank=delta_rank,
            ),
        }
    )
    rows += benchmark_method(
        method="delta_mem",
        tokenizer=delta_tokenizer,
        context_lengths=args.context_lengths,
        warmup_runs=args.warmup_runs,
        measure_runs=args.measure_runs,
        decode_tokens=args.decode_tokens,
        device=args.device,
        timed_run=lambda ids, timing: fixed_decode_standard(
            delta_model, ids, args.decode_tokens,
            before_run=lambda: reset_delta_mem_states(delta_model),
            timing=timing),
        **delta_flop_kwargs,
    )
    del delta_model, delta_tokenizer
    release()

    print("[load] TransMem D=4 dynamic gate", flush=True)
    trans_model, trans_tokenizer, trans_memory, rollout, trans_config = load_transmem(args)
    if trans_config.inject_layers != [32, 33, 34, 35]:
        raise RuntimeError(
            f"Expected D=4 layers [32,33,34,35], got {trans_config.inject_layers}")
    if trans_config.gate_mode == "constant":
        raise RuntimeError("Selected checkpoint uses a constant gate, not dynamic gate")
    transmem_flop_kwargs = (
        {
            "analytical_flops": lambda _length: transmem_memory_component_flops(
                config=trans_config,
                decode_tokens=args.decode_tokens,
            )
        }
        if args.flops_scope == "memory_only"
        else {
            "flop_run": lambda ids: fixed_decode_transmem(
                rollout, ids, args.decode_tokens, timing=False
            )
        }
    )
    rows += benchmark_method(
        method="transmem_d4_gate",
        tokenizer=trans_tokenizer,
        context_lengths=args.context_lengths,
        warmup_runs=args.warmup_runs,
        measure_runs=args.measure_runs,
        decode_tokens=args.decode_tokens,
        device=args.device,
        timed_run=lambda ids, timing: fixed_decode_transmem(
            rollout, ids, args.decode_tokens, timing=timing),
        **transmem_flop_kwargs,
    )
    del trans_model, trans_tokenizer, trans_memory, rollout
    release()

    payload = {
        "model_path": args.model_path,
        "delta_adapter_dir": str(args.delta_adapter_dir),
        "transmem_ckpt": str(args.transmem_ckpt),
        "context_lengths": args.context_lengths,
        "decode_tokens": args.decode_tokens,
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "flops_scope": args.flops_scope,
        "seed": args.seed,
        "delta_config": delta_config,
        "transmem_config": trans_config.to_dict(),
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "flop_convention": (
            (
                "Memory components only: analytical projection/attention "
                "matmuls plus Delta-Mem affine scan; frozen backbone and "
                "elementwise operations excluded"
            )
            if args.flops_scope == "memory_only"
            else (
                "PyTorch FlopCounterMode plus 11*r^2 arithmetic per Delta-Mem "
                "Triton affine-scan token/layer; elementwise nonlinearities excluded"
            )
        ),
        "results": rows,
    }
    json_path = args.output_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(rows, args.output_dir / "results.csv")
    save_plots(rows, args.output_dir, flops_scope=args.flops_scope)
    write_report(payload, args.output_dir / "report.md")
    print(f"Saved benchmark artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
