#!/usr/bin/env python3
"""122k 长上下文 forward 显存探针 (stage0 OOM 定位).

现象: extract_features._forward_ids_and_hook 的 model.model(ids, use_cache=False)
在 ~107k token 处 OOM 于 F.sdpa 内部 (S² bf16 ≈ 21GB 二次分配), 而 evaluate 的
generate() 路径 (attention_mask=ones + KV cache) 在 122k 上实证可跑.
本探针在同一模型上测 4 种组合的峰值显存, 定位关键 flag.
"""
import os
import sys
import torch

sys.path.insert(0, "/mnt/petrelfs/leihaodong/Project4")
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

MODEL = sys.argv[1]
S = int(os.environ.get("PROBE_LEN", "125000"))

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, local_files_only=True,
    trust_remote_code=True, attn_implementation="sdpa").cuda().eval()
cfg = model.config
print(f"config: use_sliding_window={getattr(cfg, 'use_sliding_window', None)} "
      f"sliding_window={getattr(cfg, 'sliding_window', None)} "
      f"layer_types={set(getattr(cfg, 'layer_types', []) or [])}", flush=True)

ids = torch.randint(0, 150000, (1, S), device="cuda")

def trial(name, ones_mask: bool, cache: bool):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    kw = dict(use_cache=cache)
    if cache:
        kw["past_key_values"] = DynamicCache()
    if ones_mask:
        kw["attention_mask"] = torch.ones_like(ids)
    try:
        with torch.inference_mode():
            out = model.model(input_ids=ids, **kw)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"[{name}] OK  peak={peak:.1f}GB", flush=True)
        del out
    except torch.OutOfMemoryError as e:
        print(f"[{name}] OOM ({str(e)[:60]})", flush=True)
    torch.cuda.empty_cache()

trial("mask=None  cache=False (现 extractor)", False, False)
trial("mask=ones  cache=False", True, False)
trial("mask=None  cache=True ", False, True)
trial("mask=ones  cache=True  (generate 同款)", True, True)
print("done", flush=True)
