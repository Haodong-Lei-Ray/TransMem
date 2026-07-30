#!/usr/bin/env python3
"""池提取双源分账: 把合并 meta.json 按文件实际所在目录拆成 本地(pool128) + S3 两份.

背景 (2026-07-09): datafrontier 桶满, 前 6,935 个样本以 512 行全网格池写在 S3
(hotpotqa_pool/stage0_train_short200_pool), 其余样本以 128 行池写本地
(pool128/stage0_train_short200_pool). 合并 meta 会把两边样本都列进本地目录的
meta.json — 训练加载会找不到 S3 部分的文件. 本脚本:
  1) 本地 meta.json 只留本地存在的文件 (pool_ns=[4..128]);
  2) S3 部分的条目写 meta_s3pool_train.json (pool_ns=[4..384], 512 行超集),
     训练包装脚本 sync S3 -> /nvme 后把它拷成该目录的 meta.json.

用法: python scripts/stage0/hotpotqa/split_pool_meta.py
"""
import json
from pathlib import Path

PROJ = Path("/mnt/petrelfs/leihaodong/Project4")
LOCAL_DIR = PROJ / "data/hotpotqa_data/Qwen3-4B-Instruct-2507/pool128/stage0_train_short200_pool"
S3_META_OUT = PROJ / "data/hotpotqa_data/meta_s3pool_train.json"

meta = json.load(open(LOCAL_DIR / "meta.json"))
local, remote = [], []
for e in meta["samples"]:
    (local if (LOCAL_DIR / e["file"]).exists() else remote).append(e)

def mk(base, samples, pool_ns):
    m = dict(base)
    m["samples"] = samples
    m["pool_ns"] = pool_ns
    m["succeeded"] = len(samples)
    m["total_pairs"] = sum(e["M"] for e in samples)
    return m

json.dump(mk(meta, local, [4, 8, 16, 32, 64, 128]),
          open(LOCAL_DIR / "meta.json", "w"), indent=2)
json.dump(mk(meta, remote, [4, 8, 16, 32, 64, 128, 256, 384]),
          open(S3_META_OUT, "w"), indent=2)
print(f"本地 pool128: {len(local)} 样本 / {sum(e['M'] for e in local)} 对")
print(f"S3   pool512: {len(remote)} 样本 / {sum(e['M'] for e in remote)} 对 -> {S3_META_OUT}")
assert len(local) + len(remote) == len(meta["samples"])
