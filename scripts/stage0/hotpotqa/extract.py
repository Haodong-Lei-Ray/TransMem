#!/usr/bin/env python3
"""
hotpotqa-agentmem Stage 0 特征抽取 — 薄封装 transmem.extract_features.

用法 (需要先 s3mount 模型, 或 MODEL_PATH 环境变量):
  python extract.py                              # 使用默认超参
  MAXN=100 python extract.py                     # 只抽前100条
  N=8 MAX_ANS=80 python extract.py               # 改超参

sbatch 方式 (推荐, 自动 s3mount):
  sbatch extract.sh
  N=8 MAX_ANS=200 THINKING=true sbatch extract.sh

继承关系:
  Stage0Extractor (transmem/extract_features.py) ← 不改, 只注入 defaults.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

from transmem.extract_features import Stage0Extractor

HERE = os.path.dirname(os.path.abspath(__file__))


def build_args():
    """从环境变量构造 args. 跟 run_stage0_short128.sh 语义一致."""
    N = int(os.environ.get("N", "4"))
    max_ans = int(os.environ.get("MAX_ANS", "128"))
    maxn = os.environ.get("MAXN", "")              # 空=全量
    thinking = os.environ.get("THINKING", "false").lower() == "true"
    model_path = os.environ.get("MODEL_PATH",
                                "/mnt/petrelfs/leihaodong/models/Qwen3-4B-Instruct-2507")
    attn = os.environ.get("ATTN", "sdpa")
    output_dir = os.environ.get("OUTPUT_DIR",
                                os.path.join(HERE, "stage0_output"))

    ns = SimpleNamespace(
        data_path=os.path.join(HERE, "hotpotqa_train_32k.parquet"),
        data_format="hotpotqa-agentmem",
        model_path=model_path,
        output_dir=output_dir,
        device="cuda:0",
        dtype="bfloat16",
        attn_impl=attn,
        N=N,
        max_answer_tokens=max_ans,
        samples_per_shard=1000,
        save_dtype="bfloat16",
        max_samples=int(maxn) if maxn else None,
        dump_lm_head=True,
        thinking=thinking,
    )
    return ns


def main():
    args = build_args()
    print(f"数据: {args.data_path} (format=hotpotqa-agentmem)")
    print(f"模型: {args.model_path}")
    print(f"输出: {args.output_dir}")
    print(f"超参: N={args.N} max_ans={args.max_answer_tokens} max_samples={args.max_samples} thinking={args.thinking}")

    extractor = Stage0Extractor(args)
    extractor.load_model()
    extractor.run()


if __name__ == "__main__":
    main()
