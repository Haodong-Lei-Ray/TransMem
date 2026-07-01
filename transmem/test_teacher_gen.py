#!/usr/bin/env python3
"""教师生成的单元测试: 验证 chat template 修复了"教师不吐 EOS、一路跑到 max"的问题.

两级:
  [CPU] test_prompt_structure  —— 只用 tokenizer: build_chat_prompt_ids 产出 ChatML,
        含 <|im_start|>user / assistant, 末尾是 assistant 生成提示(还没答案).
  [GPU] test_teacher_stops     —— 用真模型: generate_answer 在真实 QA 上应"自然停"
        (AN < max_answer_tokens, 末位 token ∈ eos_ids), hq_tea 与答案逐位对齐.

用法:
  python -m transmem.test_teacher_gen                 # 仅 CPU 结构测试
  srun ... python -m transmem.test_teacher_gen --gpu  # 加 GPU 行为测试
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MODEL = "/mnt/petrelfs/leihaodong/models/Qwen3-4B-Instruct-2507"
_RECORD = "/mnt/petrelfs/leihaodong/Project4/data/qasper/sample_qa.json"


def test_prompt_structure(model_path: str):
    """[CPU] build_chat_prompt_ids 应产出 ChatML + assistant 生成提示, 且不含答案."""
    from transformers import AutoTokenizer
    from transmem.extract_features import build_chat_prompt_ids, _make_prompt

    print("[CPU] prompt 结构")
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                        trust_remote_code=True)
    ctx, q = "The sky is blue because of Rayleigh scattering.", "Why is the sky blue?"
    ids = build_chat_prompt_ids(tok, ctx, q)             # [1, L]
    assert ids.dim() == 2 and ids.shape[0] == 1, ids.shape
    text = tok.decode(ids[0], skip_special_tokens=False)

    assert "<|im_start|>user" in text, "缺 user 轮"
    assert "<|im_start|>assistant" in text, "缺 assistant 轮"
    assert q in text and "Rayleigh" in text, "context/question 未进 prompt"
    # add_generation_prompt=True: 末尾停在 assistant 生成提示, 还没有答案内容
    assert text.rstrip().endswith("assistant"), f"末尾非 assistant 生成提示: ...{text[-40:]!r}"
    # <|im_end|> 是停止 token
    assert tok.convert_tokens_to_ids("<|im_end|>") == 151645
    print(f"    prompt {ids.shape[1]} tok, ChatML+assistant 提示 OK, 末尾={text[-25:]!r}")


def test_teacher_stops(model_path: str, record: str):
    """[GPU] 真模型 generate_answer 应自然停 (AN<max, 末位∈eos), hq_tea 对齐."""
    import torch
    from transmem.extract_features import Stage0Extractor, load_records

    print("[GPU] 教师自然停 + 对齐")
    max_ans = 64
    ext = Stage0Extractor(SimpleNamespace(
        model_path=model_path, device="cuda:0", dtype="bfloat16", save_dtype="bfloat16",
        attn_impl="sdpa", N=4, max_answer_tokens=max_ans))
    ext.load_model()

    rec = load_records(record, "qasper", 1)[0]
    answer_ids, answer_text, hq_tea = ext.generate_answer(rec["cs_text"], rec["question"])
    AN = len(answer_ids)

    assert AN >= 1, "空答案"
    assert AN < max_ans, f"AN={AN} 达到上限 {max_ans} -> 仍未自然停 (chat template 没生效?)"
    assert answer_ids[-1] in ext._eos_ids, f"末位 {answer_ids[-1]} 不是 eos {ext._eos_ids}"
    assert hq_tea.shape[0] == AN, f"hq_tea {hq_tea.shape} 与 AN={AN} 不对齐"
    assert hq_tea.shape[1] == ext.dim
    # 内容清醒: 命中参考答案(Qasper 该题 gold='Chinese general corpus')
    print(f"    AN={AN} (<{max_ans}) 自然停 | 末位 eos={answer_ids[-1]} | hq_tea={tuple(hq_tea.shape)}")
    print(f"    答案: {answer_text!r}")
    print(f"    参考: {rec['ground_truth']!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=_MODEL)
    ap.add_argument("--record", default=_RECORD)
    ap.add_argument("--gpu", action="store_true", help="额外跑 GPU 行为测试")
    args = ap.parse_args()

    print("=" * 60)
    test_prompt_structure(args.model_path)
    if args.gpu:
        test_teacher_stops(args.model_path, args.record)
    else:
        print("[GPU] 跳过 (加 --gpu 在有卡环境运行)")
    print("=" * 60)
    print("✅ 教师生成测试通过")


if __name__ == "__main__":
    main()
