#!/usr/bin/env python3
"""TransMem-Layer CPU 测试 (无 GPU / 无真模型): 小随机 Qwen3 + 小 TransMemLayered.

  [1] 零初始化恒等: LayeredRollout 生成 == 裸 LLM 贪心 (hook 接线/KV cache 全链验证)
  [2] 偏置生效: out_proj 置非零后 prefill logits 改变 (hook 确实改写了层输出流)
  [3] forward 等价: TransMemLayered.forward(堆叠) == 逐块 TransMem 前向
  [4] 训练全链: 造 layered stage0 特征 + lm_head/final_norm → LayeredTrainer 短跑,
      初始 rel==1 / improve==0 (零初始化), 过拟合后 loss 显著下降, best.pt 落盘
  [5] ckpt round-trip: config to_dict → evaluate 侧 from_dict 重建 + load_state_dict

用法: python -m transmem.test_layered
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

from transmem.layered import LayeredConfig, TransMemLayered, LayeredRollout

DIM, VOCAB, NLAYER = 64, 128, 6
EOS = 127


def tiny_llm(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=VOCAB, hidden_size=DIM, intermediate_size=128,
        num_hidden_layers=NLAYER, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512, rope_theta=10000.0,
        eos_token_id=EOS, attn_implementation="eager")
    return Qwen3ForCausalLM(cfg).eval()


def tiny_layered_cfg(inject=(3, 5)):
    return LayeredConfig(
        dim=DIM, block_depth=1, num_heads=4, num_kv_heads=2, head_dim=16,
        intermediate_size=128, rope_theta=10000.0, max_position_embeddings=512,
        n_mem=4, inject_layers=list(inject), attn_impl="eager")


def plain_greedy(model, cq_ids, max_new):
    """裸 LLM 贪心 (base-model 调用, 与 LayeredRollout 同路径但无 hook)."""
    ids = []
    with torch.no_grad():
        out = model.model(input_ids=cq_ids, use_cache=True)
        past = out.past_key_values
        logits = model.lm_head(out.last_hidden_state[:, -1, :])
        for _ in range(max_new):
            nxt = logits.argmax(dim=-1)
            t = int(nxt.item())
            ids.append(t)
            if t == EOS:
                break
            step = model.model(input_ids=nxt.view(1, 1), past_key_values=past,
                               use_cache=True)
            past = step.past_key_values
            logits = model.lm_head(step.last_hidden_state[:, -1, :])
    return ids


def test_1_identity():
    model = tiny_llm()
    layered = TransMemLayered(tiny_layered_cfg()).eval()
    ro = LayeredRollout(model, tokenizer=None, device="cpu", layered=layered,
                        dtype=torch.float32)
    torch.manual_seed(7)
    for trial in range(3):
        cq = torch.randint(0, VOCAB - 1, (1, 40 + trial * 17))
        ref = plain_greedy(model, cq, 12)
        got = ro.generate_from_ids(cq, len_cl=30 + trial * 10, max_new=12)
        assert got == ref, f"[1] 零初始化不恒等: {got} != {ref}"
    print("[1] PASS 零初始化恒等 (rollout == 裸贪心, 3 组)")


def test_2_bias_flows():
    model = tiny_llm()
    layered = TransMemLayered(tiny_layered_cfg()).eval()
    with torch.no_grad():
        for b in layered.blocks.values():
            b.out_proj.weight.normal_(0, 1.0)
    ro = LayeredRollout(model, tokenizer=None, device="cpu", layered=layered,
                        dtype=torch.float32)
    torch.manual_seed(7)
    cq = torch.randint(0, VOCAB - 1, (1, 40))
    ref = plain_greedy(model, cq, 12)
    got = ro.generate_from_ids(cq, len_cl=30, max_new=12)
    assert got != ref, "[2] out_proj 非零但生成不变 — hook 未生效?"
    print("[2] PASS 偏置生效 (非零 out_proj 改变生成)")


def test_3_forward_equiv():
    torch.manual_seed(1)
    cfg = tiny_layered_cfg()
    layered = TransMemLayered(cfg).eval()
    with torch.no_grad():
        for b in layered.blocks.values():
            b.out_proj.weight.normal_(0, 0.1)
    B, D, N, M = 2, 2, cfg.n_mem, 5
    hm = torch.randn(B, D, N, DIM)
    h_in = torch.randn(B, D, M, DIM)
    ms = layered(hm, h_in)
    assert ms.shape == (B, D, M, DIM)
    for k, l in enumerate(cfg.inject_layers):
        X = torch.cat([hm[:, k], h_in[:, k]], dim=1)
        ref = layered.block(l)(X, return_all_queries=True)
        assert torch.allclose(ms[:, k], ref, atol=1e-6), f"[3] 层 {l} 不等价"
    print("[3] PASS forward 堆叠 == 逐块前向")


def _fake_stage0_layered(root: Path, layer_ids, n_samples=8, N=4):
    torch.manual_seed(3)
    K = len(layer_ids)
    shard = root / "shard_0000"
    shard.mkdir(parents=True)
    manifest = []
    for i in range(n_samples):
        M = 4 + (i % 3)
        rec = {
            "hm_stu": torch.randn(N, DIM, dtype=torch.bfloat16),
            "hq_stu": torch.randn(M, DIM, dtype=torch.bfloat16),
            "hq_tea": torch.randn(M, DIM, dtype=torch.bfloat16),
            "hm_stu_layers": torch.randn(K, N, DIM, dtype=torch.bfloat16),
            "hq_stu_layers": torch.randn(K, M, DIM, dtype=torch.bfloat16),
            "hq_tea_layers": torch.randn(K, M, DIM, dtype=torch.bfloat16),
            "answer_ids": torch.randint(0, VOCAB, (M,)),
            "sample_idx": i, "dim": DIM, "N": N,
        }
        torch.save(rec, shard / f"sample_{i:05d}.pt")
        manifest.append({"sample_idx": i, "shard_idx": 0,
                         "file": f"shard_0000/sample_{i:05d}.pt", "M": M})
    meta = {"N": N, "dim": DIM, "save_dtype": "bfloat16",
            "dump_layers": K, "layer_ids": list(layer_ids),
            "total_records": n_samples, "succeeded": n_samples, "failed": 0,
            "total_pairs": sum(e["M"] for e in manifest), "num_shards": 1,
            "has_lm_head": True, "samples": manifest}
    (root / "meta.json").write_text(json.dumps(meta))
    torch.save({"weight": torch.randn(VOCAB, DIM), "tied": False,
                "vocab_size": VOCAB, "dim": DIM}, root / "lm_head.pt")
    torch.save({"weight": torch.ones(DIM), "eps": 1e-6}, root / "final_norm.pt")


def test_4_trainer():
    from transmem.train_layered import LayeredTrainer
    layer_ids = [2, 3, 4, 5]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "feat"
        _fake_stage0_layered(root, layer_ids)
        cfg_path = Path(td) / "cfg.json"
        cfg_path.write_text(json.dumps({
            "dim": DIM, "block_depth": 1, "num_heads": 4, "num_kv_heads": 2,
            "head_dim": 16, "intermediate_size": 128, "rope_theta": 10000.0,
            "max_position_embeddings": 512, "n_mem": 4, "attn_impl": "eager",
            "inject_layers": [5]}))
        out_dir = Path(td) / "ckpt"
        args = argparse.Namespace(
            data_dir=str(root), val_data_dir=str(root),
            lm_head_path=None, final_norm_path=None, config=str(cfg_path),
            D=3, inject_layers=None,
            divergence="forward_kl", temperature=1.0, jsd_beta=0.5,
            mse_weight=1.0, alpha_mix="uniform",
            output_dir=str(out_dir), batch_size=4, lr=3e-3, weight_decay=0.0,
            epochs=40, warmup_steps=5, grad_clip=1.0,
            log_interval=20, val_interval=20, save_interval=1000,
            max_steps=None, device="cpu", num_workers=0, dtype="float32",
            resume=None)
        tr = LayeredTrainer(args)
        assert tr.config.inject_layers == [3, 4, 5], tr.config.inject_layers

        # 初始: 零初始化 → rel==1, improve≈0
        dl = __import__("transmem.train_layered", fromlist=["make_dataloader"]) \
            .make_dataloader(str(root), tr.config.inject_layers, 4, 0,
                             torch.float32, shuffle=False)
        batch = next(iter(dl))
        with torch.no_grad():
            _, m0 = tr.compute_loss(*[t for t in batch], training=False, full=True)
        assert abs(m0["rel_mean"] - 1.0) < 1e-4, f"[4] 初始 rel!=1: {m0['rel_mean']}"
        assert abs(m0["improve"]) < 1e-4, f"[4] 初始 improve!=0: {m0['improve']}"

        tr.run()   # 40 epoch × 2 step = 80 步小过拟合
        with torch.no_grad():
            _, m1 = tr.compute_loss(*[t for t in batch], training=False, full=True)
        assert m1["loss"] < 0.5 * m0["loss"], (
            f"[4] 过拟合后 loss 未显著下降: {m0['loss']:.4f} -> {m1['loss']:.4f}")
        assert (out_dir / "best.pt").exists() and (out_dir / "latest.pt").exists()
        print(f"[4] PASS 训练全链 (loss {m0['loss']:.4f} -> {m1['loss']:.4f}, "
              f"rel {m0['rel_mean']:.3f} -> {m1['rel_mean']:.3f}, best.pt 落盘)")

        # [5] ckpt round-trip (evaluate 侧分发路径)
        ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
        assert ckpt["config"].get("layered") is True
        lcfg = LayeredConfig.from_dict(ckpt["config"])
        mem2 = TransMemLayered(lcfg)
        mem2.load_state_dict(ckpt["model_state_dict"])
        assert lcfg.inject_layers == [3, 4, 5]
        print("[5] PASS ckpt round-trip (layered 标记 + from_dict + load_state_dict)")


class _FakeTok:
    """字符级假 tokenizer: 只为驱动 Stage0Extractor 的 shape/取位测试."""
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = EOS

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        import types
        ids = [ord(c) % (VOCAB - 2) for c in text][:400]
        if return_tensors:
            return types.SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))
        return types.SimpleNamespace(input_ids=ids)

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        return "\n".join(m["content"] for m in messages) + "\nASSISTANT:"

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (i % 26)) for i in ids)


def test_6_extract_layered():
    """--dump_layers 抽取: shape + 与 output_hidden_states 独立通道的取位交叉验证.

    HF output_hidden_states 语义: hidden_states[i] = 第 i 层的输入 (=第 i-1 层输出),
    末元素是 post-final-norm — 所以第 l 层输出 = hidden_states[l+1] 仅对 l≤n-2 成立,
    末层 (l=n-1) 只做 shape 检查 (hook 代码路径与其它层完全相同).
    """
    from transmem.extract_features import Stage0Extractor, build_chat_prompt_ids

    args = argparse.Namespace(
        device="cpu", dtype="float32", save_dtype="float32", N=4, hm_mode="floor",
        pool_ns="", trajectory="teacher", max_answer_tokens=6, thinking=False,
        dump_layers=3, attn_impl="eager", data_path="", data_format="json",
        model_path="", output_dir="", samples_per_shard=1000, max_samples=None,
        num_workers=1, dump_lm_head=False, manifest_dir=None)
    ex = Stage0Extractor(args)
    ex.tokenizer = _FakeTok()
    ex.model = tiny_llm(seed=5)
    ex.dim = DIM
    ex.lm_head = ex.model.lm_head
    ex._eos_ids = [EOS]
    ex.layer_ids = [NLAYER - 3, NLAYER - 2, NLAYER - 1]     # [3,4,5]

    rec = {"question": "what is the color of x?",
           "context": "the sky is blue. " * 12,
           "cs_text": "x is blue.",
           "golden_index": None, "ground_truth": "blue", "sample_idx": 0}
    out = ex.process_sample(rec)
    assert out is not None, "[6] process_sample 返回 None"
    M = out["hq_stu"].shape[0]
    K = 3
    assert out["hm_stu_layers"].shape == (K, 4, DIM), out["hm_stu_layers"].shape
    assert out["hq_stu_layers"].shape == (K, M, DIM), out["hq_stu_layers"].shape
    assert out["hq_tea_layers"].shape == (K, M, DIM), out["hq_tea_layers"].shape
    assert out["layer_ids"] == [3, 4, 5]

    # 学生侧交叉验证 (l=3,4): hook 取位 == output_hidden_states[l+1] 同位置
    from transmem.extract_features import hm_positions
    tok = ex.tokenizer
    cq = build_chat_prompt_ids(tok, rec["context"], rec["question"], "cpu")
    len_cq = cq.shape[1]
    len_cl = len(tok(rec["context"]).input_ids)
    ans = out["answer_ids"].tolist()
    full = torch.cat([cq, torch.tensor([ans[:-1]], dtype=torch.long)], dim=1)
    with torch.no_grad():
        hs = ex.model.model(input_ids=full, output_hidden_states=True).hidden_states
    hm_idx = hm_positions(len_cl, 4, "floor")
    positions = [len_cq - 1] + [len_cq + i - 2 for i in range(2, M + 1)]
    for k, l in enumerate([3, 4]):
        ref = hs[l + 1][0]
        assert torch.allclose(out["hm_stu_layers"][k], ref[hm_idx], atol=1e-4), \
            f"[6] 层 {l} HM 取位不一致"
        assert torch.allclose(out["hq_stu_layers"][k], ref[positions], atol=1e-4), \
            f"[6] 层 {l} HQ_stu 取位不一致"

    # 教师侧交叉验证 (l=3,4): 逐步生成捕获 == teacher-forcing 并行前向同位置
    cqs = build_chat_prompt_ids(tok, rec["cs_text"], rec["question"], "cpu")
    len_cqs = cqs.shape[1]
    full_t = torch.cat([cqs, torch.tensor([ans[:-1]], dtype=torch.long)], dim=1)
    with torch.no_grad():
        hs_t = ex.model.model(input_ids=full_t, output_hidden_states=True).hidden_states
    pos_t = [len_cqs - 1] + [len_cqs + i - 2 for i in range(2, M + 1)]
    for k, l in enumerate([3, 4]):
        ref = hs_t[l + 1][0]
        assert torch.allclose(out["hq_tea_layers"][k], ref[pos_t], atol=1e-3), \
            f"[6] 层 {l} HQ_tea (生成逐步 vs 并行) 不一致: " \
            f"max_err={(out['hq_tea_layers'][k]-ref[pos_t]).abs().max():.2e}"
    print(f"[6] PASS layered 抽取 (M={M}, 学生/教师取位 == output_hidden_states 独立通道)")


if __name__ == "__main__":
    test_1_identity()
    test_2_bias_flows()
    test_3_forward_equiv()
    test_4_trainer()
    test_6_extract_layered()
    print("\n✅ test_layered 全部通过")
