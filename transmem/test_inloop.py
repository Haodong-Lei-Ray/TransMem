#!/usr/bin/env python3
"""v3.2 在环训练 CPU 测试 (无 GPU / 无真模型): 小随机 Qwen3 + 小 TransMemLayered.

  [1] 零初始化恒等: teacher_forced_forward == 裸前向同位置 hidden (hook 不改流)
  [2] 训推等价 (核心): 非零块下, TF 并行前向逐位 argmax == LayeredRollout 逐步
      贪心生成的轨迹 — 单次并行注入与增量 KV cache 注入数学等价的直接验证
  [3] 深度信用分配: 顶端 loss 反传后每个注入层的块都有非零梯度 (最低层的梯度
      必须穿过真实 LLM 上层才能到达)
  [4] TF 训练全链: 假 stage0 + 注入 records → InLoopTrainer 短跑, 零初始化基线
      val == 裸学生 KL, 过拟合后显著下降, best.pt 落盘 + ckpt round-trip
  [5] onpolicy 模式 smoke: rollout→教师重打分→梯度步, loss 有限
  [6] resume: latest.pt 断点续训 global_step 恢复

用法: python -m transmem.test_inloop
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

from transmem.layered import LayeredConfig, TransMemLayered, LayeredRollout
from transmem.test_layered import (DIM, VOCAB, NLAYER, EOS, tiny_llm,
                                   tiny_layered_cfg, plain_greedy, _FakeTok)


def _mk(inject=(3, 5), seed=0, noise=0.0):
    model = tiny_llm(seed=seed)
    layered = TransMemLayered(tiny_layered_cfg(inject)).train()
    if noise > 0:
        torch.manual_seed(seed + 100)
        with torch.no_grad():
            for b in layered.blocks.values():
                b.out_proj.weight.normal_(0, noise)
    ro = LayeredRollout(model, tokenizer=None, device="cpu", layered=layered,
                        dtype=torch.float32)
    return model, layered, ro


def test_1_identity():
    model, _, ro = _mk()
    torch.manual_seed(11)
    for trial in range(3):
        M = 3 + trial * 2
        cq = torch.randint(0, VOCAB - 1, (1, 40 + trial * 13))
        ans = torch.randint(0, VOCAB - 1, (M,)).tolist()
        len_cq = cq.shape[1]
        full = torch.cat([cq, torch.tensor([ans[:-1]])], dim=1) if M > 1 else cq
        h_q = ro.teacher_forced_forward(full, len_cl=30, len_cq=len_cq, M=M)
        with torch.no_grad():
            ref = model.model(input_ids=full,
                              attention_mask=torch.ones_like(full)
                              ).last_hidden_state[0, len_cq - 1: len_cq + M - 1]
        assert torch.allclose(h_q, ref, atol=1e-5), \
            f"[1] 零初始化 TF 前向 != 裸前向 (max {(h_q-ref).abs().max():.2e})"
    print("[1] PASS 零初始化恒等 (TF 前向 == 裸前向, 3 组)")


def test_2_train_infer_equiv():
    """非零块: 逐步 rollout 轨迹 = TF 并行前向的逐位贪心 — 训推等价的直接证明."""
    for trial, (inject, noise) in enumerate([((3, 5), 0.5), ((2, 3, 4, 5), 0.3),
                                             ((5,), 1.0)]):
        model, layered, ro = _mk(inject=inject, seed=trial, noise=noise)
        torch.manual_seed(23 + trial)
        cq = torch.randint(0, VOCAB - 1, (1, 37 + trial * 11))
        len_cl, len_cq = 25 + trial * 7, cq.shape[1]
        traj = ro.generate_from_ids(cq, len_cl=len_cl, max_new=10)
        ref = plain_greedy(model, cq, 10)
        assert traj != ref, f"[2] trial{trial}: 非零块未改变生成, 测试无效"
        M = len(traj)
        full = (torch.cat([cq, torch.tensor([traj[:-1]])], dim=1) if M > 1 else cq)
        with torch.no_grad():
            h_q = ro.teacher_forced_forward(full, len_cl, len_cq, M)
            logits = model.lm_head(h_q)                          # [M, vocab]
        got = logits.argmax(-1).tolist()
        assert got == traj, f"[2] trial{trial}: TF argmax {got} != rollout 轨迹 {traj}"
    print("[2] PASS 训推等价 (TF 并行前向 argmax == 逐步 rollout, 3 组配置)")


def test_3_deep_credit():
    """顶端 loss 对所有块可导 — 最低块的梯度穿过真实上层 (深注入的信用分配)."""
    model, layered, ro = _mk(inject=(2, 4, 5), noise=0.1)
    model.requires_grad_(False)                    # 与训练器同构: LLM 冻结
    torch.manual_seed(31)
    cq = torch.randint(0, VOCAB - 1, (1, 40))
    M = 5
    ans = torch.randint(0, VOCAB - 1, (M,)).tolist()
    full = torch.cat([cq, torch.tensor([ans[:-1]])], dim=1)
    h_q = ro.teacher_forced_forward(full, len_cl=30, len_cq=40, M=M)
    loss = model.lm_head(h_q).float().logsumexp(-1).mean()
    loss.backward()
    for l, b in layered.blocks.items():
        g = b.out_proj.weight.grad
        assert g is not None and float(g.abs().max()) > 0, f"[3] 层 {l} 块无梯度"
    for p in model.parameters():
        assert p.grad is None, "[3] 冻结 LLM 竟有梯度"
    print("[3] PASS 深度信用分配 (全部块有梯度, LLM 冻结无梯度)")


# ── [4]-[6] 训练器: 假 stage0 + 注入 records ────────────────────────────────

def _fake_data(root: Path, n=8, N=4, with_cs=True):
    torch.manual_seed(43)
    shard = root / "shard_0000"
    shard.mkdir(parents=True)
    manifest, records = [], []
    for i in range(n):
        M = 3 + (i % 3)
        rec = {"hm_stu": torch.randn(N, DIM, dtype=torch.bfloat16),
               "hq_stu": torch.randn(M, DIM, dtype=torch.bfloat16),
               "hq_tea": torch.randn(M, DIM),
               "answer_ids": torch.randint(0, VOCAB - 1, (M,)),
               "sample_idx": i, "M": M, "dim": DIM, "N": N}
        torch.save(rec, shard / f"sample_{i:05d}.pt")
        manifest.append({"sample_idx": i, "shard_idx": 0,
                         "file": f"shard_0000/sample_{i:05d}.pt", "M": M})
        records.append({"question": f"what is item {i}?",
                        "context": f"item {i} is thing-{i}. " * 8,
                        "cs_text": (f"item {i} is thing-{i}." if with_cs else ""),
                        "golden_index": None, "ground_truth": f"thing-{i}",
                        "sample_idx": i})
    meta = {"N": N, "dim": DIM, "save_dtype": "bfloat16", "total_records": n,
            "succeeded": n, "failed": 0,
            "total_pairs": sum(e["M"] for e in manifest), "num_shards": 1,
            "has_lm_head": True, "samples": manifest}
    (root / "meta.json").write_text(json.dumps(meta))
    return records


def _trainer_args(td, policy="tf", epochs=12, out="ckpt"):
    return argparse.Namespace(
        data_dir=str(Path(td) / "feat"), data_path="", data_format="json",
        val_data_dir=None, val_data_path=None, val_max=8, max_samples=None,
        model_path="", attn_impl="eager", config=str(Path(td) / "cfg.json"),
        D=3, inject_layers=None, policy=policy,
        divergence="forward_kl", temperature=1.0, jsd_beta=0.5,
        sample_temp=0.0, max_answer_tokens=8,
        output_dir=str(Path(td) / out), grad_accum=2, lr=3e-3, weight_decay=0.0,
        epochs=epochs, warmup_steps=4, grad_clip=1.0,
        log_interval=8, val_interval=8, save_interval=1000, max_steps=None,
        device="cpu", num_workers=0, resume=None)


def _write_cfg(td):
    (Path(td) / "cfg.json").write_text(json.dumps({
        "dim": DIM, "block_depth": 1, "num_heads": 4, "num_kv_heads": 2,
        "head_dim": 16, "intermediate_size": 128, "rope_theta": 10000.0,
        "max_position_embeddings": 512, "n_mem": 4, "attn_impl": "eager",
        "inject_layers": [5]}))


def test_4_trainer_tf():
    from transmem.train_inloop import InLoopTrainer, InLoopDataset

    with tempfile.TemporaryDirectory() as td:
        records = _fake_data(Path(td) / "feat")
        _write_cfg(td)
        args = _trainer_args(td, policy="tf", epochs=14)
        tr = InLoopTrainer(args, model=tiny_llm(seed=7), tokenizer=_FakeTok())
        assert tr.config.inject_layers == [3, 4, 5]
        ds = InLoopDataset(args.data_dir, "", "json", policy="tf", records=records)
        assert len(ds) == 8

        v0 = tr.validate(ds)                       # 零初始化基线 = 裸学生 KL
        exp_pos = sum(e["M"] for e in ds.meta["samples"])
        assert v0["val_loss"] > 0 and v0["val_positions"] == exp_pos, v0

        # 短跑过拟合 (8 样本 × 14 epoch, accum2 → 56 步)
        import transmem.train_inloop as ti
        orig = ti.InLoopDataset
        ti.InLoopDataset = (lambda data_dir, data_path, data_format, policy="tf",
                            max_samples=None, records=None, records_=records:
                            orig(data_dir, data_path, data_format, policy=policy,
                                 max_samples=max_samples, records=records_))
        try:
            tr.args.val_data_dir = args.data_dir   # val 同集 (过拟合检查)
            tr.args.val_data_path = ""
            tr.run()
        finally:
            ti.InLoopDataset = orig
        v1 = tr.validate(ds)
        assert v1["val_loss"] < 0.7 * v0["val_loss"], \
            f"[4] 过拟合后 KL 未显著下降: {v0['val_loss']:.4f} -> {v1['val_loss']:.4f}"
        out = Path(args.output_dir)
        assert (out / "best.pt").exists() and (out / "latest.pt").exists()

        ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        assert ckpt["config"].get("layered") is True
        assert ckpt["train_mode"] == "inloop_tf"
        mem2 = TransMemLayered(LayeredConfig.from_dict(ckpt["config"]))
        mem2.load_state_dict(ckpt["model_state_dict"])
        print(f"[4] PASS TF 训练全链 (KL {v0['val_loss']:.4f} -> {v1['val_loss']:.4f}, "
              f"best@{ckpt['global_step']}, round-trip OK)")
        return


def test_5_onpolicy_smoke():
    from transmem.train_inloop import InLoopTrainer, InLoopDataset

    with tempfile.TemporaryDirectory() as td:
        records = _fake_data(Path(td) / "feat")
        _write_cfg(td)
        args = _trainer_args(td, policy="onpolicy", epochs=1)
        tr = InLoopTrainer(args, model=tiny_llm(seed=9), tokenizer=_FakeTok())
        ds = InLoopDataset(args.data_dir, "", "json", policy="onpolicy",
                           records=records)
        n_step = 0
        for i in range(4):
            r = tr.micro_loss(ds[i], policy="onpolicy")
            if r is None:
                continue
            loss, M, m = r
            assert torch.isfinite(loss), f"[5] onpolicy loss 非有限: {loss}"
            (loss / 2).backward()
            n_step += 1
        gn, stepped = tr.sync_and_step()
        assert stepped and n_step > 0, f"[5] onpolicy 未走出优化步 (micro={n_step})"
        print(f"[5] PASS onpolicy smoke ({n_step} micro, grad_norm={gn:.3f})")


def test_5b_pool_stage0_teacher_only():
    """Pool Stage0 的 N=None 可供 in-loop 使用，因为这里只读教师字段。"""
    from transmem.train_inloop import InLoopDataset

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "feat"
        records = _fake_data(root)
        meta_path = root / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["N"] = None
        meta["pool_ns"] = [4, 8]
        meta_path.write_text(json.dumps(meta))
        ds = InLoopDataset(str(root), "", "json", policy="tf", records=records)
        assert ds.teacher_only_pool and ds.N is None and ds.pool_ns == [4, 8]
        assert len(ds) == len(records)
        print("[5b] PASS pool Stage0 teacher-only 兼容")


def test_6_resume():
    from transmem.train_inloop import InLoopTrainer

    with tempfile.TemporaryDirectory() as td:
        records = _fake_data(Path(td) / "feat")
        _write_cfg(td)
        args = _trainer_args(td, policy="tf", epochs=2, out="ck6")
        import transmem.train_inloop as ti
        orig = ti.InLoopDataset
        ti.InLoopDataset = (lambda data_dir, data_path, data_format, policy="tf",
                            max_samples=None, records=None, records_=records:
                            orig(data_dir, data_path, data_format, policy=policy,
                                 max_samples=max_samples, records=records_))
        try:
            tr = InLoopTrainer(args, model=tiny_llm(seed=7), tokenizer=_FakeTok())
            tr.run()
            step_a = tr.global_step
            assert step_a > 0
            tr2 = InLoopTrainer(args, model=tiny_llm(seed=7), tokenizer=_FakeTok())
            tr2.load(str(Path(args.output_dir) / "latest.pt"))
            assert tr2.global_step == step_a, (tr2.global_step, step_a)
        finally:
            ti.InLoopDataset = orig
        print(f"[6] PASS resume (latest.pt step={step_a} 恢复)")


if __name__ == "__main__":
    test_1_identity()
    test_2_train_infer_equiv()
    test_3_deep_credit()
    test_4_trainer_tf()
    test_5_onpolicy_smoke()
    test_5b_pool_stage0_teacher_only()
    test_6_resume()
    print("\n✅ test_inloop 全部通过")
