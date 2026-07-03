#!/usr/bin/env python3
"""无 GPU toy 验证: 随机张量跑通 X -> MS -> HQ' + 反传, 以及散度/恒等启动/各开关.

用法:
  python -m transmem.test_shapes            # 小 config 全测 + 真实 config.json 构建
  python -m transmem.test_shapes --skip-real  # 跳过真实 config (省内存/时间)
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transmem import TransMemConfig, TransMem, DistillLoss, FrozenLMHead


def _small_cfg(**over) -> TransMemConfig:
    """小 config, CPU 秒级跑通."""
    base = dict(dim=128, depth=2, num_heads=8, num_kv_heads=2, head_dim=16,
                intermediate_size=256, max_position_embeddings=4096,
                attn_impl="eager")
    base.update(over)
    return TransMemConfig(**base)


def test_forward_backward():
    """X -> MS -> HQ', 形状 + 反传."""
    print("[1] forward/backward + shapes")
    cfg = _small_cfg()
    mem = TransMem(cfg)
    B, S, D = 4, cfg.n_mem + 1, cfg.dim
    X = torch.randn(B, S, D)
    hq_stu = X[:, -1, :].clone()

    ms = mem(X)
    assert ms.shape == (B, D), ms.shape
    hq_prime = mem.correct(ms, hq_stu)
    assert hq_prime.shape == (B, D), hq_prime.shape

    loss = hq_prime.pow(2).mean()
    loss.backward()
    g = [p.grad for p in mem.parameters() if p.requires_grad and p.grad is not None]
    assert len(g) > 0, "无梯度!"
    print(f"    MS {tuple(ms.shape)}, HQ' {tuple(hq_prime.shape)}, {len(g)} 个张量有梯度  OK")


def test_zero_init_identity():
    """zero_init_out=True 时初始 MS=0, HQ'=HQ_stu 恒等."""
    print("[2] 零初始化恒等启动")
    mem = TransMem(_small_cfg(zero_init_out=True))
    B, S, D = 3, mem.config.n_mem + 1, mem.config.dim
    X = torch.randn(B, S, D)
    hq_stu = X[:, -1, :].clone()
    with torch.no_grad():
        ms = mem(X)
        hq_prime = mem.correct(ms, hq_stu)
    assert torch.allclose(ms, torch.zeros_like(ms), atol=1e-6), "初始 MS 非 0"
    assert torch.allclose(hq_prime, hq_stu, atol=1e-6), "初始 HQ' != HQ_stu"
    print("    MS=0, HQ'=HQ_stu  OK")


def test_pos_modes_and_mask():
    """三种 pos_mode + causal 开关都能前向."""
    print("[3] pos_mode {none,rope,learned} x causal {T,F}")
    for pos in ("none", "rope", "learned"):
        for causal in (True, False):
            mem = TransMem(_small_cfg(pos_mode=pos, causal=causal))
            B, S, D = 2, mem.config.n_mem + 1, mem.config.dim
            X = torch.randn(B, S, D)
            ms = mem(X)
            assert ms.shape == (B, D)
            # 非零初始化下应能产出非零 MS(排除恒等退化误判)
            mem2 = TransMem(_small_cfg(pos_mode=pos, causal=causal, zero_init_out=False))
            ms2 = mem2(torch.randn(B, S, D))
            assert ms2.abs().sum() > 0
    print("    6 组组合全部前向 OK")


def test_distill_loss():
    """散度损失穿冻结 LM head 回传到 TransMem; 教师侧 detach."""
    print("[4] DistillLoss 穿 FrozenLMHead 反传")
    cfg = _small_cfg(zero_init_out=False)   # 关零初始化, 保证有非零梯度
    mem = TransMem(cfg)
    vocab = 100
    lm_head = FrozenLMHead(torch.randn(vocab, cfg.dim) * 0.02)
    assert not lm_head.proj.weight.requires_grad, "lm_head 应冻结"

    B, S, D = 5, cfg.n_mem + 1, cfg.dim
    X = torch.randn(B, S, D)
    hq_stu = X[:, -1, :].clone()
    hq_tea = torch.randn(B, D)

    for div in ("forward_kl", "reverse_kl", "jsd"):
        mem.zero_grad()
        ms = mem(X)
        hq_prime = mem.correct(ms, hq_stu)
        student_logits = lm_head(hq_prime)
        teacher_logits = lm_head(hq_tea)
        loss_fn = DistillLoss(divergence=div, temperature=2.0, reg_weight=0.1)
        loss, metrics = loss_fn(student_logits, teacher_logits, hq_prime, hq_tea)
        loss.backward()
        grads = [p.grad for p in mem.parameters()
                 if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
        assert loss.item() > 0, f"{div} loss<=0"
        assert len(grads) > 0, f"{div} 无梯度回传到 TransMem"
        # lm_head 冻结, 不应有梯度
        assert lm_head.proj.weight.grad is None, "lm_head 不该有梯度"
        print(f"    {div:11s}: loss={metrics['loss']:.4f} div={metrics['div']:.4f} "
              f"reg={metrics.get('reg', 0):.4f}  grad OK")


def test_param_count():
    print("[5] 参数量")
    mem = TransMem(_small_cfg())
    print(f"    small: {mem.num_params():,} (trainable {mem.num_params(True):,})")


# ═══════════════════════════════════════════════════════════════════════
# 序列语义 (docs/version2/transmem正常化修改意见.md §5): 因果依赖 / KV cache /
# padding 不变性 —— 钉死 "query i 看 {HM, HQ_1..i}" 与 token-by-token 推理一致
# ═══════════════════════════════════════════════════════════════════════

def test_all_queries_causality():
    """return_all_queries: 扰动 HQ_1 -> 全部 MS 变; 扰动 HQ_M -> 只 MS_M 变 (因果)."""
    print("[6] return_all_queries 因果依赖")
    torch.manual_seed(1)
    cfg = _small_cfg(zero_init_out=False)
    mem = TransMem(cfg).eval()
    B, N, M, D = 2, cfg.n_mem, 5, cfg.dim
    X = torch.randn(B, N + M, D)
    with torch.no_grad():
        ms_all = mem(X, return_all_queries=True)                 # [B, M, D]
        assert ms_all.shape == (B, M, D), ms_all.shape
        # 默认读末位 == 并行读出的最后一个 query 位
        ms_last = mem(X)
        assert torch.allclose(ms_last, ms_all[:, -1, :], atol=1e-5), "末位读出不一致"
        # 扰动第一个 query: 它自己和所有后续 query 都能看到它 -> 全变
        X1 = X.clone(); X1[:, N, :] += 1.0
        ms1 = mem(X1, return_all_queries=True)
        assert not torch.allclose(ms1, ms_all, atol=1e-4), "扰动 HQ_1 后 MS 未变"
        assert (ms1 - ms_all).abs().amax(dim=(0, 2)).min() > 1e-6, \
            "扰动 HQ_1 后存在完全不变的后续 MS (历史没被看到)"
        # 扰动最后一个 query: 前面的 query 看不到未来 -> MS_1..M-1 不变
        X2 = X.clone(); X2[:, -1, :] += 1.0
        ms2 = mem(X2, return_all_queries=True)
        assert torch.allclose(ms2[:, :-1, :], ms_all[:, :-1, :], atol=1e-5), \
            "扰动 HQ_M 影响了更早的 MS (因果泄漏!)"
        assert not torch.allclose(ms2[:, -1, :], ms_all[:, -1, :], atol=1e-4), \
            "扰动 HQ_M 后 MS_M 未变"
    # 零初始化: 任意序列长下 MS 恒为 0 (恒等启动与历史长度无关)
    mem0 = TransMem(_small_cfg(zero_init_out=True)).eval()
    with torch.no_grad():
        ms0 = mem0(X, return_all_queries=True)
    assert torch.allclose(ms0, torch.zeros_like(ms0), atol=1e-6), "零初始化下 MS != 0"
    print("    HQ_1 全传播 / HQ_M 无泄漏 / 末位一致 / 零初始化恒等  OK")


def test_kv_cache_matches_full():
    """增量 KV cache 前向 (prefill [HM;HQ_1] + 逐 token) == 整段并行前向."""
    print("[7] KV cache 增量 == 整段并行")
    from transformers.cache_utils import DynamicCache
    torch.manual_seed(2)
    for pos in ("none", "rope", "learned"):
        cfg = _small_cfg(zero_init_out=False, pos_mode=pos)
        mem = TransMem(cfg).eval()
        N, M, D = cfg.n_mem, 6, cfg.dim
        X = torch.randn(1, N + M, D)
        with torch.no_grad():
            full = mem(X, return_all_queries=True)               # [1, M, D]
            past = DynamicCache()
            steps = [mem(X[:, :N + 1, :], past_key_values=past, use_cache=True)]
            for i in range(1, M):
                steps.append(mem(X[:, N + i:N + i + 1, :],
                                 past_key_values=past, use_cache=True))
            inc = torch.stack(steps, dim=1)                      # [1, M, D]
        err = (inc - full).abs().max().item()
        assert torch.allclose(inc, full, atol=1e-4, rtol=1e-4), \
            f"pos={pos}: KV cache 与并行前向不一致 (max_err={err:.2e})"
        print(f"    pos={pos:7s}: max_err={err:.2e}  OK")


def test_trailing_padding_invariance():
    """causal mask 下尾部 padding 不影响有效 query 位 (off-policy 批内变长的前提)."""
    print("[8] 尾部 padding 不变性")
    torch.manual_seed(3)
    cfg = _small_cfg(zero_init_out=False)
    mem = TransMem(cfg).eval()
    N, M, P, D = cfg.n_mem, 4, 3, cfg.dim
    X = torch.randn(1, N + M, D)
    X_pad = torch.cat([X, torch.randn(1, P, D)], dim=1)          # 尾部垫 P 个垃圾位
    with torch.no_grad():
        ms = mem(X, return_all_queries=True)                     # [1, M, D]
        ms_pad = mem(X_pad, return_all_queries=True)[:, :M, :]   # 取前 M 个有效位
    err = (ms_pad - ms).abs().max().item()
    assert torch.allclose(ms_pad, ms, atol=1e-5), \
        f"尾部 padding 改变了有效位输出 (max_err={err:.2e}) — causal mask 失效?"
    print(f"    max_err={err:.2e}  OK")


def test_real_config():
    """加载真实 config.json (dim=2560) 构建并前向一次, 验证配置可用."""
    print("[9] 真实 config.json 构建 + 前向 (dim=2560)")
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    cfg = TransMemConfig.from_json(cfg_path)
    cfg.attn_impl = "eager"   # CPU
    mem = TransMem(cfg)
    print(f"    params: {mem.num_params():,}")
    B, S, D = 2, cfg.n_mem + 1, cfg.dim
    X = torch.randn(B, S, D)
    ms = mem(X)
    assert ms.shape == (B, D)
    mem.correct(ms, X[:, -1, :]).pow(2).mean().backward()
    print(f"    MS {tuple(ms.shape)} + 反传 OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    print("=" * 60)
    test_forward_backward()
    test_zero_init_identity()
    test_pos_modes_and_mask()
    test_distill_loss()
    test_param_count()
    test_all_queries_causality()
    test_kv_cache_matches_full()
    test_trailing_padding_invariance()
    if not args.skip_real:
        test_real_config()
    print("=" * 60)
    print("✅ 全部通过")


if __name__ == "__main__":
    main()
