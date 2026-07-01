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


def test_real_config():
    """加载真实 config.json (dim=2560) 构建并前向一次, 验证配置可用."""
    print("[6] 真实 config.json 构建 + 前向 (dim=2560)")
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
    if not args.skip_real:
        test_real_config()
    print("=" * 60)
    print("✅ 全部通过")


if __name__ == "__main__":
    main()
