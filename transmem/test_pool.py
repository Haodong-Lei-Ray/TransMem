#!/usr/bin/env python3
"""记忆池 (pool_ns) CPU 单测: 取位嵌套性 / 池切片一致性 / Dataset 按 N 切片.

python -m transmem.test_pool
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transmem.extract_features import hm_positions, Stage0Extractor, atomic_save
from transmem.train_offpolicy import OffPolicyDataset, collate_sequences

POOL_NS = [4, 8, 16, 32, 64, 128, 256, 384]


def test_frac_nesting():
    """N | N' 时 positions(N) ⊆ positions(N'); 末槽 = len_cl-1; 位置单调不减."""
    for L in [30000, 29993, 997, 640, 511, 384, 100, 63, 8, 5, 1]:
        for n in POOL_NS:
            pos = hm_positions(L, n, "frac")
            assert len(pos) == n
            assert pos[-1] == max(L - 1, 0), (L, n, pos[-1])
            assert all(0 <= p < max(L, 1) for p in pos)
            assert pos == sorted(pos)
        for a in POOL_NS:
            for b in POOL_NS:
                if b % a == 0 and a < b:
                    pa, pb = hm_positions(L, a, "frac"), set(hm_positions(L, b, "frac"))
                    assert set(pa) <= pb, f"L={L}: pos({a}) ⊄ pos({b})"
    # floor 模式回归: 与历史公式逐点一致
    for L in [30000, 997, 63, 5]:
        for n in [4, 8]:
            seg = max(L // n, 1)
            legacy = [max(min((i + 1) * seg, L) - 1, 0) for i in range(n)]
            assert hm_positions(L, n, "floor") == legacy
    print("[1] frac 嵌套性 + floor 回归 ✓")


def _pool_extractor():
    args = SimpleNamespace(device="cpu", dtype="float32", save_dtype="float32",
                           N=4, trajectory="teacher", hm_mode="frac",
                           pool_ns=",".join(map(str, POOL_NS)))
    return Stage0Extractor(args)


def test_pool_extract_hm():
    """池行按 hm_maps 切出的 == 直接按该 N 的 frac 位置取的 hidden."""
    ex = _pool_extractor()
    dim = 16
    for L in [30000, 640, 511, 63, 5]:
        hidden = torch.randn(L + 40, dim)          # 序列比 len_cl 长 (Q+A 部分)
        hm, extras = ex._extract_hm(hidden, L)
        P = hm.shape[0]
        assert extras["hm_pos"].shape[0] == P and extras["len_cl"] == L
        assert P <= 512, (L, P)
        for n in POOL_NS:
            direct = hidden[torch.tensor(hm_positions(L, n, "frac"))]
            sliced = hm[extras["hm_maps"][str(n)]]
            assert torch.equal(direct, sliced), (L, n)
    # 30k 上下文时并集应恰为 512 (256∪384, 重叠 128)
    hm, extras = ex._extract_hm(torch.randn(30040, 4), 30000)
    assert hm.shape[0] == 512, hm.shape
    print("[2] 池切片 == 直接取位 (P_30k=512) ✓")


def _write_fake_stage0(root: Path, pool: bool, n_samples=3, dim=8):
    """最小 stage0 目录: shard_0000/*.pt + meta.json (可选池化)."""
    shard = root / "shard_0000"
    shard.mkdir(parents=True)
    ex = _pool_extractor()
    manifest = []
    total = 0
    for s in range(n_samples):
        L = 200 + 37 * s
        M = 2 + s
        hidden = torch.randn(L + 20, dim)
        if pool:
            hm, extras = ex._extract_hm(hidden, L)
        else:
            hm = hidden[torch.tensor(hm_positions(L, 4, "floor"))]
            extras = {}
        d = {"hm_stu": hm, "hq_stu": torch.randn(M, dim), "hq_tea": torch.randn(M, dim),
             "answer_ids": torch.arange(M), "answer_text": "x", "sample_idx": s,
             "N": (None if pool else 4), "dim": dim}
        d.update(extras)
        atomic_save(d, shard / f"sample_{s:05d}.pt")
        manifest.append({"sample_idx": s, "shard_idx": 0,
                         "file": f"shard_0000/sample_{s:05d}.pt", "M": M})
        total += M
    meta = {"N": (None if pool else 4), "dim": dim,
            "pool_ns": (POOL_NS if pool else None),
            "hm_mode": ("frac" if pool else "floor"), "save_dtype": "float32",
            "total_pairs": total, "samples": manifest}
    json.dump(meta, open(root / "meta.json", "w"))


def test_dataset_slicing():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "pool_dir"
        _write_fake_stage0(root, pool=True)
        for n in [4, 64, 384]:
            ds = OffPolicyDataset(str(root), load_dtype=torch.float32, n_mem=n)
            assert ds.N == n
            hm, hq_s, hq_t, ans = ds[1]
            assert hm.shape == (n, 8) and hq_s.shape[0] == 3
            # 与原始 .pt 对照: 池行 -> maps 切片
            raw = torch.load(root / "shard_0000/sample_00001.pt", weights_only=False)
            assert torch.equal(hm, raw["hm_stu"][raw["hm_maps"][str(n)]])
        # collate 形状
        ds = OffPolicyDataset(str(root), load_dtype=torch.float32, n_mem=8)
        X, hq_tea, ans, q_mask = collate_sequences([ds[i] for i in range(3)])
        assert X.shape == (3, 8 + 4, 8) and q_mask.shape == (3, 4)
        # 错误路径: 池化目录缺 n_mem / n_mem 不在 pool_ns
        for bad in [None, 5]:
            try:
                OffPolicyDataset(str(root), n_mem=bad)
                raise AssertionError(f"n_mem={bad} 应当报错")
            except RuntimeError:
                pass
        # 旧格式目录: 无 n_mem 照常, 给一致 n_mem 通过, 不一致报错
        legacy = Path(td) / "legacy_dir"
        _write_fake_stage0(legacy, pool=False)
        assert OffPolicyDataset(str(legacy)).N == 4
        assert OffPolicyDataset(str(legacy), n_mem=4).N == 4
        try:
            OffPolicyDataset(str(legacy), n_mem=8)
            raise AssertionError("旧格式 n_mem=8 应当报错")
        except RuntimeError:
            pass
    print("[3] Dataset 池切片 + 旧格式兼容 ✓")


if __name__ == "__main__":
    test_frac_nesting()
    test_pool_extract_hm()
    test_dataset_slicing()
    print("\n全部通过 ✓")
