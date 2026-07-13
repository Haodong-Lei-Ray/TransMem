#!/usr/bin/env python3
"""CPU tests for the optional online HM diagnostic override."""

from __future__ import annotations

import torch

from transmem.train_onpolicy import _apply_hm_transform


def test_hm_transform_default_and_override() -> None:
    hm = torch.randn(4, 8, dtype=torch.float32)
    assert _apply_hm_transform(hm, None) is hm

    replacement = torch.zeros_like(hm)
    seen = []

    def transform(value: torch.Tensor) -> torch.Tensor:
        seen.append(value)
        return replacement

    out = _apply_hm_transform(hm, transform)
    assert seen == [hm]
    assert out is replacement


def test_hm_transform_rejects_incompatible_output() -> None:
    hm = torch.randn(4, 8)
    bad = (
        lambda value: value[:2],
        lambda value: value.to(torch.float64),
    )
    for transform in bad:
        try:
            _apply_hm_transform(hm, transform)
            raise AssertionError("incompatible HM transform output was accepted")
        except ValueError:
            pass


if __name__ == "__main__":
    test_hm_transform_default_and_override()
    test_hm_transform_rejects_incompatible_output()
    print("HM override tests passed")
