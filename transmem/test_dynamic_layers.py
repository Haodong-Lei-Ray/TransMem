#!/usr/bin/env python3
"""Dynamic injection-window tests for the v4 intermediate-layer experiments."""

from __future__ import annotations

import pytest

from transmem.layered import resolve_inject_layers


def test_resolve_dynamic_windows_from_experiment_plan():
    assert resolve_inject_layers(n_layers=36, depth=4, stop=32) == [28, 29, 30, 31]
    assert resolve_inject_layers(n_layers=36, depth=4, stop=26) == [22, 23, 24, 25]
    assert resolve_inject_layers(n_layers=36, depth=4, stop=22) == [18, 19, 20, 21]


def test_omitting_stop_preserves_last_d_layers_behavior():
    assert resolve_inject_layers(n_layers=36, depth=4) == [32, 33, 34, 35]


def test_explicit_layers_remain_supported_and_are_canonicalized():
    assert resolve_inject_layers(
        n_layers=36, explicit="31,28,30,29"
    ) == [28, 29, 30, 31]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_layers": 36}, "depth"),
        ({"n_layers": 36, "depth": 0}, "depth"),
        ({"n_layers": 36, "depth": 5, "stop": 4}, "depth"),
        ({"n_layers": 36, "depth": 4, "stop": 37}, "stop"),
        ({"n_layers": 36, "depth": 4, "stop": 0}, "stop"),
        ({"n_layers": 36, "depth": 4, "explicit": "1,2"}, "二选一"),
        ({"n_layers": 36, "explicit": "1,1"}, "重复"),
        ({"n_layers": 36, "explicit": "-1,2"}, "范围"),
        ({"n_layers": 36, "explicit": "2,36"}, "范围"),
    ],
)
def test_invalid_windows_fail_before_model_construction(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_inject_layers(**kwargs)
