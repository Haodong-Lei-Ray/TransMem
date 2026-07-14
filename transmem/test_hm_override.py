#!/usr/bin/env python3
"""CPU tests for the optional online HM diagnostic override."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from scripts.eval.eval_hm_ablation import (
    build_run_config,
    load_progress,
    make_run_id,
    parse_args,
)
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


def test_progress_resume_rejects_other_runs_unless_forced() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "real.progress.jsonl"

        path.write_text(json.dumps({"sample_idx": 3}) + "\n")
        try:
            load_progress(path, expected_run_id="run-a")
            raise AssertionError("legacy progress without run_id was accepted")
        except RuntimeError as exc:
            assert "--force" in str(exc)

        path.write_text(json.dumps({"sample_idx": 3, "run_id": "run-b"}) + "\n")
        try:
            load_progress(path, expected_run_id="run-a")
            raise AssertionError("progress from a different run was accepted")
        except RuntimeError as exc:
            assert "run-b" in str(exc)

        path.write_text(json.dumps({"sample_idx": 3, "run_id": "run-a"}) + "\n")
        rows = load_progress(path, expected_run_id="run-a")
        assert rows == {3: {"sample_idx": 3, "run_id": "run-a"}}

        assert load_progress(path, expected_run_id="run-c", force=True) == {}
        assert not path.exists()


def test_run_id_covers_inputs_but_not_temporary_model_mount_path() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)

        def write(relative: str, content: bytes) -> Path:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            return path

        checkpoint = write("best.pt", b"AAAA")
        eval_file = write("eval.json", b"eval-v1")
        stage0_dir = root / "stage0"
        manifest = [{"sample_idx": 7, "file": "shard/sample.pt"}]
        meta_bytes = json.dumps({
            "N": 4,
            "dim": 8,
            "samples": manifest,
        }, sort_keys=True).encode()
        write("stage0/meta.json", meta_bytes)
        sample_bytes = b"stage0-sample-v1"
        write("stage0/shard/sample.pt", sample_bytes)
        transmem_config = write("transmem.json", b'{"depth": 4}')

        for model_dir in ("mount-a/model", "mount-b/model"):
            write(f"{model_dir}/config.json", b'{"model_type": "qwen3"}')
            write(f"{model_dir}/tokenizer_config.json", b'{"chat_template": "v1"}')
            write(f"{model_dir}/model.safetensors.index.json", b'{"weight_map": {}}')
            write(f"{model_dir}/model-00001-of-00001.safetensors", b"weights")

        args = SimpleNamespace(
            ckpt=str(checkpoint),
            eval_file=str(eval_file),
            data_format="longmemeval",
            model_path=str(root / "mount-a/model"),
            variant="shuffled",
            seed=11,
            max_samples=20,
            max_answer_tokens=5,
            hm_tolerance=0.05,
            dtype="bfloat16",
            attn_impl="sdpa",
        )
        config = build_run_config(
            args, stage0_dir, manifest, transmem_config)
        run_id = make_run_id(config)

        assert config["checkpoint"]["sha256"] == hashlib.sha256(b"AAAA").hexdigest()
        assert config["checkpoint"]["size"] == 4
        assert "mtime_ns" in config["checkpoint"]
        assert config["eval_file"]["sha256"] == hashlib.sha256(b"eval-v1").hexdigest()
        assert config["stage0"]["meta"]["sha256"] == hashlib.sha256(meta_bytes).hexdigest()
        assert config["stage0"]["samples"][0]["artifact"]["sha256"] == (
            hashlib.sha256(sample_bytes).hexdigest())
        assert config["model"]["config_files"]["config.json"]["sha256"] == (
            hashlib.sha256(b'{"model_type": "qwen3"}').hexdigest())
        assert config["decode"] == {
            "strategy": "greedy",
            "sample": False,
            "temperature": 1.0,
            "max_answer_tokens": 5,
        }

        relocated_args = SimpleNamespace(**vars(args))
        relocated_args.model_path = str(root / "mount-b/model")
        relocated_config = build_run_config(
            relocated_args, stage0_dir, manifest, transmem_config)
        assert make_run_id(relocated_config) == run_id

        for field, value in (
                ("variant", "zero"),
                ("seed", 12),
                ("max_samples", 19),
                ("max_answer_tokens", 6),
                ("dtype", "float32"),
                ("attn_impl", "eager")):
            changed_args = SimpleNamespace(**vars(args))
            setattr(changed_args, field, value)
            changed_config = build_run_config(
                changed_args, stage0_dir, manifest, transmem_config)
            assert make_run_id(changed_config) != run_id, field

        checkpoint_stat = checkpoint.stat()
        checkpoint.write_bytes(b"BBBB")
        os.utime(checkpoint, ns=(
            checkpoint_stat.st_atime_ns, checkpoint_stat.st_mtime_ns))
        changed_config = build_run_config(
            args, stage0_dir, manifest, transmem_config)
        assert changed_config["checkpoint"]["size"] == config["checkpoint"]["size"]
        assert changed_config["checkpoint"]["mtime_ns"] == config["checkpoint"]["mtime_ns"]
        assert make_run_id(changed_config) != run_id


def test_force_is_an_explicit_cli_opt_in() -> None:
    required = [
        "--eval_file", "eval.json",
        "--data_format", "longmemeval",
        "--stage0_dir", "stage0",
        "--model_path", "model",
        "--ckpt", "best.pt",
        "--variant", "real",
        "--output_json", "real.json",
    ]
    assert not parse_args(required).force
    assert parse_args([*required, "--force"]).force


if __name__ == "__main__":
    test_hm_transform_default_and_override()
    test_hm_transform_rejects_incompatible_output()
    test_progress_resume_rejects_other_runs_unless_forced()
    test_run_id_covers_inputs_but_not_temporary_model_mount_path()
    test_force_is_an_explicit_cli_opt_in()
    print("HM override tests passed")
