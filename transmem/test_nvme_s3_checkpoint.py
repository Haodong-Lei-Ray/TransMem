"""Behavioral tests for opt-in NVMe-to-S3 checkpoint persistence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from transmem.nvme_s3_checkpoint import (
    NvmeS3CheckpointStore,
    add_nvme_s3_checkpoint_args,
    build_nvme_s3_checkpoint_store,
)


_FAKE_AWS = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import urlparse

args = sys.argv[1:]
root = Path(os.environ["FAKE_S3_ROOT"])

def s3_path(uri):
    parsed = urlparse(uri)
    return root / parsed.netloc / parsed.path.lstrip("/")

if args[:2] == ["s3", "cp"]:
    source, destination = args[2], args[3]
    if destination.startswith("s3://") and os.environ.get("FAKE_AWS_FAIL_UPLOAD") == "1":
        print("injected upload failure", file=sys.stderr)
        raise SystemExit(9)
    source_path = s3_path(source) if source.startswith("s3://") else Path(source)
    destination_path = (s3_path(destination)
                        if destination.startswith("s3://") else Path(destination))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination_path)
elif args[:2] == ["s3api", "head-object"]:
    bucket = args[args.index("--bucket") + 1]
    key = args[args.index("--key") + 1]
    target = root / bucket / key
    if not target.exists():
        print("404 Not Found", file=sys.stderr)
        raise SystemExit(255)
    print(target.stat().st_size)
else:
    print("unsupported fake aws command: " + " ".join(args), file=sys.stderr)
    raise SystemExit(2)
'''


def _store(tmp_path: Path, monkeypatch) -> NvmeS3CheckpointStore:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    aws = fake_bin / "aws"
    aws.write_text(_FAKE_AWS)
    aws.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_S3_ROOT", str(tmp_path / "fake-s3"))
    return NvmeS3CheckpointStore(
        output_dir=tmp_path / "logical-output" / "experiment",
        nvme_dir=tmp_path / "nvme" / "experiment",
        s3_uri="s3://datafrontier/leihaodong/Project4/checkpoints/experiment",
    )


def test_nvme_s3_storage_is_strictly_opt_in(tmp_path) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(tmp_path / "original-output"))
    add_nvme_s3_checkpoint_args(parser)

    default_args = parser.parse_args([])
    assert default_args.save_nvme_s3 is False
    assert build_nvme_s3_checkpoint_store(default_args) is None

    enabled_args = parser.parse_args(["--save_nvme_s3"])
    assert build_nvme_s3_checkpoint_store(enabled_args) is not None


def test_checkpoint_roundtrip_leaves_no_durable_local_pt(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path, monkeypatch)
    payload = {"global_step": 250, "weights": torch.arange(5)}

    assert store.save_payload(payload, "latest.pt")
    assert not store.local_path("latest.pt").exists()
    assert store.remote_size(store.remote_uri("latest.pt")) is not None

    staged = store.download("latest.pt", required=True)
    assert staged is not None
    restored = torch.load(staged, map_location="cpu", weights_only=False)
    assert restored["global_step"] == 250
    assert torch.equal(restored["weights"], torch.arange(5))
    store.remove_local(staged)
    assert not staged.exists()


def test_failed_upload_keeps_previous_remote_resume_and_nvme_retry(
    tmp_path, monkeypatch,
) -> None:
    store = _store(tmp_path, monkeypatch)
    assert store.save_payload({"global_step": 250}, "latest.pt")

    monkeypatch.setenv("FAKE_AWS_FAIL_UPLOAD", "1")
    assert not store.save_payload({"global_step": 500}, "latest.pt")
    assert store.local_path("latest.pt").exists()

    monkeypatch.delenv("FAKE_AWS_FAIL_UPLOAD")
    staged = store.download("latest.pt", required=True)
    restored = torch.load(staged, map_location="cpu", weights_only=False)
    assert restored["global_step"] == 250


def test_optional_missing_remote_does_not_start_from_stale_nvme(
    tmp_path, monkeypatch,
) -> None:
    store = _store(tmp_path, monkeypatch)
    stale = store.local_path("latest.pt")
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    assert store.download("latest.pt", required=False) is None
    assert not stale.exists()
