"""Opt-in NVMe staging with durable S3 checkpoints.

The default trainers keep their historical local-file behavior.  When
``--save_nvme_s3`` is enabled, large ``.pt`` files are serialized to node-local
NVMe, uploaded with an S3 multipart PUT, size-verified, and removed from NVMe.
S3 only exposes a multipart object after CompleteMultipartUpload, so an upload
failure leaves the previous durable key available for resume.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import Mapping
from urllib.parse import urlparse
from uuid import uuid4

import torch


DEFAULT_ENDPOINT = "http://d-ceph-ssd-inside.pjlab.org.cn"


def add_nvme_s3_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    """Add storage-only arguments without changing the default save path."""
    parser.add_argument(
        "--save_nvme_s3",
        action="store_true",
        help=("checkpoint 先写计算节点 /nvme，再直接上传并校验 S3；"
              "不传时完全保留原本 output_dir 本地存储逻辑"),
    )
    parser.add_argument(
        "--nvme_checkpoint_dir",
        default=None,
        help="NVMe 临时目录；默认 /nvme/$USER/Project4/checkpoints/<run-name>",
    )
    parser.add_argument(
        "--s3_checkpoint_uri",
        default=None,
        help=("S3 checkpoint 目录；默认 "
              "s3://datafrontier/$USER/Project4/checkpoints/<run-name>"),
    )
    parser.add_argument(
        "--s3_endpoint_url",
        default=DEFAULT_ENDPOINT,
        help="S3-compatible endpoint",
    )


def build_nvme_s3_checkpoint_store(args) -> "NvmeS3CheckpointStore | None":
    if not bool(getattr(args, "save_nvme_s3", False)):
        return None
    return NvmeS3CheckpointStore(
        output_dir=getattr(args, "output_dir"),
        nvme_dir=getattr(args, "nvme_checkpoint_dir", None),
        s3_uri=getattr(args, "s3_checkpoint_uri", None),
        endpoint_url=getattr(args, "s3_endpoint_url", DEFAULT_ENDPOINT),
    )


class NvmeS3CheckpointStore:
    """Publish and recover checkpoint objects through an NVMe staging area."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        nvme_dir: str | Path | None = None,
        s3_uri: str | None = None,
        endpoint_url: str = DEFAULT_ENDPOINT,
    ) -> None:
        run_name = Path(output_dir).expanduser().resolve().name
        if not run_name:
            raise ValueError("output_dir 必须有非空目录名")
        user = os.environ.get("USER") or "leihaodong"
        self.local_dir = Path(
            nvme_dir
            or f"/nvme/{user}/Project4/checkpoints/{run_name}"
        ).expanduser()
        self.s3_uri = (
            s3_uri
            or f"s3://datafrontier/{user}/Project4/checkpoints/{run_name}"
        ).rstrip("/")
        self.endpoint_url = str(endpoint_url).rstrip("/")
        self._parse_s3_uri(self.s3_uri)
        self.failed_uploads: set[str] = set()

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"不是合法的 s3:// URI: {uri!r}")
        return parsed.netloc, parsed.path.lstrip("/")

    @staticmethod
    def _safe_name(name: str) -> str:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError(f"checkpoint 对象名不安全: {name!r}")
        return str(path)

    def local_path(self, name: str) -> Path:
        return self.local_dir / self._safe_name(name)

    def remote_uri(self, name: str) -> str:
        return f"{self.s3_uri}/{self._safe_name(name)}"

    def metadata(self) -> dict[str, str]:
        return {
            "mode": "nvme_s3",
            "s3_uri": self.s3_uri,
            "endpoint_url": self.endpoint_url,
        }

    def _aws_env(self) -> Mapping[str, str]:
        env = dict(os.environ)
        for name in (
            "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
            "all_proxy", "ALL_PROXY",
        ):
            env.pop(name, None)
        # aws-cli v1 otherwise requests CRC32 for multipart uploads, which the
        # DataFrontier Ceph gateway rejects (it requires SHA256).
        env["AWS_REQUEST_CHECKSUM_CALCULATION"] = "WHEN_REQUIRED"
        env["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "WHEN_REQUIRED"
        env.setdefault("AWS_DEFAULT_REGION", "us-east-1")
        env.setdefault("AWS_MAX_ATTEMPTS", "10")
        return env

    def _run_aws(
        self,
        arguments: list[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = ["aws", *arguments, "--endpoint-url", self.endpoint_url]
        return subprocess.run(
            command,
            check=check,
            text=True,
            capture_output=True,
            env=self._aws_env(),
        )

    def remote_size(self, uri: str) -> int | None:
        bucket, key = self._parse_s3_uri(uri)
        result = self._run_aws(
            [
                "s3api", "head-object", "--bucket", bucket, "--key", key,
                "--query", "ContentLength", "--output", "text",
            ],
            check=False,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
        message = f"{result.stdout}\n{result.stderr}"
        if any(marker in message for marker in (
                "404", "NoSuchKey", "Not Found", "HeadObject operation: Not Found")):
            return None
        raise RuntimeError(
            f"S3 head-object 失败 ({uri}): {message.strip()}")

    def _remove_remote_best_effort(self, uri: str) -> None:
        try:
            self._run_aws([
                "s3", "rm", uri, "--only-show-errors",
            ], check=False)
        except OSError:
            pass

    def _preserve_unpublished_local(self, path: Path) -> Path:
        """Move a failed staged checkpoint aside so resume cannot overwrite it."""
        preserved = path.with_name(
            f"{path.stem}.unpublished-{uuid4().hex}{path.suffix}")
        os.replace(path, preserved)
        return preserved

    @staticmethod
    def _remove_superseded_unpublished(path: Path) -> None:
        pattern = f"{path.stem}.unpublished-*{path.suffix}"
        for candidate in path.parent.glob(pattern):
            candidate.unlink(missing_ok=True)

    def upload_existing(
        self,
        local_path: str | Path,
        *,
        name: str | None = None,
        remove_local: bool = True,
    ) -> bool:
        """Upload one existing file, verify its byte size, then optionally unlink."""
        local = Path(local_path)
        object_name = self._safe_name(name or local.name)
        remote = self.remote_uri(object_name)
        pending_name = self._safe_name(
            f".pending/{PurePosixPath(object_name).name}.{uuid4().hex}.upload")
        pending_remote = self.remote_uri(pending_name)
        try:
            local_size = local.stat().st_size
            self._run_aws([
                "s3", "cp", str(local), pending_remote,
                "--only-show-errors", "--checksum-algorithm", "SHA256",
            ])
            pending_size = self.remote_size(pending_remote)
            if pending_size != local_size:
                raise RuntimeError(
                    "S3 临时对象字节校验失败: "
                    f"local={local_size}, pending={pending_size}")
            # S3→S3 cp performs a server-side (multipart for large objects)
            # publication. The final key changes only after that copy completes,
            # so a failed upload/copy cannot replace the previous checkpoint.
            self._run_aws([
                "s3", "cp", pending_remote, remote, "--only-show-errors",
            ])
            remote_size = self.remote_size(remote)
            if remote_size != local_size:
                raise RuntimeError(
                    f"S3 字节校验失败: local={local_size}, remote={remote_size}")
            self._remove_remote_best_effort(pending_remote)
            self._remove_superseded_unpublished(local)
            if remove_local:
                local.unlink(missing_ok=True)
            self.failed_uploads.discard(object_name)
            print(
                f"  NVMe→S3 checkpoint: {object_name} "
                f"({local_size} bytes) -> {remote}",
                flush=True,
            )
            return True
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            self.failed_uploads.add(object_name)
            # The NVMe source remains the retry source, so a fully uploaded but
            # unpublished 20 GiB pending object would only leak S3 capacity.
            self._remove_remote_best_effort(pending_remote)
            preserved = None
            if remove_local and local.exists():
                try:
                    preserved = self._preserve_unpublished_local(local)
                except OSError:
                    preserved = local
            print(
                f"  ⚠️ NVMe→S3 存档失败 ({object_name}): {exc}; "
                "保留旧 S3 checkpoint 和当前文件"
                + (f" {preserved}" if preserved is not None else ""),
                flush=True,
            )
            return False

    def assert_uploads_complete(self) -> None:
        """Fail the process at the end if its latest upload attempt did not publish."""
        if self.failed_uploads:
            names = ", ".join(sorted(self.failed_uploads))
            raise RuntimeError(
                "训练已完成，但这些对象的最近一次 S3 上传失败；"
                f"保留本地/NVMe 文件并以失败状态退出: {names}")

    def save_payload(self, payload: object, name: str) -> bool:
        """Serialize on NVMe and publish; no durable local checkpoint remains."""
        object_name = self._safe_name(name)
        target = self.local_path(object_name)
        temporary = target.with_suffix(target.suffix + ".tmp")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            temporary.unlink(missing_ok=True)
            torch.save(payload, temporary)
            os.replace(temporary, target)
        except (OSError, RuntimeError) as exc:
            temporary.unlink(missing_ok=True)
            self.failed_uploads.add(object_name)
            print(
                f"  ⚠️ NVMe 序列化失败 ({target.name}): {exc}; 跳过本次存档",
                flush=True,
            )
            return False
        return self.upload_existing(
            target, name=object_name, remove_local=True)

    def download_uri(self, uri: str, *, required: bool = False) -> Path | None:
        """Download and size-verify one S3 checkpoint into the staging directory."""
        remote_size = self.remote_size(uri)
        name = PurePosixPath(urlparse(uri).path).name
        target = self.local_path(name)
        if target.exists():
            preserved = self._preserve_unpublished_local(target)
            print(
                "  保留 requeue 前尚未确认发布的 NVMe candidate: "
                f"{preserved}",
                flush=True,
            )
        if remote_size is None:
            if required:
                raise FileNotFoundError(f"S3 checkpoint 不存在: {uri}")
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".download")
        # Size equality is insufficient for a stale failed-upload file, so a
        # remote resume always replaces any existing NVMe candidate.
        temporary.unlink(missing_ok=True)
        try:
            self._run_aws([
                "s3", "cp", uri, str(temporary), "--only-show-errors",
            ])
            local_size = temporary.stat().st_size
            if local_size != remote_size:
                raise RuntimeError(
                    f"S3 下载字节校验失败: remote={remote_size}, local={local_size}")
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        print(
            f"  S3→NVMe resume: {uri} -> {target} ({remote_size} bytes)",
            flush=True,
        )
        return target

    def download(self, name: str, *, required: bool = False) -> Path | None:
        return self.download_uri(self.remote_uri(name), required=required)

    def remove_local(self, path: str | Path | None) -> None:
        if path is None:
            return
        candidate = Path(path).expanduser().resolve()
        root = self.local_dir.expanduser().resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return
        candidate.unlink(missing_ok=True)
