"""Real DataFrontier round-trip smoke test for NVMe-staged checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from transmem.nvme_s3_checkpoint import NvmeS3CheckpointStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvme-dir", required=True)
    parser.add_argument("--s3-uri", required=True)
    parser.add_argument("--endpoint-url", required=True)
    args = parser.parse_args()

    store = NvmeS3CheckpointStore(
        output_dir=Path(args.nvme_dir).name,
        nvme_dir=args.nvme_dir,
        s3_uri=args.s3_uri,
        endpoint_url=args.endpoint_url,
    )
    payload = {
        "global_step": 250,
        "sentinel": torch.arange(4096, dtype=torch.int64),
    }

    if not store.save_payload(payload, "latest.pt"):
        raise RuntimeError("NVMe→S3 smoke upload failed")
    if store.local_path("latest.pt").exists():
        raise RuntimeError("verified upload did not remove the NVMe staging file")

    restored_path = store.download("latest.pt", required=True)
    if restored_path is None:
        raise RuntimeError("S3→NVMe smoke download returned no file")
    restored = torch.load(restored_path, map_location="cpu", weights_only=False)
    if restored["global_step"] != payload["global_step"]:
        raise RuntimeError("round-trip global_step mismatch")
    if not torch.equal(restored["sentinel"], payload["sentinel"]):
        raise RuntimeError("round-trip tensor mismatch")
    store.remove_local(restored_path)
    print("REAL_NVME_S3_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
