#!/usr/bin/env python3
"""Fail fast when an in-loop training log is dominated by skipped OOM samples."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--max-oom", type=int, default=0)
    args = parser.parse_args()

    text = args.log.read_bytes().replace(b"\0", b"").decode(
        "utf-8", errors="replace")
    oom_count = text.count("OOM 跳样本")
    steps = [int(x) for x in re.findall(r"step\s+(\d+)/\d+", text)]
    grads = [float(x) for x in re.findall(r"\| grad ([0-9.eE+-]+) \|", text)]
    latest_step = max(steps, default=0)
    zero_grad = sum(value == 0.0 for value in grads)
    print(
        f"log={args.log} latest_step={latest_step} "
        f"oom_count={oom_count} zero_grad_logs={zero_grad}"
    )
    if oom_count > args.max_oom:
        print(f"FAIL: OOM count {oom_count} exceeds {args.max_oom}")
        return 1
    if zero_grad:
        print("FAIL: a logged optimizer step had zero gradient")
        return 1
    print("PASS: no OOM storm or zero-gradient optimizer step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
