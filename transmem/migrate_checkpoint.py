#!/usr/bin/env python3
"""Explicitly migrate a fixed-gate TransMem checkpoint to dynamic gate format."""

from __future__ import annotations

import argparse

from .checkpoints import migrate_legacy_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_ckpt", required=True)
    parser.add_argument("--dst_ckpt", required=True)
    parser.add_argument(
        "--gate_mode", default="centered_sigmoid", choices=["centered_sigmoid"])
    parser.add_argument("--gate_max", type=float, default=2.0)
    parser.add_argument("--gate_temperature", type=float, default=1.0)
    parser.add_argument("--gate_init", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    migrate_legacy_checkpoint(
        args.src_ckpt,
        args.dst_ckpt,
        gate_config={
            "gate_mode": args.gate_mode,
            "gate_granularity": "token_scalar",
            "gate_max": args.gate_max,
            "gate_temperature": args.gate_temperature,
            "gate_init": args.gate_init,
        },
    )
    print(f"Migrated dynamic-gate checkpoint: {args.dst_ckpt}")


if __name__ == "__main__":
    main()
