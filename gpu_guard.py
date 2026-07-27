"""Reserve one physical GPU until the main experiment process asks for release."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--guard-dir", type=Path, required=True)
    parser.add_argument("--leave-mb", type=int, default=512)
    parser.add_argument("--reserve-mb", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable on physical GPU {args.gpu_id}.")

    free_bytes, _ = torch.cuda.mem_get_info(0)
    if args.reserve_mb is None:
        allocation_bytes = free_bytes - args.leave_mb * 1024 * 1024
    else:
        allocation_bytes = args.reserve_mb * 1024 * 1024
    if allocation_bytes <= 0 or allocation_bytes > free_bytes:
        raise RuntimeError(f"Cannot reserve memory on physical GPU {args.gpu_id}.")

    # cudaMalloc backs torch.empty, so this immediately claims device memory.
    buffer = torch.empty(allocation_bytes, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize(0)
    args.guard_dir.mkdir(parents=True, exist_ok=True)
    (args.guard_dir / f"ready.{args.gpu_id}").touch()
    print(f"GPU guard ready on physical GPU {args.gpu_id}.", flush=True)

    try:
        while not (args.guard_dir / "release").exists():
            time.sleep(0.05)
    finally:
        del buffer
        torch.cuda.empty_cache()
        torch.cuda.synchronize(0)
        (args.guard_dir / f"done.{args.gpu_id}").touch()


if __name__ == "__main__":
    main()
