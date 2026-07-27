"""Keep a nonzero GPU-memory anchor while lending free memory to model loading."""

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
    parser.add_argument("--anchor-mb", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable on physical GPU {args.gpu_id}.")

    args.guard_dir.mkdir(parents=True, exist_ok=True)
    anchor_bytes = args.anchor_mb * 1024 * 1024
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if anchor_bytes <= 0 or anchor_bytes > free_bytes:
        raise RuntimeError(f"Cannot allocate the guard anchor on physical GPU {args.gpu_id}.")
    anchor = torch.empty(anchor_bytes, dtype=torch.uint8, device="cuda:0")

    def reserve_extra():
        free_now, _ = torch.cuda.mem_get_info(0)
        if args.reserve_mb is None:
            allocation_bytes = free_now - args.leave_mb * 1024 * 1024
        else:
            allocation_bytes = max(0, args.reserve_mb * 1024 * 1024 - anchor_bytes)
        if allocation_bytes <= 0:
            return None
        return torch.empty(allocation_bytes, dtype=torch.uint8, device="cuda:0")

    extra = reserve_extra()
    torch.cuda.synchronize(0)
    (args.guard_dir / f"ready.{args.gpu_id}").touch()
    print(f"GPU guard ready on physical GPU {args.gpu_id}.", flush=True)

    try:
        while not (args.guard_dir / "stop").exists():
            shrink_file = args.guard_dir / f"shrink.{args.gpu_id}"
            expand_file = args.guard_dir / f"expand.{args.gpu_id}"
            if shrink_file.exists():
                if extra is not None:
                    del extra
                    extra = None
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(0)
                (args.guard_dir / f"shrinked.{args.gpu_id}").touch()
                shrink_file.unlink(missing_ok=True)
            if expand_file.exists():
                if extra is None:
                    extra = reserve_extra()
                    torch.cuda.synchronize(0)
                (args.guard_dir / f"expanded.{args.gpu_id}").touch()
                expand_file.unlink(missing_ok=True)
            time.sleep(0.02)
    finally:
        del extra, anchor
        torch.cuda.empty_cache()
        torch.cuda.synchronize(0)


if __name__ == "__main__":
    main()
