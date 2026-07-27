"""Runtime setup that must happen before importing torch or vLLM."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path


def parse_gpu_devices(gpu_devices: str) -> list[str]:
    """Validate and split a comma-separated physical GPU list."""
    devices = [device.strip() for device in gpu_devices.split(",") if device.strip()]
    if not devices or any(not device.isdigit() for device in devices):
        raise ValueError("--gpu-devices must be a comma-separated list such as 0,1,2,3.")
    if len(set(devices)) != len(devices):
        raise ValueError("--gpu-devices must not contain duplicate GPU IDs.")
    return devices


def configure_cuda_visibility(gpu_devices: str) -> list[str]:
    """Expose only the requested physical GPUs to this process."""
    devices = parse_gpu_devices(gpu_devices)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    return devices


@dataclass
class GPUReservation:
    """Temporarily hold GPU memory until vLLM begins to allocate its KV cache."""

    memory_mb: int | None
    device_count: int
    free_memory_mb: int = 512
    device_offset: int = 0
    external_guard_dir: str | None = None
    external_guard_ids: tuple[str, ...] = ()
    _buffers: list[object] = field(default_factory=list, init=False, repr=False)
    _external_guard_active: bool = field(default=False, init=False, repr=False)
    _external_guard_shrunk: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._external_guard_active = bool(self.external_guard_dir)

    def reserve(self) -> None:
        if self.memory_mb == 0:
            return
        if self._external_guard_active:
            if self._external_guard_shrunk:
                self._signal_external_guard("expand")
                self._external_guard_shrunk = False
            return
        if self._buffers:
            return

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("GPU reservation was requested, but CUDA is unavailable.")
        required_device_count = self.device_offset + self.device_count
        if torch.cuda.device_count() < required_device_count:
            raise RuntimeError(
                f"Requested visible GPU index {required_device_count - 1}, but only "
                f"{torch.cuda.device_count()} are available."
            )

        try:
            for device_index in range(self.device_offset, required_device_count):
                free_bytes, _ = torch.cuda.mem_get_info(device_index)
                if self.memory_mb is None:
                    bytes_per_device = free_bytes - self.free_memory_mb * 1024 * 1024
                    if bytes_per_device <= 0:
                        raise RuntimeError(
                            f"GPU {device_index} has no free memory beyond the "
                            f"{self.free_memory_mb} MiB safety margin."
                        )
                else:
                    bytes_per_device = self.memory_mb * 1024 * 1024
                    if bytes_per_device > free_bytes:
                        raise RuntimeError(
                            f"Cannot reserve {self.memory_mb} MiB on GPU {device_index}; "
                            f"only {free_bytes // (1024 * 1024)} MiB is currently free."
                        )

                # cudaMalloc, which backs torch.empty, immediately owns device memory.
                buffer = torch.empty(bytes_per_device, dtype=torch.uint8, device=f"cuda:{device_index}")
                self._buffers.append(buffer)
                torch.cuda.synchronize(device_index)
        except Exception:
            self.release()
            raise

        if self.memory_mb is None:
            print(
                f"Reserved all currently free GPU memory except {self.free_memory_mb} MiB "
                f"on GPU index(es) {self._device_label()} until the model starts."
            )
        else:
            print(
                f"Reserved {self.memory_mb} MiB on GPU index(es) {self._device_label()} "
                "until the model starts."
            )

    def release(self) -> None:
        if self._external_guard_active:
            if not self._external_guard_shrunk:
                self._signal_external_guard("shrink")
                self._external_guard_shrunk = True
            return
        if not self._buffers:
            return

        self._buffers.clear()
        import torch

        for device_index in range(self.device_offset, self.device_offset + self.device_count):
            with torch.cuda.device(device_index):
                torch.cuda.empty_cache()
            torch.cuda.synchronize(device_index)
        print(f"Released GPU reservation on index(es) {self._device_label()} for model loading.")

    def _device_label(self) -> str:
        return ",".join(
            str(index) for index in range(self.device_offset, self.device_offset + self.device_count)
        )

    def _signal_external_guard(self, action: str) -> None:
        guard_dir = Path(self.external_guard_dir)
        if not self.external_guard_ids:
            raise RuntimeError("External GPU guard IDs are missing.")
        for gpu_id in self.external_guard_ids:
            (guard_dir / f"{action}ed.{gpu_id}").unlink(missing_ok=True)
            (guard_dir / f"{action}.{gpu_id}").touch()
        deadline = time.monotonic() + 60
        while any(not (guard_dir / f"{action}ed.{gpu_id}").exists() for gpu_id in self.external_guard_ids):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for the GPU guard to {action} memory.")
            time.sleep(0.05)
        print(f"External GPU guard {action}ed memory on GPU(s) {','.join(self.external_guard_ids)}.")
