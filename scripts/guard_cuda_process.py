#!/usr/bin/env python3
"""Stop an owned CUDA solver if another compute process enters its GPU."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import time
from typing import Callable, Protocol


NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7


class _ProcessInfoV3(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used_gpu_memory", ctypes.c_ulonglong),
        ("gpu_instance_id", ctypes.c_uint),
        ("compute_instance_id", ctypes.c_uint),
    ]


class NvmlComputeProcesses:
    """Minimal NVML binding that never shells out to a process scanner."""

    def __init__(self, gpu_index: int) -> None:
        self._library = ctypes.CDLL("libnvidia-ml.so.1")
        self._library.nvmlInit_v2.restype = ctypes.c_int
        self._library.nvmlShutdown.restype = ctypes.c_int
        self._library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self._library.nvmlDeviceGetComputeRunningProcesses_v3.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(_ProcessInfoV3),
        ]
        self._library.nvmlDeviceGetComputeRunningProcesses_v3.restype = (
            ctypes.c_int
        )
        self._check(self._library.nvmlInit_v2(), "nvmlInit_v2")
        self._handle = ctypes.c_void_p()
        self._check(
            self._library.nvmlDeviceGetHandleByIndex_v2(
                ctypes.c_uint(gpu_index), ctypes.byref(self._handle)
            ),
            "nvmlDeviceGetHandleByIndex_v2",
        )

    @staticmethod
    def _check(return_code: int, operation: str) -> None:
        if return_code != NVML_SUCCESS:
            raise RuntimeError(f"{operation} failed with NVML code {return_code}")

    def pids(self) -> set[int]:
        for _attempt in range(3):
            count = ctypes.c_uint(0)
            return_code = self._library.nvmlDeviceGetComputeRunningProcesses_v3(
                self._handle, ctypes.byref(count), None
            )
            if return_code == NVML_SUCCESS:
                return set()
            if return_code != NVML_ERROR_INSUFFICIENT_SIZE:
                self._check(
                    return_code, "nvmlDeviceGetComputeRunningProcesses_v3"
                )
            capacity = max(1, count.value + 4)
            processes = (_ProcessInfoV3 * capacity)()
            count = ctypes.c_uint(capacity)
            return_code = self._library.nvmlDeviceGetComputeRunningProcesses_v3(
                self._handle, ctypes.byref(count), processes
            )
            if return_code == NVML_SUCCESS:
                return {int(processes[index].pid) for index in range(count.value)}
            if return_code != NVML_ERROR_INSUFFICIENT_SIZE:
                self._check(
                    return_code, "nvmlDeviceGetComputeRunningProcesses_v3"
                )
        raise RuntimeError("the CUDA process list changed during three NVML reads")

    def close(self) -> None:
        self._check(self._library.nvmlShutdown(), "nvmlShutdown")


class ComputeProcessSource(Protocol):
    def pids(self) -> set[int]: ...


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_status(path: Path | None, payload: dict[str, object]) -> None:
    message = json.dumps(payload, sort_keys=True)
    print(message, flush=True)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(f"{message}\n", encoding="utf-8")
    os.replace(temporary, path)


def guard_solver(
    monitor: ComputeProcessSource,
    *,
    gpu_index: int,
    solver_pid: int,
    poll_seconds: float,
    interrupt_grace_seconds: float,
    status_file: Path | None,
    process_exists: Callable[[int], bool] = _process_exists,
    send_signal: Callable[[int, int], None] = os.kill,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    while process_exists(solver_pid):
        compute_pids = monitor.pids()
        unexpected = sorted(compute_pids - {solver_pid})
        if unexpected:
            _write_status(
                status_file,
                {
                    "gpu_index": gpu_index,
                    "solver_pid": solver_pid,
                    "status": "foreign_compute_detected",
                    "unexpected_pids": unexpected,
                },
            )
            send_signal(solver_pid, signal.SIGINT)
            deadline = monotonic() + interrupt_grace_seconds
            while process_exists(solver_pid) and monotonic() < deadline:
                sleep(min(0.5, poll_seconds))
            if process_exists(solver_pid):
                send_signal(solver_pid, signal.SIGTERM)
            return 3
        sleep(poll_seconds)
    _write_status(
        status_file,
        {
            "gpu_index": gpu_index,
            "solver_pid": solver_pid,
            "status": "solver_exited",
        },
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--solver-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--interrupt-grace-seconds", type=float, default=30.0)
    parser.add_argument("--status-file", type=Path)
    arguments = parser.parse_args()
    if arguments.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    if arguments.solver_pid <= 0:
        parser.error("--solver-pid must be positive")
    if arguments.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")
    if arguments.interrupt_grace_seconds < 0.0:
        parser.error("--interrupt-grace-seconds must be nonnegative")

    monitor = NvmlComputeProcesses(arguments.gpu_index)
    try:
        return guard_solver(
            monitor,
            gpu_index=arguments.gpu_index,
            solver_pid=arguments.solver_pid,
            poll_seconds=arguments.poll_seconds,
            interrupt_grace_seconds=arguments.interrupt_grace_seconds,
            status_file=arguments.status_file,
        )
    finally:
        monitor.close()


if __name__ == "__main__":
    raise SystemExit(main())
