#!/usr/bin/env python3
"""Run only the seed and Torch stages of a q-e plan under an NVML guard."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from threading import Thread
import time
from typing import Callable, Protocol

from guard_cuda_process import NvmlComputeProcesses, guard_solver
from plan_wave_calibration_runs import RunPlanRow, build_plan


EX_SOFTWARE = 70
EX_TEMPFAIL = 75
EX_CONFIG = 78

# The n512 solver has a measured 19,326 MiB A10 footprint.  A 22 GiB total
# floor admits the 23,028 MiB A10 while the 21 GiB free floor preserves more
# than 2 GiB above that observation.  The n768 manifest estimate is 54 GiB;
# it remains restricted to an 80 GiB-class GPU with 64 GiB actually free.
# These floors complement, rather than replace, the NVML process collision
# guard before and during every owned stage.
MIN_TOTAL_MEMORY_MIB = {512: 22 * 1024, 768: 80 * 1024}
MIN_FREE_MEMORY_MIB = {512: 21 * 1024, 768: 64 * 1024}
GPU_STAGES = {"seed", "torch"}


class _MemoryInfo(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class GpuMonitor(Protocol):
    def pids(self) -> set[int]: ...

    def total_memory_mib(self) -> int: ...

    def free_memory_mib(self) -> int: ...

    def close(self) -> None: ...


class NvmlGpuMonitor(NvmlComputeProcesses):
    """NVML process source with a one-time physical-memory query."""

    def __init__(self, gpu_index: int) -> None:
        super().__init__(gpu_index)
        self._library.nvmlDeviceGetMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_MemoryInfo),
        ]
        self._library.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int

    def _memory_info(self) -> _MemoryInfo:
        info = _MemoryInfo()
        self._check(
            self._library.nvmlDeviceGetMemoryInfo(
                self._handle, ctypes.byref(info)
            ),
            "nvmlDeviceGetMemoryInfo",
        )
        return info

    def total_memory_mib(self) -> int:
        return int(self._memory_info().total // (1024 * 1024))

    def free_memory_mib(self) -> int:
        return int(self._memory_info().free // (1024 * 1024))


class DuplicateRunner(RuntimeError):
    pass


class GuardWaitFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class PlannerInputs:
    manifest: Path
    cases: Path
    initial_root: Path
    torch_root: Path
    pyul_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Publish JSON with file and directory durability before returning."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def acquire_gpu_lock(log_root: Path, gpu_index: int):
    """Acquire the one nonblocking runner lock for a physical GPU index."""

    log_root.mkdir(parents=True, exist_ok=True)
    path = log_root / f"qe_guard_gpu{gpu_index}.lock"
    stream = path.open("a", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        stream.close()
        raise DuplicateRunner(
            f"another guarded q-e runner holds GPU {gpu_index}"
        ) from error
    return stream


def _read_json_status(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unreadable_guard_failure_status"}
    return payload if isinstance(payload, dict) else {"status": "invalid_status"}


def _check_wait_guard_statuses(paths: tuple[Path, ...]) -> None:
    for path in paths:
        payload = _read_json_status(path)
        if payload is None:
            continue
        status = payload.get("status")
        if status == "solver_exited":
            continue
        if status == "foreign_compute_detected":
            raise GuardWaitFailure(
                f"status=foreign_compute_detected reported by {path}"
            )
        raise GuardWaitFailure(
            f"unsafe or malformed guard status {status!r} in {path}"
        )


def wait_for_path(
    path: Path,
    *,
    guard_status_paths: tuple[Path, ...],
    cadence_seconds: float,
    timeout_seconds: float | None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Wait only on file existence; never scan the process table."""

    if cadence_seconds < 5.0:
        raise ValueError("wait cadence must be at least five seconds")
    deadline = None if timeout_seconds is None else monotonic() + timeout_seconds
    while True:
        _check_wait_guard_statuses(guard_status_paths)
        if path.exists():
            return
        if deadline is not None and monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        sleep(cadence_seconds)


def _stop_owned_process(process: subprocess.Popen, *, grace_seconds: float = 20.0) -> None:
    """Stop only the child process group created by this runner."""

    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10.0)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()


class GuardedQeRunner:
    def __init__(
        self,
        *,
        planner_inputs: PlannerInputs,
        log_root: Path,
        gpu_index: int,
        poll_seconds: float,
        interrupt_grace_seconds: float,
        marker_timeout_seconds: float = 60.0,
        wait_until_gpu_empty: bool = False,
        gpu_empty_timeout_seconds: float | None = None,
        plan_builder: Callable[..., list[RunPlanRow]] = build_plan,
        monitor_factory: Callable[[int], GpuMonitor] = NvmlGpuMonitor,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        guard_function: Callable[..., int] = guard_solver,
        stop_process: Callable[[subprocess.Popen], None] = _stop_owned_process,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.inputs = planner_inputs
        self.log_root = log_root
        self.gpu_index = gpu_index
        self.poll_seconds = poll_seconds
        self.interrupt_grace_seconds = interrupt_grace_seconds
        self.marker_timeout_seconds = marker_timeout_seconds
        self.wait_until_gpu_empty = wait_until_gpu_empty
        self.gpu_empty_timeout_seconds = gpu_empty_timeout_seconds
        self.plan_builder = plan_builder
        self.monitor_factory = monitor_factory
        self.popen_factory = popen_factory
        self.guard_function = guard_function
        self.stop_process = stop_process
        self.sleep = sleep
        self.monotonic = monotonic
        self.guard_failure_status = log_root / f"qe_gpu{gpu_index}_guard_failure.json"

    def _plan(self) -> list[RunPlanRow]:
        return self.plan_builder(
            self.inputs.manifest,
            self.inputs.cases,
            self.inputs.initial_root,
            self.inputs.torch_root,
            self.inputs.pyul_path,
            device="cuda:0",
        )

    def _stage_status_path(self, row: RunPlanRow, stage: str) -> Path:
        return self.log_root / "status" / f"{row.run_id}.{stage}.json"

    def _guard_status_path(self, row: RunPlanRow, stage: str) -> Path:
        return self.log_root / "status" / f"{row.run_id}.{stage}.guard.json"

    def _pid_marker(self, row: RunPlanRow) -> Path:
        return self.log_root / "pids" / f"{row.run_id}.solver.pid"

    def _write_stage_status(
        self,
        row: RunPlanRow,
        stage: str,
        *,
        status: str,
        pid: int | None,
        started_at: str | None,
        exit_code: int | None,
        detail: str | None = None,
    ) -> None:
        now = _utc_now()
        payload: dict[str, object] = {
            "run_id": row.run_id,
            "stage": stage,
            "pid": pid,
            "status": status,
            "started_at": started_at,
            "updated_at": now,
            "ended_at": now if exit_code is not None else None,
            "exit_code": exit_code,
            "gpu_index": self.gpu_index,
        }
        if detail is not None:
            payload["detail"] = detail
        _atomic_json(self._stage_status_path(row, stage), payload)

    def _environment(self, marker: Path | None) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(self.gpu_index),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        if marker is None:
            environment.pop("FDM_SOLVER_PID_FILE", None)
        else:
            environment["FDM_SOLVER_PID_FILE"] = str(marker)
        return environment

    def _command_arguments(self, command: str) -> list[str]:
        arguments = shlex.split(command)
        if arguments and arguments[0] in {"python", "python3"}:
            arguments[0] = sys.executable
        return arguments

    def _wait_for_solver_pid(
        self, marker: Path, process: subprocess.Popen
    ) -> int | None:
        deadline = self.monotonic() + self.marker_timeout_seconds
        while not marker.is_file():
            if process.poll() is not None:
                return None
            if self.monotonic() >= deadline:
                raise TimeoutError("Torch solver PID marker was not published")
            self.sleep(0.2)
        try:
            solver_pid = int(marker.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as error:
            raise ValueError("Torch solver PID marker is invalid") from error
        if solver_pid <= 0:
            raise ValueError("Torch solver PID marker is invalid")
        return solver_pid

    def _memory_preflight(
        self, row: RunPlanRow, monitor: GpuMonitor
    ) -> tuple[str, int, str] | None:
        required_total = MIN_TOTAL_MEMORY_MIB.get(row.resolution)
        required_free = MIN_FREE_MEMORY_MIB.get(row.resolution)
        if required_total is None or required_free is None:
            return None
        total = monitor.total_memory_mib()
        if total < required_total:
            return (
                "memory_capacity_censored",
                EX_CONFIG,
                f"n{row.resolution} requires at least {required_total} MiB "
                f"total GPU memory; NVML reported {total} MiB; this GPU is "
                "outside the guarded run's supported capacity",
            )
        free = monitor.free_memory_mib()
        if free < required_free:
            return (
                "preflight_memory_busy",
                EX_TEMPFAIL,
                f"n{row.resolution} requires at least {required_free} MiB "
                f"free GPU memory; NVML reported {free} MiB of {total} MiB; "
                "retry only after the GPU is empty",
            )
        return None

    def _run_guarded_child(
        self,
        *,
        process: subprocess.Popen,
        monitor: GpuMonitor,
        solver_pid: int,
        guard_status: Path,
    ) -> tuple[int, int, int]:
        result: dict[str, int] = {}

        def run_guard() -> None:
            try:
                result["code"] = self.guard_function(
                    monitor,
                    gpu_index=self.gpu_index,
                    solver_pid=solver_pid,
                    poll_seconds=self.poll_seconds,
                    interrupt_grace_seconds=self.interrupt_grace_seconds,
                    status_file=guard_status,
                )
            except Exception:
                result["code"] = EX_SOFTWARE

        guard_thread = Thread(target=run_guard, daemon=True)
        guard_thread.start()
        while guard_thread.is_alive() and process.poll() is None:
            guard_thread.join(timeout=0.2)
        guard_code = result.get("code")
        if guard_code == 3:
            self.stop_process(process)
            child_code = process.wait()
            guard_thread.join(timeout=1.0)
            return child_code, EX_TEMPFAIL, guard_code

        child_code = process.wait()
        guard_thread.join(timeout=self.poll_seconds + 2.0)
        if guard_thread.is_alive():
            return child_code, EX_TEMPFAIL, EX_SOFTWARE
        guard_code = result.get("code", EX_SOFTWARE)
        if guard_code != 0:
            return child_code, EX_TEMPFAIL, guard_code
        return child_code, child_code, guard_code

    def _execute_stage(self, row: RunPlanRow, stage: str, command: str) -> int:
        if stage not in GPU_STAGES:
            raise ValueError(f"CPU response stage is forbidden: {stage}")
        started_at = _utc_now()
        monitor = self.monitor_factory(self.gpu_index)
        try:
            occupied = sorted(monitor.pids())
            if occupied and self.wait_until_gpu_empty:
                deadline = (
                    None
                    if self.gpu_empty_timeout_seconds is None
                    else self.monotonic() + self.gpu_empty_timeout_seconds
                )
                self._write_stage_status(
                    row,
                    stage,
                    status="waiting_for_gpu_empty",
                    pid=None,
                    started_at=started_at,
                    exit_code=None,
                    detail=(
                        "NVML compute PIDs remain active; the guarded runner "
                        f"will not signal or replace them: {occupied}"
                    ),
                )
                while occupied:
                    if deadline is not None and self.monotonic() >= deadline:
                        self._write_stage_status(
                            row,
                            stage,
                            status="gpu_empty_wait_timeout",
                            pid=None,
                            started_at=started_at,
                            exit_code=EX_TEMPFAIL,
                            detail=f"NVML compute PIDs still active: {occupied}",
                        )
                        return EX_TEMPFAIL
                    self.sleep(self.poll_seconds)
                    occupied = sorted(monitor.pids())
            if occupied:
                detail = f"GPU preflight found compute PIDs {occupied}"
                self._write_stage_status(
                    row,
                    stage,
                    status="preflight_occupied",
                    pid=None,
                    started_at=started_at,
                    exit_code=EX_TEMPFAIL,
                    detail=detail,
                )
                return EX_TEMPFAIL
            memory_failure = self._memory_preflight(row, monitor)
            if memory_failure is not None:
                memory_status, memory_code, memory_detail = memory_failure
                self._write_stage_status(
                    row,
                    stage,
                    status=memory_status,
                    pid=None,
                    started_at=started_at,
                    exit_code=memory_code,
                    detail=memory_detail,
                )
                return memory_code

            marker = self._pid_marker(row) if stage == "torch" else None
            if marker is not None:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.unlink(missing_ok=True)
            stdout_path = self.log_root / "logs" / f"{row.run_id}.{stage}.out"
            stderr_path = self.log_root / "logs" / f"{row.run_id}.{stage}.err"
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            with stdout_path.open("a", encoding="utf-8") as stdout_stream:
                with stderr_path.open("a", encoding="utf-8") as stderr_stream:
                    process = self.popen_factory(
                        self._command_arguments(command),
                        cwd=Path(__file__).resolve().parents[1],
                        env=self._environment(marker),
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        start_new_session=True,
                    )
                    self._write_stage_status(
                        row,
                        stage,
                        status="starting",
                        pid=process.pid,
                        started_at=started_at,
                        exit_code=None,
                    )
                    try:
                        solver_pid = (
                            process.pid
                            if marker is None
                            else self._wait_for_solver_pid(marker, process)
                        )
                    except (TimeoutError, ValueError) as error:
                        self.stop_process(process)
                        self._write_stage_status(
                            row,
                            stage,
                            status="pid_marker_failed",
                            pid=process.pid,
                            started_at=started_at,
                            exit_code=EX_TEMPFAIL,
                            detail=str(error),
                        )
                        return EX_TEMPFAIL
                    if solver_pid is None:
                        child_code = process.wait()
                        self._write_stage_status(
                            row,
                            stage,
                            status="failed",
                            pid=process.pid,
                            started_at=started_at,
                            exit_code=child_code,
                            detail="Torch launcher exited before publishing solver PID",
                        )
                        return child_code or EX_SOFTWARE
                    self._write_stage_status(
                        row,
                        stage,
                        status="guarded_running",
                        pid=solver_pid,
                        started_at=started_at,
                        exit_code=None,
                    )
                    child_code, effective_code, guard_code = self._run_guarded_child(
                        process=process,
                        monitor=monitor,
                        solver_pid=solver_pid,
                        guard_status=self._guard_status_path(row, stage),
                    )
            if marker is not None:
                marker.unlink(missing_ok=True)
            if effective_code == EX_TEMPFAIL:
                guard_payload = _read_json_status(self._guard_status_path(row, stage))
                failure_status = (
                    "foreign_compute_detected" if guard_code == 3 else "guard_failed"
                )
                _atomic_json(
                    self.guard_failure_status,
                    {
                        "run_id": row.run_id,
                        "stage": stage,
                        "pid": solver_pid,
                        "status": failure_status,
                        "guard": guard_payload,
                        "timestamp": _utc_now(),
                        "exit_code": EX_TEMPFAIL,
                    },
                )
                status = "guard_collision" if guard_code == 3 else "guard_failed"
            else:
                status = "child_complete" if child_code == 0 else "failed"
            self._write_stage_status(
                row,
                stage,
                status=status,
                pid=solver_pid,
                started_at=started_at,
                exit_code=effective_code,
            )
            return effective_code
        finally:
            monitor.close()

    def _mark_verification(self, row: RunPlanRow, stage: str, verified: bool) -> int:
        code = 0 if verified else EX_SOFTWARE
        previous = _read_json_status(self._stage_status_path(row, stage)) or {}
        previous_pid = previous.get("pid")
        pid = previous_pid if isinstance(previous_pid, int) else None
        previous_started = previous.get("started_at")
        started_at = previous_started if isinstance(previous_started, str) else None
        self._write_stage_status(
            row,
            stage,
            status="verified_complete" if verified else "verification_failed",
            pid=pid,
            started_at=started_at,
            exit_code=code,
            detail="completion semantics re-evaluated by build_plan",
        )
        return code

    def run(self, selected_run_ids: set[str] | None = None) -> int:
        initial_plan = self._plan()
        known = {row.run_id for row in initial_plan}
        selected = known if selected_run_ids is None else selected_run_ids
        unknown = selected - known
        if unknown:
            raise ValueError(f"unknown selected run IDs: {sorted(unknown)}")

        for run_id in sorted(selected):
            while True:
                rows = {row.run_id: row for row in self._plan()}
                row = rows[run_id]
                pending_gpu = [
                    (stage, command)
                    for stage, command in row.pending_commands
                    if stage in GPU_STAGES
                ]
                if not pending_gpu:
                    break
                stage, command = pending_gpu[0]
                status = self._execute_stage(row, stage, command)
                if status != 0:
                    return status
                refreshed = {item.run_id: item for item in self._plan()}[run_id]
                verified = refreshed.seed_ready if stage == "seed" else refreshed.torch_complete
                status = self._mark_verification(refreshed, stage, verified)
                if status != 0:
                    return status
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run q-e seed/Torch stages sequentially; never run response CPU work."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/wave_calibration_qe_extension/run_manifest.csv"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("results/wave_calibration_qe_extension/physical_cases.csv"),
    )
    parser.add_argument("--initial-root", type=Path, required=True)
    parser.add_argument("--torch-root", type=Path, required=True)
    parser.add_argument("--pyul-path", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--interrupt-grace-seconds", type=float, default=30.0)
    parser.add_argument("--marker-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--wait-until-gpu-empty",
        action="store_true",
        help=(
            "wait on the NVML compute-process list before preflight; never "
            "signal a process that was already using the GPU"
        ),
    )
    parser.add_argument("--gpu-empty-timeout-seconds", type=float)
    parser.add_argument("--wait-for-path", type=Path)
    parser.add_argument(
        "--wait-guard-status",
        type=Path,
        action="append",
        default=[],
        help=(
            "repeatable legacy/current guard JSON checked while waiting; only "
            "status=solver_exited is ignored"
        ),
    )
    parser.add_argument("--wait-cadence-seconds", type=float, default=30.0)
    parser.add_argument("--wait-timeout-seconds", type=float)
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    if arguments.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")
    if arguments.interrupt_grace_seconds < 0.0:
        parser.error("--interrupt-grace-seconds must be nonnegative")
    if arguments.marker_timeout_seconds <= 0.0:
        parser.error("--marker-timeout-seconds must be positive")
    if (
        arguments.gpu_empty_timeout_seconds is not None
        and arguments.gpu_empty_timeout_seconds <= 0.0
    ):
        parser.error("--gpu-empty-timeout-seconds must be positive")
    if (
        arguments.gpu_empty_timeout_seconds is not None
        and not arguments.wait_until_gpu_empty
    ):
        parser.error(
            "--gpu-empty-timeout-seconds requires --wait-until-gpu-empty"
        )
    if arguments.wait_cadence_seconds < 5.0:
        parser.error("--wait-cadence-seconds must be at least five")
    if arguments.wait_timeout_seconds is not None and arguments.wait_timeout_seconds <= 0:
        parser.error("--wait-timeout-seconds must be positive")
    if arguments.wait_guard_status and arguments.wait_for_path is None:
        parser.error("--wait-guard-status requires --wait-for-path")

    log_root = arguments.log_root.expanduser().resolve()
    failure_status = log_root / f"qe_gpu{arguments.gpu_index}_guard_failure.json"
    lock_stream = acquire_gpu_lock(log_root, arguments.gpu_index)
    try:
        if arguments.wait_for_path is not None:
            explicit_statuses = tuple(
                path.expanduser().resolve() for path in arguments.wait_guard_status
            )
            guard_statuses = tuple(dict.fromkeys((failure_status, *explicit_statuses)))
            try:
                wait_for_path(
                    arguments.wait_for_path.expanduser().resolve(),
                    guard_status_paths=guard_statuses,
                    cadence_seconds=arguments.wait_cadence_seconds,
                    timeout_seconds=arguments.wait_timeout_seconds,
                )
            except (GuardWaitFailure, TimeoutError, OSError, ValueError) as error:
                message = f"guarded q-e wait failed: {error}"
                _atomic_json(
                    log_root / f"qe_gpu{arguments.gpu_index}_wait_status.json",
                    {
                        "status": "wait_failed",
                        "exit_code": EX_TEMPFAIL,
                        "gpu_index": arguments.gpu_index,
                        "wait_for_path": str(
                            arguments.wait_for_path.expanduser().resolve()
                        ),
                        "guard_status_paths": [str(path) for path in guard_statuses],
                        "detail": str(error),
                        "timestamp": _utc_now(),
                    },
                )
                print(message, file=sys.stderr, flush=True)
                return EX_TEMPFAIL
        runner = GuardedQeRunner(
            planner_inputs=PlannerInputs(
                manifest=arguments.manifest.expanduser().resolve(),
                cases=arguments.cases.expanduser().resolve(),
                initial_root=arguments.initial_root.expanduser().resolve(),
                torch_root=arguments.torch_root.expanduser().resolve(),
                pyul_path=arguments.pyul_path.expanduser().resolve(),
            ),
            log_root=log_root,
            gpu_index=arguments.gpu_index,
            poll_seconds=arguments.poll_seconds,
            interrupt_grace_seconds=arguments.interrupt_grace_seconds,
            marker_timeout_seconds=arguments.marker_timeout_seconds,
            wait_until_gpu_empty=arguments.wait_until_gpu_empty,
            gpu_empty_timeout_seconds=arguments.gpu_empty_timeout_seconds,
        )
        return runner.run(None if arguments.run_ids is None else set(arguments.run_ids))
    finally:
        lock_stream.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DuplicateRunner as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(EX_TEMPFAIL) from error
