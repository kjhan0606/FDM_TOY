#!/usr/bin/env python3
"""Run the Boey n384 sequence on an otherwise idle syn101 GPU."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import fcntl
import getpass
import json
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import socket
import subprocess
import sys
from threading import Lock, Thread
import time


REPOSITORY = Path("/home/kjhan/BACKUP/FDM_SINK_MERGE/FDM_TOY")
RESULT_ROOT = Path("/gpfs/kjhan/FDM_TOY_RESULTS")
REFERENCE_ROOT = RESULT_ROOT / "torch_initial" / "tier0_n384"
OUTPUT_ROOT = RESULT_ROOT / "torch_calibration" / "tier0_n384"
LOG_ROOT = RESULT_ROOT / "logs"
CASE_IDS = (
    "boey_each02pct",
    "boey_each05pct",
    "boey_each10pct",
)


@dataclass(frozen=True)
class BoeyCase:
    case_id: str
    duration_myr: float = 0.8
    save_number: int = 2048
    movie_frame_number: int = 128
    save_3d_number: int = 16
    checkpoint_every_saves: int = 32
    rk4_substeps: int = 9
    time_step_factor: float = 1.0
    box_pc: float = 100.0

    @property
    def reference(self) -> Path:
        return REFERENCE_ROOT / f"{self.case_id}_n384"

    @property
    def output(self) -> Path:
        return OUTPUT_ROOT / f"{self.case_id}_n384"

    @property
    def pid_marker(self) -> Path:
        return LOG_ROOT / f"fdm_{self.case_id}_n384_solver.pid"


def _pmon_pid(line: str) -> int | None:
    """Return a process ID from one nvidia-smi pmon row."""

    fields = line.split()
    if not fields or fields[0].startswith("#") or len(fields) < 2:
        return None
    if fields[1] == "-":
        return None
    try:
        pid = int(fields[1])
    except ValueError:
        return None
    return pid if pid > 0 else None


def _foreign_slurm_reason(line: str, owner: str) -> str | None:
    """Describe one foreign Slurm allocation line, if present."""

    fields = line.split()
    if len(fields) < 3 or fields[0] == owner:
        return None
    return f"foreign Slurm job {fields[1]} owned by {fields[0]} is {fields[2]}"


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class GuardedSequence:
    """Run commands while persistent telemetry watches for shared-node use."""

    def __init__(self, *, gpu_index: int, poll_seconds: int) -> None:
        self.gpu_index = gpu_index
        self.poll_seconds = poll_seconds
        self.owner = getpass.getuser()
        self.guard_log = LOG_ROOT / "fdm_boey_n384_guard.log"
        self.progress_log = LOG_ROOT / "fdm_boey_n384.out"
        self.error_log = LOG_ROOT / "fdm_boey_n384.err"
        self.yield_marker = LOG_ROOT / "fdm_boey_n384_yielded.json"
        self.lock_path = LOG_ROOT / "fdm_boey_n384_guard.lock"
        self.events: Queue[str] = Queue()
        self.allowed_lock = Lock()
        self.allowed_pids: set[int] = set()
        self.current_marker: Path | None = None
        self.telemetry: list[subprocess.Popen[str]] = []
        self.telemetry_logs: list[object] = []
        self.collision_reported = False
        self.lock_stream = None

    def record(self, message: str) -> None:
        timestamp = datetime.now().astimezone().strftime("%F %T %Z")
        line = f"{timestamp} | {message}"
        with self.guard_log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    def acquire_lock(self) -> None:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        self.lock_stream = self.lock_path.open("a", encoding="utf-8")
        try:
            fcntl.flock(
                self.lock_stream.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise RuntimeError("another Boey n384 guard holds the lock") from error

    def _capture(self, arguments: list[str]) -> str:
        result = subprocess.run(
            arguments,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout

    def preflight(self) -> None:
        gpu_uuid = self._capture(
            [
                "nvidia-smi",
                "-i",
                str(self.gpu_index),
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ]
        ).strip()
        applications = self._capture(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ]
        )
        for line in applications.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and fields[0] == gpu_uuid and fields[1]:
                raise RuntimeError(
                    f"GPU {self.gpu_index} already has process PID {fields[1]}"
                )
        jobs = self._capture(
            [
                "squeue",
                "--noheader",
                "-w",
                "syn101",
                "-t",
                "RUNNING,COMPLETING,CONFIGURING",
                "-o",
                "%u %i %T",
            ]
        )
        for line in jobs.splitlines():
            reason = _foreign_slurm_reason(line, self.owner)
            if reason is not None:
                raise RuntimeError(reason)

    def _report_collision(self, reason: str) -> None:
        if self.collision_reported:
            return
        self.collision_reported = True
        self.events.put(reason)

    def request_stop(self, reason: str) -> None:
        """Request a guarded stop without inspecting unrelated processes."""

        self._report_collision(reason)

    def _marker_pid(self) -> int | None:
        with self.allowed_lock:
            marker = self.current_marker
        if marker is None or not marker.is_file():
            return None
        try:
            return int(marker.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def _read_gpu_telemetry(self, stream) -> None:
        for line in stream:
            pid = _pmon_pid(line)
            if pid is None:
                continue
            with self.allowed_lock:
                allowed = set(self.allowed_pids)
            marker_pid = self._marker_pid()
            if marker_pid is not None:
                allowed.add(marker_pid)
            if pid not in allowed:
                self._report_collision(
                    f"unmanaged process PID {pid} appeared on GPU {self.gpu_index}"
                )
                return

    def _read_slurm_telemetry(self, stream) -> None:
        for line in stream:
            reason = _foreign_slurm_reason(line, self.owner)
            if reason is not None:
                self._report_collision(reason)
                return

    def start_telemetry(self) -> None:
        error_stream = self.guard_log.open("a", encoding="utf-8")
        self.telemetry_logs.append(error_stream)
        gpu = subprocess.Popen(
            [
                "nvidia-smi",
                "pmon",
                "-i",
                str(self.gpu_index),
                "-d",
                str(self.poll_seconds),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=error_stream,
            start_new_session=True,
        )
        slurm = subprocess.Popen(
            [
                "squeue",
                "--noheader",
                f"--iterate={self.poll_seconds}",
                "-w",
                "syn101",
                "-t",
                "RUNNING,COMPLETING,CONFIGURING",
                "-o",
                "%u %i %T",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=error_stream,
            start_new_session=True,
        )
        self.telemetry.extend((gpu, slurm))
        assert gpu.stdout is not None
        assert slurm.stdout is not None
        Thread(
            target=self._read_gpu_telemetry,
            args=(gpu.stdout,),
            daemon=True,
        ).start()
        Thread(
            target=self._read_slurm_telemetry,
            args=(slurm.stdout,),
            daemon=True,
        ).start()

    def _environment(self, marker: Path | None) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(self.gpu_index),
                "SLURM_CPUS_PER_TASK": "4",
                "OMP_NUM_THREADS": "4",
                "PYTHONUNBUFFERED": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        cuda_library = (
            "/home/kjhan/miniconda3/lib/python3.13/"
            "site-packages/nvidia/cu13/lib"
        )
        previous = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            cuda_library if not previous else f"{cuda_library}:{previous}"
        )
        if marker is None:
            environment.pop("FDM_SOLVER_PID_FILE", None)
        else:
            environment["FDM_SOLVER_PID_FILE"] = str(marker)
        return environment

    def _stop_owned(self, process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=20)
            return
        except subprocess.TimeoutExpired:
            pass
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()

    def run_command(
        self,
        arguments: list[str],
        *,
        description: str,
        marker: Path | None = None,
    ) -> int:
        self.record(f"starting {description}")
        if marker is not None:
            marker.unlink(missing_ok=True)
        with self.progress_log.open("a", encoding="utf-8") as output_stream:
            with self.error_log.open("a", encoding="utf-8") as error_stream:
                process = subprocess.Popen(
                    arguments,
                    cwd=REPOSITORY,
                    env=self._environment(marker),
                    stdout=output_stream,
                    stderr=error_stream,
                    start_new_session=True,
                )
                with self.allowed_lock:
                    self.allowed_pids = {process.pid}
                    self.current_marker = marker
                while process.poll() is None:
                    for telemetry in self.telemetry:
                        if telemetry.poll() is not None:
                            self._report_collision(
                                "shared-node telemetry stopped unexpectedly"
                            )
                            break
                    try:
                        reason = self.events.get(timeout=0.5)
                    except Empty:
                        continue
                    self.record(f"yielding {description}: {reason}")
                    _atomic_json(
                        self.yield_marker,
                        {
                            "reason": reason,
                            "description": description,
                            "time": datetime.now().astimezone().isoformat(),
                        },
                    )
                    self._stop_owned(process)
                    with self.allowed_lock:
                        self.allowed_pids.clear()
                        self.current_marker = None
                    return 75
                status = process.wait()
                with self.allowed_lock:
                    self.allowed_pids.clear()
                    self.current_marker = None
        if marker is not None:
            marker.unlink(missing_ok=True)
        self.record(f"finished {description} with status {status}")
        return status

    def prepare_reference(self, case: BoeyCase) -> int:
        wave = case.reference / "Outputs" / "3Wfn" / "P3D_#000.npy"
        particles = case.reference / "Outputs" / "NBody" / "NTM_#000.npy"
        if wave.is_file() and particles.is_file():
            return 0
        if case.reference.exists():
            raise RuntimeError(
                f"incomplete reference requires inspection: {case.reference}"
            )
        arguments = [
            sys.executable,
            "scripts/run_pyul_wave_case.py",
            "--pyul-path",
            "/gpfs/kjhan/FDM_TOY_DEPS/PyUL_NBody",
            "--case-id",
            case.case_id,
            "--resolution",
            "384",
            "--duration-myr",
            "0.000001",
            "--save-number",
            "1",
            "--save-3d",
            "--rk-steps",
            "36",
            "--box-pc",
            str(case.box_pc),
            "--output",
            str(REFERENCE_ROOT),
        ]
        return self.run_command(
            arguments,
            description=f"{case.case_id} n384 reference",
        )

    def evolve(self, case: BoeyCase) -> int:
        summary_path = case.output / "torch_run_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") != "complete":
                raise RuntimeError(f"invalid completion summary: {summary_path}")
            self.record(f"{case.case_id} n384 evolution already complete")
            return 0
        resume_arguments: list[str] = []
        if case.output.exists():
            checkpoint = case.output / "Checkpoints" / "latest.json"
            if not checkpoint.is_file():
                raise RuntimeError(
                    f"partial output lacks a restart marker: {case.output}"
                )
            resume_arguments.append("--resume")
        arguments = [
            sys.executable,
            "scripts/launch_torch_wave_case.py",
            str(case.reference),
            "--output",
            str(case.output),
            "--duration-myr",
            str(case.duration_myr),
            "--save-number",
            str(case.save_number),
            "--movie-frame-number",
            str(case.movie_frame_number),
            "--save-3d-number",
            str(case.save_3d_number),
            "--checkpoint-every-saves",
            str(case.checkpoint_every_saves),
            "--rk4-substeps",
            str(case.rk4_substeps),
            "--time-step-factor",
            str(case.time_step_factor),
            "--device",
            "cuda:0",
            *resume_arguments,
        ]
        return self.run_command(
            arguments,
            description=f"{case.case_id} n384 evolution",
            marker=case.pid_marker,
        )

    def shutdown(self) -> None:
        for process in self.telemetry:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in self.telemetry:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for stream in self.telemetry_logs:
            stream.close()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("cases", nargs="*", choices=CASE_IDS)
    arguments = parser.parse_args()
    if arguments.gpu_index < 0 or arguments.poll_seconds < 1:
        parser.error("GPU index and poll interval must be positive")
    return arguments


def main() -> int:
    arguments = _parse_arguments()
    if socket.gethostname().split(".")[0] != "syn101":
        raise RuntimeError("this guarded runner is restricted to syn101")
    sequence = GuardedSequence(
        gpu_index=arguments.gpu_index,
        poll_seconds=arguments.poll_seconds,
    )
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(
            signum,
            lambda received, _frame: sequence.request_stop(
                f"guard received signal {received}"
            ),
        )
    sequence.acquire_lock()
    try:
        sequence.preflight()
        sequence.record(
            f"syn101 GPU {arguments.gpu_index} preflight passed"
        )
        if arguments.preflight_only:
            return 0
        sequence.start_telemetry()
        selected = arguments.cases or list(CASE_IDS)
        completed = []
        for case_id in selected:
            case = BoeyCase(case_id)
            status = sequence.prepare_reference(case)
            if status != 0:
                return status
            status = sequence.evolve(case)
            if status != 0:
                return status
            completed.append(case_id)
        _atomic_json(
            OUTPUT_ROOT / "boey_n384_sequence_summary.json",
            {
                "status": "complete",
                "cases": completed,
                "resolution": 384,
                "gpu_index": arguments.gpu_index,
            },
        )
        sequence.yield_marker.unlink(missing_ok=True)
        sequence.record("Boey n384 evolution sequence complete")
        return 0
    finally:
        sequence.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
