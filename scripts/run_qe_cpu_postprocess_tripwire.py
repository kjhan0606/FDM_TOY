#!/usr/bin/env python3
"""Finalize the eight added q-e runs, one run and one CPU process at a time."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from functools import partial
import json
import math
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
from typing import Callable, Iterator

from fdm_smbh_delay.run_metadata import validate_torch_calibration_completion
from plan_wave_calibration_runs import (
    DEFAULT_PYUL_PATH,
    DEFAULT_RESULT_ROOT,
    RunPlanRow,
    build_plan,
)


EX_SOFTWARE = 70
EX_TEMPFAIL = 75
EX_CONFIG = 78
ADDRESS_SPACE_LIMIT_BYTES = {512: 64 * 1024**3, 768: 192 * 1024**3}
FINALIZER_LOCK = Path("/gpfs/kjhan/FDM_TOY_RESULTS/logs/qe_finalize_all.lock")
GPU_RESOLUTIONS = {512, 768}
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}


class WaitFailed(RuntimeError):
    pass


class StageFailed(RuntimeError):
    def __init__(self, message: str, exit_code: int = EX_SOFTWARE) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _same_float(actual: object, expected: float) -> bool:
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(
        value, expected, rel_tol=0.0, abs_tol=1.0e-12
    )


def _expected_saved_3d_states(row: RunPlanRow) -> int:
    return min(row.save_number, row.save_3d_number) + 1


def select_eight_added_runs(
    plan: list[RunPlanRow], selected_run_ids: set[str] | None = None
) -> list[RunPlanRow]:
    """Select the adjacent finer run for each three-level tier-1/2 case."""

    by_id = {row.run_id: row for row in plan}
    if len(by_id) != len(plan):
        raise ValueError("the q-e plan contains duplicate run IDs")
    if selected_run_ids is not None:
        unknown = selected_run_ids - set(by_id)
        if unknown:
            raise ValueError(f"unknown run IDs: {sorted(unknown)}")
        chosen = [by_id[run_id] for run_id in sorted(selected_run_ids)]
    else:
        by_case: dict[str, list[RunPlanRow]] = {}
        for row in plan:
            by_case.setdefault(row.case_id, []).append(row)
        chosen = [
            max(rows, key=lambda row: row.resolution)
            for rows in by_case.values()
            if len(rows) == 3
        ]
        chosen.sort(key=lambda row: row.run_id)
    if len(chosen) != 8:
        raise ValueError(f"exactly eight q-e extension runs are required; got {len(chosen)}")
    if any(row.resolution not in GPU_RESOLUTIONS for row in chosen):
        raise ValueError("the eight-run finalizer accepts only n512 and n768 runs")
    return chosen


def validate_torch_summary(row: RunPlanRow) -> dict[str, object]:
    run = row.torch_directory.expanduser().resolve()
    summary, metadata = validate_torch_calibration_completion(
        run,
        expected_case_id=row.case_id,
        expected_resolution=row.resolution,
        expected_duration_myr=row.case_duration_myr,
        expected_saved_intervals=row.save_number,
        expected_saved_3d_states=_expected_saved_3d_states(row),
        expected_rk4_substeps=row.rk4_substeps,
        expected_checkpoint_interval=row.checkpoint_every_saves,
        expected_run_id=row.run_id,
    )
    if run.name != row.run_id:
        raise ValueError(f"run directory identity is invalid: {run}")
    if not _same_float(metadata.get("box_size_pc"), row.box_size_pc):
        raise ValueError(f"Torch calibration metadata has invalid box_size_pc: {run}")
    expected_reference = row.initial_directory.expanduser().resolve()
    try:
        reference = Path(str(metadata["reference_initial_state"])).resolve()
    except (KeyError, OSError) as error:
        raise ValueError(f"Torch calibration reference identity is invalid: {run}") from error
    if reference != expected_reference:
        raise ValueError(f"Torch calibration reference identity is invalid: {run}")
    return summary


def _guard_status_failure(paths: tuple[Path, ...]) -> str | None:
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = _load_json(path)
        except ValueError as error:
            return str(error)
        status = payload.get("status")
        if status == "solver_exited":
            continue
        return f"GPU guard/wait status {status!r} in {path}"
    return None


def wait_for_torch_completion(
    row: RunPlanRow,
    *,
    guard_status_paths: tuple[Path, ...],
    cadence_seconds: float,
    timeout_seconds: float | None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Wait on files only, validating completion before stale guard records."""

    if cadence_seconds < 30.0:
        raise ValueError("wait cadence must be at least 30 seconds")
    deadline = None if timeout_seconds is None else monotonic() + timeout_seconds
    summary_path = row.torch_directory / "torch_run_summary.json"
    while True:
        if summary_path.is_file():
            # A present but malformed/copied completion is not a wait condition.
            validate_torch_summary(row)
            return
        guard_failure = _guard_status_failure(guard_status_paths)
        if guard_failure is not None:
            raise WaitFailed(guard_failure)
        if deadline is not None and monotonic() >= deadline:
            raise WaitFailed(f"timed out waiting for {summary_path}")
        sleep(cadence_seconds)


@contextmanager
def shared_finalizer_lock(path: Path = FINALIZER_LOCK) -> Iterator[None]:
    """Block on the exact lock shared with the live 20-run finalizer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        stream.close()


def _available_memory_bytes(path: Path = Path("/proc/meminfo")) -> int:
    """Read Linux MemAvailable without inspecting any process."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise WaitFailed(f"cannot read available memory from {path}") from error
    for line in lines:
        fields = line.split()
        if len(fields) == 3 and fields[0] == "MemAvailable:" and fields[2] == "kB":
            try:
                available = int(fields[1]) * 1024
            except ValueError as error:
                raise WaitFailed(f"invalid MemAvailable in {path}") from error
            if available < 0:
                raise WaitFailed(f"invalid MemAvailable in {path}")
            return available
    raise WaitFailed(f"MemAvailable is missing from {path}")


def _limit_address_space(limit_bytes: int) -> None:
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = limit_bytes
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _stage_output_valid(row: RunPlanRow, stage: str) -> bool:
    run = row.torch_directory.resolve()
    try:
        metadata = _load_json(run / "fdm_adapter_metadata.json")
        if metadata.get("run_id") != row.run_id or metadata.get("case_id") != row.case_id:
            return False
        if metadata.get("resolution") != row.resolution:
            return False
        if stage == "provenance":
            summary = _load_json(run / "torch_solver_provenance" / "manifest.json")
            return (
                summary.get("status") == "source_snapshot"
                and Path(str(summary.get("run"))).resolve() == run
                and summary.get("adapter_revision") == metadata.get("adapter_revision")
            )
        expected: dict[str, tuple[str, tuple[str, ...]]] = {
            "conservation": (
                "conservation_summary.json",
                ("conservation_timeseries.csv",),
            ),
            "secular": (
                "orbit_averaged_exchange_summary.json",
                ("orbit_averaged_exchange.csv",),
            ),
            "line_density": (
                "line_density_summary.json",
                ("line_density_diagnostics.csv",),
            ),
            "wave_response": (
                "wave_response_summary.json",
                ("wave_response_timeseries.csv", "wave_radial_profiles.csv"),
            ),
            "exchange_table": (
                "wave_exchange.summary.json",
                ("wave_exchange.csv",),
            ),
        }
        summary_name, data_names = expected[stage]
        if not all(_nonempty(run / name) for name in (summary_name, *data_names)):
            return False
        summary = _load_json(run / summary_name)
        statuses = {
            "conservation": "diagnosed",
            "secular": "orbit_averaged",
            "line_density": "line_density_diagnosed",
            "wave_response": "diagnosed",
            "exchange_table": "dimensionless_exchange_table",
        }
        if summary.get("status") != statuses[stage]:
            return False
        if stage == "conservation":
            return (
                summary.get("case_id") == row.case_id
                and summary.get("resolution") == row.resolution
                and int(summary.get("samples", 0)) > 0
                and _same_float(summary.get("duration_myr"), row.case_duration_myr)
            )
        if stage == "secular":
            return int(summary.get("complete_orbits", 0)) > 0
        if stage == "line_density":
            return int(summary.get("samples", 0)) > 0
        if stage == "wave_response":
            return (
                Path(str(summary.get("run"))).resolve() == run
                and summary.get("resolution") == row.resolution
                and summary.get("samples") == _expected_saved_3d_states(row)
            )
        return (
            summary.get("runs") == 1
            and int(summary.get("cycles", 0)) > 0
            and summary.get("case_ids") == [row.case_id]
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


class QeCpuTripwire:
    def __init__(
        self,
        *,
        rows: list[RunPlanRow],
        log_root: Path,
        guard_status_paths: tuple[Path, ...],
        cadence_seconds: float,
        timeout_seconds: float | None,
        command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        wait_function: Callable[..., None] = wait_for_torch_completion,
        lock_function: Callable[..., object] = shared_finalizer_lock,
        available_memory: Callable[[], int] = _available_memory_bytes,
    ) -> None:
        self.rows = sorted(rows, key=lambda row: row.run_id)
        self.log_root = log_root
        self.guard_status_paths = guard_status_paths
        self.cadence_seconds = cadence_seconds
        self.timeout_seconds = timeout_seconds
        self.command_runner = command_runner
        self.wait_function = wait_function
        self.lock_function = lock_function
        self.available_memory = available_memory
        self.project = Path(__file__).resolve().parents[1]

    def _marker(self, row: RunPlanRow, stage: str) -> Path:
        return self.log_root / "qe_cpu_postprocess" / row.run_id / f"{stage}.json"

    def _identity(self, row: RunPlanRow, stage: str) -> dict[str, object]:
        return {
            "run_id": row.run_id,
            "case_id": row.case_id,
            "resolution": row.resolution,
            "run": str(row.torch_directory.resolve()),
            "stage": stage,
        }

    def _completed_marker_valid(self, row: RunPlanRow, stage: str) -> bool:
        marker = self._marker(row, stage)
        if not marker.is_file():
            return False
        payload = _load_json(marker)
        identity = self._identity(row, stage)
        if any(payload.get(key) != value for key, value in identity.items()):
            raise StageFailed(f"checkpoint identity mismatch: {marker}", EX_CONFIG)
        status = payload.get("status")
        if status not in {"running", "failed", "memory_rejected", "complete"}:
            raise StageFailed(f"invalid checkpoint status: {marker}", EX_CONFIG)
        return status == "complete" and _stage_output_valid(row, stage)

    def _write_marker(
        self,
        row: RunPlanRow,
        stage: str,
        *,
        status: str,
        exit_code: int | None,
        command: list[str],
    ) -> None:
        _atomic_json(
            self._marker(row, stage),
            {
                **self._identity(row, stage),
                "status": status,
                "exit_code": exit_code,
                "command": command,
                "timestamp": _utc_timestamp(),
            },
        )

    def _run_command(self, row: RunPlanRow, stage: str, command: list[str]) -> None:
        if self._completed_marker_valid(row, stage):
            return
        if _stage_output_valid(row, stage):
            self._write_marker(
                row, stage, status="complete", exit_code=0, command=command
            )
            return
        limit = ADDRESS_SPACE_LIMIT_BYTES[row.resolution]
        available = self.available_memory()
        if available < limit:
            self._write_marker(
                row,
                stage,
                status="memory_rejected",
                exit_code=EX_TEMPFAIL,
                command=command,
            )
            raise StageFailed(
                f"{row.run_id} {stage} requires {limit} available bytes; "
                f"MemAvailable is {available}",
                EX_TEMPFAIL,
            )
        self._write_marker(row, stage, status="running", exit_code=None, command=command)
        log_path = self.log_root / "qe_cpu_postprocess" / row.run_id / "postprocess.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(THREAD_ENVIRONMENT)
        with log_path.open("ab") as log_stream:
            completed = self.command_runner(
                command,
                cwd=self.project,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
                preexec_fn=partial(_limit_address_space, limit),
            )
        if completed.returncode != 0:
            self._write_marker(
                row,
                stage,
                status="failed",
                exit_code=completed.returncode,
                command=command,
            )
            raise StageFailed(
                f"{row.run_id} {stage} exited with {completed.returncode}",
                completed.returncode or EX_SOFTWARE,
            )
        if not _stage_output_valid(row, stage):
            self._write_marker(
                row,
                stage,
                status="failed",
                exit_code=EX_SOFTWARE,
                command=command,
            )
            raise StageFailed(f"{row.run_id} {stage} outputs failed validation")
        self._write_marker(row, stage, status="complete", exit_code=0, command=command)

    def _run_wave_response(self, row: RunPlanRow) -> None:
        stage = "wave_response"
        command = [
            sys.executable,
            "scripts/analyze_pyul_wave_response.py",
            str(row.torch_directory.resolve()),
            "--resume",
            "--max-new-samples",
            "1",
        ]
        if self._completed_marker_valid(row, stage):
            return
        maximum_invocations = _expected_saved_3d_states(row)
        for _invocation in range(maximum_invocations):
            if _stage_output_valid(row, stage):
                self._write_marker(
                    row, stage, status="complete", exit_code=0, command=command
                )
                return
            limit = ADDRESS_SPACE_LIMIT_BYTES[row.resolution]
            available = self.available_memory()
            if available < limit:
                self._write_marker(
                    row,
                    stage,
                    status="memory_rejected",
                    exit_code=EX_TEMPFAIL,
                    command=command,
                )
                raise StageFailed(
                    f"{row.run_id} wave_response requires {limit} available "
                    f"bytes; MemAvailable is {available}",
                    EX_TEMPFAIL,
                )
            self._write_marker(
                row, stage, status="running", exit_code=None, command=command
            )
            log_path = self.log_root / "qe_cpu_postprocess" / row.run_id / "postprocess.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update(THREAD_ENVIRONMENT)
            with log_path.open("ab") as log_stream:
                completed = self.command_runner(
                    command,
                    cwd=self.project,
                    env=environment,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                    preexec_fn=partial(_limit_address_space, limit),
                )
            if completed.returncode != 0:
                self._write_marker(
                    row,
                    stage,
                    status="failed",
                    exit_code=completed.returncode,
                    command=command,
                )
                raise StageFailed(
                    f"{row.run_id} wave response exited with {completed.returncode}",
                    completed.returncode or EX_SOFTWARE,
                )
        if _stage_output_valid(row, stage):
            self._write_marker(
                row, stage, status="complete", exit_code=0, command=command
            )
            return
        self._write_marker(
            row,
            stage,
            status="failed",
            exit_code=EX_SOFTWARE,
            command=command,
        )
        raise StageFailed(
            f"{row.run_id} wave response exceeded saved_3d_states bound"
        )

    def _commands(self, row: RunPlanRow) -> list[tuple[str, list[str]]]:
        run = str(row.torch_directory.resolve())
        return [
            ("provenance", [sys.executable, "scripts/snapshot_torch_provenance.py", run]),
            ("conservation", [sys.executable, "scripts/analyze_pyul_wave_run.py", run]),
            ("secular", [sys.executable, "scripts/analyze_pyul_secular_exchange.py", run]),
            ("line_density", [sys.executable, "scripts/analyze_pyul_line_density.py", run]),
            (
                "exchange_table",
                [
                    sys.executable,
                    "scripts/build_wave_exchange_table.py",
                    run,
                    "--output",
                    str(row.torch_directory.resolve() / "wave_exchange.csv"),
                ],
            ),
        ]

    def _process_run(self, row: RunPlanRow) -> None:
        # Revalidate after a potentially long blocking lock acquisition.
        validate_torch_summary(row)
        commands = self._commands(row)
        for stage, command in commands[:4]:
            self._run_command(row, stage, command)
        self._run_wave_response(row)
        stage, command = commands[-1]
        self._run_command(row, stage, command)
        for required_stage in (
            "provenance",
            "conservation",
            "secular",
            "line_density",
            "wave_response",
            "exchange_table",
        ):
            if not self._completed_marker_valid(row, required_stage):
                raise StageFailed(f"final verification failed: {row.run_id} {required_stage}")

    def run(self) -> int:
        status_path = self.log_root / "qe_cpu_tripwire_status.json"
        detail = ""
        try:
            for row in self.rows:
                self.wait_function(
                    row,
                    guard_status_paths=self.guard_status_paths,
                    cadence_seconds=self.cadence_seconds,
                    timeout_seconds=self.timeout_seconds,
                )
                # The lock is deliberately absent during the solver-summary wait.
                with self.lock_function(FINALIZER_LOCK):
                    self._process_run(row)
            _atomic_json(
                status_path,
                {
                    "status": "complete",
                    "exit_code": 0,
                    "run_ids": [row.run_id for row in self.rows],
                    "timestamp": _utc_timestamp(),
                },
            )
            return 0
        except WaitFailed as error:
            code = EX_TEMPFAIL
            status = "wait_failed"
            detail = str(error)
        except StageFailed as error:
            code = error.exit_code
            status = "postprocess_failed"
            detail = str(error)
        except (OSError, ValueError) as error:
            code = EX_SOFTWARE
            status = "validation_failed"
            detail = str(error)
        _atomic_json(
            status_path,
            {
                "status": status,
                "exit_code": code,
                "detail": detail,
                "run_ids": [row.run_id for row in self.rows],
                "timestamp": _utc_timestamp(),
            },
        )
        print(f"q-e CPU tripwire failed: {detail}", file=sys.stderr, flush=True)
        return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="File-only tripwire for the eight new q-e CPU postprocess runs."
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
    parser.add_argument("--initial-root", type=Path, default=DEFAULT_RESULT_ROOT / "pyul_initial")
    parser.add_argument("--torch-root", type=Path, default=DEFAULT_RESULT_ROOT / "torch")
    parser.add_argument("--pyul-path", type=Path, default=DEFAULT_PYUL_PATH)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("/gpfs/kjhan/FDM_TOY_RESULTS/logs"),
    )
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--guard-status", type=Path, action="append", default=[])
    parser.add_argument("--wait-cadence-seconds", type=float, default=300.0)
    parser.add_argument("--wait-timeout-seconds", type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.wait_cadence_seconds < 30.0:
        parser.error("--wait-cadence-seconds must be at least 30")
    if arguments.wait_timeout_seconds is not None and arguments.wait_timeout_seconds <= 0:
        parser.error("--wait-timeout-seconds must be positive")
    if not arguments.dry_run and socket.gethostname().split(".", maxsplit=1)[0] != "syntax":
        parser.error("CPU post-processing is restricted to syntax")

    plan = build_plan(
        arguments.manifest.expanduser().resolve(),
        arguments.cases.expanduser().resolve(),
        arguments.initial_root.expanduser().resolve(),
        arguments.torch_root.expanduser().resolve(),
        arguments.pyul_path.expanduser().resolve(),
    )
    selected = select_eight_added_runs(
        plan, None if arguments.run_ids is None else set(arguments.run_ids)
    )
    log_root = arguments.log_root.expanduser().resolve()
    guard_paths = tuple(
        dict.fromkeys(
            (
                log_root / "qe_gpu0_guard_failure.json",
                log_root / "qe_gpu0_wait_status.json",
                *(path.expanduser().resolve() for path in arguments.guard_status),
            )
        )
    )
    if arguments.dry_run:
        print(f"shared_lock={FINALIZER_LOCK}")
        for row in selected:
            print(f"wait run_id={row.run_id} summary={row.torch_directory / 'torch_run_summary.json'}")
            print(f"lock/process/release run_id={row.run_id} resolution=n{row.resolution}")
            commands = QeCpuTripwire(
                rows=[row],
                log_root=log_root,
                guard_status_paths=guard_paths,
                cadence_seconds=arguments.wait_cadence_seconds,
                timeout_seconds=arguments.wait_timeout_seconds,
            )._commands(row)
            for stage, command in commands[:4]:
                print(f"stage={stage} command={' '.join(command)}")
            print("stage=wave_response command=... --resume --max-new-samples 1")
            stage, command = commands[-1]
            print(f"stage={stage} command={' '.join(command)}")
        return 0
    return QeCpuTripwire(
        rows=selected,
        log_root=log_root,
        guard_status_paths=guard_paths,
        cadence_seconds=arguments.wait_cadence_seconds,
        timeout_seconds=arguments.wait_timeout_seconds,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
