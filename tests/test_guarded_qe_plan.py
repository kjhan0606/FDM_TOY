from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from plan_wave_calibration_runs import RunPlanRow  # noqa: E402
import run_guarded_qe_plan  # noqa: E402
from run_guarded_qe_plan import (  # noqa: E402
    EX_CONFIG,
    EX_TEMPFAIL,
    MIN_TOTAL_MEMORY_MIB,
    DuplicateRunner,
    GuardedQeRunner,
    PlannerInputs,
    acquire_gpu_lock,
    wait_for_path,
)


class FakeMonitor:
    def __init__(self, *, pids: set[int] | None = None, memory_mib: int = 100_000):
        self.compute_pids = set() if pids is None else set(pids)
        self.memory_mib = memory_mib
        self.closed = False

    def pids(self) -> set[int]:
        return set(self.compute_pids)

    def total_memory_mib(self) -> int:
        return self.memory_mib

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self, pid: int, exit_code: int, on_wait) -> None:
        self.pid = pid
        self.exit_code = exit_code
        self.returncode = None
        self.on_wait = on_wait

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = self.exit_code
            self.on_wait(self.exit_code)
        return self.returncode


def _state(*, seed: bool = False, torch: bool = False, started: bool = False) -> dict:
    return {"seed": seed, "torch": torch, "started": started}


def _row(
    tmp_path: Path,
    run_id: str,
    state: dict,
    *,
    resolution: int = 128,
) -> RunPlanRow:
    return RunPlanRow(
        run_id=run_id,
        case_id=run_id.rsplit("_n", maxsplit=1)[0],
        resolution=resolution,
        box_size_pc=26.4,
        case_duration_myr=0.1,
        seed_ready=state["seed"],
        torch_complete=state["torch"],
        response_complete=False,
        torch_directory_exists=state["started"],
        status_detail="test",
        cases_path=tmp_path / "cases.csv",
        initial_root=tmp_path / "initial",
        torch_root=tmp_path / "torch",
        pyul_path=tmp_path / "PyUL_NBody",
        save_number=4,
        movie_frame_number=2,
        save_3d_number=1,
        checkpoint_every_saves=1,
        rk4_substeps=1,
        device="cuda:0",
    )


def _builder(tmp_path: Path, states: dict[str, dict], resolutions=None):
    resolutions = {} if resolutions is None else resolutions

    def build(*_args, **_kwargs):
        # Deliberately reverse the source order; the runner must sort run_id.
        return [
            _row(
                tmp_path,
                run_id,
                states[run_id],
                resolution=resolutions.get(run_id, 128),
            )
            for run_id in reversed(list(states))
        ]

    return build


class FakePopenFactory:
    def __init__(self, states: dict[str, dict], *, torch_exit_codes=None) -> None:
        self.states = states
        self.torch_exit_codes = {} if torch_exit_codes is None else torch_exit_codes
        self.calls: list[tuple[str, str, list[str]]] = []
        self.next_pid = 4000

    def __call__(self, arguments, **kwargs):
        joined = " ".join(arguments)
        run_id = next(
            run_id
            for run_id in self.states
            if run_id in joined or run_id.rsplit("_n", maxsplit=1)[0] in joined
        )
        stage = "torch" if "launch_torch_wave_case.py" in joined else "seed"
        self.calls.append((run_id, stage, list(arguments)))
        self.next_pid += 1
        pid = self.next_pid
        state = self.states[run_id]
        exit_code = self.torch_exit_codes.get(run_id, 0) if stage == "torch" else 0
        if stage == "torch":
            state["started"] = True
            marker = Path(kwargs["env"]["FDM_SOLVER_PID_FILE"])
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"{pid + 1000}\n", encoding="utf-8")

        def finish(code: int) -> None:
            if code != 0:
                return
            if stage == "seed":
                state["seed"] = True
            else:
                state["torch"] = True

        return FakeProcess(pid, exit_code, finish)


def _runner(
    tmp_path: Path,
    states: dict[str, dict],
    *,
    monitor_factory=None,
    popen_factory=None,
    guard_function=None,
    stop_process=None,
    resolutions=None,
) -> GuardedQeRunner:
    return GuardedQeRunner(
        planner_inputs=PlannerInputs(
            manifest=tmp_path / "manifest.csv",
            cases=tmp_path / "cases.csv",
            initial_root=tmp_path / "initial",
            torch_root=tmp_path / "torch",
            pyul_path=tmp_path / "PyUL_NBody",
        ),
        log_root=tmp_path / "logs",
        gpu_index=0,
        poll_seconds=0.01,
        interrupt_grace_seconds=0.0,
        marker_timeout_seconds=1.0,
        plan_builder=_builder(tmp_path, states, resolutions),
        monitor_factory=(
            (lambda _index: FakeMonitor())
            if monitor_factory is None
            else monitor_factory
        ),
        popen_factory=(
            FakePopenFactory(states) if popen_factory is None else popen_factory
        ),
        guard_function=(
            (lambda *_args, **_kwargs: 0)
            if guard_function is None
            else guard_function
        ),
        stop_process=(
            (lambda process: process.wait())
            if stop_process is None
            else stop_process
        ),
        sleep=lambda _seconds: None,
    )


def test_normal_stages_run_sequentially_by_run_id_and_skip_response(tmp_path: Path) -> None:
    states = {
        "b_case_n128": _state(),
        "a_case_n128": _state(),
    }
    popen = FakePopenFactory(states)
    guarded_pids: list[int] = []

    def guard(_monitor, *, solver_pid, **_kwargs):
        guarded_pids.append(solver_pid)
        return 0

    runner = _runner(
        tmp_path,
        states,
        popen_factory=popen,
        guard_function=guard,
    )
    assert runner.run() == 0
    assert [(run_id, stage) for run_id, stage, _args in popen.calls] == [
        ("a_case_n128", "seed"),
        ("a_case_n128", "torch"),
        ("b_case_n128", "seed"),
        ("b_case_n128", "torch"),
    ]
    assert all("analyze_pyul_wave_response.py" not in " ".join(args) for *_, args in popen.calls)
    assert len(guarded_pids) == 4
    status = json.loads(
        (tmp_path / "logs/status/a_case_n128.torch.json").read_text()
    )
    assert status["status"] == "verified_complete"
    assert status["pid"] is not None
    assert status["exit_code"] == 0
    assert set(("run_id", "stage", "pid", "started_at", "ended_at")) <= set(status)


def test_gpu_index_lock_rejects_a_duplicate_runner(tmp_path: Path) -> None:
    first = acquire_gpu_lock(tmp_path, 2)
    try:
        with pytest.raises(DuplicateRunner, match="holds GPU 2"):
            acquire_gpu_lock(tmp_path, 2)
    finally:
        first.close()


def test_wait_for_path_ignores_normal_solver_exit_status(tmp_path: Path) -> None:
    target = tmp_path / "ready"
    status = tmp_path / "legacy_guard.json"
    status.write_text(json.dumps({"status": "solver_exited"}), encoding="utf-8")
    sleeps: list[float] = []

    def publish_target(seconds: float) -> None:
        sleeps.append(seconds)
        target.write_text("ready", encoding="utf-8")

    wait_for_path(
        target,
        guard_status_paths=(status,),
        cadence_seconds=30.0,
        timeout_seconds=None,
        sleep=publish_target,
    )
    assert sleeps == [30.0]


def test_wait_for_path_aborts_when_guard_reports_foreign_compute(tmp_path: Path) -> None:
    target = tmp_path / "ready"
    failure = tmp_path / "guard_failure.json"
    target.write_text("ready", encoding="utf-8")
    failure.write_text(
        json.dumps({"status": "foreign_compute_detected"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="foreign_compute_detected"):
        wait_for_path(
            target,
            guard_status_paths=(failure,),
            cadence_seconds=30.0,
            timeout_seconds=None,
            sleep=lambda _seconds: None,
        )


def test_wait_for_path_aborts_on_malformed_guard_status(tmp_path: Path) -> None:
    failure = tmp_path / "malformed.json"
    failure.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="malformed"):
        wait_for_path(
            tmp_path / "never-ready",
            guard_status_paths=(failure,),
            cadence_seconds=30.0,
            timeout_seconds=None,
            sleep=lambda _seconds: None,
        )


def test_main_converts_wait_guard_failure_to_atomic_tempfail_status(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    failure = tmp_path / "legacy_guard.json"
    failure.write_text(
        json.dumps({"status": "foreign_compute_detected"}), encoding="utf-8"
    )
    log_root = tmp_path / "logs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_guarded_qe_plan.py",
            "--initial-root",
            str(tmp_path / "initial"),
            "--torch-root",
            str(tmp_path / "torch"),
            "--pyul-path",
            str(tmp_path / "PyUL_NBody"),
            "--log-root",
            str(log_root),
            "--wait-for-path",
            str(tmp_path / "handoff-summary.json"),
            "--wait-guard-status",
            str(failure),
        ],
    )
    assert run_guarded_qe_plan.main() == EX_TEMPFAIL
    assert "guarded q-e wait failed" in capsys.readouterr().err
    payload = json.loads((log_root / "qe_gpu0_wait_status.json").read_text())
    assert payload["status"] == "wait_failed"
    assert payload["exit_code"] == EX_TEMPFAIL
    assert str(failure.resolve()) in payload["guard_status_paths"]


def test_preflight_occupied_returns_tempfail_without_launch(tmp_path: Path) -> None:
    states = {"case_n128": _state()}
    monitor = FakeMonitor(pids={91})

    def forbidden_popen(*_args, **_kwargs):
        raise AssertionError("occupied preflight must not launch a child")

    runner = _runner(
        tmp_path,
        states,
        monitor_factory=lambda _index: monitor,
        popen_factory=forbidden_popen,
    )
    assert runner.run() == EX_TEMPFAIL
    assert monitor.closed
    status = json.loads((tmp_path / "logs/status/case_n128.seed.json").read_text())
    assert status["status"] == "preflight_occupied"
    assert status["exit_code"] == EX_TEMPFAIL


def test_mid_run_foreign_pid_stops_only_owned_child_and_tempfails(tmp_path: Path) -> None:
    states = {"case_n128": _state()}
    popen = FakePopenFactory(states)
    stopped: list[int] = []

    def collide(_monitor, **_kwargs):
        return 3

    def stop(process) -> None:
        stopped.append(process.pid)
        process.wait()

    runner = _runner(
        tmp_path,
        states,
        popen_factory=popen,
        guard_function=collide,
        stop_process=stop,
    )
    assert runner.run() == EX_TEMPFAIL
    assert stopped == [4001]
    failure = json.loads((tmp_path / "logs/qe_gpu0_guard_failure.json").read_text())
    assert failure["status"] == "foreign_compute_detected"
    status = json.loads((tmp_path / "logs/status/case_n128.seed.json").read_text())
    assert status["status"] == "guard_collision"


def test_crashed_torch_stage_is_resumed_on_the_next_invocation(tmp_path: Path) -> None:
    states = {"case_n256": _state(seed=True)}
    crashing = FakePopenFactory(states, torch_exit_codes={"case_n256": 9})
    first = _runner(tmp_path, states, popen_factory=crashing)
    assert first.run() == 9
    assert states["case_n256"]["started"]
    assert not states["case_n256"]["torch"]

    resumed = FakePopenFactory(states)
    second = _runner(tmp_path, states, popen_factory=resumed)
    assert second.run() == 0
    assert len(resumed.calls) == 1
    assert resumed.calls[0][:2] == ("case_n256", "torch")
    assert "--resume" in resumed.calls[0][2]
    assert states["case_n256"]["torch"]


@pytest.mark.parametrize("resolution", [512, 768])
def test_memory_gate_rejects_below_conservative_resolution_floor(
    tmp_path: Path, resolution: int
) -> None:
    run_id = f"case_n{resolution}"
    states = {run_id: _state()}
    monitor = FakeMonitor(memory_mib=MIN_TOTAL_MEMORY_MIB[resolution] - 1)

    def forbidden_popen(*_args, **_kwargs):
        raise AssertionError("memory rejection must happen before launch")

    runner = _runner(
        tmp_path,
        states,
        resolutions={run_id: resolution},
        monitor_factory=lambda _index: monitor,
        popen_factory=forbidden_popen,
    )
    assert runner.run() == EX_CONFIG
    status = json.loads(
        (tmp_path / f"logs/status/{run_id}.seed.json").read_text()
    )
    assert status["status"] == "memory_rejected"
    assert str(MIN_TOTAL_MEMORY_MIB[resolution]) in status["detail"]


def test_only_explicitly_selected_run_ids_are_executed(tmp_path: Path) -> None:
    states = {
        "a_case_n128": _state(seed=True),
        "b_case_n128": _state(seed=True),
    }
    popen = FakePopenFactory(states)
    runner = _runner(tmp_path, states, popen_factory=popen)
    assert runner.run({"b_case_n128"}) == 0
    assert [(run_id, stage) for run_id, stage, _args in popen.calls] == [
        ("b_case_n128", "torch")
    ]
    assert not states["a_case_n128"]["torch"]
