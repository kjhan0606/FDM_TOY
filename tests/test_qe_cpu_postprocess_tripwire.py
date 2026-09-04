from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from plan_wave_calibration_runs import RunPlanRow  # noqa: E402
import run_qe_cpu_postprocess_tripwire as tripwire  # noqa: E402


def _row(
    root: Path,
    run_id: str,
    *,
    resolution: int,
    save_3d_number: int = 1,
) -> RunPlanRow:
    case_id = run_id.rsplit("_n", maxsplit=1)[0]
    return RunPlanRow(
        run_id=run_id,
        case_id=case_id,
        resolution=resolution,
        box_size_pc=26.4,
        case_duration_myr=0.1,
        seed_ready=True,
        torch_complete=True,
        response_complete=False,
        torch_directory_exists=True,
        status_detail="torch_complete,response_pending",
        cases_path=root / "cases.csv",
        initial_root=root / "initial",
        torch_root=root / "torch",
        pyul_path=root / "PyUL_NBody",
        save_number=4,
        movie_frame_number=2,
        save_3d_number=save_3d_number,
        checkpoint_every_saves=2,
        rk4_substeps=3,
        device="cuda:0",
    )


def _extension_plan(root: Path) -> list[RunPlanRow]:
    rows = []
    for index in range(5):
        for resolution in (128, 256, 512):
            rows.append(
                _row(root, f"tier1_{index}_n{resolution}", resolution=resolution)
            )
    for index in range(3):
        for resolution in (256, 512, 768):
            rows.append(
                _row(root, f"tier2_{index}_n{resolution}", resolution=resolution)
            )
    for index in range(2):
        for resolution in (512, 768):
            rows.append(
                _row(root, f"tier3_{index}_n{resolution}", resolution=resolution)
            )
    return rows


def test_default_selection_is_exactly_the_eight_added_finer_runs(tmp_path: Path) -> None:
    chosen = tripwire.select_eight_added_runs(list(reversed(_extension_plan(tmp_path))))
    assert [row.run_id for row in chosen] == sorted(
        [f"tier1_{index}_n512" for index in range(5)]
        + [f"tier2_{index}_n768" for index in range(3)]
    )
    assert {row.resolution for row in chosen} == {512, 768}


def test_explicit_selection_requires_eight_known_n512_or_n768_runs(
    tmp_path: Path,
) -> None:
    plan = _extension_plan(tmp_path)
    with pytest.raises(ValueError, match="exactly eight"):
        tripwire.select_eight_added_runs(plan, {"tier1_0_n512"})
    selected = {f"tier1_{index}_n512" for index in range(5)} | {
        f"tier2_{index}_n768" for index in range(3)
    }
    assert len(tripwire.select_eight_added_runs(plan, selected)) == 8


def test_torch_completion_contract_includes_run_reference_box_and_saved_states(
    tmp_path: Path, monkeypatch
) -> None:
    row = _row(tmp_path, "case_n512", resolution=512, save_3d_number=2)
    captured = {}

    def validate(run, **kwargs):
        captured.update(kwargs)
        return {"status": "complete"}, {
            "box_size_pc": row.box_size_pc,
            "reference_initial_state": str(row.initial_directory.resolve()),
        }

    monkeypatch.setattr(tripwire, "validate_torch_calibration_completion", validate)
    assert tripwire.validate_torch_summary(row) == {"status": "complete"}
    assert captured["expected_run_id"] == row.run_id
    assert captured["expected_case_id"] == row.case_id
    assert captured["expected_resolution"] == row.resolution
    assert captured["expected_saved_3d_states"] == 3

    def stale_reference(run, **kwargs):
        return {"status": "complete"}, {
            "box_size_pc": row.box_size_pc,
            "reference_initial_state": str(tmp_path / "copied_reference"),
        }

    monkeypatch.setattr(
        tripwire, "validate_torch_calibration_completion", stale_reference
    )
    with pytest.raises(ValueError, match="reference identity"):
        tripwire.validate_torch_summary(row)


def test_wait_ignores_solver_exited_and_validates_summary(tmp_path: Path, monkeypatch) -> None:
    row = _row(tmp_path, "case_n512", resolution=512)
    guard = tmp_path / "guard.json"
    guard.write_text(json.dumps({"status": "solver_exited"}), encoding="utf-8")
    validations = []

    def validate(candidate):
        validations.append(candidate.run_id)
        return {}

    def sleep(_seconds):
        summary = row.torch_directory / "torch_run_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(tripwire, "validate_torch_summary", validate)
    tripwire.wait_for_torch_completion(
        row,
        guard_status_paths=(guard,),
        cadence_seconds=30.0,
        timeout_seconds=60.0,
        sleep=sleep,
        monotonic=iter((0.0, 1.0)).__next__,
    )
    assert validations == [row.run_id]


def test_complete_summary_takes_precedence_over_stale_foreign_status(
    tmp_path: Path, monkeypatch
) -> None:
    row = _row(tmp_path, "case_n768", resolution=768)
    summary = row.torch_directory / "torch_run_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}", encoding="utf-8")
    guard = tmp_path / "guard.json"
    guard.write_text(
        json.dumps({"status": "foreign_compute_detected"}), encoding="utf-8"
    )
    monkeypatch.setattr(tripwire, "validate_torch_summary", lambda _row: {})
    tripwire.wait_for_torch_completion(
        row,
        guard_status_paths=(guard,),
        cadence_seconds=30.0,
        timeout_seconds=None,
    )


@pytest.mark.parametrize(
    "content",
    [
        json.dumps({"status": "foreign_compute_detected"}),
        json.dumps({"status": "wait_failed"}),
        "not-json",
    ],
)
def test_guard_or_wait_failure_aborts_before_gpu_completion(
    tmp_path: Path, content: str
) -> None:
    row = _row(tmp_path, "case_n512", resolution=512)
    guard = tmp_path / "guard.json"
    guard.write_text(content, encoding="utf-8")
    with pytest.raises(tripwire.WaitFailed):
        tripwire.wait_for_torch_completion(
            row,
            guard_status_paths=(guard,),
            cadence_seconds=30.0,
            timeout_seconds=None,
        )


def test_shared_lock_is_blocking_exclusive(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        tripwire.fcntl,
        "flock",
        lambda descriptor, operation: calls.append((descriptor, operation)),
    )
    with tripwire.shared_finalizer_lock(tmp_path / "qe_finalize_all.lock"):
        pass
    assert len(calls) == 1
    assert calls[0][1] == tripwire.fcntl.LOCK_EX
    assert not calls[0][1] & tripwire.fcntl.LOCK_NB


def test_wait_occurs_without_lock_then_each_run_locks_processes_and_releases(
    tmp_path: Path, monkeypatch
) -> None:
    rows = [
        _row(tmp_path, "b_n512", resolution=512),
        _row(tmp_path, "a_n768", resolution=768),
    ]
    events = []

    def wait(row, **_kwargs):
        events.append(f"wait:{row.run_id}")

    @contextmanager
    def lock(path):
        assert path == tripwire.FINALIZER_LOCK
        events.append("lock")
        yield
        events.append("unlock")

    runner = tripwire.QeCpuTripwire(
        rows=rows,
        log_root=tmp_path / "logs",
        guard_status_paths=(),
        cadence_seconds=30.0,
        timeout_seconds=None,
        wait_function=wait,
        lock_function=lock,
    )
    monkeypatch.setattr(
        runner, "_process_run", lambda row: events.append(f"process:{row.run_id}")
    )
    assert runner.run() == 0
    assert events == [
        "wait:a_n768",
        "lock",
        "process:a_n768",
        "unlock",
        "wait:b_n512",
        "lock",
        "process:b_n512",
        "unlock",
    ]


def test_steps_are_sequential_single_threaded_bounded_and_response_is_chunked(
    tmp_path: Path, monkeypatch
) -> None:
    row = _row(tmp_path, "case_n512", resolution=512, save_3d_number=1)
    completed_stages: set[str] = set()
    calls: list[tuple[str, dict]] = []
    response_calls = 0

    def output_valid(_row, stage):
        return stage in completed_stages

    def command_runner(command, **kwargs):
        nonlocal response_calls
        script = Path(command[1]).name
        stage = {
            "snapshot_torch_provenance.py": "provenance",
            "analyze_pyul_wave_run.py": "conservation",
            "analyze_pyul_secular_exchange.py": "secular",
            "analyze_pyul_line_density.py": "line_density",
            "analyze_pyul_wave_response.py": "wave_response",
            "build_wave_exchange_table.py": "exchange_table",
        }[script]
        calls.append((stage, kwargs))
        if stage == "wave_response":
            response_calls += 1
            assert command[-3:] == ["--resume", "--max-new-samples", "1"]
            if response_calls == 2:
                completed_stages.add(stage)
        else:
            completed_stages.add(stage)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tripwire, "_stage_output_valid", output_valid)
    monkeypatch.setattr(tripwire, "validate_torch_summary", lambda _row: {})
    runner = tripwire.QeCpuTripwire(
        rows=[row],
        log_root=tmp_path / "logs",
        guard_status_paths=(),
        cadence_seconds=30.0,
        timeout_seconds=None,
        command_runner=command_runner,
        available_memory=lambda: 10**15,
    )
    runner._process_run(row)
    assert [stage for stage, _kwargs in calls] == [
        "provenance",
        "conservation",
        "secular",
        "line_density",
        "wave_response",
        "wave_response",
        "exchange_table",
    ]
    for _stage, kwargs in calls:
        assert kwargs["preexec_fn"].func is tripwire._limit_address_space
        assert kwargs["preexec_fn"].args == (
            tripwire.ADDRESS_SPACE_LIMIT_BYTES[row.resolution],
        )
        assert all(
            kwargs["env"][name] == value
            for name, value in tripwire.THREAD_ENVIRONMENT.items()
        )
    for stage in completed_stages:
        marker = json.loads(runner._marker(row, stage).read_text(encoding="utf-8"))
        assert marker["status"] == "complete"
        assert marker["run_id"] == row.run_id


def test_crash_resume_skips_semantically_complete_stages(tmp_path: Path, monkeypatch) -> None:
    row = _row(tmp_path, "case_n768", resolution=768)
    completed_stages: set[str] = set()
    calls = []
    fail_secular = True

    def output_valid(_row, stage):
        return stage in completed_stages

    def command_runner(command, **_kwargs):
        nonlocal fail_secular
        script = Path(command[1]).name
        stage = {
            "snapshot_torch_provenance.py": "provenance",
            "analyze_pyul_wave_run.py": "conservation",
            "analyze_pyul_secular_exchange.py": "secular",
            "analyze_pyul_line_density.py": "line_density",
            "analyze_pyul_wave_response.py": "wave_response",
            "build_wave_exchange_table.py": "exchange_table",
        }[script]
        calls.append(stage)
        if stage == "secular" and fail_secular:
            fail_secular = False
            return subprocess.CompletedProcess(command, 9)
        completed_stages.add(stage)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tripwire, "_stage_output_valid", output_valid)
    monkeypatch.setattr(tripwire, "validate_torch_summary", lambda _row: {})
    runner = tripwire.QeCpuTripwire(
        rows=[row],
        log_root=tmp_path / "logs",
        guard_status_paths=(),
        cadence_seconds=30.0,
        timeout_seconds=None,
        command_runner=command_runner,
        available_memory=lambda: 10**15,
    )
    with pytest.raises(tripwire.StageFailed):
        runner._process_run(row)
    runner._process_run(row)
    assert calls.count("provenance") == 1
    assert calls.count("conservation") == 1
    assert calls.count("secular") == 2
    assert calls[-3:] == ["line_density", "wave_response", "exchange_table"]


def test_checkpoint_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    row = _row(tmp_path, "case_n512", resolution=512)
    runner = tripwire.QeCpuTripwire(
        rows=[row],
        log_root=tmp_path / "logs",
        guard_status_paths=(),
        cadence_seconds=30.0,
        timeout_seconds=None,
    )
    marker = runner._marker(row, "conservation")
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                **runner._identity(row, "conservation"),
                "run_id": "copied_run",
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(tripwire.StageFailed, match="identity mismatch"):
        runner._completed_marker_valid(row, "conservation")


@pytest.mark.parametrize("checkpoint_status", [None, "running"])
def test_valid_output_recovers_complete_marker_without_rerunning_child(
    tmp_path: Path, monkeypatch, checkpoint_status: str | None
) -> None:
    row = _row(tmp_path, "case_n512", resolution=512)
    calls = []
    runner = tripwire.QeCpuTripwire(
        rows=[row],
        log_root=tmp_path / "logs",
        guard_status_paths=(),
        cadence_seconds=30.0,
        timeout_seconds=None,
        command_runner=lambda *_args, **_kwargs: calls.append("called"),
        available_memory=lambda: 0,
    )
    command = [sys.executable, "scripts/analyze_pyul_wave_run.py", str(row.torch_directory)]
    if checkpoint_status is not None:
        runner._write_marker(
            row,
            "conservation",
            status=checkpoint_status,
            exit_code=None,
            command=command,
        )
    monkeypatch.setattr(tripwire, "_stage_output_valid", lambda *_args: True)
    runner._run_command(row, "conservation", command)
    assert calls == []
    marker = json.loads(
        runner._marker(row, "conservation").read_text(encoding="utf-8")
    )
    assert marker["status"] == "complete"
    assert marker["run_id"] == row.run_id


@pytest.mark.parametrize("resolution", [512, 768])
def test_resolution_specific_memory_rejection_starts_no_child(
    tmp_path: Path, monkeypatch, resolution: int
) -> None:
    row = _row(tmp_path, f"case_n{resolution}", resolution=resolution)
    calls = []
    limit = tripwire.ADDRESS_SPACE_LIMIT_BYTES[resolution]
    runner = tripwire.QeCpuTripwire(
        rows=[row],
        log_root=tmp_path / "logs",
        guard_status_paths=(),
        cadence_seconds=30.0,
        timeout_seconds=None,
        command_runner=lambda *_args, **_kwargs: calls.append("called"),
        available_memory=lambda: limit - 1,
    )
    monkeypatch.setattr(tripwire, "_stage_output_valid", lambda *_args: False)
    command = [sys.executable, "scripts/analyze_pyul_wave_run.py", str(row.torch_directory)]
    with pytest.raises(tripwire.StageFailed) as captured:
        runner._run_command(row, "conservation", command)
    assert captured.value.exit_code == tripwire.EX_TEMPFAIL
    assert calls == []
    marker = json.loads(
        runner._marker(row, "conservation").read_text(encoding="utf-8")
    )
    assert marker["status"] == "memory_rejected"


@pytest.mark.parametrize(
    ("resolution", "expected_gib"), [(512, 64), (768, 192)]
)
def test_resolution_specific_address_space_ceiling(
    monkeypatch, resolution: int, expected_gib: int
) -> None:
    calls = []
    monkeypatch.setattr(
        tripwire.resource,
        "getrlimit",
        lambda _resource: (tripwire.resource.RLIM_INFINITY,) * 2,
    )
    monkeypatch.setattr(
        tripwire.resource,
        "setrlimit",
        lambda resource_id, limits: calls.append((resource_id, limits)),
    )
    limit = tripwire.ADDRESS_SPACE_LIMIT_BYTES[resolution]
    assert limit == expected_gib * 1024**3
    tripwire._limit_address_space(limit)
    assert calls == [
        (
            tripwire.resource.RLIMIT_AS,
            (limit, tripwire.resource.RLIM_INFINITY),
        )
    ]


def test_wait_failure_is_atomic_tempfail_status(tmp_path: Path) -> None:
    row = _row(tmp_path, "case_n512", resolution=512)

    def wait(*_args, **_kwargs):
        raise tripwire.WaitFailed("foreign collision")

    runner = tripwire.QeCpuTripwire(
        rows=[row],
        log_root=tmp_path / "logs",
        guard_status_paths=(),
        cadence_seconds=30.0,
        timeout_seconds=None,
        wait_function=wait,
    )
    assert runner.run() == tripwire.EX_TEMPFAIL
    status = json.loads(
        (tmp_path / "logs" / "qe_cpu_tripwire_status.json").read_text()
    )
    assert status["status"] == "wait_failed"
    assert status["exit_code"] == tripwire.EX_TEMPFAIL


def test_fixed_production_lock_path_has_no_fdm_prefix() -> None:
    assert tripwire.FINALIZER_LOCK == Path(
        "/gpfs/kjhan/FDM_TOY_RESULTS/logs/qe_finalize_all.lock"
    )


def test_dry_run_prints_wait_lock_and_bounded_response_without_execution(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(tripwire, "build_plan", lambda *_args, **_kwargs: _extension_plan(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qe_cpu_postprocess_tripwire.py",
            "--dry-run",
            "--log-root",
            str(tmp_path / "logs"),
        ],
    )
    assert tripwire.main() == 0
    output = capsys.readouterr().out
    assert f"shared_lock={tripwire.FINALIZER_LOCK}" in output
    assert output.count("wait run_id=") == 8
    assert output.count("lock/process/release run_id=") == 8
    assert output.count("--resume --max-new-samples 1") == 8
