from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.plan_wave_calibration_runs import (
    _response_summary_complete,
    _torch_summary_complete,
    build_plan,
)
from fdm_smbh_delay.qe_followup_design import build_qe_followup_design


PROJECT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    cases = tmp_path / "physical_cases.csv"
    manifest = tmp_path / "run_manifest.csv"
    initial_root = tmp_path / "pyul_initial"
    torch_root = tmp_path / "torch"
    _write_csv(
        cases,
        [
            {
                "case_id": case_id,
                "target_duration_myr": duration,
                "output_cadence_myr": cadence,
            }
            for case_id, duration, cadence in (
                ("missing_seed", "0.1", "0.00026041666666666666"),
                ("ready_seed", "0.2", "0.0005208333333333333"),
                ("resume_torch", "0.3", "0.00078125"),
                ("pending_response", "0.4", "0.0010416666666666667"),
                ("complete", "0.5", "0.0013020833333333333"),
                ("small_axis", "0.01", "1.0676863030981597e-05"),
            )
        ],
    )
    _write_csv(
        manifest,
        [
            {
                "run_id": f"{case_id}_n{resolution}",
                "case_id": case_id,
                "effective_grid_cells": str(resolution),
                "box_size_pc": box,
            }
            for case_id, resolution, box in (
                ("missing_seed", 128, "26.4"),
                ("ready_seed", 256, "26.4"),
                ("resume_torch", 256, "26.4"),
                ("pending_response", 128, "26.4"),
                ("complete", 128, "26.4"),
                ("small_axis", 512, "26.4"),
            )
        ],
    )
    for run_id in (
        "ready_seed_n256",
        "resume_torch_n256",
        "pending_response_n128",
        "complete_n128",
        "small_axis_n512",
    ):
        wave = initial_root / run_id / "Outputs" / "3Wfn" / "P3D_#000.npy"
        wave.parent.mkdir(parents=True, exist_ok=True)
        wave.write_bytes(b"seed")

    (torch_root / "resume_torch_n256").mkdir(parents=True)
    for run_id in ("pending_response_n128", "complete_n128"):
        run = torch_root / run_id
        run.mkdir(parents=True)
        (run / "torch_run_summary.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
    (torch_root / "complete_n128" / "wave_response_summary.json").write_text(
        json.dumps(
            {
                "status": "diagnosed",
                "run": str((torch_root / "complete_n128").resolve()),
            }
        ),
        encoding="utf-8",
    )
    return cases, manifest, initial_root, torch_root


def _run_planner(
    tmp_path: Path, *extra: str
) -> subprocess.CompletedProcess[str]:
    cases, manifest, initial_root, torch_root = _inputs(tmp_path)
    return subprocess.run(
        [
            sys.executable,
            "scripts/plan_wave_calibration_runs.py",
            "--manifest",
            str(manifest),
            "--cases",
            str(cases),
            "--initial-root",
            str(initial_root),
            "--torch-root",
            str(torch_root),
            "--pyul-path",
            str(tmp_path / "PyUL_NBody"),
            *extra,
        ],
        cwd=PROJECT,
        check=True,
        text=True,
        capture_output=True,
    )


def _registered_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    cases = tmp_path / "cases.csv"
    manifest = tmp_path / "manifest.csv"
    _write_csv(cases, [{
        "case_id": "qe_test", "mass_ratio_q": "0.3", "eccentricity": "0.3",
        "initial_separation_pc": "0.572", "core_radius_pc": "2.2",
        "kepler_period_myr": "0.002", "target_duration_myr": "0.005",
        "output_cadence_myr": "0.0001",
    }])
    _write_csv(manifest, [
        {
            "case_id": "qe_test", "run_id": f"qe_test_n{resolution}",
            "effective_grid_cells": str(resolution), "box_size_pc": "26.4",
            "finest_cell_size_pc": str(26.4 / resolution),
            "plummer_radius_pc": str(13.2 / resolution),
            "requires_qe_design": "true",
        }
        for resolution in (256, 512)
    ])
    design = build_qe_followup_design(
        case_id="qe_test", physical_cases=cases, run_manifest=manifest,
        coarse_resolution=256, fine_resolution=512,
        separation_bin_edges_pc=(0.43, 0.45), duration_myr=0.02,
    )
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design), encoding="utf-8")
    return cases, manifest, path


def test_registered_qe_plan_requires_design_and_uses_bound_duration(
    tmp_path: Path,
) -> None:
    cases, manifest, design = _registered_inputs(tmp_path)
    kwargs = (manifest, cases, tmp_path / "initial", tmp_path / "torch",
              tmp_path / "PyUL_NBody")
    with pytest.raises(ValueError, match="requires --qe-design"):
        build_plan(*kwargs)
    rows = build_plan(*kwargs, qe_design_path=design)
    assert [row.qe_design_role for row in rows] == ["coarse", "fine"]
    assert all(row.case_duration_myr == 0.02 for row in rows)
    assert all(row.save_number == 200 for row in rows)
    assert all("--qe-design-role" in row.seed_command for row in rows)
    assert all("--duration-myr 0.02" in row.torch_command for row in rows)
    assert all("--duration-myr 1e-6" in row.seed_command for row in rows)
    with pytest.raises(ValueError, match="undersamples"):
        build_plan(*kwargs, qe_design_path=design, save_number=17)


def test_registered_qe_plan_rejects_unbound_existing_seed(tmp_path: Path) -> None:
    cases, manifest, design = _registered_inputs(tmp_path)
    initial = tmp_path / "initial"
    wave = initial / "qe_test_n256/Outputs/3Wfn/P3D_#000.npy"
    wave.parent.mkdir(parents=True)
    wave.write_bytes(b"seed")
    with pytest.raises(ValueError, match="lacks bound metadata"):
        build_plan(
            manifest, cases, initial, tmp_path / "torch",
            tmp_path / "PyUL_NBody", qe_design_path=design,
        )


def test_selected_registered_run_ignores_other_jobs_partial_output(
    tmp_path: Path,
) -> None:
    cases, manifest, design = _registered_inputs(tmp_path)
    initial = tmp_path / "initial"
    (initial / "qe_test_n512").mkdir(parents=True)
    inputs = (manifest, cases, initial, tmp_path / "torch", tmp_path / "PyUL_NBody")
    with pytest.raises(ValueError, match="seed directory is incomplete"):
        build_plan(*inputs, qe_design_path=design)
    selected = build_plan(
        *inputs, qe_design_path=design, selected_run_ids={"qe_test_n256"},
    )
    assert [row.run_id for row in selected] == ["qe_test_n256"]
    command = subprocess.run(
        [
            sys.executable, "scripts/plan_wave_calibration_runs.py",
            "--manifest", str(manifest), "--cases", str(cases),
            "--qe-design", str(design),
            "--initial-root", str(initial),
            "--torch-root", str(tmp_path / "torch"),
            "--run-id", "qe_test_n256",
        ],
        cwd=PROJECT, text=True, capture_output=True, check=True,
    )
    assert "total_runs=1" in command.stdout
    assert "qe_test_n512" not in command.stdout
    with pytest.raises(ValueError, match="absent from the manifest"):
        build_plan(
            *inputs, qe_design_path=design, selected_run_ids={"unknown"},
        )


def test_registered_qe_plan_refuses_orphaned_torch_restart(tmp_path: Path) -> None:
    cases, manifest, design = _registered_inputs(tmp_path)
    torch_run = tmp_path / "torch/qe_test_n256"
    torch_run.mkdir(parents=True)
    with pytest.raises(ValueError, match="no bound initial seed"):
        build_plan(
            manifest, cases, tmp_path / "initial", tmp_path / "torch",
            tmp_path / "PyUL_NBody", qe_design_path=design,
        )


def test_registered_qe_plan_accepts_bound_seed_but_rejects_short_torch_resume(
    tmp_path: Path,
) -> None:
    cases, manifest, design_path = _registered_inputs(tmp_path)
    design = json.loads(design_path.read_text(encoding="utf-8"))
    initial = tmp_path / "initial"
    wave = initial / "qe_test_n256/Outputs/3Wfn/P3D_#000.npy"
    wave.parent.mkdir(parents=True)
    wave.write_bytes(b"seed")
    metadata = {
        "run_id": "qe_test_n256", "case_id": "qe_test", "resolution": 256,
        "box_size_pc": 26.4, "duration_myr": 1e-6,
        "qe_design_binding": {
            "status": "qe_prospective_design_bound_not_a_calibration_release",
            "path": str(design_path.resolve()),
            "physical_cases_path": str(cases.resolve()),
            "run_manifest_path": str(manifest.resolve()),
            "file_sha256": hashlib.sha256(design_path.read_bytes()).hexdigest(),
            "design_sha256": design["design_sha256"], "role": "coarse",
        },
    }
    (initial / "qe_test_n256/fdm_adapter_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8",
    )
    kwargs = (manifest, cases, initial, tmp_path / "torch", tmp_path / "PyUL_NBody")
    rows = build_plan(*kwargs, qe_design_path=design_path)
    assert rows[0].seed_ready
    assert [stage for stage, _ in rows[0].pending_commands] == ["torch"]
    torch_run = tmp_path / "torch/qe_test_n256"
    torch_run.mkdir(parents=True)
    torch_metadata = dict(metadata, duration_myr=0.005)
    (torch_run / "fdm_adapter_metadata.json").write_text(
        json.dumps(torch_metadata), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run request disagrees"):
        build_plan(*kwargs, qe_design_path=design_path)


def test_plan_reports_each_pipeline_stage(tmp_path: Path) -> None:
    completed = _run_planner(tmp_path)
    totals = {
        line.split("=", maxsplit=1)[0]: int(line.split("=", maxsplit=1)[1])
        for line in completed.stdout.splitlines()
        if line.split("=", maxsplit=1)[0]
        in {
            "total_runs",
            "seed_ready_runs",
            "torch_completed_runs",
            "response_completed_runs",
            "completed_runs",
            "remaining_runs",
        }
    }
    assert totals == {
        "total_runs": 6,
        "seed_ready_runs": 5,
        "torch_completed_runs": 2,
        "response_completed_runs": 1,
        "completed_runs": 1,
        "remaining_runs": 5,
    }
    assert "status=seed_missing" in completed.stdout
    assert "status=seed_ready,torch_pending" in completed.stdout
    assert "status=seed_ready,torch_incomplete,resume" in completed.stdout
    assert "status=torch_complete,response_pending" in completed.stdout
    assert "status=complete" in completed.stdout


def test_commands_follow_seed_torch_response_order(tmp_path: Path) -> None:
    completed = _run_planner(tmp_path, "--emit-commands")
    commands = [
        line for line in completed.stdout.splitlines() if line.startswith("stage=")
    ]
    missing_seed = [line for line in commands if "run_id=missing_seed_n128" in line]
    assert [line.split()[0] for line in missing_seed] == ["stage=seed", "stage=torch"]
    assert "scripts/run_pyul_wave_case.py" in missing_seed[0]
    assert "--duration-myr 1e-6" in missing_seed[0]
    assert "--save-number 1" in missing_seed[0]
    assert "--save-3d" in missing_seed[0]
    assert "--box-pc 26.399999999999999" in missing_seed[0]
    assert "scripts/launch_torch_wave_case.py" in missing_seed[1]
    assert "--duration-myr 0.10000000000000001" in missing_seed[1]
    assert "--save-number 384" in missing_seed[1]
    assert "--movie-frame-number 96" in missing_seed[1]
    assert "--save-3d-number 16" in missing_seed[1]
    assert "--checkpoint-every-saves 32" in missing_seed[1]
    assert "--rk4-substeps 9" in missing_seed[1]
    assert "--device cuda:0" in missing_seed[1]
    small_axis = next(line for line in commands if "run_id=small_axis_n512" in line)
    assert small_axis.startswith("stage=torch")
    assert "--save-number 936" in small_axis
    assert "run_pyul_wave_case.py" not in "\n".join(
        line for line in commands if "run_id=ready_seed_n256" in line
    )
    resume = next(line for line in commands if "run_id=resume_torch_n256" in line)
    assert resume.startswith("stage=torch")
    assert resume.endswith("--resume")
    response = next(
        line for line in commands if "run_id=pending_response_n128" in line
    )
    assert response.startswith("stage=response")
    assert "analyze_pyul_wave_response.py" in response
    assert "--resume --max-new-samples 1" in response
    assert not any("run_id=complete_n128" in line for line in commands)


def test_output_csv_keeps_stage_specific_commands(tmp_path: Path) -> None:
    output = tmp_path / "plan.csv"
    _run_planner(tmp_path, "--output-csv", str(output))
    with output.open(encoding="utf-8", newline="") as stream:
        rows = {row["run_id"]: row for row in csv.DictReader(stream)}
    assert rows["missing_seed_n128"]["seed_command"]
    assert rows["missing_seed_n128"]["torch_command"]
    assert rows["missing_seed_n128"]["movie_frame_number"] == "96"
    assert rows["missing_seed_n128"]["save_3d_number"] == "16"
    assert not rows["missing_seed_n128"]["response_command"]
    assert not rows["ready_seed_n256"]["seed_command"]
    assert rows["resume_torch_n256"]["torch_command"].endswith("--resume")
    assert rows["pending_response_n128"]["response_command"]
    assert not rows["complete_n128"]["seed_command"]
    assert not rows["complete_n128"]["torch_command"]
    assert not rows["complete_n128"]["response_command"]


def test_save_number_override_replaces_case_cadence(tmp_path: Path) -> None:
    completed = _run_planner(
        tmp_path, "--emit-commands", "--save-number", "17"
    )
    torch_commands = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("stage=torch")
    ]
    assert torch_commands
    assert all("--save-number 17" in line for line in torch_commands)


def test_snapshot_count_overrides_are_forwarded(tmp_path: Path) -> None:
    completed = _run_planner(
        tmp_path,
        "--emit-commands",
        "--movie-frame-number",
        "48",
        "--save-3d-number",
        "8",
    )
    torch_commands = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("stage=torch")
    ]
    assert torch_commands
    assert all("--movie-frame-number 48" in line for line in torch_commands)
    assert all("--save-3d-number 8" in line for line in torch_commands)


@pytest.mark.parametrize(
    ("validator", "valid_status"),
    [
        (_torch_summary_complete, "complete"),
        (_response_summary_complete, "diagnosed"),
    ],
)
def test_summary_completion_requires_valid_json_and_expected_status(
    tmp_path: Path,
    validator,
    valid_status: str,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    summary = run / "summary.json"

    summary.write_text("{not-json", encoding="utf-8")
    assert not validator(summary, run)

    summary.write_text(json.dumps({"status": "partial"}), encoding="utf-8")
    assert not validator(summary, run)

    summary.write_text(json.dumps({"status": valid_status}), encoding="utf-8")
    assert validator(summary, run)


@pytest.mark.parametrize(
    ("validator", "valid_status"),
    [
        (_torch_summary_complete, "complete"),
        (_response_summary_complete, "diagnosed"),
    ],
)
def test_summary_run_identity_is_checked_only_when_recorded(
    tmp_path: Path,
    validator,
    valid_status: str,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    summary = run / "summary.json"
    summary.write_text(
        json.dumps({"status": valid_status, "run": str(run.resolve())}),
        encoding="utf-8",
    )
    assert validator(summary, run)

    summary.write_text(
        json.dumps({"status": valid_status, "run": str(tmp_path / "other")}),
        encoding="utf-8",
    )
    assert not validator(summary, run)


def test_summary_read_error_is_incomplete(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "run"
    run.mkdir()
    summary = run / "summary.json"
    summary.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    original_read_text = Path.read_text

    def raise_for_summary(path: Path, *args, **kwargs) -> str:
        if path == summary:
            raise OSError("simulated read failure")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raise_for_summary)
    assert not _torch_summary_complete(summary, run)
