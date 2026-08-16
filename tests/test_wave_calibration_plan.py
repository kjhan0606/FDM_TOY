from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


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
        "{}", encoding="utf-8"
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
