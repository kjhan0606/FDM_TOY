#!/usr/bin/env python3
"""Plan the seed, CUDA evolution, and response stages of q-e wave runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shlex

from fdm_smbh_delay.qe_followup_design import (
    read_verified_qe_followup_design,
    verify_qe_design_run_request,
)


DEFAULT_RESULT_ROOT = Path("/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension")
DEFAULT_PYUL_PATH = Path("/gpfs/kjhan/FDM_TOY_DEPS/PyUL_NBody")


def _shell_command(arguments: list[str | Path]) -> str:
    return shlex.join(str(argument) for argument in arguments)


@dataclass(frozen=True)
class RunPlanRow:
    run_id: str
    case_id: str
    resolution: int
    box_size_pc: float
    case_duration_myr: float
    seed_ready: bool
    torch_complete: bool
    response_complete: bool
    torch_directory_exists: bool
    status_detail: str
    cases_path: Path
    initial_root: Path
    torch_root: Path
    pyul_path: Path
    save_number: int
    movie_frame_number: int
    save_3d_number: int
    checkpoint_every_saves: int
    rk4_substeps: int
    device: str
    qe_design_path: Path | None = None
    qe_design_manifest_path: Path | None = None
    qe_design_role: str | None = None

    @property
    def completed(self) -> bool:
        return self.response_complete

    @property
    def initial_directory(self) -> Path:
        return self.initial_root / self.run_id

    @property
    def torch_directory(self) -> Path:
        return self.torch_root / self.run_id

    @property
    def seed_command(self) -> str:
        arguments: list[str | Path] = [
                "python",
                "scripts/run_pyul_wave_case.py",
                "--pyul-path",
                self.pyul_path,
                "--cases",
                self.cases_path,
                "--case-id",
                self.case_id,
                "--resolution",
                str(self.resolution),
                "--duration-myr",
                "1e-6",
                "--save-number",
                "1",
                "--save-3d",
                "--rk-steps",
                "36",
                "--box-pc",
                f"{self.box_size_pc:.17g}",
                "--output",
                self.initial_root,
            ]
        if self.qe_design_path is not None:
            arguments.extend([
                "--qe-design", self.qe_design_path,
                "--qe-design-manifest", self.qe_design_manifest_path,
                "--qe-design-role", self.qe_design_role,
            ])
        return _shell_command(arguments)

    @property
    def torch_command(self) -> str:
        arguments: list[str | Path] = [
            "python",
            "scripts/launch_torch_wave_case.py",
            self.initial_directory,
            "--output",
            self.torch_directory,
            "--duration-myr",
            f"{self.case_duration_myr:.17g}",
            "--save-number",
            str(self.save_number),
            "--movie-frame-number",
            str(self.movie_frame_number),
            "--save-3d-number",
            str(self.save_3d_number),
            "--checkpoint-every-saves",
            str(self.checkpoint_every_saves),
            "--rk4-substeps",
            str(self.rk4_substeps),
            "--device",
            self.device,
        ]
        if self.torch_directory_exists and not self.torch_complete:
            arguments.append("--resume")
        return _shell_command(arguments)

    @property
    def response_command(self) -> str:
        return _shell_command(
            [
                "python",
                "scripts/analyze_pyul_wave_response.py",
                self.torch_directory,
                "--resume",
                "--max-new-samples",
                "1",
            ]
        )

    @property
    def pending_commands(self) -> list[tuple[str, str]]:
        if self.response_complete:
            return []
        if not self.torch_complete:
            commands: list[tuple[str, str]] = []
            if not self.seed_ready:
                commands.append(("seed", self.seed_command))
            commands.append(("torch", self.torch_command))
            return commands
        return [("response", self.response_command)]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _load_case_parameters(cases_path: Path) -> dict[str, tuple[float, int, float]]:
    cases = _read_rows(cases_path)
    parameters: dict[str, tuple[float, int, float]] = {}
    for row in cases:
        duration = float(row["target_duration_myr"])
        cadence = float(row["output_cadence_myr"])
        if duration <= 0.0 or cadence <= 0.0:
            raise ValueError("case duration and output cadence must be positive")
        save_number = math.floor(duration / cadence + 1.0e-9)
        if save_number < 1:
            raise ValueError("output cadence must not exceed the case duration")
        parameters[row["case_id"]] = (duration, save_number, cadence)
    return parameters


def _verify_design_run_metadata(
    run: Path,
    *,
    design: dict,
    design_file_sha256: str,
    design_path: Path,
    cases_path: Path,
    manifest_path: Path,
    role: str,
    initial_state_only: bool,
) -> None:
    path = run / "fdm_adapter_metadata.json"
    if not path.is_file():
        raise ValueError(f"registered q/e run lacks bound metadata: {run}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    expected_binding = {
        "status": "qe_prospective_design_bound_not_a_calibration_release",
        "path": str(design_path.resolve()),
        "physical_cases_path": str(cases_path.resolve()),
        "run_manifest_path": str(manifest_path.resolve()),
        "file_sha256": design_file_sha256,
        "design_sha256": design["design_sha256"],
        "role": role,
    }
    if (
        metadata.get("qe_design_binding") != expected_binding
        or metadata.get("run_id") != run.name
    ):
        raise ValueError(f"registered q/e run has a different design binding: {run}")
    verify_qe_design_run_request(
        design, case_id=metadata["case_id"], role=role,
        resolution=metadata["resolution"], box_size_pc=metadata["box_size_pc"],
        duration_myr=metadata["duration_myr"],
        initial_state_only=initial_state_only,
    )


def _summary_complete(
    path: Path,
    *,
    expected_status: str,
    expected_run_directory: Path,
) -> bool:
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(summary, dict) or summary.get("status") != expected_status:
        return False

    # The wave-response analyzer records the resolved input run as ``run``.
    # Torch summaries currently omit a run-directory field, so do not require
    # one; when a producer supplies it, reject a stale/copied summary.
    if "run" in summary:
        run = summary["run"]
        if not isinstance(run, str):
            return False
        try:
            recorded_run = Path(run).expanduser().resolve()
            expected_run = expected_run_directory.expanduser().resolve()
        except OSError:
            return False
        if recorded_run != expected_run:
            return False
    return True


def _torch_summary_complete(path: Path, run_directory: Path) -> bool:
    return _summary_complete(
        path,
        expected_status="complete",
        expected_run_directory=run_directory,
    )


def _response_summary_complete(path: Path, run_directory: Path) -> bool:
    return _summary_complete(
        path,
        expected_status="diagnosed",
        expected_run_directory=run_directory,
    )


def _status_detail(
    *,
    seed_ready: bool,
    torch_complete: bool,
    response_complete: bool,
    torch_directory_exists: bool,
) -> str:
    if response_complete:
        return "complete"
    if torch_complete:
        return "torch_complete,response_pending"
    if not seed_ready:
        if torch_directory_exists:
            return "seed_missing,torch_incomplete"
        return "seed_missing"
    if torch_directory_exists:
        return "seed_ready,torch_incomplete,resume"
    return "seed_ready,torch_pending"


def build_plan(
    manifest_path: Path,
    cases_path: Path,
    initial_root: Path,
    torch_root: Path,
    pyul_path: Path,
    *,
    save_number: int | None = None,
    movie_frame_number: int = 96,
    save_3d_number: int = 16,
    checkpoint_every_saves: int = 32,
    rk4_substeps: int = 9,
    device: str = "cuda:0",
    qe_design_path: Path | None = None,
) -> list[RunPlanRow]:
    manifest = _read_rows(manifest_path)
    case_parameters = _load_case_parameters(cases_path)
    design = None
    design_file_sha256 = None
    if qe_design_path is not None:
        qe_design_path = qe_design_path.expanduser().resolve()
        design, design_file_sha256 = read_verified_qe_followup_design(
            qe_design_path, physical_cases=cases_path, run_manifest=manifest_path,
        )
    plan: list[RunPlanRow] = []
    for row in manifest:
        run_id = row["run_id"]
        case_id = row["case_id"]
        duration, derived_save_number, cadence = case_parameters[case_id]
        role = None
        marker = row.get("requires_qe_design", "").strip().lower()
        if marker not in ("", "false", "0", "true", "1"):
            raise ValueError("q/e design requirement marker is invalid")
        if marker in ("true", "1") and design is None:
            raise ValueError("q/e follow-up manifest requires --qe-design")
        if design is not None:
            if case_id != design["case_id"]:
                raise ValueError("registered q/e plan contains another physical case")
            resolution = int(row["effective_grid_cells"])
            box_size = float(row["box_size_pc"])
            roles = {
                "coarse": design["resolution_pair"]["coarse"],
                "fine": design["resolution_pair"]["fine"],
                "doubled_box_control": design["doubled_box_control"],
            }
            matching = [
                name for name, geometry in roles.items()
                if geometry["resolution"] == resolution
            ]
            if len(matching) != 1:
                raise ValueError("registered q/e plan resolution lacks one role")
            role = matching[0]
            if run_id != f"{case_id}_n{resolution}":
                raise ValueError("registered q/e run ID differs from seed naming")
            duration = float(design["planned_duration_myr"])
            verify_qe_design_run_request(
                design, case_id=case_id, role=role,
                resolution=resolution, box_size_pc=box_size,
                duration_myr=duration,
            )
            derived_save_number = math.floor(duration / cadence + 1.0e-9)
            if derived_save_number < 1:
                raise ValueError("registered q/e output cadence exceeds planned duration")
            if save_number is not None and save_number < derived_save_number:
                raise ValueError(
                    "registered q/e save-count override undersamples the planned cadence"
                )
        initial_directory = initial_root / run_id
        torch_directory = torch_root / run_id
        seed_ready = (
            initial_directory / "Outputs" / "3Wfn" / "P3D_#000.npy"
        ).is_file()
        if design is not None and initial_directory.exists():
            if not seed_ready:
                raise ValueError(f"registered q/e seed directory is incomplete: {initial_directory}")
            _verify_design_run_metadata(
                initial_directory, design=design, design_file_sha256=design_file_sha256,
                design_path=qe_design_path, cases_path=cases_path,
                manifest_path=manifest_path, role=role, initial_state_only=True,
            )
        torch_directory_exists = torch_directory.is_dir()
        if design is not None and torch_directory_exists:
            if not seed_ready:
                raise ValueError(
                    f"registered q/e Torch run has no bound initial seed: {torch_directory}"
                )
            _verify_design_run_metadata(
                torch_directory, design=design, design_file_sha256=design_file_sha256,
                design_path=qe_design_path, cases_path=cases_path,
                manifest_path=manifest_path, role=role, initial_state_only=False,
            )
        torch_complete = _torch_summary_complete(
            torch_directory / "torch_run_summary.json",
            torch_directory,
        )
        response_complete = torch_complete and _response_summary_complete(
            torch_directory / "wave_response_summary.json",
            torch_directory,
        )
        plan.append(
            RunPlanRow(
                run_id=run_id,
                case_id=case_id,
                resolution=int(row["effective_grid_cells"]),
                box_size_pc=float(row["box_size_pc"]),
                case_duration_myr=duration,
                seed_ready=seed_ready,
                torch_complete=torch_complete,
                response_complete=response_complete,
                torch_directory_exists=torch_directory_exists,
                status_detail=_status_detail(
                    seed_ready=seed_ready,
                    torch_complete=torch_complete,
                    response_complete=response_complete,
                    torch_directory_exists=torch_directory_exists,
                ),
                cases_path=cases_path,
                initial_root=initial_root,
                torch_root=torch_root,
                pyul_path=pyul_path,
                save_number=(
                    derived_save_number if save_number is None else save_number
                ),
                movie_frame_number=movie_frame_number,
                save_3d_number=save_3d_number,
                checkpoint_every_saves=checkpoint_every_saves,
                rk4_substeps=rk4_substeps,
                device=device,
                qe_design_path=qe_design_path,
                qe_design_manifest_path=(manifest_path if design is not None else None),
                qe_design_role=role,
            )
        )
    return plan


def _print_summary(plan: list[RunPlanRow]) -> None:
    total = len(plan)
    seed_ready = sum(row.seed_ready for row in plan)
    torch_complete = sum(row.torch_complete for row in plan)
    response_complete = sum(row.response_complete for row in plan)
    print(f"total_runs={total}")
    print(f"seed_ready_runs={seed_ready}")
    print(f"torch_completed_runs={torch_complete}")
    print(f"response_completed_runs={response_complete}")
    print(f"completed_runs={response_complete}")
    print(f"remaining_runs={total-response_complete}")
    for row in plan:
        print(
            f"run {row.case_id:>30s} n{row.resolution:<5d} "
            f"id={row.run_id} status={row.status_detail}"
        )


def _emit_csv(plan: list[RunPlanRow], path: Path) -> None:
    fieldnames = [
        "run_id",
        "case_id",
        "effective_grid_cells",
        "box_size_pc",
        "target_duration_myr",
        "save_number",
        "movie_frame_number",
        "save_3d_number",
        "seed_ready",
        "torch_complete",
        "response_complete",
        "status_detail",
        "seed_command",
        "torch_command",
        "response_command",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in plan:
            pending = dict(row.pending_commands)
            writer.writerow(
                {
                    "run_id": row.run_id,
                    "case_id": row.case_id,
                    "effective_grid_cells": row.resolution,
                    "box_size_pc": row.box_size_pc,
                    "target_duration_myr": row.case_duration_myr,
                    "save_number": row.save_number,
                    "movie_frame_number": row.movie_frame_number,
                    "save_3d_number": row.save_3d_number,
                    "seed_ready": row.seed_ready,
                    "torch_complete": row.torch_complete,
                    "response_complete": row.response_complete,
                    "status_detail": row.status_detail,
                    "seed_command": pending.get("seed", ""),
                    "torch_command": pending.get("torch", ""),
                    "response_command": pending.get("response", ""),
                }
            )


def _emit_commands(plan: list[RunPlanRow]) -> None:
    for row in plan:
        for stage, command in row.pending_commands:
            print(f"stage={stage} run_id={row.run_id} {command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan seed, resumable Torch, and response stages for q-e runs."
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
    parser.add_argument(
        "--initial-root",
        "--results-root",
        dest="initial_root",
        type=Path,
        default=DEFAULT_RESULT_ROOT / "pyul_initial",
    )
    parser.add_argument(
        "--torch-root",
        "--output-root",
        dest="torch_root",
        type=Path,
        default=DEFAULT_RESULT_ROOT / "torch",
    )
    parser.add_argument("--pyul-path", type=Path, default=DEFAULT_PYUL_PATH)
    parser.add_argument(
        "--qe-design", type=Path,
        help="verified prospective design required by a marked follow-up manifest",
    )
    parser.add_argument(
        "--save-number",
        type=int,
        help="override output-cadence-derived saved intervals for every case",
    )
    parser.add_argument("--movie-frame-number", type=int, default=96)
    parser.add_argument("--save-3d-number", type=int, default=16)
    parser.add_argument("--checkpoint-every-saves", type=int, default=32)
    parser.add_argument("--rk4-substeps", type=int, default=9)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="optional CSV dump of status and stage commands",
    )
    parser.add_argument(
        "--emit-commands",
        action="store_true",
        help="print only the next safe stage commands for incomplete runs",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.save_number is not None and args.save_number < 1:
        parser.error("--save-number must be positive")
    if args.movie_frame_number < 1:
        parser.error("--movie-frame-number must be positive")
    if args.save_3d_number < 1:
        parser.error("--save-3d-number must be positive")
    if args.checkpoint_every_saves < 1:
        parser.error("--checkpoint-every-saves must be positive")
    if args.rk4_substeps < 1:
        parser.error("--rk4-substeps must be positive")

    cases_path = args.cases.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    initial_root = args.initial_root.expanduser().resolve()
    torch_root = args.torch_root.expanduser().resolve()
    pyul_path = args.pyul_path.expanduser().resolve()
    plan = build_plan(
        manifest_path,
        cases_path,
        initial_root,
        torch_root,
        pyul_path,
        save_number=args.save_number,
        movie_frame_number=args.movie_frame_number,
        save_3d_number=args.save_3d_number,
        checkpoint_every_saves=args.checkpoint_every_saves,
        rk4_substeps=args.rk4_substeps,
        device=args.device,
        qe_design_path=args.qe_design,
    )

    _print_summary(plan)
    if args.output_csv is not None:
        _emit_csv(plan, args.output_csv.expanduser().resolve())
    if args.emit_commands:
        _emit_commands(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
