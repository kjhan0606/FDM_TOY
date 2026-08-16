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
    checkpoint_every_saves: int
    rk4_substeps: int
    device: str

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
        return _shell_command(
            [
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
        )

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


def _load_case_parameters(cases_path: Path) -> dict[str, tuple[float, int]]:
    cases = _read_rows(cases_path)
    parameters: dict[str, tuple[float, int]] = {}
    for row in cases:
        duration = float(row["target_duration_myr"])
        cadence = float(row["output_cadence_myr"])
        if duration <= 0.0 or cadence <= 0.0:
            raise ValueError("case duration and output cadence must be positive")
        save_number = math.floor(duration / cadence + 1.0e-9)
        if save_number < 1:
            raise ValueError("output cadence must not exceed the case duration")
        parameters[row["case_id"]] = (duration, save_number)
    return parameters


def _torch_summary_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return summary.get("status") == "complete"


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
    checkpoint_every_saves: int = 32,
    rk4_substeps: int = 9,
    device: str = "cuda:0",
) -> list[RunPlanRow]:
    manifest = _read_rows(manifest_path)
    case_parameters = _load_case_parameters(cases_path)
    plan: list[RunPlanRow] = []
    for row in manifest:
        run_id = row["run_id"]
        case_id = row["case_id"]
        duration, derived_save_number = case_parameters[case_id]
        initial_directory = initial_root / run_id
        torch_directory = torch_root / run_id
        seed_ready = (
            initial_directory / "Outputs" / "3Wfn" / "P3D_#000.npy"
        ).is_file()
        torch_directory_exists = torch_directory.is_dir()
        torch_complete = _torch_summary_complete(
            torch_directory / "torch_run_summary.json"
        )
        response_complete = torch_complete and (
            torch_directory / "wave_response_summary.json"
        ).is_file()
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
                checkpoint_every_saves=checkpoint_every_saves,
                rk4_substeps=rk4_substeps,
                device=device,
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
        "--save-number",
        type=int,
        help="override output-cadence-derived saved intervals for every case",
    )
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
        checkpoint_every_saves=args.checkpoint_every_saves,
        rk4_substeps=args.rk4_substeps,
        device=args.device,
    )

    _print_summary(plan)
    if args.output_csv is not None:
        _emit_csv(plan, args.output_csv.expanduser().resolve())
    if args.emit_commands:
        _emit_commands(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
