#!/usr/bin/env python3
"""Run one live-wave calibration case with the public PyUL_NBody solver."""

from __future__ import annotations

import argparse
import builtins
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


def _load_case(path: Path, case_id: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] == case_id:
                return row
    raise ValueError(f"case_id={case_id!r} is absent from {path}")


def _git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyul-path", required=True, type=Path)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("results/wave_calibration_grid/physical_cases.csv"),
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--duration-myr", type=float)
    parser.add_argument("--save-number", type=int, default=16)
    parser.add_argument("--rk-steps", type=int, default=36)
    parser.add_argument("--time-step-factor", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results/pyul_wave"))
    parser.add_argument("--box-pc", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-3d", action="store_true")
    parser.add_argument("--save-3d-number", type=int)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cases_path = args.cases.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    pyul_path = args.pyul_path.expanduser().resolve()
    if not (pyul_path / "PyUltraLight2" / "Current.py").is_file():
        raise ValueError("--pyul-path must point to a PyUL_NBody checkout")
    if args.resolution < 128:
        raise ValueError("PyUL_NBody requires resolution >= 128")
    if args.save_number < 1:
        raise ValueError("--save-number must be positive")
    if not 0.0 < args.time_step_factor <= 1.0:
        raise ValueError("--time-step-factor must satisfy 0 < factor <= 1")
    if args.duration_myr is not None and args.duration_myr <= 0.0:
        raise ValueError("--duration-myr must be positive")
    if args.save_3d and args.save_3d_number is not None:
        raise ValueError("choose either all 3D states or a sparse 3D schedule")
    if args.save_3d_number is not None:
        if not 0 < args.save_3d_number <= args.save_number:
            raise ValueError("--save-3d-number must lie between 1 and --save-number")
        if args.save_number % args.save_3d_number != 0:
            raise ValueError("--save-3d-number must divide --save-number")

    case = _load_case(cases_path, args.case_id)
    particle_mass_ev = float(case["particle_mass_ev"])
    soliton_mass = float(case["soliton_mass_msun"])
    core_radius = float(case["core_radius_pc"])
    mass1 = float(case["mass1_msun"])
    mass2 = float(case["mass2_msun"])
    separation = float(case["initial_separation_pc"])
    speed1 = float(case["initial_speed1_km_s"])
    speed2 = float(case["initial_speed2_km_s"])
    duration = (
        float(case["target_duration_myr"])
        if args.duration_myr is None
        else args.duration_myr
    )
    box_size = 100.0 if args.box_pc is None else args.box_pc
    if box_size <= 4.0 * separation:
        raise ValueError("simulation box must exceed four initial binary separations")

    total_mass = mass1 + mass2
    position1 = [separation * mass2 / total_mass, 0.0, 0.0]
    position2 = [-separation * mass1 / total_mass, 0.0, 0.0]
    particles = [
        [mass1, position1, [0.0, speed1, 0.0]],
        [mass2, position2, [0.0, -speed2, 0.0]],
    ]
    solitons = [[soliton_mass, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0]]
    run_id = f"{args.case_id}_n{args.resolution}"
    run_directory = output_root / run_id
    if run_directory.exists():
        raise FileExistsError(f"refusing to replace existing run directory: {run_directory}")
    output_root.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    original_input = builtins.input
    evolve_globals = None
    original_save_grid = None
    try:
        os.chdir(pyul_path)
        os.environ.setdefault("NUMEXPR_MAX_THREADS", "128")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "64")
        sys.path.insert(0, str(pyul_path))
        builtins.input = lambda _prompt="": f"{particle_mass_ev:.16g}"
        # SciPy 1.15 removed the deprecated sph_harm name used by PyUL_NBody.
        # Preserve the old (m, n, theta, phi) calling convention without
        # changing the upstream checkout.
        import scipy.special as scipy_special

        if not hasattr(scipy_special, "sph_harm"):
            scipy_special.sph_harm = (  # type: ignore[attr-defined]
                lambda m, n, theta, phi: scipy_special.sph_harm_y(
                    n, m, theta, phi
                )
            )
        import IPython.core.display as ipython_core_display
        from IPython.display import clear_output

        if not hasattr(ipython_core_display, "clear_output"):
            ipython_core_display.clear_output = clear_output
        import PyUltraLight2 as pyul
        # The upstream progress bar emits several terminal updates per wave
        # step and can overwhelm batch logs. This changes display only.
        quiet_progress = lambda *_args, **_kwargs: None
        pyul.prog_bar = quiet_progress
        pyul.evolve.__globals__["prog_bar"] = quiet_progress

        cell_size = box_size / args.resolution
        plummer_radius = max(0.001, 0.5 * cell_size)
        plummer_parameter = pyul.GenPlummer(plummer_radius, "pc")
        save_options = "Energy NBody DF 1Density Entropy Quadrupole"
        if args.save_3d or args.save_3d_number is not None:
            save_options += " 3Density 3Wfn"
        estimated_steps = int(
            pyul.ULDStepEst(
                duration,
                "Myr",
                box_size,
                "pc",
                args.resolution,
                args.time_step_factor,
                save_number=-1,
            )
        )
        if args.save_3d:
            saved_3d_states = args.save_number + 1
        elif args.save_3d_number is not None:
            saved_3d_states = args.save_3d_number + 1
        else:
            saved_3d_states = 0
        metadata = {
            "case_id": args.case_id,
            "run_id": run_id,
            "project_root": str(project_root),
            "pyul_repository": "https://github.com/Sifyrena/PyUL_NBody",
            "pyul_revision": _git_revision(pyul_path),
            "particle_mass_ev": particle_mass_ev,
            "resolution": args.resolution,
            "box_size_pc": box_size,
            "cell_size_pc": cell_size,
            "plummer_radius_pc": plummer_radius,
            "duration_myr": duration,
            "time_step_factor": args.time_step_factor,
            "save_number": args.save_number,
            "saved_3d_states": saved_3d_states,
            "estimated_wave_steps": estimated_steps,
            "analytic_fdm_drag": False,
            "live_wave_force_on_smbhs": True,
            "smbh_force_on_live_wave": True,
            "core_radius_reference_pc": core_radius,
            "pyul_length_unit_m": float(pyul.length_unit),
            "pyul_time_unit_s": float(pyul.time_unit),
            "pyul_mass_unit_kg": float(pyul.mass_unit),
            "pyul_energy_unit_j": float(pyul.energy_unit),
        }
        if args.dry_run:
            print(json.dumps(metadata, indent=2, sort_keys=True))
            return 0

        generated = pyul.GenerateConfig(
            args.rk_steps,
            box_size,
            "pc",
            args.resolution,
            duration,
            "Myr",
            args.time_step_factor,
            args.save_number,
            save_options,
            str(output_root),
            "npy",
            "solar_masses",
            "pc",
            "km/s",
            solitons,
            0.0,
            "solar_masses",
            "pc",
            "km/s",
            particles,
            [],
            False,
            0.0,
            "MSol/pc3",
            plummer_parameter,
            [0.0, 0.0, 0.0],
            True,
            run_id,
        )
        if generated != run_id:
            raise RuntimeError(f"PyUL generated unexpected run name {generated!r}")
        (run_directory / "fdm_adapter_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if args.save_3d_number is not None:
            stride = args.save_number // args.save_3d_number
            evolve_globals = pyul.evolve.__globals__
            original_save_grid = evolve_globals["save_grid"]

            def sparse_save_grid(
                rho,
                psi,
                resol,
                matter_state,
                phi_sp,
                phi,
                gradient_log,
                options,
                save_format,
                location,
                step_index,
                steps_per_save,
            ):
                snapshot = (
                    0
                    if step_index < 0
                    else int((step_index + 1) / steps_per_save)
                )
                selected_options = list(options)
                if snapshot % stride != 0 and snapshot != args.save_number:
                    selected_options[0] = False
                    selected_options[1] = False
                return original_save_grid(
                    rho,
                    psi,
                    resol,
                    matter_state,
                    phi_sp,
                    phi,
                    gradient_log,
                    selected_options,
                    save_format,
                    location,
                    step_index,
                    steps_per_save,
                )

            evolve_globals["save_grid"] = sparse_save_grid
        pyul.evolve(
            str(output_root),
            run_id,
            EdgeClear=False,
            IsoP=False,
            UseDispSponge=False,
            SelfGravity=True,
            NBodyInterp=True,
            NBodyGravity=True,
            Silent=True,
            GenerateLog=True,
            ComputeQuad=True,
        )
    finally:
        if evolve_globals is not None and original_save_grid is not None:
            evolve_globals["save_grid"] = original_save_grid
        builtins.input = original_input
        os.chdir(original_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
