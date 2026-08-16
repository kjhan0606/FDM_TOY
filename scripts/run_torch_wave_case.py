#!/usr/bin/env python3
"""Evolve a PyUL-compatible live FDM wave with PyTorch CUDA FFTs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from fdm_smbh_delay.pyul import pyul_unit_system
from fdm_smbh_delay.torch_wave import (
    advance_binary_rk4_patched,
    periodic_poisson_torch,
    plummer_potential_torch,
    potential_patches,
    sample_potential_and_acceleration,
    spectral_grid,
    wave_energy_components,
)


def _initial_paths(reference: Path) -> tuple[Path, Path]:
    wave = reference / "Outputs" / "3Wfn" / "P3D_#000.npy"
    particles = reference / "Outputs" / "NBody" / "NTM_#000.npy"
    if not wave.is_file() or not particles.is_file():
        raise FileNotFoundError(
            "reference run must contain the initial 3Wfn and NBody snapshots"
        )
    return wave, particles


def _selected_indices(save_number: int, requested_intervals: int) -> set[int]:
    if requested_intervals < 0:
        raise ValueError("saved interval counts cannot be negative")
    if requested_intervals == 0:
        return set()
    count = min(save_number, requested_intervals)
    return set(
        np.unique(np.rint(np.linspace(0, save_number, count + 1)).astype(int))
    )


def _estimated_remaining_seconds(
    *, elapsed_seconds: float, start_step: int, step: int, total_steps: int
) -> float | None:
    completed_steps = step - start_step
    if completed_steps <= 0:
        return None
    return elapsed_seconds * (total_steps - step) / completed_steps


def _save_array(run: Path, category: str, prefix: str, index: int, value) -> None:
    directory = run / "Outputs" / category
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{prefix}_#{index:03d}.npy", value)


def _write_compatible_outputs(
    *,
    run: Path,
    index: int,
    wavefunction: torch.Tensor,
    density: torch.Tensor,
    state: np.ndarray,
    movie_indices: set[int],
    wave_indices: set[int],
) -> None:
    _save_array(run, "NBody", "NTM", index, state)
    centre = density.shape[0] // 2
    _save_array(
        run,
        "1Density",
        "R1D",
        index,
        density[centre, :, centre].detach().cpu().numpy(),
    )
    if index in movie_indices:
        _save_array(
            run,
            "2Density",
            "R2D",
            index,
            density[:, :, centre].detach().cpu().numpy(),
        )
    if index in wave_indices:
        _save_array(
            run,
            "3Wfn",
            "P3D",
            index,
            wavefunction.detach().cpu().numpy(),
        )


def _save_energy_logs(run: Path, logs: dict[str, list[float]]) -> None:
    for name, values in logs.items():
        np.save(run / "Outputs" / f"{name}.npy", np.asarray(values))


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _save_checkpoint(
    *,
    run: Path,
    wavefunction: torch.Tensor,
    state: np.ndarray,
    step: int,
    save_index: int,
) -> None:
    """Atomically publish a restart point and retain the previous one until then."""

    checkpoint_directory = run / "Checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)
    stem = f"{save_index:06d}"
    wave_name = f"wave_{stem}.npy"
    state_name = f"state_{stem}.npz"
    wave_path = checkpoint_directory / wave_name
    state_path = checkpoint_directory / state_name
    wave_temporary = checkpoint_directory / f".{wave_name}.tmp"
    state_temporary = checkpoint_directory / f".{state_name}.tmp"

    with wave_temporary.open("wb") as stream:
        np.save(stream, wavefunction.detach().cpu().numpy())
    os.replace(wave_temporary, wave_path)
    with state_temporary.open("wb") as stream:
        np.savez(
            stream,
            state=np.asarray(state, dtype=np.float64),
            step=np.int64(step),
            save_index=np.int64(save_index),
        )
    os.replace(state_temporary, state_path)

    marker_path = checkpoint_directory / "latest.json"
    previous = None
    if marker_path.is_file():
        previous = json.loads(marker_path.read_text(encoding="utf-8"))
    _atomic_json(
        marker_path,
        {
            "wave": wave_name,
            "state": state_name,
            "step": step,
            "save_index": save_index,
        },
    )
    if previous is not None:
        for key in ("wave", "state"):
            old_path = checkpoint_directory / previous[key]
            if old_path not in (wave_path, state_path):
                old_path.unlink(missing_ok=True)


def _load_checkpoint(
    *, run: Path, device: torch.device
) -> tuple[torch.Tensor, np.ndarray, int, int]:
    checkpoint_directory = run / "Checkpoints"
    marker_path = checkpoint_directory / "latest.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"restart marker is missing: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    wave_path = checkpoint_directory / marker["wave"]
    state_path = checkpoint_directory / marker["state"]
    if not wave_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("restart marker refers to an incomplete checkpoint")
    wavefunction = torch.as_tensor(np.load(wave_path), device=device)
    with np.load(state_path) as saved:
        state = np.asarray(saved["state"], dtype=np.float64)
        step = int(saved["step"])
        save_index = int(saved["save_index"])
    if step != int(marker["step"]) or save_index != int(marker["save_index"]):
        raise ValueError("restart marker and checkpoint state disagree")
    return wavefunction, state, step, save_index


def _load_energy_logs(run: Path, save_index: int) -> dict[str, list[float]]:
    logs = {}
    for name in (
        "egylist",
        "egpcmlist",
        "egpsilist",
        "ekandqlist",
        "masseslist",
        "egpcmMlist",
        "ULDMass",
    ):
        values = np.load(run / "Outputs" / f"{name}.npy")
        if values.size < save_index + 1:
            raise ValueError(f"{name} does not reach checkpoint {save_index}")
        logs[name] = values[: save_index + 1].tolist()
    return logs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration-myr", required=True, type=float)
    parser.add_argument("--save-number", type=int, default=2048)
    parser.add_argument("--movie-frame-number", type=int, default=360)
    parser.add_argument("--save-3d-number", type=int, default=32)
    parser.add_argument("--rk4-substeps", type=int, default=9)
    parser.add_argument("--time-step-factor", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-every-saves", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.duration_myr <= 0.0 or args.save_number < 1:
        raise ValueError("duration and save number must be positive")
    if (
        args.rk4_substeps < 1
        or not 0.0 < args.time_step_factor <= 1.0
        or args.checkpoint_every_saves < 0
    ):
        raise ValueError("RK4 substeps and time-step factor are invalid")

    reference = args.reference_run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if args.resume and not output.is_dir():
        raise FileNotFoundError(f"restart output does not exist: {output}")
    initial_wave_path, initial_particle_path = _initial_paths(reference)
    reference_metadata = json.loads(
        (reference / "fdm_adapter_metadata.json").read_text(encoding="utf-8")
    )
    config = json.loads((reference / "config.uldm").read_text(encoding="utf-8"))
    resolution = int(reference_metadata["resolution"])
    box_pc = float(reference_metadata["box_size_pc"])
    units = pyul_unit_system(reference_metadata)
    box_code = box_pc / units.length_pc
    duration_code = args.duration_myr / units.time_myr
    masses_msun = np.asarray(
        [particle[0] for particle in config["Matter Particles"]["Condition"]],
        dtype=float,
    )
    if masses_msun.shape != (2,):
        raise ValueError("CUDA calibration currently requires exactly two SMBHs")
    masses_code = masses_msun / units.mass_msun
    plummer_pc = float(config["Matter Particles"]["Plummer Radius"])
    plummer_code = plummer_pc / units.length_pc

    nominal_step = (box_code / resolution) ** 2 / np.pi
    minimum_steps = int(duration_code / nominal_step + 1.0)
    minimum_steps = int(minimum_steps / args.time_step_factor)
    remainder = minimum_steps % args.save_number
    actual_steps = (
        minimum_steps
        if remainder == 0
        else minimum_steps + args.save_number - remainder
    )
    save_number = args.save_number
    if actual_steps % save_number != 0:
        raise ValueError("actual step count must be divisible by saved intervals")
    steps_per_save = actual_steps // save_number
    time_step = duration_code / actual_steps
    duration_completed_myr = time_step * actual_steps * units.time_myr

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    movie_indices = _selected_indices(save_number, args.movie_frame_number)
    wave_indices = _selected_indices(save_number, args.save_3d_number)
    metadata = dict(reference_metadata)
    metadata.update(
        {
            "run_id": output.name,
            "reference_initial_state": str(reference),
            "backend": "pytorch_cuda" if device.type == "cuda" else "pytorch_cpu",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "duration_myr": duration_completed_myr,
            "time_step_factor": args.time_step_factor,
            "nbody_rk4_substeps_per_wave_step": args.rk4_substeps,
            "save_number": save_number,
            "saved_movie_planes": len(movie_indices),
            "saved_3d_states": len(wave_indices),
            "actual_wave_steps": actual_steps,
            "wave_time_step_code": time_step,
            "checkpoint_every_saved_intervals": args.checkpoint_every_saves,
            "wave_acceleration_during_particle_rk4": (
                "interpolated_from_a_local_potential_patch_at_each_rk4_stage"
            ),
            "analytic_fdm_drag": False,
            "live_wave_force_on_smbhs": True,
            "smbh_force_on_live_wave": True,
        }
    )

    if args.resume:
        saved_metadata = json.loads(
            (output / "fdm_adapter_metadata.json").read_text(encoding="utf-8")
        )
        for key in (
            "resolution",
            "box_size_pc",
            "duration_myr",
            "save_number",
            "actual_wave_steps",
            "wave_time_step_code",
        ):
            if saved_metadata[key] != metadata[key]:
                raise ValueError(f"restart request changes {key}")
    else:
        output.mkdir(parents=True)
        (output / "Outputs").mkdir()
        config["Duration"]["Time Duration"] = duration_completed_myr
        config["Save Options"]["Number"] = save_number
        config["Save Options"]["Flags"] = "Energy NBody 1Density 2Density 3Wfn"
        (output / "config.uldm").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        shutil.copyfile(
            reference / "reproducibility.uldm", output / "reproducibility.uldm"
        )
        (output / "fdm_adapter_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    start = time.perf_counter()
    if args.resume:
        wavefunction, state, start_step, start_save_index = _load_checkpoint(
            run=output, device=device
        )
    else:
        initial_wave = np.load(initial_wave_path)
        if initial_wave.shape != (resolution, resolution, resolution):
            raise ValueError("reference wavefunction has an incompatible shape")
        wavefunction = torch.as_tensor(initial_wave, device=device)
        if wavefunction.dtype != torch.complex128:
            wavefunction = wavefunction.to(torch.complex128)
        del initial_wave
        state = np.load(initial_particle_path).astype(float)
        start_step = 0
        start_save_index = 0
    if wavefunction.shape != (resolution, resolution, resolution):
        raise ValueError("checkpoint wavefunction has an incompatible shape")
    if wavefunction.dtype != torch.complex128:
        wavefunction = wavefunction.to(torch.complex128)
    if start_step != start_save_index * steps_per_save:
        raise ValueError("checkpoint is not aligned with a saved interval")

    grid = spectral_grid(
        resolution=resolution,
        box_length=box_code,
        time_step=time_step,
        device=device,
    )
    density = wavefunction.abs().square()
    wave_potential = periodic_poisson_torch(
        density, grid.poisson_inverse_wavenumber_squared
    )
    compact_potential = plummer_potential_torch(
        coordinate=grid.coordinate,
        masses=masses_code,
        positions=state.reshape(2, 6)[:, :3],
        plummer_radius=plummer_code,
    )
    total_potential = wave_potential + compact_potential

    if args.resume:
        logs = _load_energy_logs(output, start_save_index)
    else:
        logs: dict[str, list[float]] = {
            "egylist": [],
            "egpcmlist": [],
            "egpsilist": [],
            "ekandqlist": [],
            "masseslist": [],
            "egpcmMlist": [],
            "ULDMass": [],
        }

    def save(index: int) -> None:
        kinetic, self_gravity, interaction, wave_mass = wave_energy_components(
            wavefunction=wavefunction,
            density=density,
            wave_potential=wave_potential,
            compact_potential=compact_potential,
            kinetic_wavenumber_squared=grid.kinetic_wavenumber_squared,
            cell_volume=grid.cell_volume,
        )
        point_potential, _ = sample_potential_and_acceleration(
            potential=wave_potential,
            positions=state.reshape(2, 6)[:, :3],
            box_length=box_code,
        )
        point_interaction = float(masses_code @ point_potential)
        logs["ekandqlist"].append(kinetic)
        logs["egpsilist"].append(self_gravity)
        logs["egpcmlist"].append(interaction)
        logs["egylist"].append(kinetic + self_gravity + interaction)
        logs["masseslist"].append(wave_mass)
        logs["egpcmMlist"].append(point_interaction)
        logs["ULDMass"].append(wave_mass)
        _write_compatible_outputs(
            run=output,
            index=index,
            wavefunction=wavefunction,
            density=density,
            state=state,
            movie_indices=movie_indices,
            wave_indices=wave_indices,
        )
        _save_energy_logs(output, logs)

    if not args.resume:
        save(0)
        if args.checkpoint_every_saves:
            _save_checkpoint(
                run=output,
                wavefunction=wavefunction,
                state=state,
                step=0,
                save_index=0,
            )
    for step in range(start_step + 1, actual_steps + 1):
        half_phase = torch.exp(-0.5j * time_step * total_potential)
        wavefunction.mul_(half_phase)
        del half_phase
        wavefunction_k = torch.fft.fftn(wavefunction)
        wavefunction_k.mul_(grid.kinetic_phase)
        wavefunction = torch.fft.ifftn(wavefunction_k)
        del wavefunction_k
        density = wavefunction.abs().square()
        wave_potential = periodic_poisson_torch(
            density, grid.poisson_inverse_wavenumber_squared
        )
        wave_patches, patch_starts = potential_patches(
            potential=wave_potential,
            positions=state.reshape(2, 6)[:, :3],
            box_length=box_code,
        )
        state = advance_binary_rk4_patched(
            state=state,
            masses=masses_code,
            patches=wave_patches,
            patch_starts=patch_starts,
            box_length=box_code,
            resolution=resolution,
            plummer_radius=plummer_code,
            time_step=time_step,
            substeps=args.rk4_substeps,
        )
        compact_potential = plummer_potential_torch(
            coordinate=grid.coordinate,
            masses=masses_code,
            positions=state.reshape(2, 6)[:, :3],
            plummer_radius=plummer_code,
        )
        total_potential = wave_potential + compact_potential
        wavefunction.mul_(torch.exp(-0.5j * time_step * total_potential))
        if step % steps_per_save == 0:
            save_index = step // steps_per_save
            save(save_index)
            if args.checkpoint_every_saves and (
                save_index % args.checkpoint_every_saves == 0
                or save_index == save_number
            ):
                _save_checkpoint(
                    run=output,
                    wavefunction=wavefunction,
                    state=state,
                    step=step,
                    save_index=save_index,
                )
            elapsed = time.perf_counter() - start
            print(
                json.dumps(
                    {
                        "saved_index": save_index,
                        "step": step,
                        "steps": actual_steps,
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": (
                            _estimated_remaining_seconds(
                                elapsed_seconds=elapsed,
                                start_step=start_step,
                                step=step,
                                total_steps=actual_steps,
                            )
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device)
    else:
        peak_memory = 0
    summary = {
        "status": "complete",
        "actual_wave_steps": actual_steps,
        "saved_intervals": save_number,
        "duration_myr": duration_completed_myr,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_device_memory_bytes": peak_memory,
    }
    (output / "torch_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
