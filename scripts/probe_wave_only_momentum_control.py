#!/usr/bin/env python3
"""Short source-bound wave-only momentum null control; no calibration output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import torch

from fdm_smbh_delay.pyul import pyul_unit_system
from fdm_smbh_delay.torch_wave import (
    apply_kinetic_phase_in_place,
    apply_potential_half_kick_in_place,
    periodic_poisson_torch,
    spectral_grid,
    wave_density,
    wave_kinetic_energy,
)


_LEVELS = (
    ("f100", 1.0, 10), ("f050", 0.5, 20),
    ("f025", 0.25, 40), ("f0125", 0.125, 80),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spectral_wave_diagnostics_torch(
    wave: torch.Tensor, box_length: float, *, high_shell: bool = False,
) -> dict:
    """Match the NumPy audit's signed momentum and summation floor."""

    if (
        wave.ndim != 3 or len(set(wave.shape)) != 1
        or wave.dtype != torch.complex128
        or not np.isfinite(box_length) or box_length <= 0.0
    ):
        raise ValueError("wave momentum requires a cubic complex128 field")
    resolution = wave.shape[0]
    wave_number = 2.0 * torch.pi * torch.fft.fftfreq(
        resolution, d=box_length / resolution,
        dtype=torch.float64, device=wave.device,
    )
    if resolution % 2 == 0:
        wave_number[resolution // 2] = 0.0
    spectrum = torch.fft.fftn(wave)
    power = spectrum.real.square()
    power.addcmul_(spectrum.imag, spectrum.imag)
    factor = (box_length / resolution) ** 3 / resolution**3
    components = []
    absolute_weight = 0.0
    for axes in ((1, 2), (0, 2), (0, 1)):
        marginal = power.sum(dim=axes)
        components.append(float(torch.dot(wave_number, marginal)))
        absolute_weight += float(torch.dot(wave_number.abs(), marginal))
    high_fraction = None
    if high_shell:
        high = torch.abs(torch.fft.fftfreq(
            resolution, device=wave.device
        )) >= 0.375
        shell = (
            high[:, None, None] | high[None, :, None]
            | high[None, None, :]
        )
        high_fraction = float(power[shell].sum() / power.sum())
    return {
        "momentum_code": factor * np.asarray(components, dtype=np.float64),
        "summation_rounding_floor_code": (
            32.0 * np.finfo(float).eps * factor * absolute_weight
        ),
        "high_frequency_power_fraction": high_fraction,
    }


def spectral_wave_momentum_torch(
    wave: torch.Tensor, box_length: float,
) -> np.ndarray:
    """Return the skew-adjoint signed spectral momentum."""

    return spectral_wave_diagnostics_torch(wave, box_length)["momentum_code"]


def evolve_wave_only(
    wave: torch.Tensor, *, grid, time_step: float, steps: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Advance self-gravitating FDM without a compact source or SMBH force."""

    if steps < 0 or not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("wave-only evolution requires nonnegative steps")
    potential = periodic_poisson_torch(
        wave_density(wave), grid.poisson_inverse_wavenumber_squared
    )
    history = [spectral_wave_momentum_torch(wave, grid.cell_size * wave.shape[0])]
    for _ in range(steps):
        apply_potential_half_kick_in_place(wave, potential, time_step)
        spectrum = torch.fft.fftn(wave)
        del wave
        apply_kinetic_phase_in_place(spectrum, grid.kinetic_axis_phase)
        wave = torch.fft.ifftn(spectrum)
        del spectrum, potential
        potential = periodic_poisson_torch(
            wave_density(wave), grid.poisson_inverse_wavenumber_squared
        )
        apply_potential_half_kick_in_place(wave, potential, time_step)
        history.append(spectral_wave_momentum_torch(
            wave, grid.cell_size * wave.shape[0]
        ))
    return wave, np.asarray(history)


def _atomic_json(path: Path, payload: dict) -> None:
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.link(staged, path)
    staged.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=Path)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    seed = args.seed.expanduser().resolve()
    candidate = args.candidate_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace wave-only control: {output}")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("wave-only source commit is invalid")
    wave_path = seed / "Outputs/3Wfn/P3D_#000.npy"
    body_path = seed / "Outputs/NBody/NTM_#000.npy"
    wave_hash = _sha256(wave_path)
    body_hash = _sha256(body_path)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("n256 wave-only control requires an allocated CUDA GPU")
    torch.cuda.set_device(device)
    project = Path(__file__).resolve().parents[1]
    import fdm_smbh_delay.torch_wave as wave_module
    import fdm_smbh_delay.pyul as pyul_module
    if (
        Path(wave_module.__file__).resolve()
        != project / "src/fdm_smbh_delay/torch_wave.py"
        or Path(pyul_module.__file__).resolve()
        != project / "src/fdm_smbh_delay/pyul.py"
    ):
        raise ValueError("executed wave module differs from recorded source")
    source_hashes = {
        "scripts/probe_wave_only_momentum_control.py": _sha256(Path(__file__).resolve()),
        "src/fdm_smbh_delay/torch_wave.py": _sha256(Path(wave_module.__file__)),
        "src/fdm_smbh_delay/pyul.py": _sha256(Path(pyul_module.__file__)),
    }
    for relative, digest in source_hashes.items():
        committed = subprocess.run(
            ["git", "show", f"{args.source_commit}:{relative}"],
            cwd=project, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if hashlib.sha256(committed.stdout).hexdigest() != digest:
            raise ValueError(f"executed source differs from commit: {relative}")
    level_inputs = []
    for label, factor, stop in _LEVELS:
        metadata_path = candidate / label / "fdm_adapter_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        solver = json.loads(
            (candidate / label / "torch_run_summary.json").read_text()
        )
        if (
            metadata.get("resolution") != 256
            or metadata.get("case_id") != "qe_q030_e030_a020"
            or metadata.get("time_step_factor") != factor
            or metadata.get("wave_smbh_coupling")
            != "periodic_tsc_strang_momentum"
            or metadata.get("diagnostic_stop_after_save") != stop
            or metadata.get("save_number") != metadata.get("actual_wave_steps")
            or metadata.get("reference_initial_wave_sha256") != wave_hash
            or metadata.get("reference_initial_particle_sha256") != body_hash
            or Path(metadata.get("reference_initial_state", "")).resolve() != seed
            or solver.get("status") != "diagnostic_partial"
            or solver.get("actual_wave_steps") != stop
            or solver.get("saved_intervals") != stop
            or any(
                metadata.get("solver_source_sha256", {}).get(relative)
                != source_hashes[relative]
                or _sha256(candidate / label / "torch_solver_provenance/source"
                           / relative) != source_hashes[relative]
                for relative in (
                    "src/fdm_smbh_delay/torch_wave.py",
                    "src/fdm_smbh_delay/pyul.py",
                )
            )
        ):
            raise ValueError(f"{label}: coupled reference differs from null control")
        units = pyul_unit_system(metadata)
        box_code = float(metadata["box_size_pc"]) / units.length_pc
        time_step = float(metadata["wave_time_step_code"])
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("wave-only time step is invalid")
        level_inputs.append((label, factor, stop, metadata_path,
                             _sha256(metadata_path), units, box_code, time_step))
    output.mkdir(parents=True)
    for (label, factor, stop, metadata_path, metadata_hash,
         units, box_code, time_step) in level_inputs:
        torch.cuda.reset_peak_memory_stats(device)
        initial = np.load(wave_path)
        if _sha256(wave_path) != wave_hash or initial.shape != (256, 256, 256):
            raise ValueError("initial wave changed or has wrong resolution")
        wave = torch.as_tensor(initial, dtype=torch.complex128, device=device)
        del initial
        grid = spectral_grid(
            resolution=256, box_length=box_code, time_step=time_step,
            device=device,
        )
        initial_mass = float(
            torch.sum(wave_density(wave)) * grid.cell_volume
        )
        initial_kinetic = wave_kinetic_energy(
            wavefunction=wave,
            kinetic_axis_wavenumber_squared=grid.kinetic_axis_wavenumber_squared,
            cell_volume=grid.cell_volume,
        )
        candidate_mass = float(np.load(
            candidate / label / "Outputs/ULDMass.npy", mmap_mode="r"
        )[0])
        candidate_kinetic = float(np.load(
            candidate / label / "Outputs/ekandqlist.npy", mmap_mode="r"
        )[0])
        if (
            not np.isclose(initial_mass, candidate_mass, rtol=1e-11, atol=1e-9)
            or not np.isclose(
                initial_kinetic, candidate_kinetic, rtol=1e-11, atol=1e-9
            )
        ):
            raise ValueError("wave-only initial mass or kinetic ledger differs")
        initial_diagnostic = spectral_wave_diagnostics_torch(
            wave, box_code, high_shell=True
        )
        wave, momentum = evolve_wave_only(
            wave, grid=grid, time_step=time_step, steps=stop
        )
        final_mass = float(torch.sum(wave_density(wave)) * grid.cell_volume)
        final_diagnostic = spectral_wave_diagnostics_torch(
            wave, box_code, high_shell=True
        )
        if (
            not np.all(np.isfinite(momentum))
            or not np.isfinite(initial_mass) or not np.isfinite(final_mass)
            or initial_mass <= 0.0
            or abs(final_mass / initial_mass - 1.0) > 1e-10
        ):
            raise ValueError("wave-only control produced a non-finite state")
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        path = output / label
        path.mkdir()
        final_path = path / "final_wave.npy"
        with final_path.open("xb") as stream:
            np.save(stream, wave.detach().cpu().numpy())
        del wave, grid
        if (
            _sha256(wave_path) != wave_hash
            or _sha256(metadata_path) != metadata_hash
            or any(_sha256(project / relative) != digest
                   for relative, digest in source_hashes.items())
        ):
            raise ValueError("source, seed or candidate changed during control evolution")
        change = momentum - momentum[0]
        rounding_floor = (
            initial_diagnostic["summation_rounding_floor_code"]
            + final_diagnostic["summation_rounding_floor_code"]
        )
        payload = {
            "status": "wave_only_short_prefix_null_not_a_calibration_release",
            "calibration_eligible": False,
            "candidate_run": str(candidate / label),
            "candidate_metadata_sha256": metadata_hash,
            "source_commit": args.source_commit,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_sha256": source_hashes,
            "initial_wave_sha256": wave_hash,
            "initial_body_sha256": body_hash,
            "final_wave_sha256": _sha256(final_path),
            "time_step_factor": factor,
            "wave_time_step_code": time_step,
            "wave_steps": stop,
            "elapsed_myr": time_step * stop * units.time_myr,
            "wave_momentum_code": momentum.tolist(),
            "maximum_momentum_change_code": float(
                np.max(np.linalg.norm(change, axis=1))
            ),
            "final_momentum_change_code": change[-1].tolist(),
            "estimated_endpoint_momentum_rounding_floor_code": rounding_floor,
            "endpoint_momentum_change_resolved_above_rounding_floor": bool(
                np.linalg.norm(change[-1]) > rounding_floor
            ),
            "initial_high_frequency_power_fraction": initial_diagnostic[
                "high_frequency_power_fraction"
            ],
            "final_high_frequency_power_fraction": final_diagnostic[
                "high_frequency_power_fraction"
            ],
            "initial_wave_mass_code": initial_mass,
            "final_wave_mass_code": final_mass,
            "peak_device_memory_bytes": peak,
            "interpretation": (
                "No SMBH potential or force; this is a numerical null control, "
                "not the coupled wave's momentum budget"
            ),
        }
        _atomic_json(path / "control.json", payload)
        print(json.dumps({"level": label, "maximum_momentum_change_code":
                          payload["maximum_momentum_change_code"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
