#!/usr/bin/env python3
"""Isolate fixed-SMBH wave-solver drift at the first q/e startup save time."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import torch

from fdm_smbh_delay import pyul as _loaded_pyul
from fdm_smbh_delay import torch_wave as _loaded_torch_wave
from fdm_smbh_delay.pyul import pyul_unit_system
from fdm_smbh_delay.torch_wave import (
    apply_kinetic_phase_in_place,
    apply_potential_half_kick_in_place,
    periodic_poisson_torch,
    plummer_potential_torch,
    spectral_grid,
    wave_density,
    wave_energy_components,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loaded_numerical_source_paths(project: Path) -> dict[str, Path]:
    expected = {
        "src/fdm_smbh_delay/torch_wave.py": project / "src/fdm_smbh_delay/torch_wave.py",
        "src/fdm_smbh_delay/pyul.py": project / "src/fdm_smbh_delay/pyul.py",
    }
    loaded = {
        "src/fdm_smbh_delay/torch_wave.py": Path(_loaded_torch_wave.__file__).resolve(),
        "src/fdm_smbh_delay/pyul.py": Path(_loaded_pyul.__file__).resolve(),
    }
    if any(loaded[key] != path.resolve() for key, path in expected.items()):
        raise ValueError("loaded numerical operators are not from this project")
    return expected


def _wave_budget(
    wavefunction: torch.Tensor, grid, compact: torch.Tensor,
) -> tuple[float, float, float, float]:
    density = wave_density(wavefunction)
    wave_potential = periodic_poisson_torch(
        density, grid.poisson_inverse_wavenumber_squared
    )
    return wave_energy_components(
        wavefunction=wavefunction, density=density,
        wave_potential=wave_potential, compact_potential=compact,
        kinetic_axis_wavenumber_squared=grid.kinetic_axis_wavenumber_squared,
        cell_volume=grid.cell_volume,
    )


def _evolve_fixed_sources(
    initial_wave: torch.Tensor,
    *,
    box_length_code: float,
    masses_code: np.ndarray,
    positions_code: np.ndarray,
    plummer_radius_code: float,
    physical_interval_code: float,
    steps: int,
) -> dict:
    if type(steps) is not int or steps < 1:
        raise ValueError("positive integer wave step count is required")
    if initial_wave.ndim != 3 or len(set(initial_wave.shape)) != 1:
        raise ValueError("initial wave must be cubic")
    grid = spectral_grid(
        resolution=initial_wave.shape[0], box_length=box_length_code,
        time_step=physical_interval_code / steps, device=torch.device("cpu"),
    )
    compact = plummer_potential_torch(
        coordinate=grid.coordinate, masses=masses_code,
        positions=positions_code, plummer_radius=plummer_radius_code,
    )
    wave = initial_wave.clone()
    initial = _wave_budget(wave, grid, compact)
    density = wave_density(wave)
    potential = periodic_poisson_torch(
        density, grid.poisson_inverse_wavenumber_squared
    )
    for _ in range(steps):
        apply_potential_half_kick_in_place(
            wave, potential + compact, physical_interval_code / steps
        )
        wave_k = torch.fft.fftn(wave)
        apply_kinetic_phase_in_place(wave_k, grid.kinetic_axis_phase)
        wave = torch.fft.ifftn(wave_k)
        del wave_k, density, potential
        density = wave_density(wave)
        potential = periodic_poisson_torch(
            density, grid.poisson_inverse_wavenumber_squared
        )
        apply_potential_half_kick_in_place(
            wave, potential + compact, physical_interval_code / steps
        )
    final = _wave_budget(wave, grid, compact)
    components = ("kinetic", "self_gravity", "fixed_smbh_interaction")
    initial_h = sum(initial[:3])
    final_h = sum(final[:3])
    transfers = [abs(after - before) for before, after in zip(initial[:3], final[:3])]
    transfer_scale = max(transfers)
    return {
        "wave_steps": steps,
        "wave_time_step_code": physical_interval_code / steps,
        "initial_components_code": dict(zip(components, initial[:3], strict=True)),
        "final_components_code": dict(zip(components, final[:3], strict=True)),
        "initial_hamiltonian_code": initial_h,
        "final_hamiltonian_code": final_h,
        "hamiltonian_change_code": final_h - initial_h,
        "energy_error_over_largest_component_exchange": (
            abs(final_h - initial_h) / transfer_scale if transfer_scale > 0.0 else 0.0
        ),
        "wave_mass_relative_change": final[3] / initial[3] - 1.0,
    }


def _reference_interval(
    *, seed: Path, reference: Path, seed_hashes: dict[str, str],
    numerical_hashes: dict[str, str],
) -> tuple[float, float, tuple[int, int, int], dict]:
    metadata_path = reference / "fdm_adapter_metadata.json"
    config_path = reference / "config.uldm"
    metadata = json.loads(metadata_path.read_text())
    config = json.loads(config_path.read_text())
    seed_metadata = json.loads((seed / "fdm_adapter_metadata.json").read_text())
    seed_config = json.loads((seed / "config.uldm").read_text())
    summary = json.loads((reference / "torch_run_summary.json").read_text())
    manifest = json.loads((reference / "torch_solver_provenance/manifest.json").read_text())
    if (
        metadata.get("reference_initial_state") != str(seed)
        or metadata.get("reference_initial_wave_sha256") != seed_hashes["wave"]
        or metadata.get("reference_initial_particle_sha256") != seed_hashes["particle"]
        or metadata.get("case_id") != "qe_q030_e030_a020"
        or metadata.get("resolution") != 256
        or metadata.get("backend") != "pytorch_cuda"
        or metadata.get("analytic_fdm_drag") is not False
        or metadata.get("time_step_factor") != 1.0
        or metadata.get("save_number") != 58500
        or metadata.get("diagnostic_stop_after_save") != 10
        or metadata.get("nbody_rk4_substeps_per_wave_step") != 9
        or summary.get("status") != "diagnostic_partial"
        or summary.get("saved_intervals", 0) < 1
        or manifest.get("status") != "source_snapshot"
        or manifest.get("run") != str(reference)
        or manifest.get("input_records", {}).get("fdm_adapter_metadata_sha256")
        != _sha256(metadata_path)
        or manifest.get("input_records", {}).get("config_sha256")
        != _sha256(config_path)
        or config.get("Matter Particles") != seed_config.get("Matter Particles")
        or any(metadata.get(key) != seed_metadata.get(key) for key in (
            "box_size_pc", "particle_mass_ev", "pyul_length_unit_m",
            "pyul_time_unit_s", "pyul_mass_unit_kg", "pyul_energy_unit_j",
        ))
    ):
        raise ValueError("reference Torch run lacks matching launch and provenance")
    records = manifest.get("source_files")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("reference numerical source snapshot is invalid")
    recorded = {row.get("path"): row.get("sha256") for row in records}
    for relative, digest in numerical_hashes.items():
        frozen = reference / "torch_solver_provenance/source" / relative
        if recorded.get(relative) != digest or _sha256(frozen) != digest:
            raise ValueError("current numerical source differs from reference run")
    try:
        saves = int(metadata["save_number"])
        steps = int(metadata["actual_wave_steps"])
        duration_myr = float(metadata["duration_myr"])
        wave_step_code = float(metadata["wave_time_step_code"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("reference time-step metadata is invalid") from error
    if (
        saves != 58500 or steps != 58500 or steps % saves != 0
        or not np.isfinite(duration_myr)
        or not np.isclose(duration_myr, 0.1, rtol=0.0, atol=1.0e-12)
        or not np.isfinite(wave_step_code) or wave_step_code <= 0.0
        or summary.get("planned_wave_steps") != steps
    ):
        raise ValueError("reference time-step metadata differs from startup design")
    interval_myr = duration_myr / saves
    return interval_myr, wave_step_code, (1, 2, 4), {
        "reference_run": str(reference),
        "reference_metadata_sha256": _sha256(metadata_path),
        "reference_manifest_sha256": _sha256(
            reference / "torch_solver_provenance/manifest.json"
        ),
        "reference_wave_step_code": wave_step_code,
        "reference_saved_interval_steps": steps // saves,
        "reference_torch_version": metadata.get("torch_version"),
    }


def audit(seed: Path, reference: Path) -> dict:
    metadata_path = seed / "fdm_adapter_metadata.json"
    config_path = seed / "config.uldm"
    wave_path = seed / "Outputs/3Wfn/P3D_#000.npy"
    particle_path = seed / "Outputs/NBody/NTM_#000.npy"
    project = Path(__file__).resolve().parents[1]
    source_paths = {
        "scripts/run_torch_wave_case.py": project / "scripts/run_torch_wave_case.py",
        **_loaded_numerical_source_paths(project),
    }
    input_paths = {
        "metadata": metadata_path, "config": config_path,
        "wave": wave_path, "particle": particle_path,
    }
    input_hashes = {key: _sha256(path) for key, path in input_paths.items()}
    numerical_hashes = {key: _sha256(path) for key, path in source_paths.items()}
    script_hash = _sha256(Path(__file__).resolve())
    metadata = json.loads(metadata_path.read_text())
    config = json.loads(config_path.read_text())
    if (
        metadata.get("case_id") != "qe_q030_e030_a020"
        or metadata.get("resolution") != 256
        or metadata.get("analytic_fdm_drag") is not False
    ):
        raise ValueError("wave-only audit requires the registered n256 q/e seed")
    particles = config["Matter Particles"]
    if (particles.get("Mass Units") != "solar_masses"
            or particles.get("Position Units") != "pc"
            or len(particles.get("Condition", [])) != 2):
        raise ValueError("seed compact-source units or multiplicity are invalid")
    units = pyul_unit_system(metadata)
    interval_myr, interval_code, step_ladder, reference_record = _reference_interval(
        seed=seed, reference=reference, seed_hashes=input_hashes,
        numerical_hashes=numerical_hashes,
    )
    if not np.isclose(interval_code * units.time_myr, interval_myr,
                      rtol=1.0e-12, atol=0.0):
        raise ValueError("reference wave step disagrees with the seed unit system")
    if (reference_record["reference_torch_version"] != torch.__version__):
        raise ValueError("current Torch version differs from reference run")
    masses = np.asarray([row[0] for row in particles["Condition"]], dtype=float)
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("seed SMBH masses are invalid")
    state = np.load(particle_path, allow_pickle=False).reshape(2, 6)
    if np.any(~np.isfinite(state)):
        raise ValueError("seed SMBH state is invalid")
    wave = torch.as_tensor(
        np.array(np.load(wave_path, allow_pickle=False), copy=True),
        dtype=torch.complex128,
    )
    if wave.shape != (256, 256, 256):
        raise ValueError("seed wave shape differs from n256")
    box_code = float(metadata["box_size_pc"]) / units.length_pc
    plummer_code = float(particles["Plummer Radius"]) / units.length_pc
    runs = [
        _evolve_fixed_sources(
            wave, box_length_code=box_code, masses_code=masses / units.mass_msun,
            positions_code=state[:, :3], plummer_radius_code=plummer_code,
            physical_interval_code=interval_code, steps=steps,
        )
        for steps in step_ladder
    ]
    if (input_hashes != {key: _sha256(path) for key, path in input_paths.items()}
            or numerical_hashes != {key: _sha256(path) for key, path in source_paths.items()}
            or script_hash != _sha256(Path(__file__).resolve())):
        raise ValueError("wave-only audit inputs or numerical source changed during use")
    return {
        "status": "fixed_smbh_wave_only_startup_diagnostic_not_a_calibration_release",
        "interpretation": (
            "SMBH positions and velocities are held fixed. This isolates wave "
            "operator drift but does not test coupled wave-SMBH conservation."
        ),
        "seed": str(seed),
        "audit_device": "cpu_float64_complex128",
        "first_common_time_myr": interval_myr,
        **reference_record,
        "seed_metadata_sha256": input_hashes["metadata"],
        "seed_config_sha256": input_hashes["config"],
        "seed_wave_sha256": input_hashes["wave"],
        "seed_particle_sha256": input_hashes["particle"],
        "numerical_source_sha256": numerical_hashes,
        "audit_script_sha256": script_hash,
        "runs": runs,
    }


def main() -> int:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=Path)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed = args.seed.expanduser().resolve()
    reference = args.reference_run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace wave-only audit: {output}")
    payload = audit(seed, reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, output)
    finally:
        staged.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
