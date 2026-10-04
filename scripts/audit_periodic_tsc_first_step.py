#!/usr/bin/env python3
"""Source-bound three-substep ledger for the first reciprocal TSC wave step."""

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

from fdm_smbh_delay import periodic_mesh_coupling as _loaded_tsc
from fdm_smbh_delay import pyul as _loaded_pyul
from fdm_smbh_delay import torch_wave as _loaded_torch_wave
from fdm_smbh_delay.periodic_mesh_coupling import (
    advance_binary_rk4_tsc,
    periodic_tsc_patches,
    tsc_interaction_and_force,
    tsc_source_density,
)
from fdm_smbh_delay.pyul import pyul_unit_system
from fdm_smbh_delay.torch_wave import (
    apply_kinetic_phase_in_place,
    apply_potential_half_kick_in_place,
    periodic_poisson_torch,
    spectral_grid,
    wave_density,
    wave_energy_components,
    wave_kinetic_energy,
)

if __package__:
    from .audit_wave_force_energy import _binary_energy
    from .audit_wave_only_startup import _reference_interval
else:
    from audit_wave_force_energy import _binary_energy
    from audit_wave_only_startup import _reference_interval


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logged_wave(run: Path, index: int) -> tuple[float, float, float, float]:
    return tuple(
        float(np.load(run / "Outputs" / f"{name}.npy", allow_pickle=False)[index])
        for name in ("ekandqlist", "egpsilist", "egpcmlist", "masseslist")
    )


def _grid_interaction(density: torch.Tensor, potential: torch.Tensor, volume: float) -> float:
    return float(volume * torch.sum(density * potential))


def audit(seed: Path, run: Path) -> dict:
    project = Path(__file__).resolve().parents[1]
    source_paths = {
        "scripts/run_torch_wave_case.py": project / "scripts/run_torch_wave_case.py",
        "src/fdm_smbh_delay/torch_wave.py": project / "src/fdm_smbh_delay/torch_wave.py",
        "src/fdm_smbh_delay/pyul.py": project / "src/fdm_smbh_delay/pyul.py",
        "src/fdm_smbh_delay/periodic_mesh_coupling.py": (
            project / "src/fdm_smbh_delay/periodic_mesh_coupling.py"
        ),
    }
    loaded = {
        "src/fdm_smbh_delay/torch_wave.py": Path(_loaded_torch_wave.__file__).resolve(),
        "src/fdm_smbh_delay/pyul.py": Path(_loaded_pyul.__file__).resolve(),
        "src/fdm_smbh_delay/periodic_mesh_coupling.py": Path(_loaded_tsc.__file__).resolve(),
    }
    if any(loaded[key] != source_paths[key].resolve() for key in loaded):
        raise ValueError("loaded numerical operators are not from the source-bound project")
    input_paths = {
        "metadata": seed / "fdm_adapter_metadata.json",
        "config": seed / "config.uldm",
        "wave": seed / "Outputs/3Wfn/P3D_#000.npy",
        "particle": seed / "Outputs/NBody/NTM_#000.npy",
    }
    hashes = {key: _sha256(path) for key, path in input_paths.items()}
    numerical_hashes = {key: _sha256(path) for key, path in source_paths.items()}
    interval_myr, step_code, _, reference_record = _reference_interval(
        seed=seed, reference=run,
        seed_hashes={"wave": hashes["wave"], "particle": hashes["particle"]},
        numerical_hashes=numerical_hashes,
    )
    metadata = json.loads((run / "fdm_adapter_metadata.json").read_text())
    summary = json.loads((run / "torch_run_summary.json").read_text())
    conservation = json.loads((run / "conservation_summary.json").read_text())
    if (
        metadata.get("wave_smbh_coupling") != "periodic_tsc_reciprocal"
        or metadata.get("experimental_coupling_not_a_calibration_release") is not True
        or metadata.get("torch_version") != torch.__version__
        or summary.get("status") != "diagnostic_partial"
        or conservation.get("status") != "diagnostic_partial"
        or conservation.get("samples") != 11
        or _sha256(run / "Outputs/NBody/NTM_#000.npy") != hashes["particle"]
    ):
        raise ValueError("first-step audit requires the matching reciprocal TSC diagnostic")
    seed_metadata = json.loads(input_paths["metadata"].read_text())
    config = json.loads(input_paths["config"].read_text())
    units = pyul_unit_system(seed_metadata)
    if not np.isclose(step_code * units.time_myr, interval_myr, rtol=1e-12):
        raise ValueError("first wave step does not match the saved physical interval")
    particles = config["Matter Particles"]
    if (
        particles.get("Mass Units") != "solar_masses"
        or particles.get("Position Units") != "pc"
        or particles.get("Velocity Units") != "km/s"
    ):
        raise ValueError("source particle units are not the registered physical units")
    masses = np.asarray([row[0] for row in particles["Condition"]], dtype=float)
    if masses.shape != (2,) or np.any(~np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("first-step audit requires two finite SMBH masses")
    masses /= units.mass_msun
    state0 = np.load(input_paths["particle"], allow_pickle=False).reshape(2, 6)
    wave0 = torch.as_tensor(
        np.array(np.load(input_paths["wave"], allow_pickle=False), copy=True),
        dtype=torch.complex128,
    )
    if wave0.shape != (256, 256, 256) or not np.all(np.isfinite(state0)):
        raise ValueError("registered n256 initial state is invalid")
    box_code = float(seed_metadata["box_size_pc"]) / units.length_pc
    radius = float(particles["Plummer Radius"]) / units.length_pc
    grid = spectral_grid(
        resolution=256, box_length=box_code, time_step=step_code,
        device=torch.device("cpu"),
    )

    density0 = wave_density(wave0)
    wave_potential0 = periodic_poisson_torch(
        density0, grid.poisson_inverse_wavenumber_squared
    )
    source0 = tsc_source_density(
        masses=masses, positions=state0[:, :3], resolution=256,
        box_length=box_code, device=torch.device("cpu"),
    )
    compact0 = periodic_poisson_torch(
        source0, grid.poisson_inverse_wavenumber_squared
    )
    del source0
    initial_wave = wave_energy_components(
        wavefunction=wave0, density=density0,
        wave_potential=wave_potential0, compact_potential=compact0,
        kinetic_axis_wavenumber_squared=grid.kinetic_axis_wavenumber_squared,
        cell_volume=grid.cell_volume,
    )
    logged_initial = _logged_wave(run, 0)
    if not np.allclose(initial_wave, logged_initial, rtol=1e-10, atol=1e-7):
        raise ValueError("initial reconstructed wave energy differs from the GPU log")
    apply_potential_half_kick_in_place(
        wave0, wave_potential0 + compact0, step_code
    )
    del density0, wave_potential0
    wave_k = torch.fft.fftn(wave0)
    del wave0
    apply_kinetic_phase_in_place(wave_k, grid.kinetic_axis_phase)
    post_wave = torch.fft.ifftn(wave_k)
    del wave_k
    post_density = wave_density(post_wave)
    post_potential = periodic_poisson_torch(
        post_density, grid.poisson_inverse_wavenumber_squared
    )
    post_kinetic = wave_kinetic_energy(
        wavefunction=post_wave,
        kinetic_axis_wavenumber_squared=grid.kinetic_axis_wavenumber_squared,
        cell_volume=grid.cell_volume,
    )
    post_self = float(0.5 * grid.cell_volume * torch.sum(post_density * post_potential))
    old_interaction = _grid_interaction(post_density, compact0, grid.cell_volume)
    old_point, _ = tsc_interaction_and_force(
        wave_potential=post_potential, masses=masses,
        positions=state0[:, :3], box_length=box_code,
    )
    patches, starts = periodic_tsc_patches(
        potential=post_potential, positions=state0[:, :3], box_length=box_code,
    )
    state1 = advance_binary_rk4_tsc(
        state=state0, masses=masses, patches=patches, patch_starts=starts,
        box_length=box_code, resolution=256, plummer_radius=radius,
        time_step=step_code, substeps=9,
    ).reshape(2, 6)
    saved_state1 = np.load(
        run / "Outputs/NBody/NTM_#001.npy", allow_pickle=False
    ).reshape(2, 6)
    state_error = float(np.max(np.abs(state1 - saved_state1)))
    if not np.allclose(state1, saved_state1, rtol=1e-10, atol=1e-12):
        raise ValueError("reconstructed first-step SMBH state differs from GPU output")
    source1 = tsc_source_density(
        masses=masses, positions=saved_state1[:, :3], resolution=256,
        box_length=box_code, device=torch.device("cpu"),
    )
    compact1 = periodic_poisson_torch(
        source1, grid.poisson_inverse_wavenumber_squared
    )
    del source1, compact0
    new_interaction = _grid_interaction(post_density, compact1, grid.cell_volume)
    new_point, _ = tsc_interaction_and_force(
        wave_potential=post_potential, masses=masses,
        positions=saved_state1[:, :3], box_length=box_code,
    )
    reciprocity_error = max(
        abs(old_interaction - old_point), abs(new_interaction - new_point)
    )
    reciprocity_scale = max(abs(old_interaction), abs(new_interaction), 1.0)
    if reciprocity_error > 1e-11 * reciprocity_scale:
        raise ValueError("reconstructed TSC cross energy is not reciprocal")
    apply_potential_half_kick_in_place(
        post_wave, post_potential + compact1, step_code
    )
    final_wave = wave_energy_components(
        wavefunction=post_wave, density=post_density,
        wave_potential=post_potential, compact_potential=compact1,
        kinetic_axis_wavenumber_squared=grid.kinetic_axis_wavenumber_squared,
        cell_volume=grid.cell_volume,
    )
    logged_final = _logged_wave(run, 1)
    wave_error = max(abs(a - b) for a, b in zip(final_wave, logged_final, strict=True))
    if not np.allclose(final_wave, logged_final, rtol=1e-10, atol=1e-7):
        raise ValueError("reconstructed first-step wave energy differs from GPU output")

    orbital0 = _binary_energy(state0, masses, radius)
    orbital1 = _binary_energy(saved_state1, masses, radius)
    h0 = sum(logged_initial[:3]) + orbital0
    h_after_wave = post_kinetic + post_self + old_interaction + orbital0
    h_after_bodies = post_kinetic + post_self + new_interaction + orbital1
    h1 = sum(logged_final[:3]) + orbital1
    substeps = (
        h_after_wave - h0,
        h_after_bodies - h_after_wave,
        h1 - h_after_bodies,
    )
    orbital_plus_interaction = (
        orbital1 - orbital0 + new_interaction - old_interaction
    )
    if abs(substeps[1] - orbital_plus_interaction) > 1e-3:
        raise ValueError("fixed-wave particle-substep exchange identity failed")
    series_path = run / "conservation_timeseries.csv"
    series = np.genfromtxt(series_path, delimiter=",", names=True)
    measured_change = float(series["combined_energy"][1] - series["combined_energy"][0])
    conversion = units.energy_msun_pc2_myr2
    if abs((h1 - h0) * conversion - measured_change) > 1.0:
        raise ValueError("reconstructed first-step Hamiltonian differs from analysis")
    if (
        hashes != {key: _sha256(path) for key, path in input_paths.items()}
        or numerical_hashes != {key: _sha256(path) for key, path in source_paths.items()}
    ):
        raise ValueError("TSC audit inputs or numerical source changed during use")
    return {
        "status": "periodic_tsc_first_step_diagnostic_not_a_calibration_release",
        "seed": str(seed), "run": str(run),
        "reference_provenance": reference_record,
        "seed_sha256": hashes,
        "numerical_source_sha256": numerical_hashes,
        "run_conservation_timeseries_sha256": _sha256(series_path),
        "run_first_step_particle_sha256": _sha256(
            run / "Outputs/NBody/NTM_#001.npy"
        ),
        "first_wave_substep_code": substeps[0],
        "fixed_wave_particle_substep_code": substeps[1],
        "final_wave_substep_code": substeps[2],
        "three_substep_change_code": sum(substeps),
        "three_substep_change_msun_pc2_myr2": sum(substeps) * conversion,
        "measured_csv_change_msun_pc2_myr2": measured_change,
        "fixed_wave_particle_substep_msun_pc2_myr2": substeps[1] * conversion,
        "orbital_energy_change_code": orbital1 - orbital0,
        "fixed_density_interaction_change_code": new_interaction - old_interaction,
        "post_drift_reciprocity_max_abs_error_code": reciprocity_error,
        "state_reconstruction_max_abs_error_code": state_error,
        "wave_energy_reconstruction_max_abs_error_code": wave_error,
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
    }


def main() -> int:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    seed = args.seed.expanduser().resolve()
    run = args.run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace TSC first-step audit: {output}")
    payload = audit(seed, run)
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
