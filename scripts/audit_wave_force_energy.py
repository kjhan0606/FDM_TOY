#!/usr/bin/env python3
"""Compare the runtime wave force with the force conjugate to its energy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import torch

from fdm_smbh_delay.force_energy_audit import plummer_grid_conjugate_force
from fdm_smbh_delay.pyul import pyul_unit_system
from fdm_smbh_delay.torch_wave import (
    _patched_wave_acceleration,
    apply_kinetic_phase_in_place,
    apply_potential_half_kick_in_place,
    periodic_poisson_torch,
    plummer_potential_torch,
    potential_patches,
    spectral_grid,
    wave_density,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_measured_energy_conversion(
    *, trace: Path, series: np.ndarray, masses: np.ndarray,
    plummer_radius: float, energy_conversion: float,
) -> None:
    """Rebuild the first two physical Hamiltonians from code-unit solver logs."""

    components = [
        np.load(trace / "Outputs" / f"{name}.npy")[:2]
        for name in ("ekandqlist", "egpsilist", "egpcmlist")
    ]
    states = [
        np.load(trace / f"Outputs/NBody/NTM_#{index:03d}.npy").reshape(2, 6)
        for index in range(2)
    ]
    expected = []
    for index, state in enumerate(states):
        kinetic = 0.5 * float(np.sum(masses[:, None] * state[:, 3:] ** 2))
        displacement = state[0, :3] - state[1, :3]
        mutual = -float(np.prod(masses)) / np.sqrt(
            float(displacement @ displacement) + plummer_radius**2
        )
        code_energy = sum(float(component[index]) for component in components)
        expected.append((code_energy + kinetic + mutual) * energy_conversion)
    if not np.allclose(
        np.asarray(series["combined_energy"][:2]), expected,
        rtol=1.0e-12, atol=0.0,
    ):
        raise ValueError("CSV Hamiltonian is not the physical-unit solver budget")


def main() -> int:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=Path)
    parser.add_argument("--trace-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed = args.seed.expanduser().resolve()
    trace = args.trace_run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace diagnostic output: {output}")

    seed_metadata = json.loads((seed / "fdm_adapter_metadata.json").read_text())
    trace_metadata = json.loads((trace / "fdm_adapter_metadata.json").read_text())
    trace_summary = json.loads((trace / "torch_run_summary.json").read_text())
    conservation = json.loads((trace / "conservation_summary.json").read_text())
    if (
        trace_metadata.get("reference_initial_state") != str(seed)
        or trace_metadata.get("diagnostic_stop_after_save") is None
        or trace_summary.get("status") != "diagnostic_partial"
        or conservation.get("status") != "diagnostic_partial"
        or trace_metadata.get("case_id") != seed_metadata.get("case_id")
        or trace_metadata.get("resolution") != seed_metadata.get("resolution")
        or trace_metadata.get("analytic_fdm_drag") is not False
    ):
        raise ValueError("force audit requires the matching partial live-wave trace")
    for unit_field in (
        "pyul_length_unit_m", "pyul_time_unit_s", "pyul_mass_unit_kg",
        "pyul_energy_unit_j",
    ):
        if (
            unit_field not in seed_metadata
            or unit_field not in trace_metadata
            or trace_metadata[unit_field] != seed_metadata[unit_field]
        ):
            raise ValueError(f"trace and seed unit systems differ: {unit_field}")
    derived_energy_j = (
        float(seed_metadata["pyul_mass_unit_kg"])
        * float(seed_metadata["pyul_length_unit_m"]) ** 2
        / float(seed_metadata["pyul_time_unit_s"]) ** 2
    )
    if not np.isclose(
        float(seed_metadata["pyul_energy_unit_j"]), derived_energy_j,
        rtol=1.0e-10, atol=0.0,
    ):
        raise ValueError("PyUL energy unit disagrees with mass-length-time units")

    config = json.loads((seed / "config.uldm").read_text())
    particle_config = config["Matter Particles"]
    if (
        particle_config.get("Mass Units") != "solar_masses"
        or particle_config.get("Position Units") != "pc"
        or particle_config.get("Velocity Units") != "km/s"
    ):
        raise ValueError("force audit requires the documented SMBH physical units")
    masses_msun = np.asarray(
        [entry[0] for entry in particle_config["Condition"]],
        dtype=float,
    )
    if masses_msun.shape != (2,):
        raise ValueError("force audit requires exactly two compact masses")
    units = pyul_unit_system(seed_metadata)
    masses = masses_msun / units.mass_msun
    particle_path = seed / "Outputs/NBody/NTM_#000.npy"
    trace_particle_path = trace / "Outputs/NBody/NTM_#000.npy"
    if _sha256(particle_path) != _sha256(trace_particle_path):
        raise ValueError("trace and seed initial particle states differ")
    state = np.load(particle_path).reshape(2, 6)
    positions = state[:, :3]
    velocities = state[:, 3:]

    resolution = int(seed_metadata["resolution"])
    if resolution > 256:
        raise ValueError("force audit above n256 requires a separate resource review")
    box_code = float(seed_metadata["box_size_pc"]) / units.length_pc
    radius = float(particle_config["Plummer Radius"]) / units.length_pc
    step_code = float(trace_metadata["wave_time_step_code"])
    grid = spectral_grid(
        resolution=resolution,
        box_length=box_code,
        time_step=step_code,
        device=torch.device("cpu"),
    )
    wave_path = seed / "Outputs/3Wfn/P3D_#000.npy"
    wavefunction = torch.as_tensor(np.load(wave_path), dtype=torch.complex128)
    if wavefunction.shape != (resolution, resolution, resolution):
        raise ValueError("initial wavefunction has an incompatible shape")
    initial_density = wave_density(wavefunction)
    initial_wave_potential = periodic_poisson_torch(
        initial_density, grid.poisson_inverse_wavenumber_squared
    )
    initial_compact_potential = plummer_potential_torch(
        coordinate=grid.coordinate,
        masses=masses,
        positions=positions,
        plummer_radius=radius,
    )
    first_total_potential = initial_wave_potential + initial_compact_potential
    apply_potential_half_kick_in_place(
        wavefunction, first_total_potential, step_code
    )
    del first_total_potential, initial_density, initial_wave_potential
    del initial_compact_potential
    wavefunction_k = torch.fft.fftn(wavefunction)
    del wavefunction
    apply_kinetic_phase_in_place(wavefunction_k, grid.kinetic_axis_phase)
    wavefunction = torch.fft.ifftn(wavefunction_k)
    del wavefunction_k
    density = wave_density(wavefunction)
    del wavefunction
    potential = periodic_poisson_torch(
        density, grid.poisson_inverse_wavenumber_squared
    )
    patches, starts = potential_patches(
        potential=potential,
        positions=positions,
        box_length=box_code,
    )
    solver_acceleration = np.vstack([
        _patched_wave_acceleration(
            positions[index], patches[index], starts[index], box_code, resolution
        )
        for index in range(2)
    ])
    solver_force = masses[:, None] * solver_acceleration
    conjugate_force = plummer_grid_conjugate_force(
        density=density.numpy(),
        coordinate=grid.coordinate.numpy(),
        masses=masses,
        positions=positions,
        plummer_radius=radius,
        cell_volume=grid.cell_volume,
    )
    mismatch = solver_force - conjugate_force
    predicted_change = float(np.sum(mismatch * velocities) * step_code)

    series_path = trace / "conservation_timeseries.csv"
    series = np.genfromtxt(series_path, delimiter=",", names=True)
    if series.size < 2 or not np.isclose(
        series["time_myr"][1], step_code * units.time_myr,
        rtol=1.0e-10, atol=0.0,
    ):
        raise ValueError("trace does not contain the first physical wave step")
    measured_change = float(series["combined_energy"][1] - series["combined_energy"][0])
    energy_conversion = units.energy_msun_pc2_myr2
    _verify_measured_energy_conversion(
        trace=trace,
        series=series,
        masses=masses,
        plummer_radius=radius,
        energy_conversion=energy_conversion,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "force_energy_diagnostic_not_a_calibration_release",
        "force_evaluation_phase": "after_first_wave_half_kick_and_spectral_drift_before_particle_rk4",
        "initial_wave_identity": (
            "seed path is recorded by trace metadata; original wave bytes were "
            "not independently hashed at trace launch"
        ),
        "linear_work_assumptions": (
            "initial particle velocities and one post-drift force pair held "
            "fixed through the first wave step; excludes evolving density, "
            "split-step and RK4 errors, and other budget terms"
        ),
        "seed": str(seed),
        "trace_run": str(trace),
        "resolution": resolution,
        "plummer_radius_code": radius,
        "solver_wave_force_code": solver_force.tolist(),
        "grid_energy_conjugate_force_code": conjugate_force.tolist(),
        "force_mismatch_code": mismatch.tolist(),
        "relative_force_mismatch_by_body": (
            np.linalg.norm(mismatch, axis=1)
            / np.maximum(np.linalg.norm(conjugate_force, axis=1), np.finfo(float).tiny)
        ).tolist(),
        "first_step_work_from_force_mismatch_msun_pc2_myr2": (
            predicted_change * energy_conversion
        ),
        "first_step_measured_hamiltonian_change_msun_pc2_myr2": measured_change,
        "first_step_unexplained_after_linear_mismatch_msun_pc2_myr2": (
            measured_change - predicted_change * energy_conversion
        ),
        "seed_wave_sha256": _sha256(wave_path),
        "seed_particle_sha256": _sha256(particle_path),
        "trace_conservation_timeseries_sha256": _sha256(series_path),
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
    }
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
