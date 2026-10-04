#!/usr/bin/env python3
"""Compare the runtime wave force with the force conjugate to its energy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import torch

from fdm_smbh_delay import pyul as _loaded_pyul
from fdm_smbh_delay import torch_wave as _loaded_torch_wave
from fdm_smbh_delay.force_energy_audit import plummer_grid_conjugate_force
from fdm_smbh_delay.pyul import pyul_unit_system
from fdm_smbh_delay.torch_wave import (
    _patched_binary_derivative,
    _patched_wave_acceleration,
    advance_binary_rk4,
    advance_binary_rk4_patched,
    apply_kinetic_phase_in_place,
    apply_potential_half_kick_in_place,
    periodic_poisson_torch,
    plummer_potential_torch,
    potential_patches,
    spectral_grid,
    wave_density,
    wave_energy_components,
    wave_kinetic_energy,
)
if __package__:
    from .audit_wave_only_startup import _reference_interval
else:
    from audit_wave_only_startup import _reference_interval


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


def _binary_energy(state: np.ndarray, masses: np.ndarray, radius: float) -> float:
    """Kinetic plus softened mutual energy in the runtime's code units."""

    kinetic = 0.5 * float(np.sum(masses[:, None] * state[:, 3:] ** 2))
    displacement = state[0, :3] - state[1, :3]
    mutual = -float(np.prod(masses)) / np.sqrt(
        float(displacement @ displacement) + radius**2
    )
    return kinetic + mutual


def _interaction_at_fixed_density(
    *, density: torch.Tensor, grid, masses: np.ndarray,
    positions: np.ndarray, radius: float,
) -> tuple[float, torch.Tensor]:
    compact = plummer_potential_torch(
        coordinate=grid.coordinate, masses=masses, positions=positions,
        plummer_radius=radius,
    )
    interaction = grid.cell_volume * torch.sum(density * compact)
    return float(interaction), compact


def _rk4_wave_work(
    *, state: np.ndarray, masses: np.ndarray, patches: np.ndarray,
    starts: np.ndarray, box_length: float, resolution: int,
    radius: float, step: float, substeps: int,
) -> tuple[np.ndarray, float]:
    """Quadrature of the applied wave-force power at the runtime RK4 stages."""

    advanced = np.asarray(state, dtype=np.float64).reshape(2, 6).copy()
    substep = step / substeps

    def derivative(stage: np.ndarray) -> np.ndarray:
        return _patched_binary_derivative(
            stage, masses, patches, starts, box_length, resolution, radius
        )

    def wave_power(stage: np.ndarray) -> float:
        return float(sum(
            masses[body] * np.dot(
                stage[body, 3:],
                _patched_wave_acceleration(
                    stage[body, :3], patches[body], starts[body],
                    box_length, resolution,
                ),
            )
            for body in range(2)
        ))

    work = 0.0
    for _ in range(substeps):
        k1 = derivative(advanced)
        stage2 = advanced + 0.5 * substep * k1
        k2 = derivative(stage2)
        stage3 = advanced + 0.5 * substep * k2
        k3 = derivative(stage3)
        stage4 = advanced + substep * k3
        k4 = derivative(stage4)
        work += substep * (
            wave_power(advanced) + 2.0 * wave_power(stage2)
            + 2.0 * wave_power(stage3) + wave_power(stage4)
        ) / 6.0
        advanced += substep * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return advanced, work


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
    seed_hashes = {
        "wave": _sha256(seed / "Outputs/3Wfn/P3D_#000.npy"),
        "particle": _sha256(particle_path),
    }
    project = Path(__file__).resolve().parents[1]
    numerical_paths = {
        "scripts/run_torch_wave_case.py": project / "scripts/run_torch_wave_case.py",
        "src/fdm_smbh_delay/torch_wave.py": project / "src/fdm_smbh_delay/torch_wave.py",
        "src/fdm_smbh_delay/pyul.py": project / "src/fdm_smbh_delay/pyul.py",
    }
    if (
        Path(_loaded_torch_wave.__file__).resolve()
        != numerical_paths["src/fdm_smbh_delay/torch_wave.py"].resolve()
        or Path(_loaded_pyul.__file__).resolve()
        != numerical_paths["src/fdm_smbh_delay/pyul.py"].resolve()
    ):
        raise ValueError("loaded numerical operators are not from this project")
    numerical_hashes = {key: _sha256(path) for key, path in numerical_paths.items()}
    interval_myr, _, _, reference_record = _reference_interval(
        seed=seed, reference=trace, seed_hashes=seed_hashes,
        numerical_hashes=numerical_hashes,
    )
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
    if not np.isclose(step_code * units.time_myr, interval_myr, rtol=1e-12):
        raise ValueError("reference step is not the first saved physical interval")
    if trace_metadata.get("torch_version") != torch.__version__:
        raise ValueError("Torch version differs from the reference run")
    grid = spectral_grid(
        resolution=resolution,
        box_length=box_code,
        time_step=step_code,
        device=torch.device("cpu"),
    )
    wave_path = seed / "Outputs/3Wfn/P3D_#000.npy"
    wavefunction = torch.as_tensor(
        np.array(np.load(wave_path, allow_pickle=False), copy=True),
        dtype=torch.complex128,
    )
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
    mean_density = float(torch.mean(density))
    uniform_background_force = plummer_grid_conjugate_force(
        density=np.broadcast_to(mean_density, density.shape),
        coordinate=grid.coordinate.numpy(), masses=masses,
        positions=positions, plummer_radius=radius,
        cell_volume=grid.cell_volume,
    )
    perturbation_conjugate_force = conjugate_force - uniform_background_force
    mismatch = solver_force - conjugate_force
    predicted_change = float(np.sum(mismatch * velocities) * step_code)

    saved_state = np.load(
        trace / "Outputs/NBody/NTM_#001.npy", allow_pickle=False
    ).reshape(2, 6)
    reconstructed_state = advance_binary_rk4_patched(
        state=state, masses=masses, patches=patches, patch_starts=starts,
        box_length=box_code, resolution=resolution, plummer_radius=radius,
        time_step=step_code, substeps=9,
    ).reshape(2, 6)
    work_state, solver_wave_work = _rk4_wave_work(
        state=state, masses=masses, patches=patches, starts=starts,
        box_length=box_code, resolution=resolution, radius=radius,
        step=step_code, substeps=9,
    )
    if not np.allclose(work_state, reconstructed_state, rtol=1e-14, atol=1e-14):
        raise ValueError("wave-work quadrature does not reproduce runtime RK4")
    state_reconstruction_max_error = float(np.max(np.abs(reconstructed_state - saved_state)))
    no_wave_state = advance_binary_rk4(
        state=state, masses=masses,
        external_acceleration=np.zeros((2, 3)),
        plummer_radius=radius, time_step=step_code, substeps=9,
    ).reshape(2, 6)
    wave_state_signal = float(np.max(np.abs(saved_state - no_wave_state)))
    if not np.allclose(reconstructed_state, saved_state, rtol=1e-10, atol=1e-12):
        raise ValueError("reconstructed first-step SMBH state differs from GPU output")
    initial_interaction, old_compact = _interaction_at_fixed_density(
        density=density, grid=grid, masses=masses, positions=positions,
        radius=radius,
    )
    final_interaction, new_compact = _interaction_at_fixed_density(
        density=density, grid=grid, masses=masses, positions=saved_state[:, :3],
        radius=radius,
    )
    post_drift_kinetic = wave_kinetic_energy(
        wavefunction=wavefunction,
        kinetic_axis_wavenumber_squared=grid.kinetic_axis_wavenumber_squared,
        cell_volume=grid.cell_volume,
    )
    post_drift_self_gravity = float(
        0.5 * grid.cell_volume * torch.sum(density * potential)
    )
    uniform_interaction_change = float(
        mean_density * grid.cell_volume * torch.sum(new_compact - old_compact)
    )
    del old_compact
    orbital_change = (
        _binary_energy(saved_state, masses, radius)
        - _binary_energy(state, masses, radius)
    )
    interaction_change = final_interaction - initial_interaction
    apply_potential_half_kick_in_place(
        wavefunction, potential + new_compact, step_code
    )
    reconstructed_wave = wave_energy_components(
        wavefunction=wavefunction, density=density, wave_potential=potential,
        compact_potential=new_compact,
        kinetic_axis_wavenumber_squared=grid.kinetic_axis_wavenumber_squared,
        cell_volume=grid.cell_volume,
    )
    logged_wave = tuple(
        float(np.load(trace / "Outputs" / f"{name}.npy", allow_pickle=False)[1])
        for name in ("ekandqlist", "egpsilist", "egpcmlist", "masseslist")
    )
    initial_logged_wave = tuple(
        float(np.load(trace / "Outputs" / f"{name}.npy", allow_pickle=False)[0])
        for name in ("ekandqlist", "egpsilist", "egpcmlist")
    )
    wave_component_errors = [
        abs(reconstructed - logged)
        for reconstructed, logged in zip(reconstructed_wave, logged_wave, strict=True)
    ]
    if not np.allclose(reconstructed_wave, logged_wave, rtol=1e-10, atol=1e-7):
        raise ValueError("reconstructed first-step wave budget differs from GPU output")

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
    initial_orbital = _binary_energy(state, masses, radius)
    final_orbital = _binary_energy(saved_state, masses, radius)
    initial_total = sum(initial_logged_wave) + initial_orbital
    post_drift_old_total = (
        post_drift_kinetic + post_drift_self_gravity
        + initial_interaction + initial_orbital
    )
    post_drift_new_total = (
        post_drift_kinetic + post_drift_self_gravity
        + final_interaction + final_orbital
    )
    final_total = sum(logged_wave[:3]) + final_orbital
    first_wave_substep = post_drift_old_total - initial_total
    fixed_wave_particle_substep = post_drift_new_total - post_drift_old_total
    final_wave_substep = final_total - post_drift_new_total
    substep_sum = first_wave_substep + fixed_wave_particle_substep + final_wave_substep
    if not np.isclose(
        substep_sum, final_total - initial_total, rtol=0.0, atol=1e-3
    ) or not np.isclose(
        fixed_wave_particle_substep, orbital_change + interaction_change,
        rtol=0.0, atol=1e-3
    ):
        raise ValueError("first-step Hamiltonian substep ledger does not close")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "force_energy_diagnostic_not_a_calibration_release",
        "force_evaluation_phase": "after_first_wave_half_kick_and_spectral_drift_before_particle_rk4",
        "initial_wave_identity": (
            "seed wave and particle hashes and frozen numerical-source hashes "
            "agree with the reference run's launch provenance"
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
        "uniform_background_conjugate_force_code": uniform_background_force.tolist(),
        "density_perturbation_conjugate_force_code": (
            perturbation_conjugate_force.tolist()
        ),
        "force_mismatch_after_background_subtraction_code": (
            (solver_force - perturbation_conjugate_force).tolist()
        ),
        "force_mismatch_code": mismatch.tolist(),
        "relative_force_mismatch_by_body": (
            np.linalg.norm(mismatch, axis=1)
            / np.maximum(np.linalg.norm(conjugate_force, axis=1), np.finfo(float).tiny)
        ).tolist(),
        "first_step_work_from_force_mismatch_msun_pc2_myr2": (
            predicted_change * energy_conversion
        ),
        "first_step_measured_hamiltonian_change_msun_pc2_myr2": measured_change,
        "initial_velocity_linear_work_status": (
            "superseded by RK4 stage-force work; initial velocity alone misses "
            "displacement generated during the step"
        ),
        "first_wave_substep_hamiltonian_change_code": first_wave_substep,
        "fixed_wave_particle_substep_hamiltonian_change_code": (
            fixed_wave_particle_substep
        ),
        "final_wave_substep_hamiltonian_change_code": final_wave_substep,
        "three_substep_hamiltonian_change_code": substep_sum,
        "three_substep_hamiltonian_change_msun_pc2_myr2": (
            substep_sum * energy_conversion
        ),
        "three_substep_vs_csv_difference_msun_pc2_myr2": (
            substep_sum * energy_conversion - measured_change
        ),
        "fixed_density_interaction_change_code": interaction_change,
        "uniform_background_interaction_change_code": uniform_interaction_change,
        "density_perturbation_interaction_change_code": (
            interaction_change - uniform_interaction_change
        ),
        "binary_orbital_energy_change_code": orbital_change,
        "rk4_applied_wave_force_work_code": solver_wave_work,
        "rk4_orbital_energy_defect_code": orbital_change - solver_wave_work,
        "rk4_wave_force_vs_grid_interaction_work_code": (
            solver_wave_work + interaction_change
        ),
        "fixed_density_exchange_residual_code": orbital_change + interaction_change,
        "fixed_density_exchange_residual_msun_pc2_myr2": (
            (orbital_change + interaction_change) * energy_conversion
        ),
        "density_perturbation_exchange_residual_code": (
            orbital_change + interaction_change - uniform_interaction_change
        ),
        "density_perturbation_exchange_residual_msun_pc2_myr2": (
            (orbital_change + interaction_change - uniform_interaction_change)
            * energy_conversion
        ),
        "exchange_interpretation": (
            "The fixed-post-drift-wave particle substep is an exact term in "
            "the three-substep Hamiltonian ledger, with separate first and "
            "final wave substeps. The particle-substep residual isolates "
            "force/energy conjugacy up to the measured RK4 defect; it alone "
            "does not establish conservation of the whole step. "
            "The solver wave force uses a periodic zero-mean Poisson potential, "
            "whereas the grid Plummer interaction uses a non-periodic kernel "
            "and includes the uniform wave background. Kernel and force "
            "interpolation differences are also possible."
        ),
        "reconstructed_first_step_state_matches_gpu": True,
        "reconstructed_first_step_wave_budget_matches_gpu": True,
        "state_reconstruction_max_abs_error_code": state_reconstruction_max_error,
        "wave_force_state_signal_vs_zero_wave_code": wave_state_signal,
        "state_reconstruction_error_over_wave_signal": (
            state_reconstruction_max_error / wave_state_signal
            if wave_state_signal > 0.0 else None
        ),
        "wave_budget_reconstruction_abs_errors_code": dict(zip(
            ("kinetic", "self_gravity", "compact_interaction", "mass"),
            wave_component_errors, strict=True,
        )),
        "compact_interaction_reconstruction_error_over_fixed_density_change": (
            wave_component_errors[2] / abs(interaction_change)
            if interaction_change != 0.0 else None
        ),
        "reference_provenance": reference_record,
        "numerical_source_sha256": numerical_hashes,
        "seed_wave_sha256": _sha256(wave_path),
        "seed_particle_sha256": _sha256(particle_path),
        "trace_conservation_timeseries_sha256": _sha256(series_path),
        "trace_first_step_particle_sha256": _sha256(
            trace / "Outputs/NBody/NTM_#001.npy"
        ),
        "conjugate_force_source_sha256": _sha256(
            project / "src/fdm_smbh_delay/force_energy_audit.py"
        ),
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
    }
    if (
        seed_hashes != {
            "wave": _sha256(wave_path), "particle": _sha256(particle_path)
        }
        or numerical_hashes != {
            key: _sha256(path) for key, path in numerical_paths.items()
        }
        or payload["audit_script_sha256"] != _sha256(Path(__file__).resolve())
    ):
        raise ValueError("force-work inputs or numerical source changed during use")
    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, output)
    finally:
        staged.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
