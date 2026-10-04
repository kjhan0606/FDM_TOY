#!/usr/bin/env python3
"""Attribute short coupled Strang momentum exchange at both half kicks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Callable

import numpy as np
import torch

from fdm_smbh_delay.periodic_mesh_coupling import (
    drift_binary_tsc,
    kick_binary_tsc,
    tsc_source_density,
    validate_binary_tsc_timestep,
)
from fdm_smbh_delay.pyul import pyul_unit_system
from fdm_smbh_delay.torch_wave import (
    apply_kinetic_phase_in_place,
    apply_potential_half_kick_in_place,
    periodic_poisson_torch,
    spectral_grid,
    wave_density,
)
from scripts.probe_wave_only_momentum_control import (
    spectral_wave_diagnostics_torch, spectral_wave_momentum_torch,
)


_LEVELS = {
    "f100": (1.0, 10), "f050": (0.5, 20),
    "f025": (0.25, 40), "f0125": (0.125, 80),
}
_SOURCE_PATHS = (
    "scripts/probe_coupled_kick_attribution.py",
    "scripts/probe_wave_only_momentum_control.py",
    "src/fdm_smbh_delay/periodic_mesh_coupling.py",
    "src/fdm_smbh_delay/torch_wave.py",
    "src/fdm_smbh_delay/pyul.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def body_momentum(body: np.ndarray, masses: np.ndarray) -> np.ndarray:
    state = np.asarray(body, dtype=float)
    mass = np.asarray(masses, dtype=float)
    if state.shape != (2, 6) or mass.shape != (2,):
        raise ValueError("binary momentum needs two six-component bodies")
    return np.sum(mass[:, None] * state[:, 3:], axis=0)


def spectral_wave_momentum_reordered_torch(
    wave: torch.Tensor, box_length: float,
) -> np.ndarray:
    """Same spectral momentum, with weighted 3-D reduction before marginalization.

    This changes floating-point summation order only; it is not a higher
    precision estimator or a different physical observable.
    """

    if (
        wave.ndim != 3 or len(set(wave.shape)) != 1
        or wave.dtype != torch.complex128
        or not np.isfinite(box_length) or box_length <= 0.0
    ):
        raise ValueError("reordered momentum requires a cubic complex128 wave")
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
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = resolution
        components.append(float(torch.sum(
            power * wave_number.reshape(shape)
        )))
    return factor * np.asarray(components, dtype=np.float64)


def compact_field(
    body: np.ndarray, masses: np.ndarray, *, grid,
    box_length: float, device: torch.device,
) -> torch.Tensor:
    source = tsc_source_density(
        masses=masses, positions=body[:, :3],
        resolution=body_resolution(grid), box_length=box_length,
        device=device,
    )
    return periodic_poisson_torch(
        source, grid.poisson_inverse_wavenumber_squared
    )


def body_resolution(grid) -> int:
    return int(grid.kinetic_axis_phase.numel())


def attributed_step(
    wave: torch.Tensor, body: np.ndarray, masses: np.ndarray,
    wave_potential: torch.Tensor, compact_potential: torch.Tensor, *,
    grid, box_length: float, plummer_radius: float, time_step: float,
    momentum_estimator: Callable[[torch.Tensor, float], np.ndarray]
    = spectral_wave_momentum_torch,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor, dict]:
    """Replay one symmetric step with two commuting phase factors measured."""

    resolution = wave.shape[0]
    validate_binary_tsc_timestep(
        state=body, masses=masses, box_length=box_length,
        resolution=resolution, plummer_radius=plummer_radius,
        time_step=time_step,
    )
    p_wave_start = momentum_estimator(wave, box_length)
    p_body_start = body_momentum(body, masses)
    first_phase = float(torch.max(torch.abs(
        0.5 * time_step * (wave_potential + compact_potential)
    )))
    alternate_wave = wave.clone()

    apply_potential_half_kick_in_place(wave, wave_potential, time_step)
    p_after_self_first = momentum_estimator(wave, box_length)
    apply_potential_half_kick_in_place(wave, compact_potential, time_step)
    p_after_compact_first = momentum_estimator(wave, box_length)
    apply_potential_half_kick_in_place(
        alternate_wave, compact_potential, time_step
    )
    p_after_alternate_compact_first = momentum_estimator(
        alternate_wave, box_length
    )
    apply_potential_half_kick_in_place(
        alternate_wave, wave_potential, time_step
    )
    p_after_alternate_first = momentum_estimator(
        alternate_wave, box_length
    )
    first_order_wave_difference = float(torch.max(torch.abs(
        alternate_wave - wave
    )))
    del alternate_wave
    kicked_body = kick_binary_tsc(
        state=body, masses=masses, wave_potential=wave_potential,
        box_length=box_length, plummer_radius=plummer_radius,
        time_step=0.5 * time_step, force_scheme="spectral_momentum",
    )
    p_after_body_first = body_momentum(kicked_body, masses)
    validate_binary_tsc_timestep(
        state=kicked_body, masses=masses, box_length=box_length,
        resolution=resolution, plummer_radius=plummer_radius,
        time_step=time_step,
    )
    first_joint_total = p_after_compact_first + p_after_body_first

    spectrum = torch.fft.fftn(wave)
    del wave
    apply_kinetic_phase_in_place(spectrum, grid.kinetic_axis_phase)
    wave = torch.fft.ifftn(spectrum)
    del spectrum
    p_after_drift = momentum_estimator(wave, box_length)
    drifted_body = drift_binary_tsc(
        state=kicked_body, box_length=box_length, time_step=time_step
    )
    density = wave_density(wave)
    next_wave_potential = periodic_poisson_torch(
        density, grid.poisson_inverse_wavenumber_squared
    )
    del density
    next_compact_potential = compact_field(
        drifted_body, masses, grid=grid, box_length=box_length,
        device=wave.device,
    )
    drift_total = p_after_drift + p_after_body_first

    second_phase = float(torch.max(torch.abs(
        0.5 * time_step * (next_wave_potential + next_compact_potential)
    )))
    alternate_wave = wave.clone()
    apply_potential_half_kick_in_place(wave, next_wave_potential, time_step)
    p_after_self_second = momentum_estimator(wave, box_length)
    apply_potential_half_kick_in_place(wave, next_compact_potential, time_step)
    p_after_compact_second = momentum_estimator(wave, box_length)
    apply_potential_half_kick_in_place(
        alternate_wave, next_compact_potential, time_step
    )
    p_after_alternate_compact_second = momentum_estimator(
        alternate_wave, box_length
    )
    apply_potential_half_kick_in_place(
        alternate_wave, next_wave_potential, time_step
    )
    p_after_alternate_second = momentum_estimator(
        alternate_wave, box_length
    )
    second_order_wave_difference = float(torch.max(torch.abs(
        alternate_wave - wave
    )))
    del alternate_wave
    final_body = kick_binary_tsc(
        state=drifted_body, masses=masses, wave_potential=next_wave_potential,
        box_length=box_length, plummer_radius=plummer_radius,
        time_step=0.5 * time_step, force_scheme="spectral_momentum",
    )
    p_body_final = body_momentum(final_body, masses)
    final_total = p_after_compact_second + p_body_final
    first_defect = first_joint_total - (p_wave_start + p_body_start)
    second_defect = final_total - drift_total
    components = {
        "wave_momentum_start": p_wave_start,
        "body_momentum_start": p_body_start,
        "wave_momentum_end": p_after_compact_second,
        "body_momentum_end": p_body_final,
        "wave_self_first": p_after_self_first - p_wave_start,
        "wave_compact_first": p_after_compact_first - p_after_self_first,
        "wave_self_first_alternate": (
            p_after_alternate_first - p_after_alternate_compact_first
        ),
        "wave_compact_first_alternate": (
            p_after_alternate_compact_first - p_wave_start
        ),
        "body_first": p_after_body_first - p_body_start,
        "wave_drift": p_after_drift - p_after_compact_first,
        "wave_self_second": p_after_self_second - p_after_drift,
        "wave_compact_second": p_after_compact_second - p_after_self_second,
        "wave_self_second_alternate": (
            p_after_alternate_second - p_after_alternate_compact_second
        ),
        "wave_compact_second_alternate": (
            p_after_alternate_compact_second - p_after_drift
        ),
        "body_second": p_body_final - p_after_body_first,
        "first_half_kick_defect": first_defect,
        "second_half_kick_defect": second_defect,
        "total_after_first_joint": first_joint_total,
        "total_after_drift": drift_total,
        "total_after_second_joint": final_total,
        "max_phase_radians": max(first_phase, second_phase),
        "maximum_phase_order_wave_difference": max(
            first_order_wave_difference, second_order_wave_difference
        ),
    }
    component_sum = sum((components[key] for key in (
        "wave_self_first", "wave_compact_first", "body_first",
        "wave_drift", "wave_self_second", "wave_compact_second",
        "body_second",
    )), np.zeros(3))
    telescope_floor = 64.0 * np.finfo(float).eps * (
        np.linalg.norm(p_wave_start) + np.linalg.norm(p_body_start)
        + np.linalg.norm(final_total)
    )
    if not np.allclose(
        component_sum, final_total - (p_wave_start + p_body_start),
        rtol=0, atol=telescope_floor,
    ):
        raise ValueError("attributed momentum ledger does not telescope")
    return (wave, final_body, next_wave_potential,
            next_compact_potential, components)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--momentum-estimator", choices=("marginal", "reordered"),
        default="marginal",
    )
    args = parser.parse_args()
    run = args.candidate_run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace kick attribution: {output}")
    if run.name not in _LEVELS or not re.fullmatch(
        r"[0-9a-f]{40}", args.source_commit
    ):
        raise ValueError("attribution run level or source commit is invalid")
    factor, stop = _LEVELS[run.name]
    project = Path(__file__).resolve().parents[1]
    import fdm_smbh_delay.periodic_mesh_coupling as coupling_module
    import fdm_smbh_delay.torch_wave as wave_module
    import fdm_smbh_delay.pyul as pyul_module
    import scripts.probe_wave_only_momentum_control as momentum_module
    loaded_modules = {
        "scripts/probe_wave_only_momentum_control.py": momentum_module,
        "src/fdm_smbh_delay/periodic_mesh_coupling.py": coupling_module,
        "src/fdm_smbh_delay/torch_wave.py": wave_module,
        "src/fdm_smbh_delay/pyul.py": pyul_module,
    }
    if any(
        Path(module.__file__).resolve() != (project / relative).resolve()
        for relative, module in loaded_modules.items()
    ):
        raise ValueError("attribution imported source differs from project source")
    source_hashes = {relative: _sha256(project / relative)
                     for relative in _SOURCE_PATHS}
    for relative, digest in source_hashes.items():
        committed = subprocess.run(
            ["git", "show", f"{args.source_commit}:{relative}"],
            cwd=project, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if hashlib.sha256(committed.stdout).hexdigest() != digest:
            raise ValueError(f"attribution source differs from commit: {relative}")
    metadata_path = run / "fdm_adapter_metadata.json"
    config_path = run / "config.uldm"
    metadata = json.loads(metadata_path.read_text())
    summary = json.loads((run / "torch_run_summary.json").read_text())
    manifest = json.loads(
        (run / "torch_solver_provenance/manifest.json").read_text()
    )
    expected_solver = {
        relative: source_hashes[relative] for relative in _SOURCE_PATHS
        if relative.startswith("src/")
    }
    producer_path = project / "scripts/run_torch_wave_case.py"
    producer_hash = _sha256(producer_path)
    if (
        metadata.get("wave_smbh_coupling") != "periodic_tsc_strang_momentum"
        or metadata.get("case_id") != "qe_q030_e030_a020"
        or metadata.get("resolution") != 256
        or metadata.get("time_step_factor") != factor
        or metadata.get("save_number") != metadata.get("actual_wave_steps")
        or metadata.get("save_number", 0) <= stop
        or metadata.get("diagnostic_stop_after_save") != stop
        or any(
            metadata.get("solver_source_sha256", {}).get(relative) != digest
            for relative, digest in expected_solver.items()
        )
        or metadata.get("solver_source_sha256", {}).get(
            "scripts/run_torch_wave_case.py"
        ) != producer_hash
        or summary.get("status") != "diagnostic_partial"
        or summary.get("actual_wave_steps") != stop
        or manifest.get("status") != "source_snapshot"
        or manifest.get("run") != str(run)
        or manifest.get("input_records", {}).get("fdm_adapter_metadata_sha256")
        != _sha256(metadata_path)
        or manifest.get("input_records", {}).get("config_sha256")
        != _sha256(config_path)
    ):
        raise ValueError("attribution candidate run is incompatible")
    frozen_hashes = {
        row.get("path"): row.get("sha256")
        for row in manifest.get("source_files", [])
        if isinstance(row, dict)
    }
    for relative, digest in expected_solver.items():
        if (
            _sha256(run / "torch_solver_provenance/source" / relative) != digest
            or frozen_hashes.get(relative) != digest
        ):
            raise ValueError("attribution frozen solver source differs")
    if (
        _sha256(run / "torch_solver_provenance/source/scripts/run_torch_wave_case.py")
        != producer_hash
        or frozen_hashes.get("scripts/run_torch_wave_case.py") != producer_hash
    ):
        raise ValueError("attribution frozen producer source differs")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("n256 attribution requires an allocated CUDA GPU")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    momentum_estimator = (
        spectral_wave_momentum_torch
        if args.momentum_estimator == "marginal"
        else spectral_wave_momentum_reordered_torch
    )
    seed = Path(metadata["reference_initial_state"]).resolve()
    wave_path = seed / "Outputs/3Wfn/P3D_#000.npy"
    body_path = seed / "Outputs/NBody/NTM_#000.npy"
    seed_config_path = seed / "config.uldm"
    seed_metadata_path = seed / "fdm_adapter_metadata.json"
    if (
        _sha256(wave_path) != metadata["reference_initial_wave_sha256"]
        or _sha256(body_path) != metadata["reference_initial_particle_sha256"]
        or _sha256(seed_config_path) != metadata["reference_config_sha256"]
        or _sha256(seed_metadata_path) != metadata["reference_metadata_sha256"]
    ):
        raise ValueError("attribution initial state changed")
    stable_input_hashes = {
        "metadata": _sha256(metadata_path),
        "config": _sha256(config_path),
        "seed_wave": _sha256(wave_path),
        "seed_body": _sha256(body_path),
        "seed_config": _sha256(seed_config_path),
        "seed_metadata": _sha256(seed_metadata_path),
    }
    units = pyul_unit_system(metadata)
    config = json.loads(config_path.read_text())
    masses = np.asarray([
        item[0] for item in config["Matter Particles"]["Condition"]
    ], dtype=float) / units.mass_msun
    box_code = float(metadata["box_size_pc"]) / units.length_pc
    plummer_code = (
        float(config["Matter Particles"]["Plummer Radius"])
        / units.length_pc
    )
    time_step = float(metadata["wave_time_step_code"])
    initial_wave = np.load(wave_path)
    if initial_wave.shape != (256, 256, 256) or initial_wave.dtype != np.complex128:
        raise ValueError("attribution seed wave shape or dtype differs")
    wave = torch.as_tensor(initial_wave, dtype=torch.complex128, device=device)
    del initial_wave
    body = np.load(body_path).astype(float).reshape(2, 6)
    grid = spectral_grid(resolution=256, box_length=box_code,
                         time_step=time_step, device=device)
    wave_potential = periodic_poisson_torch(
        wave_density(wave), grid.poisson_inverse_wavenumber_squared
    )
    compact_potential = compact_field(
        body, masses, grid=grid, box_length=box_code, device=device,
    )
    initial_diagnostic = spectral_wave_diagnostics_torch(wave, box_code)
    initial_total = initial_diagnostic["momentum_code"] + body_momentum(body, masses)
    previous_wave_momentum = initial_diagnostic["momentum_code"]
    previous_body_momentum = body_momentum(body, masses)
    continuity_tolerance = 10.0 * initial_diagnostic[
        "summation_rounding_floor_code"
    ]
    body_save_max_difference = 0.0
    records = []
    for step in range(1, stop + 1):
        wave, body, wave_potential, compact_potential, changes = attributed_step(
            wave, body, masses, wave_potential, compact_potential,
            grid=grid, box_length=box_code, plummer_radius=plummer_code,
            time_step=time_step, momentum_estimator=momentum_estimator,
        )
        if (
            np.linalg.norm(changes["wave_momentum_start"] - previous_wave_momentum)
            > continuity_tolerance
            or not np.allclose(
                changes["body_momentum_start"], previous_body_momentum,
                rtol=0, atol=1e-10,
            )
        ):
            raise ValueError("attribution momentum boundary differs across steps")
        previous_wave_momentum = changes["wave_momentum_end"]
        previous_body_momentum = changes["body_momentum_end"]
        saved_body = np.load(run / f"Outputs/NBody/NTM_#{step:03d}.npy")
        difference = float(np.max(np.abs(saved_body.reshape(2, 6) - body)))
        body_save_max_difference = max(body_save_max_difference, difference)
        body_scale = max(float(np.max(np.abs(saved_body))), np.finfo(float).tiny)
        if difference / body_scale > 1e-12:
            raise ValueError(f"attribution binary differs at saved step {step}")
        records.append({
            "step": step,
            "first_half_kick_defect_norm_code": float(np.linalg.norm(
                changes["first_half_kick_defect"]
            )),
            "second_half_kick_defect_norm_code": float(np.linalg.norm(
                changes["second_half_kick_defect"]
            )),
            **{
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in changes.items()
            },
        })
    torch.cuda.synchronize(device)
    peak_memory = torch.cuda.max_memory_allocated(device)
    marker = json.loads((run / "Checkpoints/latest.json").read_text())
    if (
        marker.get("step") != stop or marker.get("save_index") != stop
        or marker.get("wave") != f"wave_{stop:06d}.npy"
        or marker.get("state") != f"state_{stop:06d}.npz"
    ):
        raise ValueError("attribution target checkpoint differs")
    expected_wave_path = run / "Checkpoints" / marker["wave"]
    expected_body_path = run / "Checkpoints" / marker["state"]
    expected_wave = torch.as_tensor(
        np.load(expected_wave_path), dtype=torch.complex128, device=device
    )
    wave_max_difference = float(torch.max(torch.abs(wave - expected_wave)))
    wave_scale = max(float(torch.max(torch.abs(expected_wave))),
                     np.finfo(float).tiny)
    wave_relative_difference = wave_max_difference / wave_scale
    with np.load(expected_body_path) as checkpoint:
        expected_body = np.asarray(checkpoint["state"], dtype=float).reshape(2, 6)
        if (
            int(checkpoint["step"]) != stop
            or int(checkpoint["save_index"]) != stop
        ):
            raise ValueError("attribution body checkpoint marker disagrees")
    body_max_difference = float(np.max(np.abs(body - expected_body)))
    body_scale = max(float(np.max(np.abs(expected_body))), np.finfo(float).tiny)
    body_relative_difference = body_max_difference / body_scale
    if wave_relative_difference > 1e-10 or body_relative_difference > 1e-12:
        raise ValueError("factorized kick replay differs from candidate checkpoint")
    final_diagnostic = spectral_wave_diagnostics_torch(wave, box_code)
    endpoint_total = final_diagnostic["momentum_code"] + body_momentum(body, masses)
    audit_path = run.parent / f"momentum_{run.name}_v1.json"
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("run") != str(run)
        or audit.get("initial_wave_sha256") != _sha256(wave_path)
        or audit.get("initial_body_sha256") != _sha256(body_path)
        or audit.get("final_wave_sha256") != _sha256(expected_wave_path)
        or audit.get("final_body_sha256") != _sha256(expected_body_path)
    ):
        raise ValueError("attribution CPU momentum audit differs from replay")
    cpu_endpoint = np.asarray(audit["total_momentum_change_code"], dtype=float)
    cpu_gap = float(np.linalg.norm(endpoint_total - initial_total - cpu_endpoint))
    cpu_tolerance = 10.0 * (
        initial_diagnostic["summation_rounding_floor_code"]
        + final_diagnostic["summation_rounding_floor_code"]
    )
    if not np.isfinite(cpu_gap) or cpu_gap > cpu_tolerance:
        raise ValueError("attribution GPU momentum differs from CPU endpoint")
    if (
        any(_sha256(project / relative) != digest
            for relative, digest in source_hashes.items())
        or _sha256(producer_path) != producer_hash
        or stable_input_hashes != {
            "metadata": _sha256(metadata_path),
            "config": _sha256(config_path),
            "seed_wave": _sha256(wave_path),
            "seed_body": _sha256(body_path),
            "seed_config": _sha256(seed_config_path),
            "seed_metadata": _sha256(seed_metadata_path),
        }
    ):
        raise ValueError("attribution source or seed changed during replay")
    completed_step_residuals = [
        float(np.linalg.norm(np.asarray(row["total_after_second_joint"])
                             - initial_total)) for row in records
    ]
    drift_nulls = [float(np.linalg.norm(row["wave_drift"]))
                   for row in records]
    sums = {
        key: np.sum([np.asarray(row[key]) for row in records], axis=0)
        for key in (
            "wave_self_first", "wave_compact_first", "body_first",
            "wave_drift", "wave_self_second", "wave_compact_second",
            "body_second",
            "wave_self_first_alternate", "wave_compact_first_alternate",
            "wave_self_second_alternate", "wave_compact_second_alternate",
        )
    }
    compact_plus_body = (
        sums["wave_compact_first"] + sums["wave_compact_second"]
        + sums["body_first"] + sums["body_second"]
    )
    self_total = sums["wave_self_first"] + sums["wave_self_second"]
    self_alternate = (
        sums["wave_self_first_alternate"]
        + sums["wave_self_second_alternate"]
    )
    compact_alternate_plus_body = (
        sums["wave_compact_first_alternate"]
        + sums["wave_compact_second_alternate"]
        + sums["body_first"] + sums["body_second"]
    )
    drift_total = sums["wave_drift"]
    total_components = self_total + compact_plus_body + drift_total
    cumulative_telescope_gap = float(np.linalg.norm(
        total_components - (endpoint_total - initial_total)
    ))
    if cumulative_telescope_gap > cpu_tolerance:
        raise ValueError("attribution cumulative ledger differs from endpoint")
    half_defects = [
        np.asarray(row[key], dtype=float)
        for row in records
        for key in ("first_half_kick_defect", "second_half_kick_defect")
    ]
    half_impulses = [
        np.asarray(row[key], dtype=float)
        for row in records
        for key in ("body_first", "body_second")
    ]
    half_norms = [float(np.linalg.norm(value)) for value in half_defects]
    max_defect_over_impulse = max(
        defect / max(float(np.linalg.norm(impulse)), np.finfo(float).tiny)
        for defect, impulse in zip(half_norms, half_impulses)
    )
    phase_max = max(row["max_phase_radians"] for row in records)
    phase_order_difference = max(
        row["maximum_phase_order_wave_difference"] for row in records
    )
    payload = {
        "status": "coupled_factorized_kick_diagnostic_not_a_calibration_release",
        "calibration_eligible": False,
        "run": str(run),
        "source_commit": args.source_commit,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_sha256": source_hashes,
        "producer_source_sha256": producer_hash,
        "candidate_metadata_sha256": _sha256(metadata_path),
        "candidate_cpu_momentum_audit_sha256": _sha256(audit_path),
        "initial_wave_sha256": _sha256(wave_path),
        "initial_body_sha256": _sha256(body_path),
        "final_wave_sha256": _sha256(expected_wave_path),
        "final_body_sha256": _sha256(expected_body_path),
        "time_step_factor": factor,
        "momentum_estimator": args.momentum_estimator,
        "wave_steps": stop,
        "wave_max_replay_difference": wave_max_difference,
        "body_max_replay_difference": body_max_difference,
        "wave_relative_replay_difference": wave_relative_difference,
        "body_relative_replay_difference": body_relative_difference,
        "maximum_saved_body_difference_code": body_save_max_difference,
        "maximum_phase_radians": phase_max,
        "maximum_phase_order_wave_difference": phase_order_difference,
        "maximum_phase_order_wave_relative_difference": (
            phase_order_difference / wave_scale
        ),
        "maximum_completed_step_momentum_residual_code": max(
            completed_step_residuals
        ),
        "maximum_half_kick_action_reaction_defect_code": max(half_norms),
        "maximum_half_kick_defect_over_body_impulse": max_defect_over_impulse,
        "maximum_half_kick_defect_over_initial_spectral_floor": (
            max(half_norms) / initial_diagnostic["summation_rounding_floor_code"]
        ),
        "maximum_wave_drift_momentum_null_code": max(drift_nulls),
        "final_total_momentum_residual_code": (
            endpoint_total - initial_total
        ).tolist(),
        "cpu_endpoint_momentum_residual_code": cpu_endpoint.tolist(),
        "cpu_gpu_endpoint_momentum_gap_code": cpu_gap,
        "cpu_gpu_endpoint_tolerance_code": cpu_tolerance,
        "wave_self_attribution_code": self_total.tolist(),
        "wave_self_alternate_order_attribution_code": self_alternate.tolist(),
        "wave_compact_plus_body_attribution_code": compact_plus_body.tolist(),
        "wave_compact_plus_body_alternate_order_attribution_code": (
            compact_alternate_plus_body.tolist()
        ),
        "wave_drift_attribution_code": drift_total.tolist(),
        "cumulative_ledger_telescope_gap_code": cumulative_telescope_gap,
        "per_step": records,
        "peak_device_memory_bytes": peak_memory,
        "interpretation": (
            "Self-then-compact and compact-then-self phase-kick ordering on "
            "the coupled state. Half-kick defects are discrete operator "
            "diagnostics, not physical states or a new force law"
        ),
    }
    if output.exists():
        raise FileExistsError(f"refusing to replace kick attribution: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.link(temporary, output)
    temporary.unlink()
    print(json.dumps({key: payload[key] for key in (
        "status", "wave_steps", "maximum_completed_step_momentum_residual_code",
        "maximum_half_kick_action_reaction_defect_code",
        "wave_relative_replay_difference", "body_relative_replay_difference",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
