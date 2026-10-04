#!/usr/bin/env python3
"""Audit short-prefix wave plus SMBH momentum without modifying a live solver."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from fdm_smbh_delay.pyul import pyul_unit_system

if __package__:
    from .analyze_periodic_tsc_step_trace import _read_trace
    from .analyze_qe_startup_component_dt import _sha256
else:
    from analyze_periodic_tsc_step_trace import _read_trace
    from analyze_qe_startup_component_dt import _sha256


def spectral_wave_diagnostics(wave: np.ndarray, box_length: float) -> dict:
    """Return periodic wave momentum, mass, kinetic energy, and grid-scale warnings.

    With the solver's ``-nabla^2/2`` kinetic operator, Parseval gives
    ``P = dV / N^3 sum_k k |FFT(psi)_k|^2``. The function performs one FFT
    and never forms a three-dimensional gradient or wavenumber array.
    """

    state = np.asarray(wave)
    if (
        state.ndim != 3 or len(set(state.shape)) != 1
        or state.shape[0] < 4 or not np.iscomplexobj(state)
        or not np.isfinite(box_length) or box_length <= 0.0
        or not np.all(np.isfinite(state))
    ):
        raise ValueError("wave momentum requires a finite cubic complex field")
    state = np.asarray(state, dtype=np.complex128)
    resolution = state.shape[0]
    spectrum = np.fft.fftn(state)
    power = spectrum.real**2 + spectrum.imag**2
    del spectrum
    total_power = float(power.sum())
    if not np.isfinite(total_power) or total_power <= 0.0:
        raise ValueError("wave momentum requires positive finite wave mass")
    wave_number = 2.0 * np.pi * np.fft.fftfreq(
        resolution, d=box_length / resolution
    )
    factor = (box_length / resolution) ** 3 / resolution**3
    marginals = (
        power.sum(axis=(1, 2)), power.sum(axis=(0, 2)),
        power.sum(axis=(0, 1)),
    )
    momentum = factor * np.asarray([
        wave_number @ marginal for marginal in marginals
    ])
    kinetic = 0.5 * factor * sum(
        wave_number**2 @ marginal for marginal in marginals
    )
    absolute_weight = factor * sum(
        np.abs(wave_number) @ marginal for marginal in marginals
    )
    high_frequency = np.abs(np.fft.fftfreq(resolution)) >= 0.375
    high_shell = np.zeros(power.shape, dtype=bool)
    high_shell[high_frequency, :, :] = True
    high_shell[:, high_frequency, :] = True
    high_shell[:, :, high_frequency] = True
    return {
        "momentum_code": momentum,
        "mass_code": float(factor * total_power),
        "kinetic_code": float(kinetic),
        "high_frequency_power_fraction": float(
            power[high_shell].sum() / total_power
        ),
        "summation_rounding_floor_code": float(
            32.0 * np.finfo(float).eps * absolute_weight
        ),
    }


def spectral_wave_momentum(wave: np.ndarray, box_length: float) -> np.ndarray:
    """Return periodic Schrödinger wave momentum in code units."""

    return spectral_wave_diagnostics(wave, box_length)["momentum_code"]


def _body_momentum(state: np.ndarray, masses: np.ndarray) -> np.ndarray:
    bodies = np.asarray(state, dtype=float).reshape(2, 6)
    if not np.all(np.isfinite(bodies)):
        raise ValueError("binary momentum requires a finite two-body state")
    return np.sum(np.asarray(masses)[:, None] * bodies[:, 3:], axis=0)


def summarize(
    run: Path, *, factor: float, saves: int, stop: int,
) -> dict:
    path = run.expanduser().resolve()
    trace, _series = _read_trace(
        path, factor, saves, stop, coupling="periodic_tsc_strang"
    )
    metadata = json.loads((path / "fdm_adapter_metadata.json").read_text())
    config = json.loads((path / "config.uldm").read_text())
    reference = Path(metadata["reference_initial_state"]).resolve()
    wave_initial_path = reference / "Outputs/3Wfn/P3D_#000.npy"
    body_initial_path = reference / "Outputs/NBody/NTM_#000.npy"
    if (
        _sha256(wave_initial_path) != metadata["reference_initial_wave_sha256"]
        or _sha256(body_initial_path) != metadata["reference_initial_particle_sha256"]
    ):
        raise ValueError("initial wave or binary differs from bound reference")
    marker = json.loads((path / "Checkpoints/latest.json").read_text())
    if marker.get("step") != stop or marker.get("save_index") != stop:
        raise ValueError("momentum audit requires the final diagnostic checkpoint")
    wave_final_path = path / "Checkpoints" / marker["wave"]
    body_final_path = path / "Checkpoints" / marker["state"]
    with np.load(body_final_path) as saved:
        body_final = np.asarray(saved["state"], dtype=float)
        if int(saved["step"]) != stop or int(saved["save_index"]) != stop:
            raise ValueError("final momentum checkpoint marker disagrees with state")
    body_saved = np.load(path / f"Outputs/NBody/NTM_#{stop:03d}.npy")
    if not np.array_equal(body_final, body_saved):
        raise ValueError("final momentum checkpoint differs from saved binary")
    units = pyul_unit_system(metadata)
    mass = np.asarray(
        [body[0] for body in config["Matter Particles"]["Condition"]],
        dtype=float,
    ) / units.mass_msun
    box_code = float(metadata["box_size_pc"]) / units.length_pc
    wave_initial = np.load(wave_initial_path, mmap_mode="r")
    initial_wave = spectral_wave_diagnostics(wave_initial, box_code)
    del wave_initial
    wave_final = np.load(wave_final_path, mmap_mode="r")
    final_wave = spectral_wave_diagnostics(wave_final, box_code)
    del wave_final
    recorded_mass = np.load(path / "Outputs/ULDMass.npy")
    recorded_kinetic = np.load(path / "Outputs/ekandqlist.npy")
    for index, measured in ((0, initial_wave), (stop, final_wave)):
        if (
            not np.isclose(
                measured["mass_code"], recorded_mass[index], rtol=1e-11, atol=1e-9
            )
            or not np.isclose(
                measured["kinetic_code"], recorded_kinetic[index],
                rtol=1e-11, atol=1e-9,
            )
        ):
            raise ValueError("wave checkpoint mass or kinetic energy disagrees with ledger")
    wave_momentum_initial = initial_wave["momentum_code"]
    wave_momentum_final = final_wave["momentum_code"]
    body_momentum_initial = _body_momentum(np.load(body_initial_path), mass)
    body_momentum_final = _body_momentum(body_final, mass)
    wave_change = wave_momentum_final - wave_momentum_initial
    body_change = body_momentum_final - body_momentum_initial
    total_change = wave_change + body_change
    exchange = max(np.linalg.norm(wave_change), np.linalg.norm(body_change))
    rounding_floor = (
        initial_wave["summation_rounding_floor_code"]
        + final_wave["summation_rounding_floor_code"]
        + 32.0 * np.finfo(float).eps * (
            np.linalg.norm(body_momentum_initial)
            + np.linalg.norm(body_momentum_final)
        )
    )
    momentum_scale = units.mass_msun * units.length_pc / units.time_myr
    return {
        "status": "periodic_tsc_strang_momentum_diagnostic_not_a_calibration_release",
        "run": str(path),
        "slurm_job_id": trace["slurm_job_id"],
        "source_commit": trace["source_commit"],
        "source_fingerprints": trace["source_fingerprints"],
        "launch_owner_sha256": trace["launch_owner_sha256"],
        "conservation_summary_sha256": trace["conservation_summary_sha256"],
        "conservation_timeseries_sha256": trace["conservation_timeseries_sha256"],
        "provenance_manifest_sha256": trace["provenance_manifest_sha256"],
        "analysis_source_sha256": _sha256(Path(__file__).resolve()),
        "factor": factor,
        "last_wave_step": stop,
        "initial_wave_sha256": _sha256(wave_initial_path),
        "final_wave_sha256": _sha256(wave_final_path),
        "initial_body_sha256": _sha256(body_initial_path),
        "final_body_sha256": _sha256(body_final_path),
        "wave_momentum_initial_code": wave_momentum_initial.tolist(),
        "wave_momentum_final_code": wave_momentum_final.tolist(),
        "body_momentum_initial_code": body_momentum_initial.tolist(),
        "body_momentum_final_code": body_momentum_final.tolist(),
        "total_momentum_change_code": total_change.tolist(),
        "total_momentum_change_msun_pc_myr": (
            total_change * momentum_scale
        ).tolist(),
        "wave_mass_initial_code": initial_wave["mass_code"],
        "wave_mass_final_code": final_wave["mass_code"],
        "wave_kinetic_initial_code": initial_wave["kinetic_code"],
        "wave_kinetic_final_code": final_wave["kinetic_code"],
        "initial_high_frequency_power_fraction": (
            initial_wave["high_frequency_power_fraction"]
        ),
        "final_high_frequency_power_fraction": (
            final_wave["high_frequency_power_fraction"]
        ),
        "estimated_momentum_rounding_floor_code": rounding_floor,
        "momentum_change_resolved_above_rounding_floor": bool(
            np.linalg.norm(total_change) > rounding_floor
        ),
        "momentum_error_over_exchange": (
            None if exchange <= rounding_floor
            else float(np.linalg.norm(total_change) / exchange)
        ),
        "calibration_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--saves", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace momentum audit: {output}")
    if run in output.parents:
        raise ValueError("momentum audit output must be outside the immutable run")
    payload = summarize(
        run, factor=args.factor, saves=args.saves, stop=args.stop
    )
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
