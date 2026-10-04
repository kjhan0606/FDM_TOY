#!/usr/bin/env python3
"""Source-bound endpoint audit of the experimental spectral-momentum force."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

if __package__:
    from .analyze_periodic_tsc_dt_followup import _launch_owner
    from .analyze_qe_startup_component_dt import _sha256
    from .audit_periodic_tsc_momentum import (
        _body_momentum, spectral_wave_diagnostics,
    )
else:
    from analyze_periodic_tsc_dt_followup import _launch_owner
    from analyze_qe_startup_component_dt import _sha256
    from audit_periodic_tsc_momentum import (
        _body_momentum, spectral_wave_diagnostics,
    )

from fdm_smbh_delay.pyul import pyul_unit_system


_LEVELS = {
    "f100": (1.0, 58500, 10),
    "f050": (0.5, 117000, 20),
    "f025": (0.25, 234000, 40),
    "f0125": (0.125, 468000, 80),
}
_SOURCES = {
    "scripts/run_torch_wave_case.py",
    "src/fdm_smbh_delay/torch_wave.py",
    "src/fdm_smbh_delay/periodic_mesh_coupling.py",
    "src/fdm_smbh_delay/pyul.py",
}


def _validate_scope(metadata: dict, solver: dict, conservation: dict,
                    label: str) -> tuple[float, int]:
    if label not in _LEVELS:
        raise ValueError("unknown momentum-candidate time-step level")
    factor, saves, stop = _LEVELS[label]
    expected = {
        "case_id": "qe_q030_e030_a020",
        "resolution": 256,
        "backend": "pytorch_cuda",
        "wave_smbh_coupling": "periodic_tsc_strang_momentum",
        "binary_integrator": "joint_kick_drift_kick_spectral_momentum_v1",
        "smbh_force_is_interaction_energy_gradient": False,
        "coupled_hamiltonian_ledger": True,
        "experimental_coupling_not_a_calibration_release": True,
        "analytic_fdm_drag": False,
        "time_step_factor": factor,
        "save_number": saves,
        "actual_wave_steps": saves,
        "diagnostic_stop_after_save": stop,
        "nbody_rk4_substeps_per_wave_step": 0,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("momentum-candidate run metadata differs from design")
    if (
        solver.get("status") != "diagnostic_partial"
        or solver.get("actual_wave_steps") != stop
        or solver.get("saved_intervals") != stop
        or not np.isfinite(solver.get("peak_device_memory_bytes", np.nan))
        or solver.get("peak_device_memory_bytes", 0) <= 0
        or not np.isfinite(conservation.get(
            "max_total_energy_drift_over_energy_transfer", np.nan
        ))
        or conservation.get("status") != "diagnostic_partial"
        or conservation.get("samples") != stop + 1
        or conservation.get("wave_smbh_coupling")
        != "periodic_tsc_strang_momentum"
        or conservation.get("smbh_force_is_interaction_energy_gradient") is not False
    ):
        raise ValueError("momentum-candidate result is not the requested prefix")
    return factor, stop


def summarize(run: Path) -> dict:
    path = run.expanduser().resolve()
    project = Path(__file__).resolve().parents[1]
    metadata_path = path / "fdm_adapter_metadata.json"
    config_path = path / "config.uldm"
    manifest_path = path / "torch_solver_provenance/manifest.json"
    solver_path = path / "torch_run_summary.json"
    conservation_path = path / "conservation_summary.json"
    series_path = path / "conservation_timeseries.csv"
    metadata = json.loads(metadata_path.read_text())
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    solver = json.loads(solver_path.read_text())
    conservation = json.loads(conservation_path.read_text())
    if (project / "scripts/analyze_pyul_wave_run.py").stat().st_mtime_ns > (
        conservation_path.stat().st_mtime_ns
    ):
        raise ValueError("conservation analyser changed after diagnostic output")
    factor, stop = _validate_scope(
        metadata, solver, conservation, path.name
    )
    series = np.genfromtxt(series_path, delimiter=",", names=True)
    required_fields = (
        "time_myr", "separation_pc", "combined_energy",
        "energy_error_over_transfer", "wave_mass_msun",
    )
    if (
        series.ndim != 1 or series.size != stop + 1
        or series.dtype.names is None
        or any(name not in series.dtype.names for name in required_fields)
        or any(not np.all(np.isfinite(series[name])) for name in required_fields)
        or not np.allclose(
            series["time_myr"],
            metadata["duration_myr"] * np.arange(stop + 1)
            / metadata["save_number"],
            rtol=1e-12, atol=0.0,
        )
        or not np.isclose(
            series["separation_pc"][-1], conservation["final_separation_pc"],
            rtol=1e-12, atol=0.0,
        )
    ):
        raise ValueError("momentum-candidate conservation series differs")
    if (
        manifest.get("status") != "source_snapshot"
        or manifest.get("run") != str(path)
        or manifest.get("input_records", {}).get("fdm_adapter_metadata_sha256")
        != _sha256(metadata_path)
        or manifest.get("input_records", {}).get("config_sha256")
        != _sha256(config_path)
        or json.loads((path.parent / f"guard_{path.name}.json").read_text())
        .get("status") != "solver_exited"
    ):
        raise ValueError("momentum-candidate provenance or GPU guard is incomplete")
    records = manifest.get("source_files")
    if not isinstance(records, list) or not all(
        isinstance(row, dict) for row in records
    ):
        raise ValueError("momentum-candidate source snapshot is malformed")
    sources = tuple(sorted((row.get("path"), row.get("sha256")) for row in records))
    if {name for name, _ in sources} != _SOURCES:
        raise ValueError("momentum-candidate numerical source inventory differs")
    for relative, digest in sources:
        if not isinstance(digest, str) or _sha256(
            path / "torch_solver_provenance/source" / relative
        ) != digest or metadata.get("solver_source_sha256", {}).get(relative) != digest:
            raise ValueError("momentum-candidate frozen numerical source differs")
    owner = _launch_owner(path, project, sources)
    marker = json.loads((path / "Checkpoints/latest.json").read_text())
    if (
        marker.get("step") != stop or marker.get("save_index") != stop
        or marker.get("wave") != f"wave_{stop:06d}.npy"
        or marker.get("state") != f"state_{stop:06d}.npz"
    ):
        raise ValueError("momentum-candidate final checkpoint is incomplete")
    final_wave_path = path / "Checkpoints" / marker["wave"]
    final_body_path = path / "Checkpoints" / marker["state"]
    with np.load(final_body_path) as checkpoint:
        final_body = np.asarray(checkpoint["state"], dtype=float)
        if int(checkpoint["step"]) != stop or int(checkpoint["save_index"]) != stop:
            raise ValueError("momentum-candidate checkpoint marker disagrees")
    if not np.array_equal(
        final_body, np.load(path / f"Outputs/NBody/NTM_#{stop:03d}.npy")
    ):
        raise ValueError("momentum-candidate saved binary differs from checkpoint")
    reference = Path(metadata["reference_initial_state"]).resolve()
    initial_wave_path = reference / "Outputs/3Wfn/P3D_#000.npy"
    initial_body_path = reference / "Outputs/NBody/NTM_#000.npy"
    if (
        _sha256(initial_wave_path) != metadata["reference_initial_wave_sha256"]
        or _sha256(initial_body_path) != metadata["reference_initial_particle_sha256"]
    ):
        raise ValueError("momentum-candidate reference state differs from launch")
    units = pyul_unit_system(metadata)
    mass = np.asarray(
        [body[0] for body in config["Matter Particles"]["Condition"]],
        dtype=float,
    ) / units.mass_msun
    if mass.shape != (2,) or np.any(mass <= 0.0):
        raise ValueError("momentum-candidate binary masses are invalid")
    box_code = float(metadata["box_size_pc"]) / units.length_pc
    initial_wave = np.load(initial_wave_path, mmap_mode="r")
    initial = spectral_wave_diagnostics(initial_wave, box_code)
    del initial_wave
    final_wave = np.load(final_wave_path, mmap_mode="r")
    final = spectral_wave_diagnostics(final_wave, box_code)
    del final_wave
    mass_log = np.load(path / "Outputs/ULDMass.npy")
    kinetic_log = np.load(path / "Outputs/ekandqlist.npy")
    for index, measured in ((0, initial), (stop, final)):
        if (
            not np.isclose(measured["mass_code"], mass_log[index], rtol=1e-11, atol=1e-9)
            or not np.isclose(measured["kinetic_code"], kinetic_log[index], rtol=1e-11, atol=1e-9)
        ):
            raise ValueError("momentum-candidate wave checkpoint disagrees with ledger")
    wave_change = final["momentum_code"] - initial["momentum_code"]
    body_change = _body_momentum(final_body, mass) - _body_momentum(
        np.load(initial_body_path), mass
    )
    exchange = max(np.linalg.norm(wave_change), np.linalg.norm(body_change))
    if exchange <= 0.0:
        raise ValueError("momentum-candidate exchange is unresolved")
    total = wave_change + body_change
    return {
        "status": "experimental_spectral_momentum_short_prefix_not_a_calibration_release",
        "calibration_eligible": False,
        "run": str(path),
        "factor": factor,
        "last_wave_step": stop,
        **owner,
        "source_fingerprints": sources,
        "solver_summary_sha256": _sha256(solver_path),
        "conservation_summary_sha256": _sha256(conservation_path),
        "conservation_timeseries_sha256": _sha256(series_path),
        "provenance_manifest_sha256": _sha256(manifest_path),
        "analysis_source_sha256": _sha256(Path(__file__).resolve()),
        "initial_wave_sha256": _sha256(initial_wave_path),
        "final_wave_sha256": _sha256(final_wave_path),
        "initial_body_sha256": _sha256(initial_body_path),
        "final_body_sha256": _sha256(final_body_path),
        "wave_momentum_change_code": wave_change.tolist(),
        "body_momentum_change_code": body_change.tolist(),
        "total_momentum_change_code": total.tolist(),
        "momentum_change_over_exchange": float(np.linalg.norm(total) / exchange),
        "max_energy_error_over_transfer": conservation[
            "max_total_energy_drift_over_energy_transfer"
        ],
        "wave_mass_relative_change": float(final["mass_code"] / initial["mass_code"] - 1),
        "initial_high_frequency_power_fraction": initial[
            "high_frequency_power_fraction"
        ],
        "final_high_frequency_power_fraction": final[
            "high_frequency_power_fraction"
        ],
        "peak_device_memory_bytes": solver["peak_device_memory_bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace momentum audit: {output}")
    payload = summarize(args.run)
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
