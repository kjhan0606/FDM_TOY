"""Common-interval diagnostics for live-wave convergence calculations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


_RATE_FIELDS = (
    "binary_orbital_energy",
    "binary_angular_momentum_msun_pc2_myr",
    "wave_intrinsic_energy",
    "wave_bh_interaction_grid",
    "bh_com_kinetic_energy",
    "combined_energy",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fractional_difference(value: float, reference: float) -> float | None:
    if abs(reference) <= np.finfo(float).tiny:
        return None
    return float((value - reference) / abs(reference))


def load_convergence_run(label: str, run: Path) -> dict:
    """Load the saved diagnostics needed for a convergence comparison."""
    resolved = run.expanduser().resolve()
    metadata = _read_json(resolved / "fdm_adapter_metadata.json")
    config = _read_json(resolved / "config.uldm")
    conservation = _read_json(resolved / "conservation_summary.json")
    orbit = _read_json(resolved / "orbit_averaged_exchange_summary.json")
    series = np.genfromtxt(
        resolved / "conservation_timeseries.csv",
        delimiter=",",
        names=True,
        ndmin=1,
    )
    if series.size < 2:
        raise ValueError(f"{label}: conservation table has fewer than two states")
    if np.any(np.diff(series["time_myr"]) <= 0.0):
        raise ValueError(f"{label}: saved times are not strictly increasing")
    missing = sorted(set(("time_myr", "separation_pc", *_RATE_FIELDS)) - set(series.dtype.names or ()))
    if missing:
        raise ValueError(f"{label}: missing conservation fields: {', '.join(missing)}")
    return {
        "label": label,
        "run": str(resolved),
        "metadata": metadata,
        "config": config,
        "conservation": conservation,
        "orbit": orbit,
        "series": series,
    }


def summarize_convergence(runs: Iterable[dict]) -> dict:
    """Compare numerical variants over their shared resolved time interval."""
    loaded = list(runs)
    if len(loaded) < 2:
        raise ValueError("at least two calculations are required")
    labels = [item["label"] for item in loaded]
    if len(labels) != len(set(labels)):
        raise ValueError("calculation labels must be unique")

    common_end = min(
        min(
            float(item["series"]["time_myr"][-1]),
            float(item["conservation"]["initial_spatially_resolved_duration_myr"]),
        )
        for item in loaded
    )
    if common_end <= 0.0:
        raise ValueError("the calculations have no shared resolved interval")

    rows: list[dict] = []
    for item in loaded:
        series = item["series"]
        time = series["time_myr"]
        initial_separation = float(series["separation_pc"][0])
        common: dict[str, float] = {
            "end_time_myr": common_end,
            "separation_pc": float(
                np.interp(common_end, time, series["separation_pc"])
            ),
            "separation_change_over_initial": float(
                (
                    np.interp(common_end, time, series["separation_pc"])
                    - initial_separation
                )
                / initial_separation
            ),
            "energy_error_over_transfer": float(
                np.interp(common_end, time, series["energy_error_over_transfer"])
            ),
        }
        for field in _RATE_FIELDS:
            final = float(np.interp(common_end, time, series[field]))
            common[f"mean_{field}_rate"] = float(
                (final - float(series[field][0])) / common_end
            )

        bootstrap = item["orbit"]["initial_resolved_window_block_bootstrap"]
        metadata = item["metadata"]
        config = item["config"]
        conservation = item["conservation"]
        rows.append(
            {
                "label": item["label"],
                "run": item["run"],
                "resolution": int(metadata["resolution"]),
                "cell_size_pc": float(metadata["cell_size_pc"]),
                "time_step_factor": float(
                    metadata.get("time_step_factor", config["Temporal Step Factor"])
                ),
                "nbody_rk4_substeps_per_wave_step": int(
                    metadata.get(
                        "nbody_rk4_substeps_per_wave_step",
                        int(config["RK Steps"]) // 4,
                    )
                ),
                "initial_resolved_duration_myr": float(
                    conservation["initial_spatially_resolved_duration_myr"]
                ),
                "maximum_energy_error_over_transfer": float(
                    conservation["max_total_energy_drift_over_energy_transfer"]
                ),
                "common_interval": common,
                "orbit_averaged_resolved_window": {
                    "orbits": int(bootstrap["window_orbits"]),
                    "orbital_power": bootstrap["orbital_power"],
                    "orbital_torque": bootstrap["orbital_torque"],
                },
            }
        )

    reference = rows[0]
    reference_common = reference["common_interval"]
    initial_reference = float(loaded[0]["series"]["separation_pc"][0])
    for row in rows:
        common = row["common_interval"]
        differences = {
            "separation_difference_over_reference_initial": float(
                (common["separation_pc"] - reference_common["separation_pc"])
                / initial_reference
            )
        }
        for field in _RATE_FIELDS:
            key = f"mean_{field}_rate"
            differences[f"{key}_fractional_difference"] = _fractional_difference(
                common[key], reference_common[key]
            )
        row["difference_from_reference"] = differences

    return {
        "status": "common_resolved_interval_compared",
        "reference_label": reference["label"],
        "common_interval_end_myr": common_end,
        "runs": rows,
        "interpretation": (
            "common-interval rates include reversible orbital-phase and interaction-energy "
            "variations; orbit-averaged estimates remain separate, and numerical convergence "
            "requires agreement of both diagnostics"
        ),
        "automatic_convergence_decision": False,
    }
