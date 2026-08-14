"""Common-interval diagnostics for live-wave convergence calculations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .secular_exchange import moving_block_bootstrap_rate


_RATE_FIELDS = (
    "binary_orbital_energy",
    "binary_angular_momentum_msun_pc2_myr",
    "wave_intrinsic_energy",
    "wave_bh_interaction_grid",
    "bh_com_kinetic_energy",
    "combined_energy",
)

_CONSERVATION_FIELDS = (
    "time_myr",
    "separation_pc",
    "energy_error_over_transfer",
    *_RATE_FIELDS,
)

_ORBIT_RATE_FIELDS = (
    "orbital_power",
    "orbital_torque",
    "wave_intrinsic_energy_rate",
    "wave_bh_interaction_energy_rate",
    "wave_total_energy_rate",
    "bh_com_kinetic_energy_rate",
    "combined_energy_residual_rate",
)

_ORBIT_TABLE_RATE_FIELDS = tuple(
    field for field in _ORBIT_RATE_FIELDS if field != "wave_total_energy_rate"
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
    missing = sorted(
        set(_CONSERVATION_FIELDS) - set(series.dtype.names or ())
    )
    if missing:
        raise ValueError(
            f"{label}: missing conservation fields: {', '.join(missing)}"
        )
    for field in _CONSERVATION_FIELDS:
        if np.any(~np.isfinite(series[field])):
            raise ValueError(
                f"{label}: conservation field {field} contains non-finite values"
            )
    if np.any(np.diff(series["time_myr"]) <= 0.0):
        raise ValueError(f"{label}: saved times are not strictly increasing")
    orbit_path = resolved / "orbit_averaged_exchange.csv"
    orbit_series = None
    if orbit_path.is_file():
        orbit_series = np.genfromtxt(
            orbit_path,
            delimiter=",",
            names=True,
            ndmin=1,
        )
        orbit_required = {
            "start_time_myr",
            "end_time_myr",
            "orbital_period_myr",
            "mean_separation_pc",
            "mean_separation_over_cell_size",
            *_ORBIT_TABLE_RATE_FIELDS,
        }
        orbit_missing = sorted(
            orbit_required - set(orbit_series.dtype.names or ())
        )
        if orbit_missing:
            raise ValueError(
                f"{label}: missing orbit-averaged fields: "
                f"{', '.join(orbit_missing)}"
            )
        if orbit_series.size < 2:
            raise ValueError(
                f"{label}: orbit-averaged table has fewer than two cycles"
            )
        for field in orbit_required:
            if np.any(~np.isfinite(orbit_series[field])):
                raise ValueError(
                    f"{label}: orbit-averaged field {field} contains "
                    "non-finite values"
                )
        cycle_duration = (
            orbit_series["end_time_myr"] - orbit_series["start_time_myr"]
        )
        if (
            np.any(cycle_duration <= 0.0)
            or np.any(orbit_series["orbital_period_myr"] <= 0.0)
            or np.any(np.diff(orbit_series["start_time_myr"]) <= 0.0)
            or not np.allclose(
                cycle_duration,
                orbit_series["orbital_period_myr"],
                rtol=1.0e-10,
                atol=1.0e-12,
            )
        ):
            raise ValueError(
                f"{label}: orbit-averaged cycle times are inconsistent"
            )
    return {
        "label": label,
        "run": str(resolved),
        "metadata": metadata,
        "config": config,
        "conservation": conservation,
        "orbit": orbit,
        "orbit_series": orbit_series,
        "series": series,
    }


def _common_orbit_window(
    loaded: list[dict], common_start: float, common_end: float
) -> tuple[float, list[dict]] | None:
    """Measure rates over one local time interval shared by all calculations."""
    if any(item["orbit_series"] is None for item in loaded):
        return None

    maximum_orbits = 32
    starts: list[float] = []
    valid_indices: list[np.ndarray] = []
    for item in loaded:
        orbit = item["orbit_series"]
        valid = np.flatnonzero(
            (orbit["start_time_myr"] >= common_start - 1.0e-12)
            & (orbit["end_time_myr"] <= common_end + 1.0e-12)
            & (orbit["mean_separation_over_cell_size"] >= 2.0)
        )
        if valid.size < 2:
            return None
        valid = valid[-maximum_orbits:]
        valid_indices.append(valid)
        starts.append(float(orbit["start_time_myr"][valid[0]]))

    orbit_window_start = max(starts)
    rows: list[dict] = []
    for item, initially_valid in zip(loaded, valid_indices):
        orbit = item["orbit_series"]
        selection = initially_valid[
            orbit["start_time_myr"][initially_valid]
            >= orbit_window_start - 1.0e-12
        ]
        if selection.size < 2:
            return None
        duration = np.asarray(orbit["orbital_period_myr"][selection], dtype=float)
        selected_start = float(orbit["start_time_myr"][selection[0]])
        selected_end = float(orbit["end_time_myr"][selection[-1]])
        requested_duration = common_end - orbit_window_start
        row: dict[str, object] = {
            "start_time_myr": selected_start,
            "end_time_myr": selected_end,
            "requested_start_time_myr": orbit_window_start,
            "requested_end_time_myr": common_end,
            "covered_duration_myr": float(np.sum(duration)),
            "coverage_fraction_of_requested_window": float(
                np.sum(duration) / requested_duration
            ),
            "complete_orbits": int(selection.size),
            "mean_separation_pc": float(
                np.sum(orbit["mean_separation_pc"][selection] * duration)
                / np.sum(duration)
            ),
            "rates": {},
        }
        rates = row["rates"]
        assert isinstance(rates, dict)
        for field in _ORBIT_RATE_FIELDS:
            if field == "wave_total_energy_rate":
                values = np.asarray(
                    orbit["wave_intrinsic_energy_rate"][selection]
                    + orbit["wave_bh_interaction_energy_rate"][selection],
                    dtype=float,
                )
            else:
                values = np.asarray(orbit[field][selection], dtype=float)
            interval = moving_block_bootstrap_rate(
                rate=values,
                duration=duration,
                block_length=min(8, selection.size),
                samples=2000,
            )
            rates[field] = {
                "estimate": interval.estimate,
                "lower_95": interval.lower_95,
                "upper_95": interval.upper_95,
            }
        rows.append(row)
    return orbit_window_start, rows


def summarize_convergence(runs: Iterable[dict]) -> dict:
    """Compare numerical variants over their shared resolved time interval."""
    loaded = list(runs)
    if len(loaded) < 2:
        raise ValueError("at least two calculations are required")
    labels = [item["label"] for item in loaded]
    if len(labels) != len(set(labels)):
        raise ValueError("calculation labels must be unique")

    common_start = max(
        float(item["series"]["time_myr"][0]) for item in loaded
    )
    common_end = min(
        min(
            float(item["series"]["time_myr"][-1]),
            float(item["conservation"]["initial_spatially_resolved_duration_myr"]),
        )
        for item in loaded
    )
    if common_end <= common_start:
        raise ValueError("the calculations have no shared resolved interval")
    common_duration = common_end - common_start

    rows: list[dict] = []
    for item in loaded:
        series = item["series"]
        time = series["time_myr"]
        initial_separation = float(series["separation_pc"][0])
        common: dict[str, float] = {
            "start_time_myr": common_start,
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
            initial = float(np.interp(common_start, time, series[field]))
            final = float(np.interp(common_end, time, series[field]))
            common[f"mean_{field}_rate"] = float(
                (final - initial) / common_duration
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
                    metadata["time_step_factor"]
                    if "time_step_factor" in metadata
                    else config["Temporal Step Factor"]
                ),
                "nbody_rk4_substeps_per_wave_step": int(
                    metadata["nbody_rk4_substeps_per_wave_step"]
                    if "nbody_rk4_substeps_per_wave_step" in metadata
                    else int(config["RK Steps"]) // 4
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

    orbit_window = _common_orbit_window(loaded, common_start, common_end)
    common_orbit_start = None
    if orbit_window is not None:
        common_orbit_start, orbit_rows = orbit_window
        reference_rates = orbit_rows[0]["rates"]
        assert isinstance(reference_rates, dict)
        for row, orbit_row in zip(rows, orbit_rows):
            rates = orbit_row["rates"]
            assert isinstance(rates, dict)
            differences = {
                field: _fractional_difference(
                    float(rates[field]["estimate"]),
                    float(reference_rates[field]["estimate"]),
                )
                for field in _ORBIT_RATE_FIELDS
            }
            orbit_row["fractional_rate_difference_from_reference"] = differences
            row["common_orbit_window"] = orbit_row

    return {
        "status": "common_resolved_interval_compared",
        "reference_label": reference["label"],
        "common_interval_start_myr": common_start,
        "common_interval_end_myr": common_end,
        "common_orbit_window_start_myr": common_orbit_start,
        "runs": rows,
        "interpretation": (
            "common-interval rates include reversible orbital-phase and interaction-energy "
            "variations; orbit-averaged estimates remain separate, and numerical convergence "
            "requires agreement of both diagnostics"
        ),
        "automatic_convergence_decision": False,
    }
