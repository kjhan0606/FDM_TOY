#!/usr/bin/env python3
"""Measure orbit-resolved exchange rates in a completed PyUL_NBody run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fdm_smbh_delay.pyul import ordered_output_paths
from fdm_smbh_delay.secular_exchange import (
    moving_block_bootstrap_rate,
    phase_cycle_average,
    unwrapped_orbital_phase,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--rate-window-orbits", type=int, default=32)
    parser.add_argument("--block-orbits", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.rate_window_orbits < 2:
        raise ValueError("--rate-window-orbits must be at least two")
    if args.block_orbits < 1:
        raise ValueError("--block-orbits must be positive")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    run = args.run.expanduser().resolve()
    metadata = json.loads(
        (run / "fdm_adapter_metadata.json").read_text(encoding="utf-8")
    )
    cell_size_pc = float(metadata["box_size_pc"]) / int(metadata["resolution"])

    timeseries_path = run / "conservation_timeseries.csv"
    if not timeseries_path.exists():
        raise FileNotFoundError(
            "run scripts/analyze_pyul_wave_run.py before the secular analysis"
        )
    data = np.genfromtxt(timeseries_path, delimiter=",", names=True)
    states = np.asarray(
        [
            np.load(path).reshape(2, 6)
            for path in ordered_output_paths(run / "Outputs" / "NBody", "NTM_#*.npy")
        ]
    )
    if states.shape[0] != data.shape[0]:
        raise ValueError("particle states and energy table have different lengths")
    displacement = states[:, 0, :3] - states[:, 1, :3]
    relative_velocity = states[:, 0, 3:] - states[:, 1, 3:]
    phase = unwrapped_orbital_phase(displacement, relative_velocity)

    fields = {
        "separation_pc": "mean_separation_pc",
        "semimajor_axis_osculating_pc": "mean_semimajor_axis_osculating_pc",
        "eccentricity_osculating": "mean_eccentricity_osculating",
        "binary_orbital_energy": "orbital_power",
        "binary_angular_momentum_msun_pc2_myr": "orbital_torque",
        "wave_intrinsic_energy": "wave_intrinsic_energy_rate",
        "wave_bh_interaction_grid": "wave_bh_interaction_energy_rate",
        "bh_com_kinetic_energy": "bh_com_kinetic_energy_rate",
        "combined_energy": "combined_energy_residual_rate",
    }
    averaged = {
        source: phase_cycle_average(
            time=data["time_myr"], phase=phase, value=data[source]
        )
        for source in fields
    }
    reference = averaged["separation_pc"]
    columns = [
        reference.cycle_index,
        reference.start_time,
        reference.end_time,
        reference.mean_time,
        reference.duration,
    ]
    header = [
        "cycle",
        "start_time_myr",
        "end_time_myr",
        "mean_time_myr",
        "orbital_period_myr",
    ]
    for source, output_name in fields.items():
        series = averaged[source]
        if output_name.startswith("mean_"):
            columns.append(series.mean_value)
        else:
            columns.append(series.rate)
        header.append(output_name)
    mean_separation_over_cell = (
        averaged["separation_pc"].mean_value / cell_size_pc
    )
    columns.append(mean_separation_over_cell)
    header.append("mean_separation_over_cell_size")
    orbital_frequency = 2.0 * np.pi / reference.duration
    frequency_torque = orbital_frequency * averaged[
        "binary_angular_momentum_msun_pc2_myr"
    ].rate
    exchange_mode_ratio = np.full(reference.cycle_index.size, np.nan)
    nonzero_torque = np.abs(frequency_torque) > np.finfo(float).tiny
    exchange_mode_ratio[nonzero_torque] = averaged[
        "binary_orbital_energy"
    ].rate[nonzero_torque] / frequency_torque[nonzero_torque]
    columns.extend((orbital_frequency, exchange_mode_ratio))
    header.extend(
        (
            "orbital_frequency_myr_inverse",
            "orbital_power_over_frequency_times_torque",
        )
    )

    output_path = run / "orbit_averaged_exchange.csv"
    np.savetxt(
        output_path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(header),
        comments="",
    )
    orbital_energy = averaged["binary_orbital_energy"]
    angular_momentum = averaged["binary_angular_momentum_msun_pc2_myr"]
    window_orbits = min(args.rate_window_orbits, reference.cycle_index.size)
    bootstrap_summary = None
    if window_orbits >= 2 and args.bootstrap_samples > 0:
        selection = slice(-window_orbits, None)
        block_orbits = min(args.block_orbits, window_orbits)
        power_interval = moving_block_bootstrap_rate(
            rate=orbital_energy.rate[selection],
            duration=orbital_energy.duration[selection],
            block_length=block_orbits,
            samples=args.bootstrap_samples,
        )
        torque_interval = moving_block_bootstrap_rate(
            rate=angular_momentum.rate[selection],
            duration=angular_momentum.duration[selection],
            block_length=block_orbits,
            samples=args.bootstrap_samples,
        )
        bootstrap_summary = {
            "window_orbits": window_orbits,
            "block_orbits": block_orbits,
            "samples": args.bootstrap_samples,
            "orbital_power": {
                "estimate": power_interval.estimate,
                "lower_95": power_interval.lower_95,
                "upper_95": power_interval.upper_95,
            },
            "orbital_torque": {
                "estimate": torque_interval.estimate,
                "lower_95": torque_interval.lower_95,
                "upper_95": torque_interval.upper_95,
            },
            "scope": (
                "correlated cycle variation in the final local window; excludes "
                "spatial-resolution systematics and secular variation outside the window"
            ),
        }
    summary = {
        "status": "orbit_averaged",
        "complete_orbits": int(reference.cycle_index.size),
        "start_time_myr": float(reference.start_time[0]),
        "end_time_myr": float(reference.end_time[-1]),
        "median_orbital_period_myr": float(np.median(reference.duration)),
        "cell_size_pc": cell_size_pc,
        "minimum_cycle_mean_separation_over_cell_size": float(
            np.min(mean_separation_over_cell)
        ),
        "cycles_with_mean_separation_above_two_cells": int(
            np.count_nonzero(mean_separation_over_cell >= 2.0)
        ),
        "median_orbital_power_over_frequency_times_torque": (
            None
            if np.all(~np.isfinite(exchange_mode_ratio))
            else float(np.nanmedian(exchange_mode_ratio))
        ),
        "mean_orbital_power": float(
            (orbital_energy.end_value[-1] - orbital_energy.start_value[0])
            / (orbital_energy.end_time[-1] - orbital_energy.start_time[0])
        ),
        "mean_orbital_torque": float(
            (angular_momentum.end_value[-1] - angular_momentum.start_value[0])
            / (angular_momentum.end_time[-1] - angular_momentum.start_time[0])
        ),
        "late_window_block_bootstrap": bootstrap_summary,
        "energy_ledger": (
            "orbital, wave intrinsic, wave-SMBH interaction, SMBH centre-of-mass, "
            "and combined-Hamiltonian residual rates are retained separately"
        ),
        "exchange_mode_ratio_role": (
            "a rigidly rotating perturbation gives power/(orbital frequency "
            "times torque)=1; departures require additional harmonics or radial response"
        ),
    }
    summary_path = run / "orbit_averaged_exchange_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
