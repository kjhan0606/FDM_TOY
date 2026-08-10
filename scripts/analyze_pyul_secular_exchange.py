#!/usr/bin/env python3
"""Measure orbit-resolved exchange rates in a completed PyUL_NBody run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fdm_smbh_delay.pyul import ordered_output_paths
from fdm_smbh_delay.secular_exchange import (
    phase_cycle_average,
    unwrapped_orbital_phase,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
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
        "mean_orbital_power": float(
            (orbital_energy.end_value[-1] - orbital_energy.start_value[0])
            / (orbital_energy.end_time[-1] - orbital_energy.start_time[0])
        ),
        "mean_orbital_torque": float(
            (angular_momentum.end_value[-1] - angular_momentum.start_value[0])
            / (angular_momentum.end_time[-1] - angular_momentum.start_time[0])
        ),
        "energy_ledger": (
            "orbital, wave intrinsic, wave-SMBH interaction, SMBH centre-of-mass, "
            "and combined-Hamiltonian residual rates are retained separately"
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
