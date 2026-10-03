#!/usr/bin/env python3
"""Measure orbit-resolved exchange rates in a completed PyUL_NBody run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fdm_smbh_delay.empirical import (
    koo_kepler_inferred_orbital_power,
    koo_q0_pc_m5half_myr,
    koo_separation_rate_at_separation_pc_myr,
)
from fdm_smbh_delay.pyul import ordered_output_paths, pyul_unit_system
from fdm_smbh_delay.secular_exchange import (
    moving_block_bootstrap_rate,
    phase_cycle_average,
    unwrapped_orbital_phase,
)


def _bootstrap_window(
    *,
    orbital_energy,
    angular_momentum,
    selection: slice,
    window_orbits: int,
    requested_block_orbits: int,
    samples: int,
    scope: str,
) -> dict | None:
    if window_orbits < 2 or samples <= 0:
        return None
    block_orbits = min(requested_block_orbits, window_orbits)
    power_interval = moving_block_bootstrap_rate(
        rate=orbital_energy.rate[selection],
        duration=orbital_energy.duration[selection],
        block_length=block_orbits,
        samples=samples,
    )
    torque_interval = moving_block_bootstrap_rate(
        rate=angular_momentum.rate[selection],
        duration=angular_momentum.duration[selection],
        block_length=block_orbits,
        samples=samples,
    )
    return {
        "window_orbits": window_orbits,
        "block_orbits": block_orbits,
        "samples": samples,
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
        "scope": scope,
    }


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
    config = json.loads((run / "config.uldm").read_text(encoding="utf-8"))
    cell_size_pc = float(metadata["box_size_pc"]) / int(metadata["resolution"])

    timeseries_path = run / "conservation_timeseries.csv"
    if not timeseries_path.exists():
        raise FileNotFoundError(
            "run scripts/analyze_pyul_wave_run.py before the secular analysis"
        )
    data = np.genfromtxt(timeseries_path, delimiter=",", names=True)
    state_paths = ordered_output_paths(run / "Outputs" / "NBody", "NTM_#*.npy")
    if len(state_paths) < data.shape[0]:
        raise ValueError("particle states are shorter than the energy table")
    states = np.asarray(
        [np.load(path).reshape(2, 6) for path in state_paths[: data.shape[0]]]
    )
    displacement = states[:, 0, :3] - states[:, 1, :3]
    relative_velocity = states[:, 0, 3:] - states[:, 1, 3:]
    units = pyul_unit_system(metadata)
    phase = unwrapped_orbital_phase(
        displacement,
        relative_velocity,
        time=data["time_myr"] / units.time_myr,
    )

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
    optional_nonfinite_fields = {"semimajor_axis_osculating_pc"}
    averaged = {
        source: phase_cycle_average(
            time=data["time_myr"],
            phase=phase,
            value=data[source],
            allow_nonfinite_value=source in optional_nonfinite_fields,
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
    finite_semimajor_axis_cycle = np.isfinite(
        averaged["semimajor_axis_osculating_pc"].mean_value
    )
    columns.append(finite_semimajor_axis_cycle.astype(int))
    header.append("mean_semimajor_axis_osculating_finite")
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
    wave_rotating_pattern_power = -frequency_torque
    wave_radial_residual_power = (
        -averaged["binary_orbital_energy"].rate - wave_rotating_pattern_power
    )
    simultaneous_losses = (
        averaged["binary_orbital_energy"].rate < 0.0
    ) & (
        averaged["binary_angular_momentum_msun_pc2_myr"].rate < 0.0
    )
    columns.extend(
        (
            orbital_frequency,
            exchange_mode_ratio,
            wave_rotating_pattern_power,
            wave_radial_residual_power,
        )
    )
    header.extend(
        (
            "orbital_frequency_myr_inverse",
            "orbital_power_over_frequency_times_torque",
            "wave_rotating_pattern_power",
            "wave_radial_residual_power",
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
    bootstrap_summary = _bootstrap_window(
        orbital_energy=orbital_energy,
        angular_momentum=angular_momentum,
        selection=slice(-window_orbits, None),
        window_orbits=window_orbits,
        requested_block_orbits=args.block_orbits,
        samples=args.bootstrap_samples,
        scope=(
            "correlated cycle variation in the final local window; excludes "
            "spatial-resolution systematics and secular variation outside the window"
        ),
    )
    underresolved_cycles = np.flatnonzero(mean_separation_over_cell < 2.0)
    initial_resolved_orbits = (
        reference.cycle_index.size
        if underresolved_cycles.size == 0
        else int(underresolved_cycles[0])
    )
    resolved_window_orbits = min(args.rate_window_orbits, initial_resolved_orbits)
    resolved_bootstrap_summary = _bootstrap_window(
        orbital_energy=orbital_energy,
        angular_momentum=angular_momentum,
        selection=slice(
            initial_resolved_orbits - resolved_window_orbits,
            initial_resolved_orbits,
        ),
        window_orbits=resolved_window_orbits,
        requested_block_orbits=args.block_orbits,
        samples=args.bootstrap_samples,
        scope=(
            "correlated cycle variation in the last initial window whose mean "
            "separation remains at least two cell widths; excludes convergence "
            "systematics between resolutions"
        ),
    )
    resolved_selection = slice(
        initial_resolved_orbits - resolved_window_orbits,
        initial_resolved_orbits,
    )
    resolved_losses = simultaneous_losses[resolved_selection]
    resolved_mode_ratio = exchange_mode_ratio[resolved_selection]
    resolved_radial_power = wave_radial_residual_power[resolved_selection]
    resolved_finite_ratio = np.isfinite(resolved_mode_ratio)
    resolved_mode_summary = None
    if resolved_window_orbits > 0:
        resolved_mode_summary = {
            "window_orbits": resolved_window_orbits,
            "fraction_with_energy_and_angular_momentum_loss": float(
                np.mean(resolved_losses)
            ),
            "median_power_over_frequency_times_torque": (
                None
                if not np.any(resolved_finite_ratio)
                else float(np.nanmedian(resolved_mode_ratio))
            ),
            "median_ratio_during_simultaneous_losses": (
                None
                if not np.any(resolved_losses & resolved_finite_ratio)
                else float(
                    np.nanmedian(
                        resolved_mode_ratio[resolved_losses & resolved_finite_ratio]
                    )
                )
            ),
            "fraction_of_simultaneous_losses_with_nonnegative_radial_residual": (
                None
                if not np.any(resolved_losses)
                else float(np.mean(resolved_radial_power[resolved_losses] >= 0.0))
            ),
        }
    published_fit_cross_check = None
    particles = config["Matter Particles"]["Condition"]
    solitons = config["ULDM Solitons"]["Condition"]
    if (
        str(metadata["case_id"]).startswith("koo_")
        and len(particles) == 2
        and len(solitons) == 1
        and np.isclose(float(particles[0][0]), float(particles[1][0]))
        and resolved_window_orbits > 0
    ):
        mass1 = float(particles[0][0])
        mass2 = float(particles[1][0])
        soliton_mass = float(solitons[0][0])
        q0 = koo_q0_pc_m5half_myr(
            soliton_mass_msun=soliton_mass,
            black_hole_mass_msun=mass1,
            particle_mass_ev=float(metadata["particle_mass_ev"]),
        )
        resolved_durations = reference.duration[resolved_selection]
        mean_separation = float(
            np.sum(
                averaged["separation_pc"].mean_value[resolved_selection]
                * resolved_durations
            )
            / np.sum(resolved_durations)
        )
        reference_power = koo_kepler_inferred_orbital_power(
            separation_pc=mean_separation,
            mass1_msun=mass1,
            mass2_msun=mass2,
            q0=q0,
        )
        measured_power = (
            None
            if resolved_bootstrap_summary is None
            else resolved_bootstrap_summary["orbital_power"]["estimate"]
        )
        published_fit_cross_check = {
            "reference": "Koo et al. (2024), equation 18",
            "window_orbits": resolved_window_orbits,
            "time_weighted_mean_separation_pc": mean_separation,
            "separation_rate_pc_myr": (
                koo_separation_rate_at_separation_pc_myr(mean_separation, q0)
            ),
            "kepler_inferred_orbital_power": reference_power,
            "measured_to_kepler_inferred_power_ratio": (
                None if measured_power is None else measured_power / reference_power
            ),
            "interpretation": (
                "the published separation fit is mapped through an isolated "
                "circular Kepler relation; it does not measure energy deposited "
                "in the live FDM wave"
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
        "initial_cycles_before_first_underresolved_orbit": int(
            initial_resolved_orbits
        ),
        "osculating_semimajor_axis": {
            "role": (
                "point-mass Kepler diagnostic; undefined values are retained "
                "when the instantaneous relative state is not point-mass bound"
            ),
            "finite_input_samples": int(
                np.count_nonzero(
                    np.isfinite(data["semimajor_axis_osculating_pc"])
                )
            ),
            "total_input_samples": int(data.shape[0]),
            "cycles_with_finite_mean": int(
                np.count_nonzero(finite_semimajor_axis_cycle)
            ),
            "total_cycles": int(reference.cycle_index.size),
        },
        "median_orbital_power_over_frequency_times_torque": (
            None
            if np.all(~np.isfinite(exchange_mode_ratio))
            else float(np.nanmedian(exchange_mode_ratio))
        ),
        "fraction_of_cycles_with_energy_and_angular_momentum_loss": float(
            np.mean(simultaneous_losses)
        ),
        "median_exchange_mode_ratio_during_simultaneous_losses": (
            None
            if not np.any(simultaneous_losses & np.isfinite(exchange_mode_ratio))
            else float(
                np.nanmedian(
                    exchange_mode_ratio[
                        simultaneous_losses & np.isfinite(exchange_mode_ratio)
                    ]
                )
            )
        ),
        "fraction_of_simultaneous_losses_with_nonnegative_radial_residual": (
            None
            if not np.any(simultaneous_losses)
            else float(
                np.mean(wave_radial_residual_power[simultaneous_losses] >= 0.0)
            )
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
        "initial_resolved_window_block_bootstrap": resolved_bootstrap_summary,
        "initial_resolved_window_mode_diagnostic": resolved_mode_summary,
        "published_separation_fit_cross_check": published_fit_cross_check,
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
