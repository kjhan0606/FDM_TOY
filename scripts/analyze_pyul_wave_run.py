#!/usr/bin/env python3
"""Measure conservation and binary separation in a PyUL_NBody run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fdm_smbh_delay.interaction import coupled_hamiltonian
from fdm_smbh_delay.orbital_exchange import keplerian_elements_from_relative_state
from fdm_smbh_delay.pyul import ordered_output_paths, output_index, pyul_unit_system


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _first_below(time: np.ndarray, value: np.ndarray, threshold: float) -> float | None:
    indices = np.flatnonzero(value < threshold)
    return None if indices.size == 0 else float(time[indices[0]])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-energy-error-over-transfer", type=float, default=0.01)
    args = parser.parse_args()
    if args.max_energy_error_over_transfer <= 0.0:
        raise ValueError("energy-error tolerance must be positive")
    run = args.run.expanduser().resolve()
    metadata = json.loads(
        (run / "fdm_adapter_metadata.json").read_text(encoding="utf-8")
    )
    config = json.loads((run / "config.uldm").read_text(encoding="utf-8"))
    particles = config["Matter Particles"]["Condition"]
    if len(particles) != 2:
        raise ValueError("binary diagnostic requires exactly two SMBHs")
    mass1 = float(particles[0][0])
    mass2 = float(particles[1][0])
    plummer_radius_pc = float(config["Matter Particles"]["Plummer Radius"])
    cell_size_pc = float(metadata["box_size_pc"]) / int(metadata["resolution"])

    units = pyul_unit_system(metadata)
    length_code_to_pc = units.length_pc
    energy_code_to_internal = units.energy_msun_pc2_myr2
    mass_code_to_msun = units.mass_msun
    mass1_code = mass1 / mass_code_to_msun
    mass2_code = mass2 / mass_code_to_msun
    plummer_radius_code = plummer_radius_pc / length_code_to_pc

    state_paths = ordered_output_paths(run / "Outputs" / "NBody", "NTM_#*.npy")
    states = [np.load(path) for path in state_paths]
    saved_wave_total = (
        np.load(run / "Outputs" / "egylist.npy") * energy_code_to_internal
    )
    wave_kinetic = (
        np.load(run / "Outputs" / "ekandqlist.npy") * energy_code_to_internal
    )
    wave_self_gravity = (
        np.load(run / "Outputs" / "egpsilist.npy") * energy_code_to_internal
    )
    wave_bh_interaction = (
        np.load(run / "Outputs" / "egpcmlist.npy") * energy_code_to_internal
    )
    point_interaction_estimator = (
        np.load(run / "Outputs" / "egpcmMlist.npy") * energy_code_to_internal
    )
    wave_intrinsic = wave_kinetic + wave_self_gravity
    wave_total = wave_intrinsic + wave_bh_interaction
    wave_mass = np.load(run / "Outputs" / "ULDMass.npy") * mass_code_to_msun
    if not (
        len(states)
        == wave_total.size
        == saved_wave_total.size
        == point_interaction_estimator.size
        == wave_mass.size
    ):
        raise ValueError("PyUL diagnostic arrays have inconsistent lengths")

    separation: list[float] = []
    bh_kinetic: list[float] = []
    bh_com_kinetic: list[float] = []
    binary_relative_kinetic: list[float] = []
    binary_mutual: list[float] = []
    binary_angular_momentum: list[float] = []
    osculating_semimajor_axis: list[float] = []
    osculating_eccentricity: list[float] = []
    for state in states:
        bodies = state.reshape(2, 6)
        positions_code = bodies[:, :3]
        velocities_code = bodies[:, 3:]
        distance_code = float(
            np.linalg.norm(positions_code[0] - positions_code[1])
        )
        kinetic_code = 0.5 * mass1_code * float(
            velocities_code[0] @ velocities_code[0]
        )
        kinetic_code += 0.5 * mass2_code * float(
            velocities_code[1] @ velocities_code[1]
        )
        total_mass_code = mass1_code + mass2_code
        reduced_mass_code = mass1_code * mass2_code / total_mass_code
        com_velocity_code = (
            mass1_code * velocities_code[0] + mass2_code * velocities_code[1]
        ) / total_mass_code
        com_kinetic_code = 0.5 * total_mass_code * float(
            com_velocity_code @ com_velocity_code
        )
        relative_velocity_code = velocities_code[0] - velocities_code[1]
        relative_position_code = positions_code[0] - positions_code[1]
        relative_kinetic_code = 0.5 * reduced_mass_code * float(
            relative_velocity_code @ relative_velocity_code
        )
        mutual_code = -mass1_code * mass2_code / np.sqrt(
            distance_code**2 + plummer_radius_code**2
        )
        separation.append(distance_code * length_code_to_pc)
        bh_kinetic.append(kinetic_code * energy_code_to_internal)
        bh_com_kinetic.append(com_kinetic_code * energy_code_to_internal)
        binary_relative_kinetic.append(
            relative_kinetic_code * energy_code_to_internal
        )
        binary_mutual.append(float(mutual_code * energy_code_to_internal))
        elements = keplerian_elements_from_relative_state(
            total_mass=total_mass_code,
            displacement=relative_position_code,
            relative_velocity=relative_velocity_code,
            gravitational_constant=1.0,
        )
        binary_angular_momentum.append(
            reduced_mass_code
            * float(np.linalg.norm(elements.specific_angular_momentum))
            * units.angular_momentum_msun_pc2_myr
        )
        osculating_semimajor_axis.append(
            np.nan
            if elements.semimajor_axis is None
            else elements.semimajor_axis * length_code_to_pc
        )
        osculating_eccentricity.append(elements.eccentricity)

    separation_array = np.asarray(separation)
    bh_kinetic_array = np.asarray(bh_kinetic)
    bh_com_kinetic_array = np.asarray(bh_com_kinetic)
    binary_orbital_energy = np.asarray(binary_relative_kinetic) + np.asarray(
        binary_mutual
    )
    binary_angular_momentum_array = np.asarray(binary_angular_momentum)
    osculating_semimajor_axis_array = np.asarray(osculating_semimajor_axis)
    osculating_eccentricity_array = np.asarray(osculating_eccentricity)
    finite_semimajor_axes = osculating_semimajor_axis_array[
        np.isfinite(osculating_semimajor_axis_array)
    ]
    bh_kinetic_decomposition_error = bh_kinetic_array - (
        bh_com_kinetic_array + np.asarray(binary_relative_kinetic)
    )
    combined_energy = coupled_hamiltonian(
        wave_kinetic=wave_kinetic,
        wave_self_gravity=wave_self_gravity,
        wave_bh_interaction=wave_bh_interaction,
        bh_kinetic=bh_kinetic_array,
        bh_mutual_gravity=np.asarray(binary_mutual),
    )
    energy_scale = max(
        abs(combined_energy[0]),
        abs(wave_total[0]),
        abs(binary_orbital_energy[0]),
        np.finfo(float).tiny,
    )
    transferred_energy_scale = max(
        np.max(np.abs(binary_orbital_energy - binary_orbital_energy[0])),
        np.max(np.abs(bh_com_kinetic_array - bh_com_kinetic_array[0])),
        np.max(np.abs(wave_intrinsic - wave_intrinsic[0])),
        np.max(np.abs(wave_bh_interaction - wave_bh_interaction[0])),
        np.finfo(float).tiny,
    )
    maximum_energy_error_over_transfer = float(
        np.max(np.abs(combined_energy - combined_energy[0]))
        / transferred_energy_scale
    )
    interaction_scale = np.maximum(
        np.maximum(
            np.abs(wave_bh_interaction), np.abs(point_interaction_estimator)
        ),
        np.finfo(float).tiny,
    )
    save_number = int(metadata.get("save_number", config["Save Options"]["Number"]))
    time = (
        float(metadata["duration_myr"])
        * np.asarray([output_index(path) for path in state_paths], dtype=float)
        / save_number
    )
    minimum_separation_pc = float(np.min(separation_array))
    plummer_force_fraction = minimum_separation_pc**3 / (
        minimum_separation_pc**2 + plummer_radius_pc**2
    ) ** 1.5
    summary = {
        "status": "diagnosed",
        "case_id": metadata["case_id"],
        "resolution": metadata["resolution"],
        "duration_myr": metadata["duration_myr"],
        "samples": len(states),
        "initial_separation_pc": float(separation_array[0]),
        "final_separation_pc": float(separation_array[-1]),
        "minimum_separation_pc": minimum_separation_pc,
        "fractional_separation_change": float(
            separation_array[-1] / separation_array[0] - 1.0
        ),
        "minimum_separation_over_plummer_radius": float(
            minimum_separation_pc / plummer_radius_pc
        ),
        "cell_size_pc": cell_size_pc,
        "plummer_radius_pc": plummer_radius_pc,
        "initial_separation_over_cell_size": float(
            separation_array[0] / cell_size_pc
        ),
        "minimum_separation_over_cell_size": float(
            minimum_separation_pc / cell_size_pc
        ),
        "first_time_below_two_cell_sizes_myr": _first_below(
            time, separation_array, 2.0 * cell_size_pc
        ),
        "first_time_below_one_cell_size_myr": _first_below(
            time, separation_array, cell_size_pc
        ),
        "minimum_plummer_to_newtonian_mutual_force_ratio": float(
            plummer_force_fraction
        ),
        "initial_osculating_semimajor_axis_pc": _finite_or_none(
            osculating_semimajor_axis_array[0]
        ),
        "final_osculating_semimajor_axis_pc": _finite_or_none(
            osculating_semimajor_axis_array[-1]
        ),
        "minimum_osculating_semimajor_axis_pc": (
            None
            if finite_semimajor_axes.size == 0
            else float(np.min(finite_semimajor_axes))
        ),
        "initial_osculating_eccentricity": float(
            osculating_eccentricity_array[0]
        ),
        "final_osculating_eccentricity": float(
            osculating_eccentricity_array[-1]
        ),
        "maximum_osculating_eccentricity": float(
            np.max(osculating_eccentricity_array)
        ),
        "fractional_angular_momentum_change": float(
            binary_angular_momentum_array[-1]
            / binary_angular_momentum_array[0]
            - 1.0
        ),
        "max_wave_mass_relative_error": float(
            np.max(np.abs(wave_mass / wave_mass[0] - 1.0))
        ),
        "max_combined_energy_relative_error": float(
            np.max(np.abs(combined_energy - combined_energy[0])) / energy_scale
        ),
        "max_total_energy_drift_over_energy_transfer": float(
            maximum_energy_error_over_transfer
        ),
        "maximum_energy_error_over_transfer_tolerance": float(
            args.max_energy_error_over_transfer
        ),
        "energy_transfer_conservation_passed": bool(
            maximum_energy_error_over_transfer
            <= args.max_energy_error_over_transfer
        ),
        "max_wave_energy_decomposition_relative_error": float(
            np.max(np.abs(saved_wave_total - wave_total))
            / max(np.max(np.abs(saved_wave_total)), np.finfo(float).tiny)
        ),
        "max_bh_kinetic_decomposition_relative_error": float(
            np.max(np.abs(bh_kinetic_decomposition_error))
            / max(np.max(np.abs(bh_kinetic_array)), np.finfo(float).tiny)
        ),
        "max_unaligned_point_estimator_relative_difference": float(
            np.max(
                np.abs(wave_bh_interaction - point_interaction_estimator)
                / interaction_scale
            )
        ),
        "interaction_energy_definition": "integral rho*Phi_BH dV, counted once",
        "point_estimator_role": (
            "gauge-dependent force diagnostic; not included in the Hamiltonian"
        ),
        "analytic_fdm_drag": metadata["analytic_fdm_drag"],
        "osculating_elements_role": (
            "point-mass diagnostics; orbit averaging is required because the "
            "FDM core and Plummer softening perturb the Kepler elements"
        ),
    }
    table = np.column_stack(
        (
            time,
            separation_array,
            wave_kinetic,
            wave_self_gravity,
            wave_intrinsic,
            wave_bh_interaction,
            point_interaction_estimator,
            binary_orbital_energy,
            binary_angular_momentum_array,
            osculating_semimajor_axis_array,
            osculating_eccentricity_array,
            bh_com_kinetic_array,
            bh_kinetic_array,
            np.asarray(binary_mutual),
            combined_energy,
            wave_mass,
        )
    )
    table_path = run / "conservation_timeseries.csv"
    np.savetxt(
        table_path,
        table,
        delimiter=",",
        header=(
            "time_myr,separation_pc,wave_kinetic_energy,wave_self_gravity_energy,"
            "wave_intrinsic_energy,wave_bh_interaction_grid,"
            "bh_wave_point_estimator,binary_orbital_energy,"
            "binary_angular_momentum_msun_pc2_myr,semimajor_axis_osculating_pc,"
            "eccentricity_osculating,"
            "bh_com_kinetic_energy,"
            "bh_total_kinetic_energy,bh_mutual_gravity_energy,combined_energy,"
            "wave_mass_msun"
        ),
        comments="",
    )
    output = run / "conservation_summary.json" if args.output is None else args.output
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
