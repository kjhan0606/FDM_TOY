#!/usr/bin/env python3
"""Measure conservation and binary separation in a PyUL_NBody run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.constants import G

from fdm_smbh_delay.interaction import coupled_hamiltonian


def _ordered_arrays(directory: Path, pattern: str) -> list[np.ndarray]:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no arrays match {directory / pattern}")
    return [np.load(path) for path in paths]


def _pyul_units(metadata: dict) -> tuple[float, float, float, float]:
    keys = (
        "pyul_length_unit_m",
        "pyul_time_unit_s",
        "pyul_mass_unit_kg",
        "pyul_energy_unit_j",
    )
    if all(key in metadata for key in keys):
        return tuple(float(metadata[key]) for key in keys)
    # Exact constants and normalization used by PyUL_NBody Current.py v30.37.
    axion_mass = metadata["particle_mass_ev"] * 1.78266191e-36
    hbar_si = 1.0545718e-34
    parsec_m = 3.0857e16
    gravitational_si = 6.67e-11
    omega_m = 0.31
    hubble_si = 67.7 / (parsec_m * 1.0e3)
    time_unit = (3.0 * hubble_si**2 * omega_m / (8.0 * np.pi)) ** -0.5
    length_unit = (
        8.0
        * np.pi
        * hbar_si**2
        / (3.0 * axion_mass**2 * hubble_si**2 * omega_m)
    ) ** 0.25
    mass_unit = (
        (3.0 * hubble_si**2 * omega_m / (8.0 * np.pi)) ** 0.25
        * hbar_si**1.5
        / (axion_mass**1.5 * gravitational_si)
    )
    energy_unit = mass_unit * length_unit**2 / time_unit**2
    return length_unit, time_unit, mass_unit, energy_unit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
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

    length_unit_m, time_unit_s, mass_unit_kg, energy_unit_j = _pyul_units(metadata)
    length_code_to_pc = (length_unit_m * u.m).to_value(u.pc)
    velocity_code_to_pc_myr = (
        length_unit_m / time_unit_s * u.m / u.s
    ).to_value(u.pc / u.Myr)
    energy_internal_j = (1.0 * u.Msun * (u.pc / u.Myr) ** 2).to_value(u.J)
    energy_code_to_internal = energy_unit_j / energy_internal_j
    mass_code_to_msun = (mass_unit_kg * u.kg).to_value(u.Msun)
    gravitational_constant = G.to_value(u.pc**3 / (u.Msun * u.Myr**2))

    states = _ordered_arrays(run / "Outputs" / "NBody", "NTM_#*.npy")
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
    for state in states:
        bodies = state.reshape(2, 6)
        positions = bodies[:, :3] * length_code_to_pc
        velocities = bodies[:, 3:] * velocity_code_to_pc_myr
        distance = float(np.linalg.norm(positions[0] - positions[1]))
        kinetic = 0.5 * mass1 * float(velocities[0] @ velocities[0])
        kinetic += 0.5 * mass2 * float(velocities[1] @ velocities[1])
        total_mass = mass1 + mass2
        reduced_mass = mass1 * mass2 / total_mass
        com_velocity = (mass1 * velocities[0] + mass2 * velocities[1]) / total_mass
        com_kinetic = 0.5 * total_mass * float(com_velocity @ com_velocity)
        relative_velocity = velocities[0] - velocities[1]
        relative_kinetic = 0.5 * reduced_mass * float(
            relative_velocity @ relative_velocity
        )
        mutual = -gravitational_constant * mass1 * mass2 / np.sqrt(
            distance**2 + plummer_radius_pc**2
        )
        separation.append(distance)
        bh_kinetic.append(kinetic)
        bh_com_kinetic.append(com_kinetic)
        binary_relative_kinetic.append(relative_kinetic)
        binary_mutual.append(float(mutual))

    separation_array = np.asarray(separation)
    bh_kinetic_array = np.asarray(bh_kinetic)
    bh_com_kinetic_array = np.asarray(bh_com_kinetic)
    binary_orbital_energy = np.asarray(binary_relative_kinetic) + np.asarray(
        binary_mutual
    )
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
    interaction_scale = np.maximum(
        np.maximum(
            np.abs(wave_bh_interaction), np.abs(point_interaction_estimator)
        ),
        np.finfo(float).tiny,
    )
    time = np.linspace(0.0, metadata["duration_myr"], len(states))
    summary = {
        "status": "diagnosed",
        "case_id": metadata["case_id"],
        "resolution": metadata["resolution"],
        "duration_myr": metadata["duration_myr"],
        "samples": len(states),
        "initial_separation_pc": float(separation_array[0]),
        "final_separation_pc": float(separation_array[-1]),
        "fractional_separation_change": float(
            separation_array[-1] / separation_array[0] - 1.0
        ),
        "max_wave_mass_relative_error": float(
            np.max(np.abs(wave_mass / wave_mass[0] - 1.0))
        ),
        "max_combined_energy_relative_error": float(
            np.max(np.abs(combined_energy - combined_energy[0])) / energy_scale
        ),
        "max_total_energy_drift_over_energy_transfer": float(
            np.max(np.abs(combined_energy - combined_energy[0]))
            / transferred_energy_scale
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
            "bh_wave_point_estimator,binary_orbital_energy,bh_com_kinetic_energy,"
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
