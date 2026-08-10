#!/usr/bin/env python3
"""Combine orbit-resolved PyUL measurements into a dimensionless table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fdm_smbh_delay.exchange_scaling import (
    exchange_scales,
    schrodinger_poisson_similarity_parameter,
)
from fdm_smbh_delay.orbital_exchange import orbital_frame_from_relative_state
from fdm_smbh_delay.pyul import ordered_output_paths, pyul_unit_system
from fdm_smbh_delay.wave_response import (
    MultipoleAmplitudes,
    rotate_multipoles_to_frame,
)


_MODE_REGIONS = (
    "core",
    "near",
    "outer",
    "binary_core",
    "binary_near",
    "binary_outer",
)
_MODE_COEFFICIENTS = (
    "l1_m0_real",
    "l1_m0_imag",
    "l1_m1_real",
    "l1_m1_imag",
    "l2_m0_real",
    "l2_m0_imag",
    "l2_m1_real",
    "l2_m1_imag",
    "l2_m2_real",
    "l2_m2_imag",
)
_MODE_COLUMNS = tuple(
    f"{region}_{quantity}"
    for region in _MODE_REGIONS
    for quantity in ("l1_fraction", "l2_fraction", *_MODE_COEFFICIENTS)
)
_ORBITAL_MODE_COLUMNS = tuple(f"orbital_{column}" for column in _MODE_COLUMNS)
_WAVE_STATE_COLUMNS = (
    "wave_binary_com_offset_pc",
    "central_density_msun_pc3",
    "core_radius_pc",
    "outer_mass_msun",
    "outer_intrinsic_energy",
    "mass_flux_2rc_msun_myr",
    "mass_flux_4rc_msun_myr",
    "mass_flux_8rc_msun_myr",
    "schrodinger_energy_flux_2rc",
    "schrodinger_energy_flux_4rc",
    "schrodinger_energy_flux_8rc",
)
_ORBITAL_FRAME_COLUMNS = tuple(
    f"orbital_{axis}_unit_{component}"
    for axis in ("radial", "tangential", "normal")
    for component in ("x", "y", "z")
)


def _nearest_wave_state(
    response: np.ndarray | None, time_myr: float, orbital_period_myr: float
) -> dict[str, float]:
    values = {
        column: np.nan for column in (*_WAVE_STATE_COLUMNS, *_MODE_COLUMNS)
    }
    values["wave_mode_sample_time_offset_over_orbital_period"] = np.nan
    values["wave_snapshot_time_myr"] = np.nan
    if response is None:
        return values
    index = int(np.argmin(np.abs(response["time_myr"] - time_myr)))
    values["wave_snapshot_time_myr"] = float(response["time_myr"][index])
    available = set(response.dtype.names or ())
    for column in (*_WAVE_STATE_COLUMNS, *_MODE_COLUMNS):
        if column in available:
            values[column] = float(response[column][index])
    values["wave_mode_sample_time_offset_over_orbital_period"] = float(
        (response["time_myr"][index] - time_myr) / orbital_period_myr
    )
    return values


def _orbital_frame_state(
    *,
    wave_time_myr: float,
    state_times_myr: np.ndarray | None,
    states: np.ndarray | None,
    box_size_code: float | None,
) -> dict[str, float]:
    values = {column: np.nan for column in _ORBITAL_FRAME_COLUMNS}
    if (
        not np.isfinite(wave_time_myr)
        or state_times_myr is None
        or states is None
        or box_size_code is None
    ):
        return values
    index = int(np.argmin(np.abs(state_times_myr - wave_time_myr)))
    bodies = states[index]
    displacement = bodies[0, :3] - bodies[1, :3]
    displacement -= box_size_code * np.floor(
        displacement / box_size_code + 0.5
    )
    relative_velocity = bodies[0, 3:] - bodies[1, 3:]
    frame = orbital_frame_from_relative_state(displacement, relative_velocity)
    for axis, vector in (
        ("radial", frame.radial_unit),
        ("tangential", frame.tangential_unit),
        ("normal", frame.normal_unit),
    ):
        for component, value in zip(("x", "y", "z"), vector, strict=True):
            values[f"orbital_{axis}_unit_{component}"] = float(value)
    return values


def _multipoles_in_orbital_frame(
    wave_snapshot: dict[str, float], frame_state: dict[str, float]
) -> dict[str, float]:
    values = {column: np.nan for column in _ORBITAL_MODE_COLUMNS}
    radial = np.asarray(
        [frame_state[f"orbital_radial_unit_{component}"] for component in "xyz"]
    )
    tangential = np.asarray(
        [
            frame_state[f"orbital_tangential_unit_{component}"]
            for component in "xyz"
        ]
    )
    normal = np.asarray(
        [frame_state[f"orbital_normal_unit_{component}"] for component in "xyz"]
    )
    if np.any(~np.isfinite(np.concatenate((radial, tangential, normal)))):
        return values
    for region in _MODE_REGIONS:
        multipoles = MultipoleAmplitudes(
            mass=1.0,
            l1_fraction=wave_snapshot[f"{region}_l1_fraction"],
            l2_fraction=wave_snapshot[f"{region}_l2_fraction"],
            l1_m0=complex(
                wave_snapshot[f"{region}_l1_m0_real"],
                wave_snapshot[f"{region}_l1_m0_imag"],
            ),
            l1_m1=complex(
                wave_snapshot[f"{region}_l1_m1_real"],
                wave_snapshot[f"{region}_l1_m1_imag"],
            ),
            l2_m0=complex(
                wave_snapshot[f"{region}_l2_m0_real"],
                wave_snapshot[f"{region}_l2_m0_imag"],
            ),
            l2_m1=complex(
                wave_snapshot[f"{region}_l2_m1_real"],
                wave_snapshot[f"{region}_l2_m1_imag"],
            ),
            l2_m2=complex(
                wave_snapshot[f"{region}_l2_m2_real"],
                wave_snapshot[f"{region}_l2_m2_imag"],
            ),
        )
        rotated = rotate_multipoles_to_frame(
            multipoles, radial, tangential, normal
        )
        values[f"orbital_{region}_l1_fraction"] = rotated.l1_fraction
        values[f"orbital_{region}_l2_fraction"] = rotated.l2_fraction
        for ell, orders in ((1, (0, 1)), (2, (0, 1, 2))):
            for order in orders:
                coefficient = getattr(rotated, f"l{ell}_m{order}")
                values[f"orbital_{region}_l{ell}_m{order}_real"] = float(
                    coefficient.real
                )
                values[f"orbital_{region}_l{ell}_m{order}_imag"] = float(
                    coefficient.imag
                )
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, float | int | str]] = []
    for run_argument in args.runs:
        run = run_argument.expanduser().resolve()
        metadata = json.loads(
            (run / "fdm_adapter_metadata.json").read_text(encoding="utf-8")
        )
        config = json.loads((run / "config.uldm").read_text(encoding="utf-8"))
        particles = config["Matter Particles"]["Condition"]
        if len(particles) != 2:
            raise ValueError(f"{run} does not contain an SMBH binary")
        mass1, mass2 = (float(particle[0]) for particle in particles)
        solitons = config["ULDM Solitons"]["Condition"]
        if len(solitons) != 1:
            raise ValueError(f"{run} must contain one calibration soliton")
        soliton_mass = float(solitons[0][0])
        core_radius = float(metadata["core_radius_reference_pc"])
        scales = exchange_scales(
            mass1_msun=mass1,
            mass2_msun=mass2,
            soliton_mass_msun=soliton_mass,
            core_radius_pc=core_radius,
        )
        similarity_parameter = schrodinger_poisson_similarity_parameter(
            particle_mass_ev=float(metadata["particle_mass_ev"]),
            soliton_mass_msun=soliton_mass,
            core_radius_pc=core_radius,
        )
        conservation_summary = json.loads(
            (run / "conservation_summary.json").read_text(encoding="utf-8")
        )
        energy_conservation_passed = bool(
            conservation_summary["energy_transfer_conservation_passed"]
        )
        initial_resolved_energy_passed = bool(
            conservation_summary["initial_resolved_energy_conservation_passed"]
        )
        cell_size = float(metadata["box_size_pc"]) / int(metadata["resolution"])
        table = np.genfromtxt(
            run / "orbit_averaged_exchange.csv",
            delimiter=",",
            names=True,
            ndmin=1,
        )
        response_path = run / "wave_response_timeseries.csv"
        response = (
            np.genfromtxt(response_path, delimiter=",", names=True, ndmin=1)
            if response_path.is_file()
            else None
        )
        state_times_myr = None
        states = None
        box_size_code = None
        if response is not None:
            conservation = np.genfromtxt(
                run / "conservation_timeseries.csv",
                delimiter=",",
                names=True,
                ndmin=1,
            )
            state_paths = ordered_output_paths(
                run / "Outputs" / "NBody", "NTM_#*.npy"
            )
            if len(state_paths) != conservation.size:
                raise ValueError("SMBH states and conservation times are inconsistent")
            states = np.asarray(
                [np.load(path).reshape(2, 6) for path in state_paths]
            )
            state_times_myr = conservation["time_myr"]
            box_size_code = float(metadata["box_size_pc"]) / pyul_unit_system(
                metadata
            ).length_pc
        initial_resolved = np.minimum.accumulate(
            table["mean_separation_over_cell_size"] >= 2.0
        )
        for row_index, cycle in enumerate(table):
            wave_snapshot = _nearest_wave_state(
                response,
                float(cycle["mean_time_myr"]),
                float(cycle["orbital_period_myr"]),
            )
            initial_central_density = (
                np.nan
                if response is None
                else float(response["central_density_msun_pc3"][0])
            )
            measured_half_density_radius = wave_snapshot.pop("core_radius_pc")
            wave_snapshot_time = wave_snapshot.pop("wave_snapshot_time_myr")
            frame_state = _orbital_frame_state(
                wave_time_myr=wave_snapshot_time,
                state_times_myr=state_times_myr,
                states=states,
                box_size_code=box_size_code,
            )
            orbital_modes = _multipoles_in_orbital_frame(
                wave_snapshot, frame_state
            )
            wave_state = {
                "wave_binary_com_offset_over_core_radius": (
                    wave_snapshot.pop("wave_binary_com_offset_pc") / core_radius
                ),
                "central_density_over_initial": (
                    wave_snapshot.pop("central_density_msun_pc3")
                    / initial_central_density
                ),
                "measured_half_density_radius_over_reference": (
                    measured_half_density_radius / core_radius
                ),
                "measured_half_density_radius_over_cell_size": (
                    measured_half_density_radius / cell_size
                ),
                "half_density_radius_spatially_resolved": int(
                    measured_half_density_radius >= 2.0 * cell_size
                ),
                "outer_mass_fraction": (
                    wave_snapshot.pop("outer_mass_msun") / soliton_mass
                ),
                "dimensionless_outer_intrinsic_energy": (
                    wave_snapshot.pop("outer_intrinsic_energy")
                    / scales.orbital_energy_msun_pc2_myr2
                ),
                **{
                    f"dimensionless_mass_flux_{factor}rc": (
                        wave_snapshot.pop(f"mass_flux_{factor}rc_msun_myr")
                        * scales.soliton_dynamical_time_myr
                        / soliton_mass
                    )
                    for factor in (2, 4, 8)
                },
                **{
                    f"dimensionless_schrodinger_energy_flux_{factor}rc": (
                        wave_snapshot.pop(f"schrodinger_energy_flux_{factor}rc")
                        / scales.orbital_power_msun_pc2_myr3
                    )
                    for factor in (2, 4, 8)
                },
            }
            spatially_resolved = bool(
                cycle["mean_separation_over_cell_size"] >= 2.0
            )
            rows.append(
                {
                    "case_id": metadata["case_id"],
                    "resolution": int(metadata["resolution"]),
                    "cycle": int(cycle["cycle"]),
                    "particle_mass_ev": float(metadata["particle_mass_ev"]),
                    "schrodinger_poisson_similarity_parameter": (
                        similarity_parameter
                    ),
                    "mass_ratio_q": min(mass1, mass2) / max(mass1, mass2),
                    "binary_to_soliton_mass": (mass1 + mass2) / soliton_mass,
                    "mean_time_over_soliton_dynamical_time": cycle[
                        "mean_time_myr"
                    ]
                    / scales.soliton_dynamical_time_myr,
                    "mean_separation_over_core_radius": cycle[
                        "mean_separation_pc"
                    ]
                    / core_radius,
                    "mean_semimajor_axis_over_core_radius": cycle[
                        "mean_semimajor_axis_osculating_pc"
                    ]
                    / core_radius,
                    "mean_eccentricity_osculating": cycle[
                        "mean_eccentricity_osculating"
                    ],
                    "mean_separation_over_cell_size": cycle[
                        "mean_separation_over_cell_size"
                    ],
                    "orbital_power_over_frequency_times_torque": cycle[
                        "orbital_power_over_frequency_times_torque"
                    ],
                    "plummer_radius_over_core_radius": float(
                        metadata["plummer_radius_pc"]
                    )
                    / core_radius,
                    "dimensionless_orbital_power": cycle["orbital_power"]
                    / scales.orbital_power_msun_pc2_myr3,
                    "dimensionless_orbital_torque": cycle["orbital_torque"]
                    / scales.orbital_torque_msun_pc2_myr2,
                    "dimensionless_wave_rotating_pattern_power": cycle[
                        "wave_rotating_pattern_power"
                    ]
                    / scales.orbital_power_msun_pc2_myr3,
                    "dimensionless_wave_radial_residual_power": cycle[
                        "wave_radial_residual_power"
                    ]
                    / scales.orbital_power_msun_pc2_myr3,
                    "dimensionless_wave_intrinsic_energy_rate": cycle[
                        "wave_intrinsic_energy_rate"
                    ]
                    / scales.orbital_power_msun_pc2_myr3,
                    "dimensionless_wave_bh_interaction_energy_rate": cycle[
                        "wave_bh_interaction_energy_rate"
                    ]
                    / scales.orbital_power_msun_pc2_myr3,
                    "dimensionless_bh_com_kinetic_energy_rate": cycle[
                        "bh_com_kinetic_energy_rate"
                    ]
                    / scales.orbital_power_msun_pc2_myr3,
                    "dimensionless_hamiltonian_residual_rate": cycle[
                        "combined_energy_residual_rate"
                    ]
                    / scales.orbital_power_msun_pc2_myr3,
                    "soliton_dynamical_time_myr": scales.soliton_dynamical_time_myr,
                    "cell_size_pc": cell_size,
                    "minimum_spatial_sampling_passed": int(spatially_resolved),
                    "before_first_underresolved_orbit": int(
                        initial_resolved[row_index]
                    ),
                    "run_energy_conservation_passed": int(
                        energy_conservation_passed
                    ),
                    "initial_resolved_energy_conservation_passed": int(
                        initial_resolved_energy_passed
                    ),
                    "provisional_numerical_acceptance": int(
                        initial_resolved[row_index]
                        and initial_resolved_energy_passed
                    ),
                    **wave_state,
                    **frame_state,
                    **orbital_modes,
                    **wave_snapshot,
                }
            )
    if not rows:
        raise ValueError("no orbit-resolved measurements were supplied")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "dimensionless_exchange_table",
        "runs": len(args.runs),
        "cycles": len(rows),
        "case_ids": sorted({str(row["case_id"]) for row in rows}),
        "normalization": {
            "time": "sqrt(r_core^3/(G*M_soliton))",
            "energy": "G*M1*M2/r_core",
            "angular_momentum": "mu*sqrt(G*M_binary*r_core)",
        },
        "selection": (
            "rows are retained without deletion; provisional numerical "
            "acceptance ends at the first orbit below two cell widths and "
            "requires the initial resolved interval to pass the Hamiltonian "
            "limit; calibration still requires convergence between spatial and "
            "temporal resolutions"
        ),
        "exchange_mode_diagnostic": (
            "power/(orbital frequency times torque) equals one for exchange "
            "through one rigidly rotating pattern"
        ),
        "wave_mode_state": (
            "nearest saved three-dimensional density multipoles, core state, "
            "and radial fluxes; the time offset in orbital periods must be "
            "small before phase-dependent fits are attempted"
        ),
        "half_density_radius_rule": (
            "the measured half-density radius is provisionally resolved only "
            "when it spans at least two cell widths"
        ),
        "similarity_parameter": "hbar^2/(G*m^2*M_soliton*r_core)",
        "orbital_frame_state": (
            "right-handed radial, tangential, and normal unit vectors at the "
            "saved three-dimensional wave state; rotate complex multipoles "
            "into this frame before fitting phase-dependent transfer"
        ),
        "orbital_frame_multipoles": (
            "complex density multipoles rotated into the instantaneous binary "
            "frame; each invariant l amplitude must match its unrotated value"
        ),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
