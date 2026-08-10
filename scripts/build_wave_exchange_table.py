#!/usr/bin/env python3
"""Combine orbit-resolved PyUL measurements into a dimensionless table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fdm_smbh_delay.exchange_scaling import exchange_scales


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


def _nearest_mode_state(
    response: np.ndarray | None, time_myr: float, orbital_period_myr: float
) -> dict[str, float]:
    values = {column: np.nan for column in _MODE_COLUMNS}
    values["wave_mode_sample_time_offset_over_orbital_period"] = np.nan
    if response is None:
        return values
    index = int(np.argmin(np.abs(response["time_myr"] - time_myr)))
    available = set(response.dtype.names or ())
    for column in _MODE_COLUMNS:
        if column in available:
            values[column] = float(response[column][index])
    values["wave_mode_sample_time_offset_over_orbital_period"] = float(
        (response["time_myr"][index] - time_myr) / orbital_period_myr
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
        for cycle in table:
            mode_state = _nearest_mode_state(
                response,
                float(cycle["mean_time_myr"]),
                float(cycle["orbital_period_myr"]),
            )
            rows.append(
                {
                    "case_id": metadata["case_id"],
                    "resolution": int(metadata["resolution"]),
                    "cycle": int(cycle["cycle"]),
                    "particle_mass_ev": float(metadata["particle_mass_ev"]),
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
                    **mode_state,
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
            "no physical or numerical selection applied; spatial-scale columns "
            "must be used before fitting"
        ),
        "exchange_mode_diagnostic": (
            "power/(orbital frequency times torque) equals one for exchange "
            "through one rigidly rotating pattern"
        ),
        "wave_mode_state": (
            "nearest saved three-dimensional density multipoles; the time "
            "offset in orbital periods must be small before phase-dependent "
            "fits are attempted"
        ),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
