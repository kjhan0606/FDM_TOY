#!/usr/bin/env python3
"""Reproduce published Koo and Boey separation-curve benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fdm_smbh_delay.constants import KM_S_TO_PC_MYR
from fdm_smbh_delay.empirical import (
    BOEY_2025_FITS,
    koo_q0_pc_m5half_myr,
    koo_separation_pc,
)
from fdm_smbh_delay.literature import (
    boey2025_schive_soliton,
    koo2024_equivalent_schive_soliton,
    soliton_definition_summary,
)
from fdm_smbh_delay.orbit import make_orbital_state


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    output: Path,
    koo_rows: list[dict[str, float]],
    boey_rows: list[dict[str, float]],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), constrained_layout=True)
    for mass in sorted({row["black_hole_mass_msun"] for row in koo_rows}):
        selected = [row for row in koo_rows if row["black_hole_mass_msun"] == mass]
        axes[0].plot(
            [row["time_myr"] * 1.0e3 for row in selected],
            [row["separation_pc"] for row in selected],
            label=rf"$M_{{\rm BH}}={mass / 1.0e8:g}\times10^8\,M_\odot$",
        )
    for ratio in sorted({int(row["single_bh_to_soliton_percent"]) for row in boey_rows}):
        selected = [
            row
            for row in boey_rows
            if int(row["single_bh_to_soliton_percent"]) == ratio
        ]
        axes[1].plot(
            [row["time_myr"] for row in selected],
            [row["separation_pc"] for row in selected],
            label=rf"$M_{{\rm BH}}/M_s={ratio}\%$",
        )
    axes[0].set(
        xlabel="Time [kyr]", ylabel="Binary separation [pc]", xlim=(0, 100)
    )
    axes[1].set(
        xlabel="Time [Myr]", ylabel="Binary separation [pc]", xlim=(0, 0.8)
    )
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/literature_reproduction")
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    koo_soliton = koo2024_equivalent_schive_soliton()
    boey_soliton = boey2025_schive_soliton()
    boey_initial = make_orbital_state(
        mass1_msun=1.0e8,
        mass2_msun=1.0e8,
        separation_pc=3.0,
        eccentricity=0.0,
        soliton=boey_soliton,
    )

    koo_rows: list[dict[str, float]] = []
    for black_hole_mass in (0.6e8, 1.0e8, 1.5e8):
        q0 = koo_q0_pc_m5half_myr(
            soliton_mass_msun=1.0e9,
            black_hole_mass_msun=black_hole_mass,
            particle_mass_ev=1.0e-21,
        )
        for time in np.linspace(0.0, 0.1, 201):
            koo_rows.append(
                {
                    "time_myr": float(time),
                    "black_hole_mass_msun": black_hole_mass,
                    "q0_pc_m5half_myr": q0,
                    "separation_pc": float(koo_separation_pc(time, 0.8, q0)),
                }
            )

    boey_rows: list[dict[str, float]] = []
    for ratio, fit in BOEY_2025_FITS.items():
        for time in np.linspace(0.0, 0.8, 401):
            boey_rows.append(
                {
                    "time_myr": float(time),
                    "single_bh_to_soliton_percent": float(ratio),
                    "fit_A_pc": fit.a_pc,
                    "fit_B_myr_inverse": fit.b_per_myr,
                    "fit_C": fit.c,
                    "separation_pc": float(fit.separation_pc(time)),
                }
            )

    _write_csv(
        args.output / "koo2024_eq18_curves.csv",
        [
            "time_myr",
            "black_hole_mass_msun",
            "q0_pc_m5half_myr",
            "separation_pc",
        ],
        koo_rows,
    )
    _write_csv(
        args.output / "boey2025_eq26_curves.csv",
        [
            "time_myr",
            "single_bh_to_soliton_percent",
            "fit_A_pc",
            "fit_B_myr_inverse",
            "fit_C",
            "separation_pc",
        ],
        boey_rows,
    )
    summary = {
        "koo2024": {
            "source": "arXiv:2311.03412v2",
            "curve": "equations 7 and 18; averaged curve, not digitized simulation data",
            "initial_binary_separation_pc": 0.9,
            "averaged_fit_D0_pc": 0.8,
            "box_size_pc": 40.0,
            "grid_cells_per_dimension": 500,
            "spatial_resolution_pc": 0.08,
            "soliton": soliton_definition_summary(koo_soliton),
        },
        "boey2025": {
            "source": "arXiv:2504.16348v3",
            "curve": "equation 26 and table I; empirical fits, not digitized simulation data",
            "initial_binary_separation_pc": 3.0,
            "reported_initial_speed_each_bh_km_s": 584.14,
            "schive_initializer_speed_each_bh_km_s": float(
                np.linalg.norm(boey_initial.velocity1_pc_myr) / KM_S_TO_PC_MYR
            ),
            "box_size_pc": 100.0,
            "effective_grid_cells_per_dimension": 1024,
            "finest_spatial_resolution_pc": 100.0 / 1024.0,
            "plummer_radius_pc": 0.001,
            "soliton": soliton_definition_summary(boey_soliton),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.plot:
        _plot(args.output / "literature_separation_curves.png", koo_rows, boey_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
