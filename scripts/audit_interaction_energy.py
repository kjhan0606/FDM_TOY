#!/usr/bin/env python3
"""Audit the live-wave--SMBH cross energy with a controlled single SMBH."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fdm_smbh_delay.interaction import (
    gauge_align_point_estimator,
    periodic_poisson_potential,
    single_bh_interaction_snapshot,
)
from fdm_smbh_delay.literature import koo2024_equivalent_schive_soliton


def _relative_difference(value1: float, value2: float) -> float:
    scale = max(abs(value1), abs(value2), np.finfo(float).tiny)
    return float(abs(value1 - value2) / scale)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument(
        "--offsets-pc", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 4.0]
    )
    parser.add_argument("--box-pc", type=float, default=40.0)
    parser.add_argument("--bh-mass-msun", type=float, default=1.0e8)
    parser.add_argument("--softening-cells", type=float, default=0.5)
    parser.add_argument(
        "--output", type=Path, default=Path("results/interaction_energy_audit")
    )
    args = parser.parse_args()
    if any(value < 32 for value in args.resolutions):
        raise ValueError("every resolution must be at least 32")
    if any(value < 0.0 or value >= 0.25 * args.box_pc for value in args.offsets_pc):
        raise ValueError("offsets must satisfy 0 <= offset < box_size/4")
    if 0.0 not in args.offsets_pc:
        raise ValueError("offsets must include zero as the gauge reference")
    if args.box_pc <= 0.0 or args.bh_mass_msun <= 0.0:
        raise ValueError("box size and SMBH mass must be positive")
    if args.softening_cells <= 0.0:
        raise ValueError("softening in cells must be positive")

    soliton = koo2024_equivalent_schive_soliton()
    rows: list[dict[str, float | int]] = []
    summaries: list[dict[str, float | int]] = []
    for resolution in args.resolutions:
        cell_size = args.box_pc / resolution
        coordinates = np.linspace(
            -0.5 * args.box_pc,
            0.5 * args.box_pc,
            resolution,
            endpoint=False,
        )
        radius = np.sqrt(
            coordinates[:, None, None] ** 2
            + coordinates[None, :, None] ** 2
            + coordinates[None, None, :] ** 2
        )
        density = soliton.density(radius)
        del radius
        wave_potential = periodic_poisson_potential(density, cell_size)
        wave_gradient = np.gradient(wave_potential, cell_size, edge_order=2)
        snapshots = [
            single_bh_interaction_snapshot(
                density_msun_pc3=density,
                wave_potential_pc2_myr2=wave_potential,
                box_size_pc=args.box_pc,
                bh_mass_msun=args.bh_mass_msun,
                bh_position_pc=np.array([offset, 0.0, 0.0]),
                plummer_radius_pc=args.softening_cells * cell_size,
                wave_gradient_pc_myr2=wave_gradient,
            )
            for offset in args.offsets_pc
        ]
        grid_energies = np.array(
            [snapshot.grid_interaction_energy for snapshot in snapshots]
        )
        point_energies = np.array(
            [snapshot.point_interaction_energy for snapshot in snapshots]
        )
        aligned_point, gauge_offset = gauge_align_point_estimator(
            point_energies, grid_energies
        )
        reference_grid = grid_energies[0]
        nonzero_energy_errors: list[float] = []
        nonzero_force_errors: list[float] = []
        for offset, snapshot, aligned in zip(
            args.offsets_pc, snapshots, aligned_point, strict=True
        ):
            delta_grid = snapshot.grid_interaction_energy - reference_grid
            delta_point = aligned - reference_grid
            force_grid = float(snapshot.force_on_bh_from_grid_energy[0])
            force_point = float(snapshot.force_on_bh_from_point_potential[0])
            energy_error = (
                0.0
                if offset == 0.0
                else _relative_difference(delta_grid, delta_point)
            )
            force_error = (
                0.0
                if offset == 0.0
                else _relative_difference(force_grid, force_point)
            )
            if offset != 0.0:
                nonzero_energy_errors.append(energy_error)
                nonzero_force_errors.append(force_error)
            rows.append(
                {
                    "resolution": resolution,
                    "cell_size_pc": cell_size,
                    "plummer_radius_pc": args.softening_cells * cell_size,
                    "offset_pc": offset,
                    "grid_interaction_energy": snapshot.grid_interaction_energy,
                    "point_interaction_energy": snapshot.point_interaction_energy,
                    "gauge_aligned_point_energy": float(aligned),
                    "delta_grid_energy": float(delta_grid),
                    "delta_point_energy": float(delta_point),
                    "delta_energy_relative_difference": energy_error,
                    "grid_force_x": force_grid,
                    "point_force_x": force_point,
                    "force_relative_difference": force_error,
                }
            )
        summaries.append(
            {
                "resolution": resolution,
                "cell_size_pc": cell_size,
                "plummer_radius_pc": args.softening_cells * cell_size,
                "unaligned_central_energy_relative_difference": _relative_difference(
                    grid_energies[0], point_energies[0]
                ),
                "point_energy_gauge_offset": gauge_offset,
                "maximum_nonzero_delta_energy_relative_difference": max(
                    nonzero_energy_errors
                ),
                "maximum_nonzero_force_relative_difference": max(
                    nonzero_force_errors
                ),
            }
        )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "interaction_energy_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "diagnosed",
        "interaction_energy_definition": "integral rho*Phi_BH dV, counted once",
        "wave_self_gravity_definition": "one half integral rho*Phi_wave dV",
        "point_estimator_role": (
            "force and convergence diagnostic after gauge alignment; "
            "not the Hamiltonian cross term"
        ),
        "boundary_conditions": {
            "wave_potential": "periodic FFT with zero mean",
            "smbh_potential": "Plummer potential with zero at infinity",
        },
        "box_size_pc": args.box_pc,
        "bh_mass_msun": args.bh_mass_msun,
        "offsets_pc": args.offsets_pc,
        "softening_cells": args.softening_cells,
        "resolutions": summaries,
    }
    (output / "interaction_energy_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
