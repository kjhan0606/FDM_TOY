#!/usr/bin/env python3
"""Measure reduced live-wave response diagnostics from PyUL 3D snapshots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fdm_smbh_delay.pyul import ordered_output_paths, output_index, pyul_unit_system
from fdm_smbh_delay.wave_response import (
    centred_grid,
    multipole_amplitudes,
    periodic_centre_of_mass,
    periodic_point_centre,
    periodic_poisson_code,
    plummer_potential_code,
    spectral_wave_fields,
)


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _core_radius(radius: np.ndarray, density: np.ndarray, central: float) -> float:
    finite = np.isfinite(density) & (density > 0.0)
    indices = np.flatnonzero(finite & (density <= 0.5 * central))
    if indices.size == 0:
        return np.nan
    index = int(indices[0])
    if index == 0:
        return float(radius[0])
    x0, x1 = radius[index - 1 : index + 1]
    y0, y1 = density[index - 1 : index + 1]
    if y1 == y0:
        return float(x1)
    return float(x0 + (0.5 * central - y0) * (x1 - x0) / (y1 - y0))


def _nearest(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--radial-bins", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.radial_bins < 16:
        raise ValueError("at least 16 radial bins are required")

    run = args.run.expanduser().resolve()
    output = run if args.output is None else args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(
        (run / "fdm_adapter_metadata.json").read_text(encoding="utf-8")
    )
    config = json.loads((run / "config.uldm").read_text(encoding="utf-8"))
    units = pyul_unit_system(metadata)
    resolution = int(metadata["resolution"])
    box_pc = float(metadata["box_size_pc"])
    box_code = box_pc / units.length_pc
    cell_code = box_code / resolution
    cell_volume_code = cell_code**3
    core_pc = float(metadata["core_radius_reference_pc"])
    core_code = core_pc / units.length_pc
    plummer_pc = float(metadata["plummer_radius_pc"])
    plummer_code = plummer_pc / units.length_pc

    particles = config["Matter Particles"]["Condition"]
    masses_msun = np.asarray([particle[0] for particle in particles], dtype=float)
    masses_code = masses_msun / units.mass_msun
    wave_paths = ordered_output_paths(run / "Outputs" / "3Wfn", "P3D_#*.npy")
    density_paths = ordered_output_paths(
        run / "Outputs" / "3Density", "R3D_#*.npy"
    )
    wave_indices = [output_index(path) for path in wave_paths]
    density_indices = [output_index(path) for path in density_paths]
    if wave_indices != density_indices:
        raise ValueError("3D wavefunction and density outputs are inconsistent")
    all_state_paths = ordered_output_paths(
        run / "Outputs" / "NBody", "NTM_#*.npy"
    )
    state_by_index = {output_index(path): path for path in all_state_paths}
    try:
        state_paths = [state_by_index[index] for index in wave_indices]
    except KeyError as exc:
        raise ValueError("a 3D field has no matching SMBH state") from exc
    samples = len(wave_paths)
    save_number = int(
        metadata.get("save_number", config["Save Options"]["Number"])
    )
    times_myr = (
        float(metadata["duration_myr"])
        * np.asarray(wave_indices, dtype=float)
        / save_number
    )
    saved_kinetic_all = np.load(run / "Outputs" / "ekandqlist.npy")
    saved_self_all = np.load(run / "Outputs" / "egpsilist.npy")
    saved_cross_all = np.load(run / "Outputs" / "egpcmlist.npy")
    if not (
        saved_kinetic_all.size
        == saved_self_all.size
        == saved_cross_all.size
        == len(all_state_paths)
    ):
        raise ValueError("saved energy arrays do not match the SMBH states")
    saved_kinetic = saved_kinetic_all[wave_indices]
    saved_self = saved_self_all[wave_indices]
    saved_cross = saved_cross_all[wave_indices]

    maximum_radius_code = min(0.45 * box_code, 8.0 * core_code)
    radial_edges = np.linspace(0.0, maximum_radius_code, args.radial_bins + 1)
    radial_centres = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    shell_widths = np.diff(radial_edges)
    shell_geometric_volumes = 4.0 * np.pi / 3.0 * (
        radial_edges[1:] ** 3 - radial_edges[:-1] ** 3
    )
    response_rows: list[dict] = []
    radial_rows: list[dict] = []
    maximum_density_snapshot_error = 0.0
    maximum_energy_snapshot_error = 0.0

    for sample, (wave_path, density_path, state_path, time_myr) in enumerate(
        zip(wave_paths, density_paths, state_paths, times_myr, strict=True)
    ):
        wavefunction = np.load(wave_path)
        saved_density = np.load(density_path)
        density = np.abs(wavefunction) ** 2
        density_scale = max(float(np.max(saved_density)), np.finfo(float).tiny)
        maximum_density_snapshot_error = max(
            maximum_density_snapshot_error,
            float(np.max(np.abs(saved_density - density)) / density_scale),
        )
        state = np.load(state_path).reshape(len(particles), 6)
        positions_code = state[:, :3]
        binary_centre = periodic_point_centre(
            positions_code, masses_code, box_code
        )
        self_potential = periodic_poisson_code(density, box_code)
        bh_potential = plummer_potential_code(
            shape=density.shape,
            box_size=box_code,
            masses=masses_code,
            positions=positions_code,
            plummer_radius=plummer_code,
        )
        fields = spectral_wave_fields(
            wavefunction, self_potential + bh_potential, box_code
        )
        wave_centre = periodic_centre_of_mass(density, box_code)
        x, y, z, radius = centred_grid(resolution, box_code, wave_centre)
        safe_radius = np.where(radius > 0.0, radius, 1.0)
        radial_mass_current = (
            fields.mass_current[0] * x
            + fields.mass_current[1] * y
            + fields.mass_current[2] * z
        ) / safe_radius
        radial_energy_current = (
            fields.schrodinger_energy_current[0] * x
            + fields.schrodinger_energy_current[1] * y
            + fields.schrodinger_energy_current[2] * z
        ) / safe_radius
        wave_self_density = 0.5 * density * self_potential
        wave_cross_density = density * bh_potential
        wave_intrinsic_density = fields.kinetic_energy_density + wave_self_density

        bin_index = np.searchsorted(radial_edges, radius, side="right") - 1
        valid = (bin_index >= 0) & (bin_index < args.radial_bins)
        indices = bin_index[valid]

        def sum_shell(values: np.ndarray) -> np.ndarray:
            return np.bincount(
                indices,
                weights=np.asarray(values)[valid] * cell_volume_code,
                minlength=args.radial_bins,
            )

        shell_mass = sum_shell(density)
        shell_kinetic = sum_shell(fields.kinetic_energy_density)
        shell_self = sum_shell(wave_self_density)
        shell_cross = sum_shell(wave_cross_density)
        shell_mass_flux = sum_shell(radial_mass_current) / shell_widths
        shell_energy_flux = sum_shell(radial_energy_current) / shell_widths
        shell_density = shell_mass / shell_geometric_volumes
        central_selection = radius < cell_code
        central_density_code = float(np.mean(density[central_selection]))
        evolved_core_code = _core_radius(
            radial_centres, shell_density, central_density_code
        )

        regions = {
            "core": radius < core_code,
            "near": (radius >= core_code) & (radius < 4.0 * core_code),
            "outer": (radius >= 4.0 * core_code) & (radius < 8.0 * core_code),
        }
        modes = {
            name: multipole_amplitudes(
                density, x, y, z, radius, selection, cell_volume_code
            )
            for name, selection in regions.items()
        }
        binary_x, binary_y, binary_z, binary_radius = centred_grid(
            resolution, box_code, binary_centre
        )
        binary_regions = {
            "binary_core": binary_radius < core_code,
            "binary_near": (binary_radius >= core_code)
            & (binary_radius < 4.0 * core_code),
            "binary_outer": (binary_radius >= 4.0 * core_code)
            & (binary_radius < 8.0 * core_code),
        }
        binary_modes = {
            name: multipole_amplitudes(
                density,
                binary_x,
                binary_y,
                binary_z,
                binary_radius,
                selection,
                cell_volume_code,
            )
            for name, selection in binary_regions.items()
        }
        all_modes = {**modes, **binary_modes}
        centre_offset = wave_centre - binary_centre
        centre_offset -= box_code * np.floor(centre_offset / box_code + 0.5)
        offline_kinetic = float(np.sum(fields.kinetic_energy_density) * cell_volume_code)
        offline_self = float(np.sum(wave_self_density) * cell_volume_code)
        offline_cross = float(np.sum(wave_cross_density) * cell_volume_code)
        energy_scale = max(
            abs(saved_kinetic[sample]),
            abs(saved_self[sample]),
            abs(saved_cross[sample]),
            np.finfo(float).tiny,
        )
        maximum_energy_snapshot_error = max(
            maximum_energy_snapshot_error,
            abs(offline_kinetic - saved_kinetic[sample]) / energy_scale,
            abs(offline_self - saved_self[sample]) / energy_scale,
            abs(offline_cross - saved_cross[sample]) / energy_scale,
        )
        outer_selection = radius >= 4.0 * core_code
        outer_mass = float(np.sum(density[outer_selection]) * cell_volume_code)
        outer_intrinsic = float(
            np.sum(wave_intrinsic_density[outer_selection]) * cell_volume_code
        )
        flux_indices = {
            factor: _nearest(radial_centres, factor * core_code)
            for factor in (2, 4, 8)
        }
        response_rows.append(
            {
                "sample": sample,
                "time_myr": time_myr,
                "wave_com_x_pc": wave_centre[0] * units.length_pc,
                "wave_com_y_pc": wave_centre[1] * units.length_pc,
                "wave_com_z_pc": wave_centre[2] * units.length_pc,
                "binary_com_x_pc": binary_centre[0] * units.length_pc,
                "binary_com_y_pc": binary_centre[1] * units.length_pc,
                "binary_com_z_pc": binary_centre[2] * units.length_pc,
                "wave_binary_com_offset_pc": float(np.linalg.norm(centre_offset))
                * units.length_pc,
                "central_density_msun_pc3": central_density_code
                * units.density_msun_pc3,
                "core_radius_pc": evolved_core_code * units.length_pc,
                "wave_kinetic_energy": saved_kinetic[sample]
                * units.energy_msun_pc2_myr2,
                "wave_self_gravity_energy": saved_self[sample]
                * units.energy_msun_pc2_myr2,
                "wave_intrinsic_energy": (saved_kinetic[sample] + saved_self[sample])
                * units.energy_msun_pc2_myr2,
                "wave_bh_interaction_energy": saved_cross[sample]
                * units.energy_msun_pc2_myr2,
                "outer_mass_msun": outer_mass * units.mass_msun,
                "outer_intrinsic_energy": outer_intrinsic
                * units.energy_msun_pc2_myr2,
                "mass_flux_2rc_msun_myr": shell_mass_flux[flux_indices[2]]
                * units.mass_rate_msun_myr,
                "mass_flux_4rc_msun_myr": shell_mass_flux[flux_indices[4]]
                * units.mass_rate_msun_myr,
                "mass_flux_8rc_msun_myr": shell_mass_flux[flux_indices[8]]
                * units.mass_rate_msun_myr,
                "schrodinger_energy_flux_2rc": shell_energy_flux[flux_indices[2]]
                * units.energy_rate_msun_pc2_myr3,
                "schrodinger_energy_flux_4rc": shell_energy_flux[flux_indices[4]]
                * units.energy_rate_msun_pc2_myr3,
                "schrodinger_energy_flux_8rc": shell_energy_flux[flux_indices[8]]
                * units.energy_rate_msun_pc2_myr3,
                **{
                    f"{region}_l{ell}_fraction": getattr(
                        multipoles, f"l{ell}_fraction"
                    )
                    for region, multipoles in all_modes.items()
                    for ell in (1, 2)
                },
                **{
                    f"{region}_l{ell}_m{order}_{component}": float(
                        getattr(multipoles, f"l{ell}_m{order}").real
                        if component == "real"
                        else getattr(multipoles, f"l{ell}_m{order}").imag
                    )
                    for region, multipoles in all_modes.items()
                    for ell, orders in ((1, (0, 1)), (2, (0, 1, 2)))
                    for order in orders
                    for component in ("real", "imag")
                },
            }
        )
        for radial_bin in range(args.radial_bins):
            radial_rows.append(
                {
                    "sample": sample,
                    "time_myr": time_myr,
                    "radius_pc": radial_centres[radial_bin] * units.length_pc,
                    "shell_mass_msun": shell_mass[radial_bin] * units.mass_msun,
                    "mean_density_msun_pc3": shell_density[radial_bin]
                    * units.density_msun_pc3,
                    "wave_kinetic_energy": shell_kinetic[radial_bin]
                    * units.energy_msun_pc2_myr2,
                    "wave_self_gravity_energy": shell_self[radial_bin]
                    * units.energy_msun_pc2_myr2,
                    "wave_bh_interaction_energy": shell_cross[radial_bin]
                    * units.energy_msun_pc2_myr2,
                    "radial_mass_flux_msun_myr": shell_mass_flux[radial_bin]
                    * units.mass_rate_msun_myr,
                    "radial_schrodinger_energy_flux": shell_energy_flux[radial_bin]
                    * units.energy_rate_msun_pc2_myr3,
                }
            )

    _write_rows(output / "wave_response_timeseries.csv", response_rows)
    _write_rows(output / "wave_radial_profiles.csv", radial_rows)
    initial = response_rows[0]
    final = response_rows[-1]
    summary = {
        "status": "diagnosed",
        "run": str(run),
        "samples": samples,
        "duration_myr": float(metadata["duration_myr"]),
        "resolution": resolution,
        "radial_bins": args.radial_bins,
        "interaction_energy_definition": "integral rho*Phi_BH dV, counted once",
        "energy_flux_scope": (
            "local Schrodinger-field current; global Hamiltonian ledger remains "
            "authoritative because self-gravity is non-local"
        ),
        "maximum_density_snapshot_relative_error": maximum_density_snapshot_error,
        "maximum_offline_energy_relative_error": maximum_energy_snapshot_error,
        "central_density_fractional_change": (
            final["central_density_msun_pc3"]
            / initial["central_density_msun_pc3"]
            - 1.0
        ),
        "core_radius_fractional_change": final["core_radius_pc"]
        / initial["core_radius_pc"]
        - 1.0,
        "outer_mass_change_msun": final["outer_mass_msun"]
        - initial["outer_mass_msun"],
        "outer_intrinsic_energy_change": final["outer_intrinsic_energy"]
        - initial["outer_intrinsic_energy"],
        "maximum_core_l1_fraction": max(row["core_l1_fraction"] for row in response_rows),
        "maximum_core_l2_fraction": max(row["core_l2_fraction"] for row in response_rows),
        "maximum_near_l2_fraction": max(row["near_l2_fraction"] for row in response_rows),
        "maximum_outer_l2_fraction": max(row["outer_l2_fraction"] for row in response_rows),
        "maximum_binary_core_l1_fraction": max(
            row["binary_core_l1_fraction"] for row in response_rows
        ),
        "maximum_binary_core_l2_fraction": max(
            row["binary_core_l2_fraction"] for row in response_rows
        ),
        "maximum_wave_binary_com_offset_pc": max(
            row["wave_binary_com_offset_pc"] for row in response_rows
        ),
        "density_multipole_coefficients": (
            "standard complex spherical-harmonic coefficients normalized by "
            "selected mass/sqrt(4*pi); negative-m coefficients follow from "
            "the reality condition; core/near/outer use the wave centre and "
            "binary_core/binary_near/binary_outer use the binary centre"
        ),
    }
    conservation_path = run / "conservation_timeseries.csv"
    if conservation_path.is_file():
        conservation = np.genfromtxt(
            conservation_path, delimiter=",", names=True, ndmin=1
        )

        def change(field: str) -> float:
            return float(conservation[field][-1] - conservation[field][0])

        orbital_change = change("binary_orbital_energy")
        intrinsic_change = change("wave_intrinsic_energy")
        interaction_change = change("wave_bh_interaction_grid")
        com_change = change("bh_com_kinetic_energy")
        total_change = change("combined_energy")
        orbital_loss = -orbital_change
        summary["energy_transfer"] = {
            "delta_binary_orbital_energy": orbital_change,
            "delta_bh_com_kinetic_energy": com_change,
            "delta_wave_intrinsic_energy": intrinsic_change,
            "delta_wave_bh_interaction_energy": interaction_change,
            "delta_total_hamiltonian": total_change,
            "wave_intrinsic_to_orbital_loss": (
                intrinsic_change / orbital_loss if orbital_loss > 0.0 else None
            ),
            "outer_intrinsic_to_wave_intrinsic_gain": (
                summary["outer_intrinsic_energy_change"] / intrinsic_change
                if intrinsic_change > 0.0
                else None
            ),
        }
    (output / "wave_response_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
