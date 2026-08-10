#!/usr/bin/env python3
"""Plot the physical separation of the SMBH pair in a PyUL calculation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fdm_smbh_delay.empirical import koo_q0_pc_m5half_myr, koo_separation_pc
from fdm_smbh_delay.pyul import ordered_output_paths, pyul_unit_system


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wave-density-panel", action="store_true")
    parser.add_argument("--mark-resolution-limit", action="store_true")
    parser.add_argument("--koo-reference", action="store_true")
    args = parser.parse_args()
    run = args.run.expanduser().resolve()
    table = np.genfromtxt(
        run / "conservation_timeseries.csv",
        delimiter=",",
        names=True,
    )
    time_myr = table["time_myr"]
    separation_pc = table["separation_pc"]
    if float(time_myr[-1]) >= 0.1:
        plot_time = time_myr
        time_label = r"Time [Myr]"
    else:
        plot_time = time_myr * 1.0e3
        time_label = r"Time [kyr]"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    if args.wave_density_panel:
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(3.35, 4.45),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.0, 1.15]},
        )
        axis = axes[0]
    else:
        figure, axis = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
    line_options = {
        "color": "#254F73",
        "linewidth": 0.85 if separation_pc.size > 200 else 1.35,
    }
    if args.koo_reference:
        line_options["label"] = "Live wave"
    if separation_pc.size <= 200:
        line_options.update(
            {
                "marker": "o",
                "markersize": 2.2,
                "markerfacecolor": "white",
                "markeredgewidth": 0.65,
            }
        )
    axis.plot(plot_time, separation_pc, **line_options)
    separation_for_limits = separation_pc
    if args.koo_reference:
        config = json.loads((run / "config.uldm").read_text(encoding="utf-8"))
        particles = config["Matter Particles"]["Condition"]
        masses = np.asarray([particle[0] for particle in particles], dtype=float)
        if masses.size != 2 or not np.isclose(masses[0], masses[1]):
            raise ValueError("the Koo reference requires an equal-mass binary")
        soliton_mass = float(config["ULDM Solitons"]["Condition"][0][0])
        metadata_for_reference = json.loads(
            (run / "fdm_adapter_metadata.json").read_text(encoding="utf-8")
        )
        q0 = koo_q0_pc_m5half_myr(
            soliton_mass_msun=soliton_mass,
            black_hole_mass_msun=float(masses[0]),
            particle_mass_ev=float(metadata_for_reference["particle_mass_ev"]),
        )
        reference_separation = koo_separation_pc(time_myr, 0.8, q0)
        axis.plot(
            plot_time,
            reference_separation,
            color="#B65E2E",
            linewidth=1.0,
            linestyle=(0, (4, 2)),
            label="Koo et al. (2024)",
        )
        separation_for_limits = np.concatenate(
            (separation_pc, np.asarray(reference_separation))
        )
        axis.legend(loc="upper right", fontsize=7.0, frameon=False)
    axis.set_xlabel(time_label)
    axis.set_ylabel(r"SMBH separation [pc]")
    axis.set_xlim(float(plot_time[0]), float(plot_time[-1]))
    span = float(np.ptp(separation_for_limits))
    axis.set_ylim(
        float(np.min(separation_for_limits) - 0.08 * span),
        float(np.max(separation_for_limits) + 0.08 * span),
    )
    axis.tick_params(top=True, right=True)
    metadata = None
    if args.mark_resolution_limit or args.wave_density_panel:
        metadata = json.loads(
            (run / "fdm_adapter_metadata.json").read_text(encoding="utf-8")
        )
    if args.mark_resolution_limit:
        assert metadata is not None
        two_cells_pc = 2.0 * float(metadata["box_size_pc"]) / int(
            metadata["resolution"]
        )
        unresolved_indices = np.flatnonzero(separation_pc < two_cells_pc)
        if unresolved_indices.size:
            first_unresolved_time = float(plot_time[unresolved_indices[0]])
            axis.axvspan(
                first_unresolved_time,
                float(plot_time[-1]),
                color="0.92",
                linewidth=0.0,
                zorder=-10,
            )
        lower_limit, upper_limit = axis.get_ylim()
        if lower_limit < two_cells_pc < upper_limit:
            axis.axhline(
                two_cells_pc,
                color="0.35",
                linewidth=0.7,
                linestyle=(0, (3, 2)),
            )
            axis.text(
                0.97,
                two_cells_pc,
                "2 cell widths",
                color="0.25",
                fontsize=7.0,
                ha="right",
                va="bottom",
                transform=axis.get_yaxis_transform(),
            )
    if args.wave_density_panel:
        axis.set_xlabel("")
        axis.text(0.03, 0.94, "(a)", transform=axis.transAxes, va="top")
        assert metadata is not None
        units = pyul_unit_system(metadata)
        density_paths = ordered_output_paths(
            run / "Outputs" / "1Density", "R1D_#*.npy"
        )
        if len(density_paths) != time_myr.size:
            raise ValueError("line-density and orbital outputs have different lengths")
        line_density = (
            np.asarray([np.load(path) for path in density_paths])
            * units.density_msun_pc3
        )
        cell_size_pc = float(metadata["box_size_pc"]) / int(
            metadata["resolution"]
        )
        coordinate_pc = (
            np.arange(line_density.shape[1]) - line_density.shape[1] // 2
        ) * cell_size_pc
        core_radius_pc = float(metadata["core_radius_reference_pc"])
        selection = np.abs(coordinate_pc) <= 4.0 * core_radius_pc
        density_floor = np.finfo(float).tiny
        log_density = np.log10(np.maximum(line_density[:, selection], density_floor))
        lower, upper = np.percentile(log_density, [1.0, 99.0])
        density_axis = axes[1]
        image = density_axis.pcolormesh(
            plot_time,
            coordinate_pc[selection],
            log_density.T,
            shading="auto",
            cmap="magma",
            vmin=float(lower),
            vmax=float(upper),
            rasterized=True,
        )
        density_axis.set_xlabel(time_label)
        density_axis.set_ylabel(r"$y$ [pc]")
        density_axis.tick_params(top=True, right=True)
        density_axis.text(
            0.03,
            0.94,
            "(b)",
            color="white",
            transform=density_axis.transAxes,
            va="top",
        )
        colour_bar = figure.colorbar(image, ax=density_axis, pad=0.02)
        colour_bar.set_label(
            r"$\log_{10}[\rho_\psi/(M_\odot\,{\rm pc}^{-3})]$",
            fontsize=7.0,
        )
        colour_bar.ax.tick_params(labelsize=7.0, width=0.7)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
