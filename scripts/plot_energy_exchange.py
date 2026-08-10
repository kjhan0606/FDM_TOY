#!/usr/bin/env python3
"""Plot the global energy ledger of a completed live-wave calculation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mark-resolution-limit", action="store_true")
    args = parser.parse_args()
    run = args.run.expanduser().resolve()
    table = np.genfromtxt(
        run / "conservation_timeseries.csv", delimiter=",", names=True
    )
    time_myr = table["time_myr"]
    if float(time_myr[-1]) >= 0.1:
        plot_time = time_myr
        time_label = r"Time [Myr]"
    else:
        plot_time = 1.0e3 * time_myr
        time_label = r"Time [kyr]"

    changes = {
        "SMBH orbit": table["binary_orbital_energy"]
        - table["binary_orbital_energy"][0],
        "FDM intrinsic": table["wave_intrinsic_energy"]
        - table["wave_intrinsic_energy"][0],
        "FDM-SMBH interaction": table["wave_bh_interaction_grid"]
        - table["wave_bh_interaction_grid"][0],
        "Total Hamiltonian": table["combined_energy"] - table["combined_energy"][0],
    }
    transfer_components = (
        changes["SMBH orbit"],
        table["bh_com_kinetic_energy"] - table["bh_com_kinetic_energy"][0],
        changes["FDM intrinsic"],
        changes["FDM-SMBH interaction"],
    )
    energy_scale = max(
        max(float(np.max(np.abs(values))) for values in transfer_components),
        np.finfo(float).tiny,
    )
    if "energy_error_over_transfer" in (table.dtype.names or ()):
        energy_error = table["energy_error_over_transfer"]
    else:
        energy_error = np.abs(changes["Total Hamiltonian"]) / energy_scale
    hamiltonian_error_envelope = np.maximum.accumulate(energy_error)
    styles = {
        "SMBH orbit": {"color": "#254F73", "linewidth": 1.1},
        "FDM intrinsic": {"color": "#B65E2E", "linewidth": 1.0},
        "FDM-SMBH interaction": {
            "color": "#6D4C7D",
            "linewidth": 1.0,
            "linestyle": (0, (4, 2)),
        },
        "Total Hamiltonian": {"color": "0.15", "linewidth": 0.75},
    }
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
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(3.35, 4.1),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.25, 0.75]},
    )
    axis = axes[0]
    for label, values in changes.items():
        axis.plot(plot_time, values / energy_scale, label=label, **styles[label])
    axis.axhline(0.0, color="0.55", linewidth=0.55)
    axis.set_ylabel(r"$\Delta E/\Delta E_{\rm exch,max}$")
    axis.tick_params(top=True, right=True)
    axis.set_ylim(-1.05, 1.35)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.55, 0.99),
        ncol=2,
        columnspacing=0.9,
        handlelength=2.2,
        fontsize=7.0,
        frameon=False,
    )
    axis.text(0.03, 0.97, "(a)", transform=axis.transAxes, va="top")

    error_axis = axes[1]
    error_axis.plot(
        plot_time,
        hamiltonian_error_envelope,
        color="0.15",
        linewidth=0.9,
    )
    error_axis.axhline(
        0.01,
        color="#B65E2E",
        linewidth=0.75,
        linestyle=(0, (4, 2)),
    )
    error_axis.text(
        0.97,
        0.01,
        "1% acceptance limit",
        color="#8E421E",
        fontsize=7.0,
        ha="right",
        va="bottom",
        transform=error_axis.get_yaxis_transform(),
    )
    error_axis.set_xlabel(time_label)
    error_axis.set_ylabel(r"$\max |\Delta H|/\Delta E_{\rm exch,max}$")
    error_axis.tick_params(top=True, right=True)
    error_axis.text(0.03, 0.91, "(b)", transform=error_axis.transAxes, va="top")
    upper_error = max(0.012, 1.08 * float(np.max(hamiltonian_error_envelope)))
    error_axis.set_ylim(0.0, upper_error)
    error_axis.set_xlim(float(plot_time[0]), float(plot_time[-1]))

    if args.mark_resolution_limit:
        metadata = json.loads(
            (run / "fdm_adapter_metadata.json").read_text(encoding="utf-8")
        )
        two_cells_pc = 2.0 * float(metadata["box_size_pc"]) / int(
            metadata["resolution"]
        )
        unresolved_indices = np.flatnonzero(table["separation_pc"] < two_cells_pc)
        if unresolved_indices.size:
            first_unresolved_time = float(plot_time[unresolved_indices[0]])
            for panel in axes:
                panel.axvspan(
                    first_unresolved_time,
                    float(plot_time[-1]),
                    color="0.92",
                    linewidth=0.0,
                    zorder=-10,
                )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
