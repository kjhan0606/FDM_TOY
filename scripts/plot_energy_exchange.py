#!/usr/bin/env python3
"""Plot the global energy ledger of a completed live-wave calculation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
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
    energy_scale = max(
        max(float(np.max(np.abs(values))) for values in changes.values()),
        np.finfo(float).tiny,
    )
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
    figure, axis = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
    for label, values in changes.items():
        axis.plot(plot_time, values / energy_scale, label=label, **styles[label])
    axis.axhline(0.0, color="0.55", linewidth=0.55)
    axis.set_xlim(float(plot_time[0]), float(plot_time[-1]))
    axis.set_xlabel(time_label)
    axis.set_ylabel(r"$\Delta E/\Delta E_{\rm exch,max}$")
    axis.tick_params(top=True, right=True)
    axis.legend(loc="best", fontsize=7.0, frameon=False)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
