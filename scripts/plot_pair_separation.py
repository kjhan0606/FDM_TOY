#!/usr/bin/env python3
"""Plot the physical separation of the SMBH pair in a PyUL calculation."""

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
    table = np.genfromtxt(
        args.run.expanduser().resolve() / "conservation_timeseries.csv",
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
    figure, axis = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
    axis.plot(
        plot_time,
        separation_pc,
        color="#254F73",
        linewidth=1.35,
        marker="o",
        markersize=2.2,
        markerfacecolor="white",
        markeredgewidth=0.65,
    )
    axis.set_xlabel(time_label)
    axis.set_ylabel(r"SMBH separation [pc]")
    axis.set_xlim(float(plot_time[0]), float(plot_time[-1]))
    span = float(np.ptp(separation_pc))
    axis.set_ylim(
        float(np.min(separation_pc) - 0.08 * span),
        float(np.max(separation_pc) + 0.08 * span),
    )
    axis.tick_params(top=True, right=True)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
