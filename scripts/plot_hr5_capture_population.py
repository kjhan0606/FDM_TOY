#!/usr/bin/env python3
"""Plot the reconstructed HR5 sink-capture population used by the JKAS paper."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


COLORS = ("#000000", "#0072B2", "#D55E00", "#009E73")


def _read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    integer_columns = {"output_number", "capture_count"}
    result: dict[str, np.ndarray] = {}
    for name in rows[0]:
        dtype = np.int64 if name in integer_columns else np.float64
        result[name] = np.asarray([row[name] for row in rows], dtype=dtype)
    return result


def _read_events(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=(2, 3, 6, 7),
        dtype=np.float64,
    )
    return data[:, 0].astype(np.int64), data[:, 1], data[:, 2], data[:, 3]


def make_figure(history_path: Path, catalog_path: Path, output: Path, volume_cmpc3: float) -> None:
    history = _read_history(history_path)
    event_output, event_redshift, mass_ratio, chirp_mass = _read_events(catalog_path)
    valid = np.isfinite(chirp_mass) & np.isfinite(mass_ratio) & (chirp_mass > 0.0)
    event_output = event_output[valid]
    event_redshift = event_redshift[valid]
    mass_ratio = mass_ratio[valid]
    chirp_mass = chirp_mass[valid]

    output_number = history["output_number"]
    event_index = np.searchsorted(output_number, event_output)
    interval_gyr = history["interval_gyr"]
    redshift = history["redshift"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.50), gridspec_kw={"wspace": 0.48})

    thresholds = (1.0e4, 1.0e5, 1.0e6, 1.0e7)
    for threshold, color in zip(thresholds, COLORS):
        counts = np.bincount(event_index[chirp_mass >= threshold], minlength=redshift.size)
        rate = counts / (volume_cmpc3 * interval_gyr)
        axes[0].plot(
            redshift[1:],
            rate[1:],
            color=color,
            lw=1.15,
            label=rf"$\mathcal{{M}}_\mathrm{{c}}\geq 10^{{{int(np.log10(threshold))}}}\,M_\odot$",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlim(0.55, 10.1)
    axes[0].set_ylim(1.0e-6, 3.5e-2)
    axes[0].set_xlabel("capture redshift")
    axes[0].set_ylabel(r"$\mathcal{R}_\mathrm{cap}$ [cMpc$^{-3}$ Gyr$^{-1}$]")
    axes[0].legend(frameon=False, fontsize=6.2, handlelength=1.4, labelspacing=0.25)
    axes[0].text(0.03, 0.95, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold")

    z_edges = np.linspace(0.6, 10.0, 48)
    log_mass_edges = np.linspace(3.8, 9.8, 52)
    histogram, _, _ = np.histogram2d(event_redshift, np.log10(chirp_mass), bins=(z_edges, log_mass_edges))
    mesh = axes[1].pcolormesh(
        z_edges,
        log_mass_edges,
        histogram.T,
        cmap="viridis",
        norm=LogNorm(vmin=1.0, vmax=max(10.0, float(histogram.max()))),
        shading="flat",
        rasterized=True,
    )
    axes[1].set_xlim(0.6, 10.0)
    axes[1].set_ylim(3.8, 9.8)
    axes[1].set_xlabel("capture redshift")
    axes[1].set_ylabel(r"$\log_{10}(\mathcal{M}_\mathrm{c}/M_\odot)$")
    axes[1].text(0.03, 0.95, "(b)", transform=axes[1].transAxes, va="top", color="white", fontweight="bold")
    colorbar = figure.colorbar(mesh, ax=axes[1], pad=0.02, fraction=0.05)
    colorbar.set_label(r"$N$", fontsize=7, labelpad=2)
    colorbar.ax.tick_params(labelsize=6)

    bins = np.linspace(0.0, 1.0, 41)
    for threshold, color in zip(thresholds, COLORS):
        selected = chirp_mass >= threshold
        axes[2].hist(
            mass_ratio[selected],
            bins=bins,
            density=True,
            histtype="step",
            color=color,
            lw=1.15,
        )
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.12, 80.0)
    axes[2].set_xlabel(r"mass ratio $q$")
    axes[2].set_ylabel(r"$p(q)$")
    axes[2].yaxis.set_label_position("right")
    axes[2].text(0.03, 0.95, "(c)", transform=axes[2].transAxes, va="top", fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={"Title": "HR5 numerical SMBH capture population"},
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("results/hr5/hr5_sink_history.csv"))
    parser.add_argument("--catalog", type=Path, default=Path("results/hr5/hr5_capture_catalog.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/hr5/hr5_capture_population.pdf"))
    parser.add_argument("--volume-cmpc3", type=float, default=1.087e7)
    args = parser.parse_args()
    make_figure(args.history, args.catalog, args.output, args.volume_cmpc3)


if __name__ == "__main__":
    main()
