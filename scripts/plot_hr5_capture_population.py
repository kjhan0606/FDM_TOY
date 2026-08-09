#!/usr/bin/env python3
"""Plot the possible HR5 binary-capture population used by the JKAS paper."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.colors import LogNorm


COLORS = ("#000000", "#0072B2", "#D55E00", "#009E73")
LINE_STYLES = ("-", "--", "-.", ":")


def _panel_label(axis: plt.Axes, label: str) -> None:
    text = axis.text(
        0.97,
        0.95,
        label,
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="black",
        fontweight="bold",
        zorder=20,
    )
    text.set_path_effects(
        [path_effects.withStroke(linewidth=1.6, foreground="white")]
    )


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
    with path.open(encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    requested = (
        "assigned_capture_output",
        "assigned_capture_redshift",
        "mass_ratio_last_resolved",
        "chirp_mass_last_resolved_msun",
    )
    missing = sorted(set(requested) - set(header))
    if missing:
        raise ValueError(f"The HR5 capture catalog is missing columns: {missing}")
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=tuple(header.index(name) for name in requested),
        dtype=np.float64,
    )
    return data[:, 0].astype(np.int64), data[:, 1], data[:, 2], data[:, 3]


def make_figure(
    history_path: Path,
    catalog_path: Path,
    output: Path,
    volume_cmpc3: float,
    mass_ratio_output: Path | None = None,
) -> None:
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
            "font.size": 10.0,
            "axes.linewidth": 0.8,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 8.0,
            "legend.title_fontsize": 8.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    capture_redshift_coordinate = np.log10(1.0 + redshift)
    event_redshift_coordinate = np.log10(1.0 + event_redshift)
    coordinate_limits = (np.log10(1.0 + 0.55), np.log10(1.0 + 10.1))

    figure, axes = plt.subplots(
        1, 2, figsize=(7.15, 2.70), gridspec_kw={"wspace": 0.34}
    )

    thresholds = (1.0e4, 1.0e5, 1.0e6, 1.0e7)
    for threshold, color, line_style in zip(thresholds, COLORS, LINE_STYLES):
        counts = np.bincount(event_index[chirp_mass >= threshold], minlength=redshift.size)
        rate = counts / (volume_cmpc3 * interval_gyr)
        axes[0].plot(
            capture_redshift_coordinate[1:],
            rate[1:],
            color=color,
            ls=line_style,
            lw=1.15,
            label=rf"$10^{{{int(np.log10(threshold))}}}$",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlim(*coordinate_limits)
    axes[0].set_ylim(1.0e-6, 3.0e-1)
    axes[0].set_xlabel(r"$\log_{10}(1+z_{\rm cap})$")
    axes[0].set_ylabel(r"$\mathcal{R}_\mathrm{cap}$ [cMpc$^{-3}$ Gyr$^{-1}$]")
    axes[0].legend(
        title=r"$\mathcal{M}_{\rm c}/M_\odot\geq$",
        frameon=False,
        handlelength=1.4,
        labelspacing=0.2,
        columnspacing=0.7,
        handletextpad=0.3,
        ncol=2,
        loc="upper left",
    )
    _panel_label(axes[0], "(a)")

    redshift_coordinate_edges = np.linspace(
        np.log10(1.0 + 0.6), np.log10(1.0 + 10.0), 48
    )
    log_mass_edges = np.linspace(3.8, 9.8, 52)
    histogram, _, _ = np.histogram2d(
        event_redshift_coordinate,
        np.log10(chirp_mass),
        bins=(redshift_coordinate_edges, log_mass_edges),
    )
    mesh = axes[1].pcolormesh(
        redshift_coordinate_edges,
        log_mass_edges,
        histogram.T,
        cmap="viridis",
        norm=LogNorm(vmin=1.0, vmax=max(10.0, float(histogram.max()))),
        shading="flat",
        rasterized=True,
    )
    axes[1].set_xlim(
        redshift_coordinate_edges[0], redshift_coordinate_edges[-1]
    )
    axes[1].set_ylim(3.8, 9.8)
    axes[1].set_xlabel(r"$\log_{10}(1+z_{\rm cap})$")
    axes[1].set_ylabel(r"$\log_{10}(\mathcal{M}_\mathrm{c}/M_\odot)$")
    _panel_label(axes[1], "(b)")
    colorbar = figure.colorbar(mesh, ax=axes[1], pad=0.02, fraction=0.05)
    colorbar.set_label(r"$N$", fontsize=10, labelpad=2)
    colorbar.ax.tick_params(labelsize=10)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(figure)

    if mass_ratio_output is None:
        mass_ratio_output = output.with_name("hr5_capture_mass_ratio.pdf")
    mass_ratio_figure, mass_ratio_axis = plt.subplots(figsize=(3.35, 3.05))
    bins = np.linspace(0.0, 1.0, 41)
    for threshold, color, line_style in zip(thresholds, COLORS, LINE_STYLES):
        selected = chirp_mass >= threshold
        mass_ratio_axis.hist(
            mass_ratio[selected],
            bins=bins,
            density=True,
            histtype="step",
            color=color,
            linestyle=line_style,
            lw=1.15,
            label=rf"$10^{{{int(np.log10(threshold))}}}$",
        )
    mass_ratio_axis.set_xlim(0.0, 1.0)
    mass_ratio_axis.set_yscale("log")
    mass_ratio_axis.set_ylim(0.12, 80.0)
    mass_ratio_axis.set_xlabel(r"mass ratio $q$")
    mass_ratio_axis.set_ylabel(r"$p(q)$")
    mass_ratio_axis.legend(
        title=r"$\mathcal{M}_{\rm c}/M_\odot\geq$",
        frameon=False,
        handlelength=1.4,
        labelspacing=0.2,
        handletextpad=0.3,
        loc="upper right",
    )
    mass_ratio_output.parent.mkdir(parents=True, exist_ok=True)
    mass_ratio_figure.savefig(
        mass_ratio_output,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(mass_ratio_figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("results/hr5/hr5_sink_history.csv"))
    parser.add_argument("--catalog", type=Path, default=Path("results/hr5/hr5_capture_catalog.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/hr5/hr5_capture_population.pdf"))
    parser.add_argument(
        "--mass-ratio-output",
        type=Path,
        default=Path("results/hr5/hr5_capture_mass_ratio.pdf"),
    )
    parser.add_argument("--volume-cmpc3", type=float, default=1.087e7)
    args = parser.parse_args()
    make_figure(
        args.history,
        args.catalog,
        args.output,
        args.volume_cmpc3,
        args.mass_ratio_output,
    )


if __name__ == "__main__":
    main()
