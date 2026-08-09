#!/usr/bin/env python3
"""Measure HR5 dual-AGN probability densities and their later sink captures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from fdm_smbh_delay.hr5 import find_dual_agn_pairs, read_mkagn_snapshot


DEFAULT_AGN_DIRECTORY = Path(
    "/home/kjhan/BACKUP/GalFinder/"
    "SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/SRC(MkAGN)/HR5_AGN_DATA"
)
COLORS = ("#D55E00", "#0072B2", "#009E73")
MARKERS = ("o", "s", "D")


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
    return {
        "output": np.asarray([row["output_number"] for row in rows], dtype=np.int64),
        "time": np.asarray([row["cosmic_time_gyr"] for row in rows], dtype=np.float64),
    }


def _read_capture_links(path: Path, maximum_sink_id: int) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    requested = ("sink_id", "receiver_id", "assigned_capture_output")
    missing = sorted(set(requested) - set(header))
    if missing:
        raise ValueError(f"The HR5 capture catalog is missing columns: {missing}")
    values = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=tuple(header.index(name) for name in requested),
        dtype=np.int64,
    )
    lower = np.minimum(values[:, 0], values[:, 1])
    upper = np.maximum(values[:, 0], values[:, 1])
    base = maximum_sink_id + 1
    key = lower * base + upper
    order = np.argsort(key, kind="stable")
    return key[order], values[:, 2][order]


def _attach_capture_history(
    pairs: dict[str, np.ndarray],
    snapshot_output: int,
    snapshot_time_gyr: float,
    capture_key: np.ndarray,
    capture_output: np.ndarray,
    output_to_time: dict[int, float],
    previous_output: dict[int, int],
    maximum_sink_id: int,
) -> dict[str, np.ndarray]:
    lower = np.minimum(pairs["id_1"], pairs["id_2"])
    upper = np.maximum(pairs["id_1"], pairs["id_2"])
    pair_key = lower * (maximum_sink_id + 1) + upper
    position = np.searchsorted(capture_key, pair_key)
    matched = position < capture_key.size
    matched[matched] &= capture_key[position[matched]] == pair_key[matched]
    event_output = np.full(pair_key.size, -1, dtype=np.int64)
    event_output[matched] = capture_output[position[matched]]
    future = matched & (event_output > snapshot_output)
    delay_lower_gyr = np.full(pair_key.size, np.nan)
    delay_upper_gyr = np.full(pair_key.size, np.nan)
    for pair_number in np.flatnonzero(future):
        assigned_output = int(event_output[pair_number])
        event_time_upper = output_to_time.get(assigned_output)
        event_time_lower = output_to_time.get(previous_output.get(assigned_output, -1))
        if event_time_upper is not None and event_time_lower is not None:
            delay_lower_gyr[pair_number] = max(0.0, event_time_lower - snapshot_time_gyr)
            delay_upper_gyr[pair_number] = event_time_upper - snapshot_time_gyr
    pairs["assigned_capture_output"] = event_output
    pairs["capture_delay_lower_gyr"] = delay_lower_gyr
    pairs["capture_delay_upper_gyr"] = delay_upper_gyr
    pairs["future_direct_capture"] = future
    pairs["capture_within_1_gyr"] = (
        future & np.isfinite(delay_upper_gyr) & (delay_upper_gyr <= 1.0)
    )
    pairs["capture_boundary_crosses_1_gyr"] = (
        future
        & np.isfinite(delay_lower_gyr)
        & (delay_lower_gyr <= 1.0)
        & (delay_upper_gyr > 1.0)
    )
    return pairs


def _pair_probability_density(
    separation_pkpc: np.ndarray,
    active_count: int,
    log_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count, _ = np.histogram(np.log10(separation_pkpc), bins=log_edges)
    width = np.diff(log_edges)
    density = count / (active_count * width)
    error = np.sqrt(count) / (active_count * width)
    return density, error


def _plot(
    pair_sets: dict[int, dict[float, dict[str, np.ndarray]]],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.0,
            "mathtext.fontset": "stix",
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
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.75), gridspec_kw={"wspace": 0.48})
    log_edges = np.linspace(np.log10(0.5), np.log10(30.0), 10)
    center = 10.0 ** (0.5 * (log_edges[:-1] + log_edges[1:]))
    for (output_number, threshold_data), color, marker in zip(pair_sets.items(), COLORS, MARKERS):
        redshift = float(threshold_data[1.0e43]["redshift"])
        for threshold, line_style in ((1.0e43, "-"), (1.0e44, "--")):
            pair = threshold_data[threshold]
            density, error = _pair_probability_density(
                pair["separation_pkpc"], int(pair["active_count"]), log_edges
            )
            selected = density > 0.0
            axes[0].errorbar(
                center[selected],
                density[selected],
                yerr=error[selected],
                marker=marker,
                ms=3.2,
                mfc="white",
                mec=color,
                mew=0.7,
                lw=0.9,
                ls=line_style,
                color=color,
                capsize=1.2,
                capthick=0.6,
            )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(0.5, 30.0)
    axes[0].set_ylim(1.0e-5, 1.0)
    axes[0].set_xlabel(r"$r_{\rm 3D}$ [pkpc]")
    axes[0].set_ylabel(r"$dP_{\rm dual}/d\log_{10}r$")
    redshift_handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            ls="none",
            mfc="white",
            mec=color,
            label=rf"${float(threshold_data[1.0e43]['redshift']):.2f}$",
        )
        for (_, threshold_data), color, marker in zip(pair_sets.items(), COLORS, MARKERS)
    ]
    redshift_legend = axes[0].legend(
        handles=redshift_handles,
        title=r"$z$",
        frameon=False,
        loc="lower center",
        ncol=3,
        columnspacing=0.6,
        handletextpad=0.3,
        borderaxespad=0.3,
    )
    axes[0].add_artist(redshift_legend)
    threshold_handles = [
        Line2D(
            [],
            [],
            color="0.2",
            ls=line_style,
            label=rf"$10^{{{power}}}$",
        )
        for power, line_style in ((43, "-"), (44, "--"))
    ]
    axes[0].legend(
        handles=threshold_handles,
        title=r"$L_{\rm bol}\,[{\rm erg\,s^{-1}}]\geq$",
        frameon=False,
        loc="upper left",
        handletextpad=0.3,
        borderaxespad=0.3,
    )
    _panel_label(axes[0], "(a)")

    for (output_number, threshold_data), color, marker in zip(pair_sets.items(), COLORS, MARKERS):
        pair = threshold_data[1.0e43]
        redshift = float(pair["redshift"])
        if output_number == max(pair_sets):
            continue
        total, _ = np.histogram(np.log10(pair["separation_pkpc"]), bins=log_edges)
        captured, _ = np.histogram(
            np.log10(pair["separation_pkpc"][pair["capture_within_1_gyr"]]), bins=log_edges
        )
        probability = np.divide(
            captured,
            total,
            out=np.full(total.size, np.nan, dtype=np.float64),
            where=total > 0,
        )
        error = np.sqrt(probability * (1.0 - probability) / np.maximum(total, 1))
        selected = total >= 5
        axes[1].errorbar(
            center[selected],
            probability[selected],
            yerr=error[selected],
            color=color,
            marker=marker,
            ms=3.6,
            mfc="white",
            mec=color,
            mew=0.7,
            lw=0.9,
            capsize=1.2,
            capthick=0.6,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.5, 30.0)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel(r"$r_{\rm 3D}$ [pkpc]")
    axes[1].set_ylabel(r"$P(\Delta t_{\rm cap,upper}\leq1\,{\rm Gyr}\mid r)$")
    _panel_label(axes[1], "(b)")

    earliest_output = min(pair_sets)
    pair = pair_sets[earliest_output][1.0e43]
    log_lambda_1 = np.log10(pair["eddington_ratio_1"])
    log_lambda_2 = np.log10(pair["eddington_ratio_2"])
    lambda_edges = np.linspace(-4.0, 0.5, 28)
    histogram, _, _ = np.histogram2d(log_lambda_1, log_lambda_2, bins=(lambda_edges, lambda_edges))
    width = np.diff(lambda_edges)[0]
    density = histogram / (np.sum(histogram) * width**2)
    mesh = axes[2].pcolormesh(
        lambda_edges,
        lambda_edges,
        density.T,
        cmap="magma",
        norm=LogNorm(vmin=max(1.0e-3, np.min(density[density > 0])), vmax=np.max(density)),
        shading="flat",
    )
    axes[2].plot([-4.0, 0.5], [-4.0, 0.5], color="white", ls=":", lw=0.8)
    axes[2].set_xlim(-4.0, 0.5)
    axes[2].set_ylim(-4.0, 0.5)
    axes[2].set_xlabel(r"$\log_{10}\lambda_{\rm Edd,1}$")
    axes[2].set_ylabel(r"$\log_{10}\lambda_{\rm Edd,2}$")
    _panel_label(axes[2], "(c)")
    colorbar = figure.colorbar(mesh, ax=axes[2], pad=0.02, fraction=0.05)
    colorbar.set_label(r"$p(\log\lambda_1,\log\lambda_2)$", fontsize=10)
    colorbar.ax.tick_params(labelsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(figure)


def analyze(
    agn_directory: Path,
    outputs: tuple[int, ...],
    history_path: Path,
    capture_catalog_path: Path,
    output_directory: Path,
    dimensionless_hubble: float,
    box_size_cmpc_over_h: float,
) -> None:
    history = _read_history(history_path)
    output_to_time = dict(zip(history["output"].tolist(), history["time"].tolist()))
    previous_output = {
        int(history["output"][index]): int(history["output"][index - 1])
        for index in range(1, history["output"].size)
    }
    maximum_sink_id = 1_688_677
    capture_key, capture_output = _read_capture_links(capture_catalog_path, maximum_sink_id)
    pair_sets: dict[int, dict[float, dict[str, np.ndarray]]] = {}
    pair_rows = []
    summary: dict[str, object] = {
        "selection": {
            "minimum_separation_pkpc": 0.5,
            "maximum_separation_pkpc": 30.0,
            "luminosity_threshold_erg_s": [1.0e43, 1.0e44],
            "pair_fraction_definition": "N_dual_pair/N_active_AGN",
            "capture_link": "direct legacy receiver assignment",
            "capture_time": "interval between the last resolved and assigned capture outputs",
            "within_1_gyr": "the upper edge of the capture interval is at most 1 Gyr",
        },
        "snapshots": {},
    }
    for output_number in outputs:
        path = agn_directory / f"agn.{output_number:05d}.dat"
        redshift, _, records = read_mkagn_snapshot(path)
        pair_sets[output_number] = {}
        snapshot_summary = {}
        for threshold in (1.0e43, 1.0e44):
            pairs = find_dual_agn_pairs(
                records,
                redshift,
                dimensionless_hubble,
                luminosity_threshold_erg_s=threshold,
                box_size_cmpc_over_h=box_size_cmpc_over_h,
            )
            pairs["redshift"] = np.array(redshift)
            pairs = _attach_capture_history(
                pairs,
                output_number,
                output_to_time[output_number],
                capture_key,
                capture_output,
                output_to_time,
                previous_output,
                maximum_sink_id,
            )
            pair_sets[output_number][threshold] = pairs
            n_pair = pairs["id_1"].size
            n_active = int(pairs["active_count"])
            snapshot_summary[f"Lbol_ge_{threshold:.0e}"] = {
                "active_agn_count": n_active,
                "dual_pair_count": n_pair,
                "dual_pair_fraction": n_pair / n_active,
                "future_direct_capture_count": int(np.count_nonzero(pairs["future_direct_capture"])),
                "capture_within_1_gyr_count": int(np.count_nonzero(pairs["capture_within_1_gyr"])),
                "capture_interval_crosses_1_gyr_count": int(
                    np.count_nonzero(pairs["capture_boundary_crosses_1_gyr"])
                ),
            }
        fiducial = pair_sets[output_number][1.0e43]
        for row_number in range(fiducial["id_1"].size):
            pair_rows.append(
                (
                    output_number,
                    redshift,
                    int(fiducial["id_1"][row_number]),
                    int(fiducial["id_2"][row_number]),
                    fiducial["separation_pkpc"][row_number],
                    fiducial["mass_1_msun"][row_number],
                    fiducial["mass_2_msun"][row_number],
                    fiducial["luminosity_1_erg_s"][row_number],
                    fiducial["luminosity_2_erg_s"][row_number],
                    fiducial["eddington_ratio_1"][row_number],
                    fiducial["eddington_ratio_2"][row_number],
                    int(fiducial["assigned_capture_output"][row_number]),
                    fiducial["capture_delay_lower_gyr"][row_number],
                    fiducial["capture_delay_upper_gyr"][row_number],
                    bool(fiducial["capture_within_1_gyr"][row_number]),
                    bool(fiducial["capture_boundary_crosses_1_gyr"][row_number]),
                )
            )
        summary["snapshots"][str(output_number)] = {"redshift": redshift, **snapshot_summary}
        print(
            f"Output {output_number:05d} at z={redshift:.3f} has "
            f"{fiducial['id_1'].size:,} fiducial dual AGN pairs",
            flush=True,
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "hr5_dual_agn_pairs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "output_number",
                "redshift",
                "primary_sink_id",
                "secondary_sink_id",
                "separation_pkpc",
                "primary_mass_msun",
                "secondary_mass_msun",
                "primary_lbol_erg_s",
                "secondary_lbol_erg_s",
                "primary_eddington_ratio",
                "secondary_eddington_ratio",
                "assigned_capture_output",
                "capture_delay_lower_gyr",
                "capture_delay_upper_gyr",
                "capture_within_1_gyr",
                "capture_interval_crosses_1_gyr",
            )
        )
        writer.writerows(pair_rows)
    (output_directory / "hr5_dual_agn_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot(pair_sets, output_directory / "hr5_dual_agn_probability.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agn-directory", type=Path, default=DEFAULT_AGN_DIRECTORY)
    parser.add_argument("--outputs", nargs="+", type=int, default=(89, 117, 296))
    parser.add_argument("--history", type=Path, default=Path("results/hr5/hr5_sink_history.csv"))
    parser.add_argument("--capture-catalog", type=Path, default=Path("results/hr5/hr5_capture_catalog.csv"))
    parser.add_argument("--output-directory", type=Path, default=Path("results/hr5/dual_agn"))
    parser.add_argument("--dimensionless-hubble", type=float, default=0.684)
    parser.add_argument("--box-size-cmpc-over-h", type=float, default=717.229040)
    args = parser.parse_args()
    analyze(
        args.agn_directory,
        tuple(args.outputs),
        args.history,
        args.capture_catalog,
        args.output_directory,
        args.dimensionless_hubble,
        args.box_size_cmpc_over_h,
    )


if __name__ == "__main__":
    main()
