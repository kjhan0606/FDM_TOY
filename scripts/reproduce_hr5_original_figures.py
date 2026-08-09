#!/usr/bin/env python3
"""Recompute and redraw Figures 1--13 of the original HR5 SMBH draft."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Polygon

from fdm_smbh_delay.hr5 import (
    HEADER_DTYPE,
    SINK_DTYPE,
    binned_source_rate,
    bootstrap_binned_source_rate,
    bootstrap_redshift_rate,
    cumulative_active_sources,
    delayed_redshift,
    fit_redshift_rate,
    histogram_quantiles,
    read_tree_header,
    redshift_rate_model,
)


DEFAULT_TREE = Path(
    "/home/kjhan/BACKUP/GalFinder/"
    "SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/BinarySMBH/"
    "Sink_Merging_Tree.dat.Updated"
)
VOLUME_CMPC3 = 1.087e7
SNAPSHOT_TARGETS = np.array([0.625, 1.0, 2.0, 3.0, 5.0, 6.0])
GROWTH_Z_EDGES = np.array(
    [0.55, 0.75, 0.95, 1.15, 1.4, 1.7, 2.0, 2.3, 2.7, 3.1,
     3.5, 3.9, 4.4, 4.9, 5.5, 6.1, 6.8, 7.6, 8.5]
)
GROWTH_MASS_BINS = np.array(
    [[1.0e4, 3.2e4], [1.0e5, 3.2e5], [1.0e6, 3.2e6], [3.2e6, 1.0e7]]
)
TRACK_MASS_BINS = np.array([[1.0e4, 1.0e5], [1.0e5, 1.0e6], [1.0e6, 1.0e7]])
GROWTH_LOG_RATE_EDGES = np.linspace(-14.0, -7.0, 141)

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY_BLUE = "#56B4E9"
GRAY = "#4D4D4D"
SNAPSHOT_COLORS = (BLUE, VERMILLION, GREEN, ORANGE, PURPLE, SKY_BLUE)
SNAPSHOT_MARKERS = ("o", "s", "D", "^", "v", "P")
THRESHOLD_COLORS = (BLUE, VERMILLION, GREEN)
THRESHOLD_MARKERS = ("o", "s", "D")
THRESHOLD_LINESTYLES = ("-", "--", "-.")
DELAY_COLORS = (BLUE, VERMILLION, GREEN, PURPLE, ORANGE)
DELAY_MARKERS = ("o", "s", "D", "^", "v")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.0,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
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


def _save(figure: plt.Figure, path: Path, _title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(figure)


def _read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No HR5 history rows were found in {path}")
    integer_columns = {
        "history_index",
        "output_number",
        "active_sink_count",
        "seed_birth_count",
        "capture_count",
    }
    result: dict[str, np.ndarray] = {}
    for name in rows[0]:
        dtype = np.int64 if name in integer_columns else np.float64
        result[name] = np.asarray([row[name] for row in rows], dtype=dtype)
    return result


def _read_events(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    requested = (
        "sink_id",
        "receiver_id",
        "assigned_capture_output",
        "assigned_capture_redshift",
        "minor_mass_last_resolved_msun",
        "receiver_mass_last_resolved_msun",
        "chirp_mass_last_resolved_msun",
    )
    missing = sorted(set(requested) - set(header))
    if missing:
        raise ValueError(f"The HR5 capture catalog is missing columns: {missing}")
    columns = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=tuple(header.index(name) for name in requested),
        dtype=np.float64,
        unpack=True,
    )
    names = (
        "sink_id",
        "receiver_id",
        "assigned_capture_output",
        "capture_redshift",
        "minor_mass",
        "major_mass",
        "chirp_mass",
    )
    result = {name: values for name, values in zip(names, columns)}
    for name in ("sink_id", "receiver_id", "assigned_capture_output"):
        result[name] = result[name].astype(np.int64)
    return result


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Return a deterministic 64-bit hash for reproducible history sampling."""

    with np.errstate(over="ignore"):
        x = values.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def _scan_tree(
    tree: Path,
    cache: Path,
    history: dict[str, np.ndarray],
    chunk_records: int,
) -> None:
    """Scan all sink histories once and retain the statistics needed by Figures 2--5."""

    header = read_tree_header(tree)
    nstep = int(header["nstep"])
    nsink = int(header["nsink"])
    redshift = np.asarray(header["redshift"][:nstep], dtype=np.float64)
    if not np.allclose(redshift, history["redshift"], rtol=0.0, atol=2.0e-6):
        raise ValueError("The tree redshifts do not match the extracted HR5 history")
    cosmic_time = history["cosmic_time_gyr"]
    interval_yr = np.diff(cosmic_time) * 1.0e9

    snapshot_indices = np.array(
        [int(np.argmin(np.abs(redshift - target))) for target in SNAPSHOT_TARGETS]
    )
    z3_index = int(np.argmin(np.abs(redshift - 3.0)))
    snapshot_mass = np.zeros((SNAPSHOT_TARGETS.size, nsink), dtype=np.float32)
    growth_histogram = np.zeros(
        (GROWTH_Z_EDGES.size - 1, GROWTH_MASS_BINS.shape[0], GROWTH_LOG_RATE_EDGES.size - 1),
        dtype=np.int64,
    )
    growth_z = 0.5 * (redshift[:-1] + redshift[1:])
    growth_z_bin = np.digitize(growth_z, GROWTH_Z_EDGES) - 1
    track_id_parts: list[list[np.ndarray]] = [[] for _ in range(TRACK_MASS_BINS.shape[0])]
    track_mass_parts: list[list[np.ndarray]] = [[] for _ in range(TRACK_MASS_BINS.shape[0])]
    sampling_limit = np.uint64(np.iinfo(np.uint64).max // 50)

    with tree.open("rb") as stream:
        stream.seek(HEADER_DTYPE.itemsize)
        records_read = 0
        while records_read < nsink:
            count = min(chunk_records, nsink - records_read)
            block = np.fromfile(stream, dtype=SINK_DTYPE, count=count)
            if block.size != count:
                raise ValueError(f"The HR5 tree ended after {records_read + block.size} histories")
            sink_id = block["sink_id"].astype(np.int64)
            sink_index = np.arange(records_read, records_read + count, dtype=np.int64)
            expected_id = sink_index + 1
            unused = sink_id == 0
            if np.any((sink_id != expected_id) & ~unused):
                raise ValueError("A populated HR5 record is inconsistent with its fixed tree index")
            mass = block["state"][:, :nstep, 0]
            if np.any(unused & np.any(mass > 0.0, axis=1)):
                raise ValueError("An unused HR5 record contains a positive sink mass")
            snapshot_mass[:, sink_index] = mass[:, snapshot_indices].T

            mass_at_z3 = mass[:, z3_index]
            hashed = _splitmix64(sink_id)
            for sample_number, (lower, upper) in enumerate(TRACK_MASS_BINS):
                selected = (
                    (mass_at_z3 >= lower)
                    & (mass_at_z3 < upper)
                    & (hashed <= sampling_limit)
                )
                if np.any(selected):
                    track_id_parts[sample_number].append(sink_id[selected].copy())
                    track_mass_parts[sample_number].append(mass[selected].copy())

            earlier_mass = mass[:, :-1]
            later_mass = mass[:, 1:]
            with np.errstate(divide="ignore", invalid="ignore"):
                growth_rate = (later_mass / earlier_mass - 1.0) / interval_yr[None, :]
            mass_bin = np.full(earlier_mass.shape, -1, dtype=np.int8)
            for bin_number, (lower, upper) in enumerate(GROWTH_MASS_BINS):
                selected = (earlier_mass >= lower) & (earlier_mass < upper)
                mass_bin[selected] = bin_number
            with np.errstate(divide="ignore", invalid="ignore"):
                log_rate = np.log10(growth_rate)
            rate_bin = np.full(log_rate.shape, -1, dtype=np.int16)
            finite_rate = np.isfinite(log_rate)
            rate_bin[finite_rate] = np.floor(
                (log_rate[finite_rate] - GROWTH_LOG_RATE_EDGES[0])
                / (GROWTH_LOG_RATE_EDGES[1] - GROWTH_LOG_RATE_EDGES[0])
            ).astype(np.int16)
            valid = (
                (mass_bin >= 0)
                & (rate_bin >= 0)
                & (rate_bin < GROWTH_LOG_RATE_EDGES.size - 1)
            )
            row, column = np.nonzero(valid)
            selected_z_bin = growth_z_bin[column]
            inside_redshift = (selected_z_bin >= 0) & (selected_z_bin < GROWTH_Z_EDGES.size - 1)
            row = row[inside_redshift]
            column = column[inside_redshift]
            selected_z_bin = selected_z_bin[inside_redshift]
            flat_key = (
                (selected_z_bin * GROWTH_MASS_BINS.shape[0] + mass_bin[row, column])
                * (GROWTH_LOG_RATE_EDGES.size - 1)
                + rate_bin[row, column]
            )
            growth_histogram += np.bincount(
                flat_key,
                minlength=growth_histogram.size,
            ).reshape(growth_histogram.shape)

            records_read += count
            if records_read == nsink or records_read % (chunk_records * 50) == 0:
                print(f"Population scan read {records_read:,} of {nsink:,} histories", flush=True)

    payload: dict[str, np.ndarray] = {
        "source_size_bytes": np.array(tree.stat().st_size, dtype=np.int64),
        "source_mtime_ns": np.array(tree.stat().st_mtime_ns, dtype=np.int64),
        "redshift": redshift,
        "snapshot_indices": snapshot_indices,
        "snapshot_redshifts": redshift[snapshot_indices],
        "snapshot_mass": snapshot_mass,
        "growth_histogram": growth_histogram,
    }
    for sample_number in range(TRACK_MASS_BINS.shape[0]):
        payload[f"track_id_{sample_number}"] = np.concatenate(track_id_parts[sample_number])
        payload[f"track_mass_{sample_number}"] = np.concatenate(track_mass_parts[sample_number], axis=0)
        print(
            f"Retained {payload[f'track_id_{sample_number}'].size:,} histories for "
            f"mass sample {sample_number + 1}",
            flush=True,
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    print(f"Wrote population cache to {cache}", flush=True)


def _load_population_cache(
    tree: Path,
    cache: Path,
    history: dict[str, np.ndarray],
    chunk_records: int,
    rebuild: bool,
) -> dict[str, np.ndarray]:
    if rebuild or not cache.exists():
        _scan_tree(tree, cache, history, chunk_records)
    stored = np.load(cache)
    if int(stored["source_size_bytes"]) != tree.stat().st_size:
        raise ValueError("The population cache does not describe the supplied HR5 tree")
    return {name: stored[name] for name in stored.files}


def _bootstrap_mass_functions(
    snapshot_mass: np.ndarray,
    volume_cmpc3: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.arange(4.0, 10.25, 0.25)
    center = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    mass_function = np.zeros((snapshot_mass.shape[0], center.size))
    scatter = np.zeros_like(mass_function)
    rng = np.random.default_rng(seed)
    for snapshot_number, mass in enumerate(snapshot_mass):
        selected = mass[mass > 0.0]
        count, _ = np.histogram(np.log10(selected), bins=edges)
        mass_function[snapshot_number] = count / (volume_cmpc3 * width)
        in_range = int(np.sum(count))
        if in_range:
            bootstrap_count = rng.multinomial(in_range, count / in_range, size=100)
            bootstrap_phi = bootstrap_count / (volume_cmpc3 * width[None, :])
            scatter[snapshot_number] = np.std(bootstrap_phi, axis=0, ddof=1)
    return center, width, mass_function, scatter


def _merger_mass_fraction(
    snapshot_mass: np.ndarray,
    snapshot_indices: np.ndarray,
    history: dict[str, np.ndarray],
    events: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output_to_index = {
        int(output): int(index) for index, output in enumerate(history["output_number"])
    }
    event_index = np.array(
        [output_to_index[int(output)] for output in events["assigned_capture_output"]], dtype=np.int64
    )
    order = np.argsort(event_index, kind="stable")
    event_index = event_index[order]
    receiver_index = events["receiver_id"][order] - 1
    minor_mass = events["minor_mass"][order]
    valid_event = (
        (receiver_index >= 0)
        & (receiver_index < snapshot_mass.shape[1])
        & np.isfinite(minor_mass)
        & (minor_mass > 0.0)
    )
    event_index = event_index[valid_event]
    receiver_index = receiver_index[valid_event]
    minor_mass = minor_mass[valid_event]

    mass_edges = np.arange(6.0, 10.25, 0.5)
    mass_center = 0.5 * (mass_edges[:-1] + mass_edges[1:])
    mean = np.full((snapshot_indices.size, mass_center.size), np.nan)
    standard_deviation = np.full_like(mean, np.nan)
    count = np.zeros_like(mean, dtype=np.int64)
    merger_component = np.zeros(snapshot_mass.shape[1], dtype=np.float64)
    event_cursor = 0
    snapshot_order = np.argsort(snapshot_indices)
    for snapshot_number in snapshot_order:
        step = int(snapshot_indices[snapshot_number])
        event_end = int(np.searchsorted(event_index, step, side="right"))
        np.add.at(
            merger_component,
            receiver_index[event_cursor:event_end],
            minor_mass[event_cursor:event_end],
        )
        event_cursor = event_end
        mass = snapshot_mass[snapshot_number].astype(np.float64)
        active = mass > 0.0
        fraction = np.zeros_like(mass)
        fraction[active] = merger_component[active] / mass[active]
        log_mass = np.full_like(mass, np.nan)
        log_mass[active] = np.log10(mass[active])
        for mass_bin in range(mass_center.size):
            selected = (
                active
                & (log_mass >= mass_edges[mass_bin])
                & (log_mass < mass_edges[mass_bin + 1])
            )
            values = fraction[selected]
            count[snapshot_number, mass_bin] = values.size
            if values.size:
                mean[snapshot_number, mass_bin] = np.mean(values)
                standard_deviation[snapshot_number, mass_bin] = np.std(values)
    return mass_center, mean, standard_deviation, count


def _plot_figure_1(history: dict[str, np.ndarray], output: Path, volume_cmpc3: float) -> None:
    redshift = history["redshift"]
    interval_yr = history["interval_gyr"] * 1.0e9
    birth_rate = history["seed_birth_count"] / (volume_cmpc3 * interval_yr)
    z_edges = np.array(
        [0.55, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75,
         5.25, 5.75, 6.25, 6.75, 7.5, 8.5, 9.5, 10.75]
    )
    center, mean, scatter, half_width = [], [], [], []
    for lower, upper in zip(z_edges[:-1], z_edges[1:]):
        selected = (
            (redshift >= lower)
            & (redshift < upper)
            & np.isfinite(birth_rate)
            & (birth_rate > 0.0)
        )
        if np.any(selected):
            center.append(np.mean(redshift[selected]))
            mean.append(np.mean(birth_rate[selected]))
            scatter.append(np.std(birth_rate[selected]))
            half_width.append(0.5 * (upper - lower))
    figure, axis = plt.subplots(figsize=(3.35, 3.15))
    axis.errorbar(
        center,
        mean,
        xerr=half_width,
        yerr=scatter,
        fmt="D",
        ms=4.2,
        color=BLUE,
        mfc="white",
        mec=GRAY,
        mew=0.7,
        ecolor=BLUE,
        elinewidth=0.65,
        capsize=1.3,
        capthick=0.6,
    )
    axis.set_yscale("log")
    axis.set_xlim(0.0, 11.0)
    axis.set_ylim(3.0e-14, 4.0e-10)
    axis.set_xlabel("redshift")
    axis.set_ylabel(r"$\dot n_{\rm seed}$ [cMpc$^{-3}$ yr$^{-1}$]")
    _save(figure, output, "HR5 SMBH seed birth rate")


def _plot_figure_2(
    mass_center: np.ndarray,
    mean: np.ndarray,
    scatter: np.ndarray,
    count: np.ndarray,
    snapshot_redshifts: np.ndarray,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(3.35, 4.05))
    for number, (color, marker) in enumerate(zip(SNAPSHOT_COLORS[:4], SNAPSHOT_MARKERS[:4])):
        selected = np.isfinite(mean[number]) & (mean[number] > 0.0) & (count[number] >= 5)
        lower = np.minimum(scatter[number, selected], 0.85 * mean[number, selected])
        axis.errorbar(
            10.0**mass_center[selected],
            mean[number, selected],
            xerr=(10.0 ** (mass_center[selected] + 0.25) - 10.0 ** mass_center[selected]),
            yerr=np.vstack([lower, scatter[number, selected]]),
            fmt=marker,
            color=color,
            mfc="white",
            mec=color,
            mew=0.75,
            ms=4.8,
            ecolor=color,
            elinewidth=0.7,
            capsize=1.3,
            capthick=0.6,
            label=rf"$z={snapshot_redshifts[number]:.3g}$",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1.0e6, 1.0e10)
    axis.set_ylim(3.0e-4, 1.0)
    axis.set_xlabel(r"$M_{\rm BH}$ [$M_\odot$]")
    axis.set_ylabel(r"$M_{\rm merge}/M_{\rm BH}$")
    axis.legend(frameon=False, loc="lower right", handletextpad=0.4)
    _save(figure, output, "HR5 merger mass fraction")


def _plot_figure_3(
    mass_center: np.ndarray,
    width: np.ndarray,
    mass_function: np.ndarray,
    scatter: np.ndarray,
    snapshot_redshifts: np.ndarray,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(3.35, 3.35))
    for number, (color, marker) in enumerate(zip(SNAPSHOT_COLORS, SNAPSHOT_MARKERS)):
        selected = mass_function[number] > 0.0
        axis.errorbar(
            10.0**mass_center[selected],
            mass_function[number, selected],
            xerr=(10.0 ** (mass_center[selected] + 0.5 * width[selected]) - 10.0 ** mass_center[selected]),
            yerr=scatter[number, selected],
            fmt=marker,
            ms=3.2,
            color=color,
            mfc="white",
            mec=color,
            mew=0.65,
            ecolor=color,
            elinewidth=0.55,
            capsize=1.0,
            capthick=0.5,
            label=rf"$z={snapshot_redshifts[number]:.3g}$",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1.0e4, 1.0e10)
    axis.set_ylim(1.0e-7, 3.0e-2)
    axis.set_xlabel(r"$M_{\rm BH}$ [$M_\odot$]")
    axis.set_ylabel(r"$\Phi_{\rm BH}$ [cMpc$^{-3}$ dex$^{-1}$]")
    axis.legend(
        frameon=False,
        loc="upper right",
        ncol=2,
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0.3,
    )
    _save(figure, output, "HR5 SMBH mass functions")


def _plot_figure_4(cache: dict[str, np.ndarray], output: Path) -> None:
    redshift = cache["redshift"]
    display_redshifts = np.array([0.625, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0])
    display_index = np.array([np.argmin(np.abs(redshift - z)) for z in display_redshifts])
    z3_index = int(np.argmin(np.abs(redshift - 3.0)))
    colors = THRESHOLD_COLORS
    figure, axis = plt.subplots(figsize=(3.35, 3.35))
    for sample_number, color in enumerate(colors):
        mass = cache[f"track_mass_{sample_number}"].astype(np.float64)
        denominator = mass[:, z3_index]
        ratio = mass[:, display_index] / denominator[:, None]
        ratio[ratio <= 0.0] = np.nan
        quantile = np.nanquantile(ratio, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
        axis.vlines(display_redshifts, quantile[0], quantile[4], color="0.55", lw=0.65)
        axis.fill_between(
            display_redshifts,
            quantile[1],
            quantile[3],
            color=color,
            alpha=0.20,
            edgecolor=color,
            linewidth=0.6,
        )
        lower, upper = TRACK_MASS_BINS[sample_number]
        axis.plot(
            display_redshifts,
            quantile[2],
            marker=THRESHOLD_MARKERS[sample_number],
            ls="none",
            ms=3.1,
            color=color,
            mfc="white",
            mec=color,
            mew=0.7,
            label=rf"$10^{{{int(np.log10(lower))}}}$--$10^{{{int(np.log10(upper))}}}$",
        )
    axis.set_yscale("log")
    axis.set_xlim(0.45, 8.3)
    axis.set_ylim(1.0e-2, 2.0e2)
    axis.set_xlabel("redshift")
    axis.set_ylabel(r"$M_{\rm BH}(z)/M_{\rm BH}(z=3)$")
    axis.legend(
        title=r"$M_{\rm BH}(z=3)/M_\odot$",
        frameon=False,
        loc="upper right",
        handletextpad=0.4,
        borderaxespad=0.3,
    )
    _save(figure, output, "HR5 SMBH mass growth from redshift three")


def _plot_figure_5(cache: dict[str, np.ndarray], output: Path) -> None:
    histogram = cache["growth_histogram"]
    quantile_log = histogram_quantiles(
        histogram,
        GROWTH_LOG_RATE_EDGES,
        (0.025, 0.16, 0.5, 0.84, 0.975),
    )
    quantile = 10.0**quantile_log
    z_center = 0.5 * (GROWTH_Z_EDGES[:-1] + GROWTH_Z_EDGES[1:])
    colors = (BLUE, VERMILLION, GREEN, PURPLE)
    line_styles = ("-", "--", "-.", ":")
    markers = ("o", "s", "D", "^")
    mass_labels = (
        r"$4.0\!-\!4.5$",
        r"$5.0\!-\!5.5$",
        r"$6.0\!-\!6.5$",
        r"$6.5\!-\!7.0$",
    )
    figure, axis = plt.subplots(figsize=(3.35, 3.35))
    for sample_number, (color, line_style, marker, mass_label) in enumerate(
        zip(colors, line_styles, markers, mass_labels)
    ):
        valid = np.isfinite(quantile[:, sample_number, 2])
        axis.fill_between(
            z_center[valid],
            quantile[valid, sample_number, 0],
            quantile[valid, sample_number, 4],
            facecolor="none",
            edgecolor=color,
            linewidth=0.55,
            alpha=0.55,
        )
        axis.fill_between(
            z_center[valid],
            quantile[valid, sample_number, 1],
            quantile[valid, sample_number, 3],
            color=color,
            alpha=0.16,
            linewidth=0.0,
        )
        axis.plot(
            z_center[valid],
            quantile[valid, sample_number, 2],
            marker=marker,
            ms=3.0,
            lw=0.9,
            ls=line_style,
            color=color,
            mfc="white",
            mec=color,
            mew=0.65,
        )
    handles = []
    for color, line_style, marker, mass_label in zip(
        colors, line_styles, markers, mass_labels
    ):
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                ls=line_style,
                mfc="white",
                mec=color,
                ms=4,
                lw=1.0,
                label=mass_label,
            )
        )
    axis.set_yscale("log")
    axis.set_xlim(0.55, 8.0)
    axis.set_ylim(3.0e-14, 1.0e-6)
    axis.set_xlabel("redshift")
    axis.set_ylabel(r"$d(\delta M)/dt$ [yr$^{-1}$]")
    axis.legend(
        handles=handles,
        title=r"$\log_{10}(M_{\rm BH}/M_\odot)$",
        frameon=False,
        loc="upper left",
        handletextpad=0.4,
        borderaxespad=0.3,
    )
    _save(figure, output, "HR5 SMBH fractional mass growth rate")


def _fixed_delay_statistics(
    history: dict[str, np.ndarray],
    events: dict[str, np.ndarray],
    cosmology: FlatLambdaCDM,
    volume_cmpc3: float,
    fit_bootstrap_realizations: int,
    fit_rng: np.random.Generator,
    rate_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[float, dict[float, dict[str, object]]]]:
    output_to_index = {
        int(output): int(index) for index, output in enumerate(history["output_number"])
    }
    event_index = np.array(
        [output_to_index[int(output)] for output in events["assigned_capture_output"]], dtype=np.int64
    )
    capture_time = history["cosmic_time_gyr"][event_index]
    valid_mass = np.isfinite(events["chirp_mass"]) & (events["chirp_mass"] > 0.0)
    redshift_edges = np.geomspace(0.18, 10.5, 19)
    redshift_center = np.sqrt(redshift_edges[:-1] * redshift_edges[1:])
    age = np.asarray(cosmology.age(redshift_edges).value)
    exposure_cmpc3_gyr = volume_cmpc3 * (age[:-1] - age[1:])
    delays = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 2.0, 3.0, 4.0)
    thresholds = (1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8)
    statistics: dict[float, dict[float, dict[str, object]]] = {}
    for delay in delays:
        event_redshift, censored = delayed_redshift(capture_time, delay, cosmology)
        statistics[delay] = {}
        for threshold in thresholds:
            selected = valid_mass & ~censored & (events["chirp_mass"] >= threshold)
            count, rate, error = binned_source_rate(
                event_redshift[selected], redshift_edges, volume_cmpc3, cosmology
            )
            fit = fit_redshift_rate(redshift_center, rate, count)
            fit_bootstrap_samples = bootstrap_redshift_rate(
                redshift_center,
                count,
                exposure_cmpc3_gyr,
                fit_bootstrap_realizations,
                fit_rng,
            )
            rate_bootstrap_quantiles = bootstrap_binned_source_rate(
                count,
                exposure_cmpc3_gyr,
                fit_bootstrap_realizations,
                rate_rng,
            )
            if fit_bootstrap_samples.size:
                fit_bootstrap_quantiles = np.quantile(
                    fit_bootstrap_samples, (0.16, 0.5, 0.84), axis=0
                )
            else:
                fit_bootstrap_quantiles = np.full((3, 4), np.nan)
            statistics[delay][threshold] = {
                "count": count,
                "rate": rate,
                "error": error,
                "fit": fit,
                "fit_bootstrap_quantiles": fit_bootstrap_quantiles,
                "fit_bootstrap_success_count": int(fit_bootstrap_samples.shape[0]),
                "fit_bootstrap_realization_count": fit_bootstrap_realizations,
                "rate_bootstrap_quantiles": rate_bootstrap_quantiles,
                "rate_bootstrap_realization_count": fit_bootstrap_realizations,
                "n_event": int(np.count_nonzero(selected)),
                "n_censored": int(np.count_nonzero(valid_mass & censored & (events["chirp_mass"] >= threshold))),
            }
    return redshift_edges, redshift_center, statistics


def _plot_figure_6(
    redshift_edges: np.ndarray,
    redshift_center: np.ndarray,
    statistics: dict[float, dict[float, dict[str, object]]],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(3.35, 3.35))
    for threshold, color, marker in zip((1.0e4, 1.0e5, 1.0e6), THRESHOLD_COLORS, THRESHOLD_MARKERS):
        data = statistics[0.5][threshold]
        selected = np.asarray(data["count"]) > 0
        rate = np.asarray(data["rate"])
        rate_quantiles = np.asarray(data["rate_bootstrap_quantiles"])
        axis.errorbar(
            redshift_center[selected],
            rate[selected] / 1.0e9,
            xerr=np.vstack(
                [redshift_center[selected] - redshift_edges[:-1][selected],
                 redshift_edges[1:][selected] - redshift_center[selected]]
            ),
            yerr=np.vstack(
                (
                    rate[selected] - rate_quantiles[0, selected],
                    rate_quantiles[2, selected] - rate[selected],
                )
            ) / 1.0e9,
            fmt=marker,
            ms=4.2,
            color=color,
            mfc="white",
            mec=color,
            mew=0.7,
            ecolor=color,
            elinewidth=0.65,
            capsize=1.3,
            capthick=0.6,
            label=rf"$10^{{{int(np.log10(threshold))}}}$",
        )
    axis.text(0.08, 0.92, r"$\Delta t_{\rm ref}=0.5\,$Gyr", transform=axis.transAxes)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(0.18, 10.5)
    axis.set_ylim(3.0e-16, 5.0e-11)
    axis.set_xlabel("reference-event redshift")
    axis.set_ylabel(r"$\Phi$ [cMpc$^{-3}$ yr$^{-1}$]")
    axis.legend(
        title=r"$\mathcal{M}_{\rm c}/M_\odot\geq$",
        frameon=False,
        loc="lower left",
        handletextpad=0.4,
        borderaxespad=0.3,
    )
    _save(figure, output, "HR5 fixed-reference-delay event rate")


def _plot_rate_grid(
    delays: tuple[float, ...],
    redshift_edges: np.ndarray,
    redshift_center: np.ndarray,
    statistics: dict[float, dict[float, dict[str, object]]],
    output: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(3.35, 5.35), sharex=True, gridspec_kw={"hspace": 0.04})
    fit_redshift = np.geomspace(0.16, 10.5, 500)
    for axis, delay in zip(axes, delays):
        for threshold, color, marker, line_style in zip(
            (1.0e4, 1.0e5, 1.0e6),
            THRESHOLD_COLORS,
            THRESHOLD_MARKERS,
            THRESHOLD_LINESTYLES,
        ):
            data = statistics[delay][threshold]
            count = np.asarray(data["count"])
            selected = count > 0
            rate = np.asarray(data["rate"])
            rate_quantiles = np.asarray(data["rate_bootstrap_quantiles"])
            rate_lower = rate_quantiles[0]
            rate_upper = rate_quantiles[2]
            axis.errorbar(
                redshift_center[selected],
                rate[selected] / 1.0e9,
                yerr=np.vstack(
                    (
                        rate[selected] - rate_lower[selected],
                        rate_upper[selected] - rate[selected],
                    )
                ) / 1.0e9,
                fmt=marker,
                color=color,
                ms=3.4,
                mfc="white",
                mec=color,
                mew=0.7,
                ecolor=color,
                elinewidth=0.6,
                capsize=1.2,
                capthick=0.6,
            )
            fit = data["fit"]
            if fit.success:
                axis.plot(
                    fit_redshift,
                    redshift_rate_model(
                        fit_redshift, fit.phi_star, fit.z_star, fit.alpha, fit.beta
                    ) / 1.0e9,
                    color=color,
                    ls=line_style,
                    lw=1.0,
                )
        axis.text(0.08, 0.83, rf"$\Delta t_{{\rm ref}}={delay:g}\,$Gyr", transform=axis.transAxes)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(0.18, 10.5)
        axis.set_ylim(2.0e-16, 5.0e-11)
    axes[1].set_ylabel(r"$\Phi$ [cMpc$^{-3}$ yr$^{-1}$]")
    axes[-1].set_xlabel("reference-event redshift")
    legend = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            ls=line_style,
            lw=1.0,
            ms=4.0,
            mfc="white",
            mec=color,
            label=rf"$10^{{{power}}}$",
        )
        for power, color, marker, line_style in zip(
            (4, 5, 6), THRESHOLD_COLORS, THRESHOLD_MARKERS, THRESHOLD_LINESTYLES
        )
    ]
    axes[-1].legend(
        handles=legend,
        title=r"$\mathcal{M}_{\rm c}/M_\odot\geq$",
        frameon=False,
        loc="lower left",
        ncol=3,
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0.3,
    )
    _save(figure, output, title)


def _plot_fit_parameters(
    delays: tuple[float, ...],
    statistics: dict[float, dict[float, dict[str, object]]],
    output: Path,
    title: str,
) -> None:
    thresholds = np.array([1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8])
    figure, axes = plt.subplots(4, 1, figsize=(3.35, 5.05), sharex=True, gridspec_kw={"hspace": 0.04})
    line_styles = ("-", ":", "--", "-.", (0, (5, 2, 1, 2)))
    for delay, line_style, color, marker in zip(
        delays, line_styles, DELAY_COLORS, DELAY_MARKERS
    ):
        fits = [statistics[delay][threshold]["fit"] for threshold in thresholds]
        direct_parameters = np.asarray(
            [[fit.phi_star, fit.z_star, fit.alpha, fit.beta] for fit in fits]
        )
        quantiles = np.stack(
            [
                np.asarray(statistics[delay][threshold]["fit_bootstrap_quantiles"])
                for threshold in thresholds
            ]
        )
        parameter_indices = (2, 1, 3, 0)
        for axis, parameter_index in zip(axes, parameter_indices):
            scale = 1.0e9 if parameter_index == 0 else 1.0
            direct = direct_parameters[:, parameter_index] / scale
            lower, center, upper = (
                quantiles[:, quantile_index, parameter_index] / scale
                for quantile_index in range(3)
            )
            axis.plot(
                thresholds,
                direct,
                color=color,
                lw=0.9,
                ls=line_style,
                label=rf"$\Delta t_{{\rm ref}}={delay:g}\,$Gyr",
            )
            axis.errorbar(
                thresholds,
                center,
                yerr=np.vstack((center - lower, upper - center)),
                fmt=marker,
                ms=2.3,
                mfc="white",
                mec=color,
                mew=0.65,
                color=color,
                ecolor=color,
                elinewidth=0.55,
                capsize=1.4,
                capthick=0.55,
                ls="none",
            )
    axes[0].set_ylabel(r"$\alpha$")
    axes[0].set_ylim(-8.0, 4.5)
    axes[1].set_ylabel(r"$z_*$")
    axes[2].set_ylabel(r"$\beta$")
    axes[3].set_ylabel(r"$\phi_*$")
    axes[3].set_yscale("log")
    axes[3].set_ylim(1.0e-15, 5.0e-11)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlim(8.0e3, 1.2e8)
        axis.margins(y=0.08)
    axes[-1].set_xlabel(r"chirp-mass threshold $\mathcal{M}_{\rm c}$ [$M_\odot$]")
    legend_handles = [
        Line2D(
            [],
            [],
            color=color,
            ls=line_style,
            marker=marker,
            mfc="white",
            mec=color,
            label=rf"${delay:g}$",
        )
        for delay, line_style, color, marker in zip(
            delays, line_styles, DELAY_COLORS, DELAY_MARKERS
        )
    ]
    axes[0].legend(
        handles=legend_handles,
        title=r"$\Delta t_{\rm ref}$ [Gyr]",
        frameon=False,
        loc="lower left",
        ncol=3,
        columnspacing=0.7,
        handlelength=1.3,
        handletextpad=0.3,
        borderaxespad=0.3,
    )
    _save(figure, output, title)


def _plot_lightcone_cartoon(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(3.35, 3.55))
    axis.set_aspect("equal")
    axis.set_xlim(-3.5, 3.5)
    axis.set_ylim(-0.35, 5.7)
    axis.axis("off")
    theta_low, theta_high = 55.0, 125.0
    radius_inner, radius_outer = 4.45, 5.15
    theta = np.deg2rad(np.linspace(theta_low, theta_high, 200))
    inner = np.c_[radius_inner * np.cos(theta), radius_inner * np.sin(theta)]
    outer = np.c_[radius_outer * np.cos(theta[::-1]), radius_outer * np.sin(theta[::-1])]
    axis.add_patch(Polygon(np.vstack([inner, outer]), closed=True, facecolor="0.82", edgecolor="0.4"))
    for angle in (theta_low, theta_high):
        angle_rad = np.deg2rad(angle)
        axis.plot([0.0, 5.45 * np.cos(angle_rad)], [0.0, 5.45 * np.sin(angle_rad)], color="black", lw=1.0)
    axis.add_patch(Arc((0.0, 0.0), 1.0, 1.0, theta1=theta_low, theta2=theta_high, color="black"))
    axis.text(0.0, 0.62, r"$\Omega$", ha="center")
    axis.text(2.75, 3.65, r"$r$", rotation=55)
    axis.text(3.15, 4.18, r"$r+dr$", rotation=55)
    for angle, radius in ((68.0, 4.78), (90.0, 4.72), (114.0, 4.82)):
        angle_rad = np.deg2rad(angle)
        x, y = radius * np.cos(angle_rad), radius * np.sin(angle_rad)
        for wave_radius, alpha in ((0.30, 0.65), (0.52, 0.4), (0.78, 0.25)):
            axis.add_patch(plt.Circle((x, y), wave_radius, fill=False, color="0.25", lw=0.65, alpha=alpha))
        axis.plot(x, y, marker="*", ms=7.0, mfc="white", mec="black")
    _save(figure, output, "Past-light-cone shell for numerical SMBH captures")


def _plot_cumulative_grid(
    delays: tuple[float, ...],
    statistics: dict[float, dict[float, dict[str, object]]],
    cosmology: FlatLambdaCDM,
    output: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(3.35, 5.35), sharex=True, gridspec_kw={"hspace": 0.04})
    redshift = np.r_[0.0, np.geomspace(1.0e-3, 10.0, 1800)]
    thresholds = (1.0e4, 1.0e5, 1.0e6, 1.0e7)
    colors = (BLUE, VERMILLION, GREEN, PURPLE)
    line_styles = ("-", "--", "-.", ":")
    for axis, delay in zip(axes, delays):
        for threshold, color, line_style in zip(thresholds, colors, line_styles):
            fit = statistics[delay][threshold]["fit"]
            if not fit.success:
                continue
            rate = redshift_rate_model(
                redshift, fit.phi_star, fit.z_star, fit.alpha, fit.beta
            )
            cumulative = cumulative_active_sources(
                redshift,
                rate,
                1.0e4,
                cosmology,
                solid_angle_sr=4.0 * np.pi,
            )
            axis.plot(redshift, cumulative, color=color, ls=line_style, lw=1.0)
        axis.text(0.08, 0.82, rf"$\Delta t_{{\rm ref}}={delay:g}\,$Gyr", transform=axis.transAxes)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(0.18, 10.0)
        axis.set_ylim(5.0e-2, 3.0e5)
    axes[0].text(
        0.97,
        0.08,
        r"all-sky, $\Omega=4\pi\,$sr",
        transform=axes[0].transAxes,
        ha="right",
        fontsize=10.0,
    )
    axes[1].set_ylabel(r"$N_{\rm active}(<z)$")
    axes[-1].set_xlabel(r"maximum survey redshift $z$")
    handles = [
        Line2D(
            [],
            [],
            color=color,
            ls=line_style,
            lw=1.2,
            label=rf"$10^{{{power}}}$",
        )
        for power, color, line_style in zip((4, 5, 6, 7), colors, line_styles)
    ]
    axes[-1].legend(
        handles=handles,
        title=r"$\mathcal{M}_{\rm c}/M_\odot\geq$",
        frameon=False,
        loc="lower right",
        ncol=2,
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0.3,
    )
    _save(figure, output, title)


def _binned_seed_birth_rate(
    history: dict[str, np.ndarray], volume_cmpc3: float
) -> tuple[np.ndarray, np.ndarray]:
    redshift = history["redshift"]
    rate = history["seed_birth_count"] / (
        volume_cmpc3 * history["interval_gyr"] * 1.0e9
    )
    edges = np.array(
        [
            0.55,
            1.25,
            1.75,
            2.25,
            2.75,
            3.25,
            3.75,
            4.25,
            4.75,
            5.25,
            5.75,
            6.25,
            6.75,
            7.5,
            8.5,
            9.5,
            10.75,
        ]
    )
    center = []
    mean = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (
            (redshift >= lower)
            & (redshift < upper)
            & np.isfinite(rate)
            & (rate > 0.0)
        )
        if np.any(selected):
            center.append(float(np.mean(redshift[selected])))
            mean.append(float(np.mean(rate[selected])))
    return np.asarray(center), np.asarray(mean)


def _write_reproduction_validation(
    path: Path,
    history: dict[str, np.ndarray],
    volume_cmpc3: float,
    snapshot_redshifts: np.ndarray,
    merger_center: np.ndarray,
    merger_mean: np.ndarray,
    delay_statistics: dict[float, dict[float, dict[str, object]]],
    cosmology: FlatLambdaCDM,
) -> None:
    seed_redshift, seed_rate = _binned_seed_birth_rate(history, volume_cmpc3)
    seed_peak = int(np.argmax(seed_rate))

    low_redshift_snapshot = int(np.argmin(np.abs(snapshot_redshifts - 0.625)))
    massive_bin = int(np.argmin(np.abs(merger_center - np.log10(3.0e9))))
    merger_fraction = float(merger_mean[low_redshift_snapshot, massive_bin])

    fixed_delay = delay_statistics[0.5][1.0e4]
    fixed_rate = np.asarray(fixed_delay["rate"]) / 1.0e9
    redshift_edges = np.geomspace(0.18, 10.5, 19)
    redshift_center = np.sqrt(redshift_edges[:-1] * redshift_edges[1:])
    fixed_peak = int(np.nanargmax(fixed_rate))

    redshift = np.r_[0.0, np.geomspace(1.0e-3, 10.0, 1800)]
    cumulative: dict[str, float] = {}
    corrected: dict[str, float] = {}
    for threshold in (1.0e4, 1.0e6):
        fit = delay_statistics[0.2][threshold]["fit"]
        rate = redshift_rate_model(redshift, fit.phi_star, fit.z_star, fit.alpha, fit.beta)
        legacy_count = cumulative_active_sources(
            redshift, rate, 1.0e4, cosmology, solid_angle_sr=4.0 * np.pi / 3.0
        )[-1]
        physical_count = cumulative_active_sources(
            redshift, rate, 1.0e4, cosmology, solid_angle_sr=4.0 * np.pi
        )[-1]
        cumulative[f"chirp_mass_ge_{threshold:.0e}"] = float(legacy_count)
        corrected[f"chirp_mass_ge_{threshold:.0e}"] = float(physical_count)

    checks = {
        "figure_1_peak_redshift": {
            "original_reported_range": [3.0, 5.0],
            "reproduced": float(seed_redshift[seed_peak]),
        },
        "figure_2_merger_fraction": {
            "original_reported_range": [0.2, 0.3],
            "original_reference": "z=0.625 and M_BH approximately 3e9",
            "reproduced_mass_bin_center_msun": float(10.0 ** merger_center[massive_bin]),
            "reproduced": merger_fraction,
        },
        "figure_6_peak_rate": {
            "original_visual_range_cmpc3_yr": [5.0e-12, 3.0e-11],
            "reproduced": float(fixed_rate[fixed_peak]),
            "units": "cMpc^-3 yr^-1",
            "reproduced_peak_redshift": float(redshift_center[fixed_peak]),
        },
        "figure_12_count_chirp_mass_ge_1e6": {
            "original_reported_range": [40.0, 100.0],
            "reproduced": cumulative["chirp_mass_ge_1e+06"],
            "comparison_basis": "legacy Omega/3 recomputation rather than the plotted all-sky curve",
        },
        "figure_12_count_chirp_mass_ge_1e4": {
            "original_reported_lower_bound": 1000.0,
            "reproduced": cumulative["chirp_mass_ge_1e+04"],
            "comparison_basis": "legacy Omega/3 recomputation rather than the plotted all-sky curve",
        },
    }
    for check in checks.values():
        value = check["reproduced"]
        if "original_reported_range" in check:
            lower, upper = check["original_reported_range"]
            check["passes_reported_benchmark"] = bool(lower <= value <= upper)
        elif "original_visual_range_cmpc3_yr" in check:
            lower, upper = check["original_visual_range_cmpc3_yr"]
            check["passes_reported_benchmark"] = bool(lower <= value <= upper)
        else:
            check["passes_reported_benchmark"] = bool(
                value >= check["original_reported_lower_bound"]
            )

    payload = {
        "reference": "HR5-BH-ver1.pdf supplied by the author",
        "scope": {
            "pointwise_original_series_available": False,
            "reported_benchmarks": "checked below",
            "figure_definitions_and_selections": "reproduced for Figures 1 through 11",
            "figures_12_and_13": "updated to the physical all-sky volume requested for PTA work",
        },
        "reported_benchmark_checks": checks,
        "lightcone_normalization": {
            "figures_12_and_13": "physical all-sky Omega=4pi sr",
            "physical_differential_volume": "Omega r^2 dr",
            "physical_counts_at_z_10_delay_0p2_gyr": corrected,
            "legacy_Omega_over_3_counts_at_z_10_delay_0p2_gyr": cumulative,
            "physical_to_legacy_Omega_over_3_factor": 3.0,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_fit_table(
    path: Path,
    statistics: dict[float, dict[float, dict[str, object]]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "delay_gyr",
                "chirp_mass_threshold_msun",
                "phi_star_cmpc3_gyr",
                "z_star",
                "alpha",
                "beta",
                "phi_star_q16_cmpc3_gyr",
                "phi_star_q50_cmpc3_gyr",
                "phi_star_q84_cmpc3_gyr",
                "z_star_q16",
                "z_star_q50",
                "z_star_q84",
                "alpha_q16",
                "alpha_q50",
                "alpha_q84",
                "beta_q16",
                "beta_q50",
                "beta_q84",
                "fit_success",
                "fit_bin_count",
                "fit_bootstrap_success_count",
                "fit_bootstrap_realization_count",
                "event_count",
                "future_censored_count",
            )
        )
        for delay, threshold_data in statistics.items():
            for threshold, data in threshold_data.items():
                fit = data["fit"]
                quantiles = np.asarray(data["fit_bootstrap_quantiles"])
                writer.writerow(
                    (
                        delay,
                        threshold,
                        fit.phi_star,
                        fit.z_star,
                        fit.alpha,
                        fit.beta,
                        quantiles[0, 0],
                        quantiles[1, 0],
                        quantiles[2, 0],
                        quantiles[0, 1],
                        quantiles[1, 1],
                        quantiles[2, 1],
                        quantiles[0, 2],
                        quantiles[1, 2],
                        quantiles[2, 2],
                        quantiles[0, 3],
                        quantiles[1, 3],
                        quantiles[2, 3],
                        fit.success,
                        fit.n_bin,
                        data["fit_bootstrap_success_count"],
                        data["fit_bootstrap_realization_count"],
                        data["n_event"],
                        data["n_censored"],
                    )
                )


def _write_rate_bootstrap_table(
    path: Path,
    redshift_edges: np.ndarray,
    redshift_center: np.ndarray,
    statistics: dict[float, dict[float, dict[str, object]]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "delay_gyr",
                "chirp_mass_threshold_msun",
                "redshift_lower",
                "redshift_center",
                "redshift_upper",
                "event_count",
                "rate_cmpc3_gyr",
                "rate_q16_cmpc3_gyr",
                "rate_q50_cmpc3_gyr",
                "rate_q84_cmpc3_gyr",
                "bootstrap_realization_count",
            )
        )
        for delay, threshold_data in statistics.items():
            for threshold, data in threshold_data.items():
                count = np.asarray(data["count"])
                rate = np.asarray(data["rate"])
                quantiles = np.asarray(data["rate_bootstrap_quantiles"])
                for bin_number in range(redshift_center.size):
                    writer.writerow(
                        (
                            delay,
                            threshold,
                            redshift_edges[bin_number],
                            redshift_center[bin_number],
                            redshift_edges[bin_number + 1],
                            count[bin_number],
                            rate[bin_number],
                            quantiles[0, bin_number],
                            quantiles[1, bin_number],
                            quantiles[2, bin_number],
                            data["rate_bootstrap_realization_count"],
                        )
                    )


def reproduce(
    tree: Path,
    history_path: Path,
    catalog_path: Path,
    output_dir: Path,
    cache: Path,
    volume_cmpc3: float,
    chunk_records: int,
    rebuild_cache: bool,
    seed: int,
    fit_bootstrap_realizations: int,
) -> None:
    _style()
    history = _read_history(history_path)
    events = _read_events(catalog_path)
    population = _load_population_cache(
        tree, cache, history, chunk_records, rebuild_cache
    )
    header = read_tree_header(tree)
    cosmology = FlatLambdaCDM(
        H0=float(header["h0"]),
        Om0=float(header["omega_m"]),
        Tcmb0=2.7255,
    )
    snapshot_mass = population["snapshot_mass"]
    snapshot_indices = population["snapshot_indices"]
    snapshot_redshifts = population["snapshot_redshifts"]

    mass_center, mass_width, mass_function, mass_scatter = _bootstrap_mass_functions(
        snapshot_mass, volume_cmpc3, seed
    )
    merger_center, merger_mean, merger_scatter, merger_count = _merger_mass_fraction(
        snapshot_mass, snapshot_indices, history, events
    )
    redshift_edges, redshift_center, delay_statistics = _fixed_delay_statistics(
        history,
        events,
        cosmology,
        volume_cmpc3,
        fit_bootstrap_realizations,
        np.random.default_rng(seed + 10000),
        np.random.default_rng(seed + 20000),
    )

    _plot_figure_1(history, output_dir / "hr5_fig01_seed_birth_rate.pdf", volume_cmpc3)
    _plot_figure_2(
        merger_center,
        merger_mean,
        merger_scatter,
        merger_count,
        snapshot_redshifts,
        output_dir / "hr5_fig02_merger_mass_fraction.pdf",
    )
    _plot_figure_3(
        mass_center,
        mass_width,
        mass_function,
        mass_scatter,
        snapshot_redshifts,
        output_dir / "hr5_fig03_mass_functions.pdf",
    )
    _plot_figure_4(population, output_dir / "hr5_fig04_mass_growth.pdf")
    _plot_figure_5(population, output_dir / "hr5_fig05_growth_rate.pdf")
    _plot_figure_6(
        redshift_edges,
        redshift_center,
        delay_statistics,
        output_dir / "hr5_fig06_fixed_delay_rate.pdf",
    )
    _plot_lightcone_cartoon(output_dir / "hr5_fig07_lightcone_shell.pdf")
    _plot_rate_grid(
        (0.3, 0.2, 0.1, 0.0),
        redshift_edges,
        redshift_center,
        delay_statistics,
        output_dir / "hr5_fig08_rate_fits_short_delay.pdf",
        "HR5 rate fits for short fixed delays",
    )
    _plot_rate_grid(
        (3.0, 2.0, 1.0, 0.5),
        redshift_edges,
        redshift_center,
        delay_statistics,
        output_dir / "hr5_fig09_rate_fits_long_delay.pdf",
        "HR5 rate fits for long fixed delays",
    )
    _plot_fit_parameters(
        (0.0, 0.1, 0.2, 0.3, 0.4),
        delay_statistics,
        output_dir / "hr5_fig10_fit_parameters_short_delay.pdf",
        "HR5 fit parameters for short fixed delays",
    )
    _plot_fit_parameters(
        (0.8, 1.0, 2.0, 3.0, 4.0),
        delay_statistics,
        output_dir / "hr5_fig11_fit_parameters_long_delay.pdf",
        "HR5 fit parameters for long fixed delays",
    )
    _plot_cumulative_grid(
        (0.3, 0.2, 0.1, 0.0),
        delay_statistics,
        cosmology,
        output_dir / "hr5_fig12_cumulative_short_delay.pdf",
        "HR5 active all-sky population for short fixed delays",
    )
    _plot_cumulative_grid(
        (3.0, 2.0, 1.0, 0.5),
        delay_statistics,
        cosmology,
        output_dir / "hr5_fig13_cumulative_long_delay.pdf",
        "HR5 active all-sky population for long fixed delays",
    )
    _write_fit_table(output_dir / "hr5_fixed_delay_fit_parameters.csv", delay_statistics)
    _write_rate_bootstrap_table(
        output_dir / "hr5_fixed_delay_rate_bootstrap.csv",
        redshift_edges,
        redshift_center,
        delay_statistics,
    )
    _write_reproduction_validation(
        output_dir / "hr5_original_figure_validation.json",
        history,
        volume_cmpc3,
        snapshot_redshifts,
        merger_center,
        merger_mean,
        delay_statistics,
        cosmology,
    )
    summary = {
        "tree": str(tree),
        "history": str(history_path),
        "catalog": str(catalog_path),
        "volume_cmpc3": volume_cmpc3,
        "mass_function_bootstrap_realizations": 100,
        "fit_parameter_bootstrap": {
            "method": "independent Poisson resampling of redshift-bin counts",
            "realizations": fit_bootstrap_realizations,
            "central_interval": [0.16, 0.84],
            "seed": seed + 10000,
        },
        "redshift_bin_rate_bootstrap": {
            "method": "independent Poisson resampling of redshift-bin counts",
            "realizations": fit_bootstrap_realizations,
            "central_interval": [0.16, 0.84],
            "seed": seed + 20000,
        },
        "history_sampling_fraction": 0.02,
        "fixed_reference_delay_gyr": list(delay_statistics),
        "chirp_mass_threshold_msun": list(next(iter(delay_statistics.values()))),
        "active_residence_time_yr": 1.0e4,
        "survey_solid_angle_sr": float(4.0 * np.pi),
        "survey_solid_angle_deg2": 41252.96124941927,
        "lightcone_shell_normalization": "physical all-sky Omega r^2 dr",
        "legacy_comparison_shell_normalization": "Omega r^2 dr / 3",
        "figure_count": 13,
    }
    (output_dir / "hr5_original_figure_reproduction.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote 13 reproduced figures to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--history", type=Path, default=Path("results/hr5/hr5_sink_history.csv"))
    parser.add_argument("--catalog", type=Path, default=Path("results/hr5/hr5_capture_catalog.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/hr5/original_figures"))
    parser.add_argument("--cache", type=Path, default=Path("results/hr5/hr5_population_cache.npz"))
    parser.add_argument("--volume-cmpc3", type=float, default=VOLUME_CMPC3)
    parser.add_argument("--chunk-records", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--fit-bootstrap-realizations", type=int, default=200)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    reproduce(
        args.tree,
        args.history,
        args.catalog,
        args.output_dir,
        args.cache,
        args.volume_cmpc3,
        args.chunk_records,
        args.rebuild_cache,
        args.seed,
        args.fit_bootstrap_realizations,
    )


if __name__ == "__main__":
    main()
