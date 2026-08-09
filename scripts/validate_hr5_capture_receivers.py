#!/usr/bin/env python3
"""Validate the surviving SMBHs assigned to HR5 sink disappearances."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from fdm_smbh_delay.hr5 import (
    HEADER_DTYPE,
    SINK_DTYPE,
    infer_capture_receivers,
    read_mkagn_snapshot,
    read_tree_header,
)


DEFAULT_TREE = Path(
    "/home/kjhan/BACKUP/GalFinder/"
    "SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/BinarySMBH/"
    "Sink_Merging_Tree.dat.Updated"
)
DEFAULT_AGN_DIRECTORY = Path(
    "/home/kjhan/BACKUP/GalFinder/"
    "SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/SRC(MkAGN)/HR5_AGN_DATA"
)
COLORS = ("#D55E00", "#0072B2", "#009E73", "#7A5195")
GRAVITATIONAL_CONSTANT_KPC_KMS2_MSUN = 4.300917270e-6


def _plot_settings() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.0,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "pdf.fonttype": 42,
        }
    )


def _panel_label(
    axis: plt.Axes,
    label: str,
    x: float = 0.96,
    y: float = 0.94,
    horizontal_alignment: str = "right",
) -> None:
    label_text = axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha=horizontal_alignment,
        va="top",
        color="black",
        fontweight="bold",
        zorder=30,
    )
    label_text.set_path_effects(
        [path_effects.withStroke(linewidth=1.6, foreground="white")]
    )


def _read_catalog(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    names = (
        "sink_id",
        "receiver_id",
        "last_resolved_history_index",
        "assigned_capture_history_index",
        "last_resolved_output",
        "assigned_capture_output",
        "last_resolved_redshift",
        "assigned_capture_redshift",
        "minor_mass_last_resolved_msun",
        "receiver_mass_last_resolved_msun",
        "receiver_mass_assigned_output_msun",
        "mass_ratio_last_resolved",
        "chirp_mass_last_resolved_msun",
        "minor_x_last_resolved_cmpc",
        "minor_y_last_resolved_cmpc",
        "minor_z_last_resolved_cmpc",
        "minor_vx_last_resolved_kms",
        "minor_vy_last_resolved_kms",
        "minor_vz_last_resolved_kms",
    )
    missing = sorted(set(names) - set(header))
    if missing:
        raise ValueError(f"The capture catalog is missing columns: {missing}")
    values = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=tuple(header.index(name) for name in names),
        dtype=np.float64,
    )
    result = {name: values[:, index] for index, name in enumerate(names)}
    for name in names[:6]:
        result[name] = result[name].astype(np.int64)
    return result


def _read_receiver_states(
    tree: Path,
    header: np.void,
    receiver_id: np.ndarray,
    previous_index: np.ndarray,
    current_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    receiver_index = np.asarray(receiver_id, dtype=np.int64) - 1
    nsink = int(header["nsink"])
    if np.any(receiver_index < 0) or np.any(receiver_index >= nsink):
        raise ValueError("At least one assigned SMBH identifier lies outside the sink tree")
    previous_state = np.full((receiver_index.size, 7), np.nan)
    current_state = np.full_like(previous_state, np.nan)
    order = np.argsort(receiver_index, kind="stable")
    sorted_receiver = receiver_index[order]
    boundary = np.r_[0, np.flatnonzero(np.diff(sorted_receiver)) + 1, order.size]
    with tree.open("rb") as stream:
        for group_number, (begin, end) in enumerate(zip(boundary[:-1], boundary[1:])):
            sink_index = int(sorted_receiver[begin])
            stream.seek(HEADER_DTYPE.itemsize + sink_index * SINK_DTYPE.itemsize)
            surviving = np.fromfile(stream, dtype=SINK_DTYPE, count=1)
            if surviving.size != 1:
                raise ValueError(f"Could not read assigned SMBH record {sink_index + 1}")
            row = order[begin:end]
            previous_state[row] = surviving["state"][0, previous_index[row], :]
            current_state[row] = surviving["state"][0, current_index[row], :]
            if group_number and group_number % 50000 == 0:
                print(f"Read {group_number:,} distinct assigned SMBH histories", flush=True)
    return previous_state, current_state


def _minimum_image(delta: np.ndarray, box_size_cmpc: float) -> np.ndarray:
    return delta - box_size_cmpc * np.rint(delta / box_size_cmpc)


def _quantiles(value: np.ndarray) -> dict[str, float]:
    finite = np.asarray(value)[np.isfinite(value)]
    return {
        name: float(np.quantile(finite, level))
        for name, level in (("q05", 0.05), ("q16", 0.16), ("q50", 0.50), ("q84", 0.84), ("q95", 0.95))
    }


def _disappeared(previous_id: np.ndarray, current_id: np.ndarray) -> np.ndarray:
    position = np.searchsorted(current_id, previous_id)
    survives = position < current_id.size
    survives[survives] &= current_id[position[survives]] == previous_id[survives]
    return np.flatnonzero(~survives)


def _cross_validate_consecutive_outputs(
    tree: Path,
    header: np.void,
    agn_directory: Path,
    dimensionless_hubble: float,
    box_size_cmpc: float,
) -> dict[str, object]:
    output_numbers = np.asarray(
        header["output_number"][: int(header["nstep"])], dtype=np.int64
    )
    tree_map = np.memmap(
        tree,
        mode="r",
        dtype=SINK_DTYPE,
        offset=HEADER_DTYPE.itemsize,
        shape=(int(header["nsink"]),),
    )
    rows = []
    for current_output in range(21, 27):
        previous_path = agn_directory / f"agn.{current_output - 1:05d}.dat"
        current_path = agn_directory / f"agn.{current_output:05d}.dat"
        if not previous_path.exists() or not current_path.exists():
            continue
        _, _, previous = read_mkagn_snapshot(previous_path)
        _, _, current = read_mkagn_snapshot(current_path)
        previous.sort(order="sink_id")
        current.sort(order="sink_id")
        disappeared = previous[_disappeared(previous["sink_id"], current["sink_id"])]
        inferred = infer_capture_receivers(
            disappeared["sink_id"],
            disappeared["mass"] / dimensionless_hubble,
            np.column_stack((disappeared["x"], disappeared["y"], disappeared["z"]))
            / dimensionless_hubble,
            current["sink_id"],
            current["mass"] / dimensionless_hubble,
            np.column_stack((current["x"], current["y"], current["z"]))
            / dimensionless_hubble,
            box_size_cmpc_over_h=box_size_cmpc,
        )
        history_position = np.flatnonzero(output_numbers == current_output)
        if history_position.size != 1:
            raise ValueError(f"Output {current_output} is not unique in the legacy tree")
        legacy = tree_map[disappeared["sink_id"].astype(np.int64) - 1]
        legacy_is_event = legacy["capture_index"] == int(history_position[0])
        exact = legacy_is_event & (legacy["receiver_id"] == inferred)
        rows.append(
            {
                "previous_output": current_output - 1,
                "current_output": current_output,
                "disappearance_count": int(disappeared.size),
                "inferred_receiver_count": int(np.count_nonzero(inferred)),
                "legacy_event_count": int(np.count_nonzero(legacy_is_event)),
                "exact_receiver_count": int(np.count_nonzero(exact)),
            }
        )
    del tree_map
    total_inferred = sum(row["inferred_receiver_count"] for row in rows)
    total_exact = sum(row["exact_receiver_count"] for row in rows)
    return {
        "outputs": rows,
        "inferred_receiver_count": total_inferred,
        "exact_receiver_count": total_exact,
        "exact_receiver_fraction": total_exact / total_inferred if total_inferred else None,
    }


def _write_validation_catalog(path: Path, data: dict[str, np.ndarray]) -> None:
    columns = (
        "sink_id",
        "receiver_id",
        "assigned_capture_output",
        "last_resolved_separation_pkpc",
        "assignment_search_distance_cmpc",
        "relative_speed_last_resolved_kms",
        "escape_speed_last_resolved_kms",
        "speed_to_escape_ratio_last_resolved",
        "receiver_mass_factor_at_assignment",
        "simultaneous_assignment_multiplicity",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*(data[name] for name in columns)))


def _plot_validation(path: Path, data: dict[str, np.ndarray]) -> None:
    _plot_settings()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 5.55),
        gridspec_kw={"wspace": 0.36, "hspace": 0.38},
    )
    axes = axes.ravel()
    separation_edges = np.logspace(-1, 4, 46)
    for selected, color, line_style, label in (
        (np.ones(data["sink_id"].size, dtype=bool), COLORS[0], "-", "possible binary captures"),
        (data["chirp_mass_last_resolved_msun"] >= 1.0e6, COLORS[1], "--", r"$\mathcal{M}_{\rm c}\geq10^6\,M_\odot$"),
    ):
        axes[0].hist(
            data["last_resolved_separation_pkpc"][selected],
            bins=separation_edges,
            density=True,
            histtype="step",
            color=color,
            ls=line_style,
            lw=1.2,
            label=label,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"separation at last common output [pkpc]")
    axes[0].set_ylabel(r"probability density")
    axes[0].legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 1.0),
        borderaxespad=0.0,
        fontsize=7.0,
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    _panel_label(axes[0], "(a)", x=0.04, horizontal_alignment="left")

    ratio = data["speed_to_escape_ratio_last_resolved"]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    axes[1].hist(
        ratio,
        bins=np.logspace(-3, 4, 55),
        density=True,
        histtype="step",
        color=COLORS[1],
        lw=1.2,
    )
    axes[1].axvline(1.0, color="black", lw=0.9, ls=":", label=r"$v_{\rm rel}=v_{\rm esc}$")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$v_{\rm rel}/v_{\rm esc}$ at last common output")
    axes[1].set_ylabel(r"probability density")
    axes[1].legend(frameon=False, loc="upper right")
    _panel_label(axes[1], "(b)", x=0.10)

    distance = data["assignment_search_distance_cmpc"]
    axes[2].hist(
        distance[np.isfinite(distance) & (distance > 0.0)],
        bins=np.logspace(-4, np.log10(0.5), 50),
        density=True,
        histtype="step",
        color=COLORS[2],
        lw=1.2,
    )
    axes[2].axvline(0.5, color="black", lw=0.9, ls=":", label="search limit")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"distance used to assign companion [cMpc]")
    axes[2].set_ylabel(r"probability density")
    axes[2].legend(frameon=False, loc="upper left", fontsize=7.0)
    _panel_label(axes[2], "(c)")

    multiplicity = data["simultaneous_assignment_multiplicity"].astype(np.int64)
    value, count = np.unique(multiplicity, return_counts=True)
    visible = value <= 6
    axes[3].bar(
        value[visible],
        count[visible] / multiplicity.size,
        width=0.72,
        facecolor="white",
        edgecolor=COLORS[3],
        linewidth=1.1,
    )
    axes[3].set_yscale("log")
    axes[3].set_xlabel(r"disappearances assigned to same SMBH")
    axes[3].set_ylabel(r"event fraction")
    axes[3].set_xticks(value[visible])
    _panel_label(axes[3], "(d)")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)


def validate(
    tree: Path,
    capture_catalog: Path,
    agn_directory: Path,
    output_directory: Path,
    dimensionless_hubble: float,
    box_size_cmpc: float,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    header = read_tree_header(tree)
    catalog = _read_catalog(capture_catalog)
    previous_state, current_state = _read_receiver_states(
        tree,
        header,
        catalog["receiver_id"],
        catalog["last_resolved_history_index"],
        catalog["assigned_capture_history_index"],
    )
    minor_position = np.column_stack(
        (
            catalog["minor_x_last_resolved_cmpc"],
            catalog["minor_y_last_resolved_cmpc"],
            catalog["minor_z_last_resolved_cmpc"],
        )
    )
    minor_velocity = np.column_stack(
        (
            catalog["minor_vx_last_resolved_kms"],
            catalog["minor_vy_last_resolved_kms"],
            catalog["minor_vz_last_resolved_kms"],
        )
    )
    previous_delta = _minimum_image(previous_state[:, 1:4] - minor_position, box_size_cmpc)
    current_delta = _minimum_image(current_state[:, 1:4] - minor_position, box_size_cmpc)
    last_separation_pkpc = (
        np.linalg.norm(previous_delta, axis=1)
        * 1000.0
        / (1.0 + catalog["last_resolved_redshift"])
    )
    assignment_distance_cmpc = np.linalg.norm(current_delta, axis=1)
    relative_speed = np.linalg.norm(previous_state[:, 4:7] - minor_velocity, axis=1)
    total_mass = (
        catalog["minor_mass_last_resolved_msun"]
        + catalog["receiver_mass_last_resolved_msun"]
    )
    escape_speed = np.sqrt(
        np.divide(
            2.0 * GRAVITATIONAL_CONSTANT_KPC_KMS2_MSUN * total_mass,
            last_separation_pkpc,
            out=np.full(total_mass.size, np.nan),
            where=last_separation_pkpc > 0.0,
        )
    )
    speed_ratio = relative_speed / escape_speed
    mass_factor = catalog["receiver_mass_assigned_output_msun"] / catalog[
        "minor_mass_last_resolved_msun"
    ]
    group_key = (
        catalog["receiver_id"] * (int(np.max(catalog["assigned_capture_output"])) + 1)
        + catalog["assigned_capture_output"]
    )
    _, inverse, group_count = np.unique(group_key, return_inverse=True, return_counts=True)
    simultaneous_multiplicity = group_count[inverse]
    result = {
        **catalog,
        "last_resolved_separation_pkpc": last_separation_pkpc,
        "assignment_search_distance_cmpc": assignment_distance_cmpc,
        "relative_speed_last_resolved_kms": relative_speed,
        "escape_speed_last_resolved_kms": escape_speed,
        "speed_to_escape_ratio_last_resolved": speed_ratio,
        "receiver_mass_factor_at_assignment": mass_factor,
        "simultaneous_assignment_multiplicity": simultaneous_multiplicity,
    }
    finite_phase_space = (
        np.isfinite(last_separation_pkpc)
        & np.isfinite(relative_speed)
        & np.isfinite(escape_speed)
        & (escape_speed > 0.0)
    )
    cross_validation = _cross_validate_consecutive_outputs(
        tree,
        header,
        agn_directory,
        dimensionless_hubble,
        box_size_cmpc,
    )
    summary = {
        "source_tree": str(tree),
        "capture_catalog": str(capture_catalog),
        "event_count": int(catalog["sink_id"].size),
        "phase_space_event_count": int(np.count_nonzero(finite_phase_space)),
        "receiver_mass_factor_ge_2_fraction": float(np.mean(mass_factor >= 2.0)),
        "assignment_within_0p5_cmpc_fraction": float(
            np.mean(assignment_distance_cmpc <= 0.5 + 1.0e-10)
        ),
        "last_resolved_speed_below_escape_fraction": float(
            np.mean(speed_ratio[finite_phase_space] <= 1.0)
        ),
        "simultaneous_multiple_assignment_fraction": float(
            np.mean(simultaneous_multiplicity > 1)
        ),
        "last_resolved_separation_pkpc": _quantiles(last_separation_pkpc),
        "assignment_search_distance_cmpc": _quantiles(assignment_distance_cmpc),
        "relative_speed_last_resolved_kms": _quantiles(relative_speed),
        "speed_to_escape_ratio_last_resolved": _quantiles(speed_ratio),
        "receiver_mass_factor_at_assignment": _quantiles(mass_factor),
        "cross_validation_outputs_20_to_26": cross_validation,
        "interpretation": {
            "assignment": "The assigned surviving SMBH follows the legacy mkmerging.c distance and mass criteria rather than a direct record of the partner selected by RAMSES.",
            "last_resolved_phase_space": "The phase-space test precedes the unresolved removal interval and does not reproduce the instantaneous simulation merger criterion.",
        },
    }
    _write_validation_catalog(output_directory / "hr5_receiver_validation.csv", result)
    (output_directory / "hr5_receiver_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot_validation(output_directory / "hr5_receiver_validation.pdf", result)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument(
        "--capture-catalog",
        type=Path,
        default=Path("results/hr5/hr5_capture_catalog.csv"),
    )
    parser.add_argument("--agn-directory", type=Path, default=DEFAULT_AGN_DIRECTORY)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/hr5/receiver_validation"),
    )
    parser.add_argument("--dimensionless-hubble", type=float, default=0.684)
    parser.add_argument("--box-size-cmpc", type=float, default=1048.5)
    args = parser.parse_args()
    validate(
        args.tree,
        args.capture_catalog,
        args.agn_directory,
        args.output_directory,
        args.dimensionless_hubble,
        args.box_size_cmpc,
    )


if __name__ == "__main__":
    main()
