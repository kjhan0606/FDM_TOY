#!/usr/bin/env python3
"""Measure HR5 dual and offset AGN demographics and later sink captures."""

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
from astropy.cosmology import FlatLambdaCDM

from fdm_smbh_delay.hr5 import (
    fibonacci_sightlines,
    find_agn_pair_population,
    interval_censored_cumulative_bounds,
    match_population_by_properties,
    pair_component_multiplicity,
    pair_component_labels,
    project_pair_observables,
    read_mkagn_snapshot,
    spatial_jackknife_pair_statistics,
)


DEFAULT_AGN_DIRECTORY = Path(
    "/home/kjhan/BACKUP/GalFinder/"
    "SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/SRC(MkAGN)/HR5_AGN_DATA"
)
FIDUCIAL_VOLUME_CMPC3 = 1.087e7
COLORS = ("#D55E00", "#0072B2", "#009E73", "#7A5195")
MARKERS = ("o", "s", "D", "^")


def _panel_label(
    axis: plt.Axes,
    label: str,
    *,
    x: float = 0.97,
    y: float = 0.95,
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


def _plot_settings(font_size: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": font_size,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "legend.title_fontsize": font_size,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "pdf.fonttype": 42,
        }
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
    key = lower * (maximum_sink_id + 1) + upper
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
    return pairs


def _add_dual_system_information(pairs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pair_label, pair_system_multiplicity, _, _ = pair_component_labels(
        pairs["id_1"], pairs["id_2"]
    )
    pairs["pair_system_label"] = pair_label
    pairs["pair_system_multiplicity"] = pair_system_multiplicity
    pairs["relative_speed_kms"] = np.linalg.norm(
        pairs["velocity_2_kms"] - pairs["velocity_1_kms"], axis=1
    )
    dual = pairs["is_dual"]
    pair_multiplicity, member, member_multiplicity = pair_component_multiplicity(
        pairs["id_1"][dual], pairs["id_2"][dual]
    )
    multiplicity = np.zeros(dual.size, dtype=np.int64)
    multiplicity[dual] = pair_multiplicity
    pairs["dual_system_multiplicity"] = multiplicity
    pairs["dual_member_id"] = member
    pairs["dual_member_multiplicity"] = member_multiplicity
    return pairs


def _add_projection_information(
    pairs: dict[str, np.ndarray],
    redshift: float,
    dimensionless_hubble: float,
    cosmology: FlatLambdaCDM,
    sightlines: np.ndarray,
    box_size_cmpc_over_h: float,
) -> dict[str, np.ndarray]:
    projected, line_velocity = project_pair_observables(
        pairs["position_1_cmpc_over_h"],
        pairs["position_2_cmpc_over_h"],
        pairs["velocity_1_kms"],
        pairs["velocity_2_kms"],
        sightlines,
        redshift,
        dimensionless_hubble,
        float(cosmology.H(redshift).value),
        box_size_cmpc_over_h,
    )
    radial_selection = (projected >= 0.5) & (projected <= 30.0)
    pairs["projected_selection_probability_dv300"] = np.mean(
        radial_selection & (line_velocity <= 300.0), axis=1
    )
    pairs["projected_selection_probability_dv600"] = np.mean(
        radial_selection & (line_velocity <= 600.0), axis=1
    )
    return pairs


def _read_nhi_sightlines(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"Unexpected N_H table shape in {path}")
    sink_id = values[:, 0].astype(np.int64)
    order = np.argsort(sink_id, kind="stable")
    return sink_id[order], values[order, 1:]


def _lookup_nhi(
    requested_id: np.ndarray,
    sorted_id: np.ndarray,
    sorted_nhi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.searchsorted(sorted_id, requested_id)
    available = position < sorted_id.size
    available[available] &= sorted_id[position[available]] == requested_id[available]
    result = np.full((requested_id.size, sorted_nhi.shape[1]), np.nan)
    result[available] = sorted_nhi[position[available]]
    return result, available


def _add_obscuration_information(
    pairs: dict[str, np.ndarray],
    sorted_id: np.ndarray,
    sorted_nhi: np.ndarray,
) -> dict[str, np.ndarray]:
    first_nhi, first_available = _lookup_nhi(pairs["id_1"], sorted_id, sorted_nhi)
    second_nhi, second_available = _lookup_nhi(pairs["id_2"], sorted_id, sorted_nhi)
    active_available = (
        (~pairs["active_1"] | first_available)
        & (~pairs["active_2"] | second_available)
    )
    for threshold, suffix in ((1.0e23, "nhi_ge_1e23"), (1.0e24, "nhi_ge_1e24")):
        obscured_count = np.zeros(pairs["id_1"].size)
        sightline_count = np.zeros(pairs["id_1"].size)
        obscured_count += np.sum(first_nhi >= threshold, axis=1) * pairs["active_1"]
        obscured_count += np.sum(second_nhi >= threshold, axis=1) * pairs["active_2"]
        sightline_count += sorted_nhi.shape[1] * pairs["active_1"]
        sightline_count += sorted_nhi.shape[1] * pairs["active_2"]
        fraction = np.divide(
            obscured_count,
            sightline_count,
            out=np.full(obscured_count.size, np.nan),
            where=(sightline_count > 0.0) & active_available,
        )
        pairs[f"active_sightline_fraction_{suffix}"] = fraction
    pairs["nhi_available"] = active_available
    return pairs


def _selection_statistics(
    pairs: dict[str, np.ndarray],
    volume_cmpc3: float,
    box_size_cmpc_over_h: float,
    spatial_region_count: int = 8,
) -> dict[str, float | int]:
    dual = pairs["is_dual"]
    offset = pairs["is_offset"]
    active_count = int(pairs["active_count"])
    dual_count = int(np.count_nonzero(dual))
    member = pairs["dual_member_id"]
    member_multiplicity = pairs["dual_member_multiplicity"]
    pure_member_count = int(np.count_nonzero(member_multiplicity == 2))
    multiple_member_count = int(np.count_nonzero(member_multiplicity > 2))
    spatial = spatial_jackknife_pair_statistics(
        pairs["active_position_x_cmpc_over_h"],
        pairs["position_1_cmpc_over_h"][:, 0],
        pairs["position_2_cmpc_over_h"][:, 0],
        dual,
        volume_cmpc3,
        box_size_cmpc_over_h,
        region_count=spatial_region_count,
    )
    return {
        "active_agn_count": active_count,
        "dual_pair_count": dual_count,
        "offset_pair_count": int(np.count_nonzero(offset)),
        "dual_pair_number_density_cmpc3": spatial["number_density"],
        "dual_pair_number_density_jackknife_error_cmpc3": spatial[
            "number_density_jackknife_error"
        ],
        "dual_pair_fraction": spatial["pair_fraction"],
        "dual_pair_fraction_jackknife_error": spatial[
            "pair_fraction_jackknife_error"
        ],
        "spatial_jackknife_region_count": spatial["region_count"],
        "pure_two_member_dual_pair_count": int(
            np.count_nonzero(dual & (pairs["pair_system_multiplicity"] == 2))
        ),
        "pure_two_member_offset_pair_count": int(
            np.count_nonzero(offset & (pairs["pair_system_multiplicity"] == 2))
        ),
        "dual_member_count": int(member.size),
        "dual_member_fraction": member.size / active_count if active_count else np.nan,
        "pure_dual_member_count": pure_member_count,
        "pure_dual_member_fraction": pure_member_count / active_count if active_count else np.nan,
        "multiple_agn_member_count": multiple_member_count,
        "multiple_agn_member_fraction": multiple_member_count / active_count if active_count else np.nan,
    }


def _population_diagnostics(pairs: dict[str, np.ndarray]) -> dict[str, object]:
    """Summarize pair properties used in the physical interpretation."""

    result: dict[str, object] = {}
    mass_ratio = pairs["mass_2_msun"] / pairs["mass_1_msun"]
    for population_name, selected in (
        ("dual", pairs["is_dual"]),
        ("offset", pairs["is_offset"]),
    ):
        if not np.any(selected):
            result[population_name] = {"pair_count": 0}
            continue
        result[population_name] = {
            "pair_count": int(np.count_nonzero(selected)),
            "median_separation_pkpc": float(np.median(pairs["separation_pkpc"][selected])),
            "median_mass_ratio": float(np.median(mass_ratio[selected])),
            "brighter_secondary_fraction": float(
                np.mean(pairs["lbol_2_erg_s"][selected] > pairs["lbol_1_erg_s"][selected])
            ),
            "mean_projected_selection_probability_dv300": float(
                np.mean(pairs["projected_selection_probability_dv300"][selected])
            ),
            "mean_projected_selection_probability_dv600": float(
                np.mean(pairs["projected_selection_probability_dv600"][selected])
            ),
            "direct_numerical_capture_count": int(
                np.count_nonzero(pairs["future_direct_capture"][selected])
            ),
        }
        for field, label in (
            ("active_sightline_fraction_nhi_ge_1e23", "mean_active_sightline_fraction_nhi_ge_1e23"),
            ("active_sightline_fraction_nhi_ge_1e24", "mean_active_sightline_fraction_nhi_ge_1e24"),
        ):
            value = pairs[field][selected]
            result[population_name][label] = (
                float(np.nanmean(value)) if np.any(np.isfinite(value)) else None
            )
    return result


def _density_per_log_separation(
    separation_pkpc: np.ndarray,
    denominator: int,
    log_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count, _ = np.histogram(np.log10(separation_pkpc), bins=log_edges)
    density = count / (denominator * np.diff(log_edges))
    error = np.sqrt(count) / (denominator * np.diff(log_edges))
    return density, error


def _binned_mean_interval(
    coordinate: np.ndarray,
    value: np.ndarray,
    edges: np.ndarray,
    rng: np.random.Generator,
    realizations: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinate = np.asarray(coordinate)
    value = np.asarray(value)
    mean = np.full(edges.size - 1, np.nan)
    lower = np.full(edges.size - 1, np.nan)
    upper = np.full(edges.size - 1, np.nan)
    count = np.zeros(edges.size - 1, dtype=np.int64)
    for bin_number in range(edges.size - 1):
        selected = (
            (coordinate >= edges[bin_number])
            & (coordinate < edges[bin_number + 1])
            & np.isfinite(value)
        )
        sample = value[selected]
        count[bin_number] = sample.size
        if sample.size == 0:
            continue
        mean[bin_number] = np.mean(sample)
        index = rng.integers(0, sample.size, size=(realizations, sample.size))
        bootstrap_mean = np.mean(sample[index], axis=1)
        lower[bin_number], upper[bin_number] = np.quantile(bootstrap_mean, (0.16, 0.84))
    return count, mean, lower, upper


def _bootstrap_capture_bounds(
    event_lower_gyr: np.ndarray,
    event_upper_gyr: np.ndarray,
    time_grid_gyr: np.ndarray,
    followup_gyr: float,
    rng: np.random.Generator,
    realizations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower_samples = np.empty((realizations, time_grid_gyr.size))
    upper_samples = np.empty_like(lower_samples)
    for realization in range(realizations):
        index = rng.integers(0, event_lower_gyr.size, size=event_lower_gyr.size)
        lower_samples[realization], upper_samples[realization] = (
            interval_censored_cumulative_bounds(
                event_lower_gyr[index],
                event_upper_gyr[index],
                time_grid_gyr,
                followup_gyr,
            )
        )
    lower_quantile = np.nanquantile(lower_samples, (0.16, 0.84), axis=0)
    upper_quantile = np.nanquantile(upper_samples, (0.16, 0.84), axis=0)
    return lower_quantile[0], lower_quantile[1], upper_quantile[0], upper_quantile[1]


def _standardized_mean_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    numerator = np.mean(first, axis=0) - np.mean(second, axis=0)
    denominator = np.sqrt(
        0.5 * (np.var(first, axis=0, ddof=1) + np.var(second, axis=0, ddof=1))
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def _matched_capture_comparison(
    pairs: dict[str, np.ndarray],
    followup_gyr: float,
    rng: np.random.Generator,
    realizations: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    pure = pairs["pair_system_multiplicity"] == 2
    dual_index = np.flatnonzero(pure & pairs["is_dual"])
    offset_index = np.flatnonzero(pure & pairs["is_offset"])
    feature_names = (
        "log10_primary_mass_msun",
        "log10_mass_ratio",
        "log10_separation_pkpc",
        "log10_relative_speed_plus_10_kms",
    )

    def properties(index: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                np.log10(pairs["mass_1_msun"][index]),
                np.log10(pairs["mass_2_msun"][index] / pairs["mass_1_msun"][index]),
                np.log10(pairs["separation_pkpc"][index]),
                np.log10(pairs["relative_speed_kms"][index] + 10.0),
            )
        )

    dual_properties = properties(dual_index)
    offset_properties = properties(offset_index)
    dual_match, offset_match, match_distance = match_population_by_properties(
        dual_properties, offset_properties
    )
    matched_dual_index = dual_index[dual_match]
    matched_offset_index = offset_index[offset_match]
    evaluation_time = min(1.0, followup_gyr)

    def outcomes(index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = pairs["capture_delay_lower_gyr"][index]
        upper = pairs["capture_delay_upper_gyr"][index]
        certain = np.isfinite(upper) & (upper <= evaluation_time)
        possible = np.isfinite(lower) & (lower <= evaluation_time)
        return certain.astype(np.float64), possible.astype(np.float64)

    dual_certain, dual_possible = outcomes(matched_dual_index)
    offset_certain, offset_possible = outcomes(matched_offset_index)
    difference_lower = dual_certain - offset_possible
    difference_upper = dual_possible - offset_certain
    difference_midpoint = 0.5 * (difference_lower + difference_upper)
    bootstrap_midpoint = np.empty(realizations)
    for realization in range(realizations):
        index = rng.integers(0, difference_midpoint.size, size=difference_midpoint.size)
        bootstrap_midpoint[realization] = np.mean(difference_midpoint[index])

    matched_dual_properties = dual_properties[dual_match]
    matched_offset_properties = offset_properties[offset_match]
    pre_match_smd = _standardized_mean_difference(dual_properties, offset_properties)
    post_match_smd = _standardized_mean_difference(
        matched_dual_properties, matched_offset_properties
    )
    summary: dict[str, object] = {
        "feature_names": feature_names,
        "pure_dual_pair_count": int(dual_index.size),
        "pure_offset_pair_count": int(offset_index.size),
        "matched_pair_count": int(matched_dual_index.size),
        "match_distance_quantiles": {
            name: float(np.quantile(match_distance, level))
            for name, level in (("q16", 0.16), ("q50", 0.50), ("q84", 0.84))
        },
        "absolute_standardized_mean_difference_before": np.abs(pre_match_smd).tolist(),
        "absolute_standardized_mean_difference_after": np.abs(post_match_smd).tolist(),
        "evaluation_time_gyr": evaluation_time,
        "matched_dual_capture_fraction_certain": float(np.mean(dual_certain)),
        "matched_dual_capture_fraction_possible": float(np.mean(dual_possible)),
        "matched_offset_capture_fraction_certain": float(np.mean(offset_certain)),
        "matched_offset_capture_fraction_possible": float(np.mean(offset_possible)),
        "dual_minus_offset_capture_fraction_lower_bound": float(
            np.mean(difference_lower)
        ),
        "dual_minus_offset_capture_fraction_upper_bound": float(
            np.mean(difference_upper)
        ),
        "dual_minus_offset_midpoint": float(np.mean(difference_midpoint)),
        "dual_minus_offset_midpoint_bootstrap_16": float(
            np.quantile(bootstrap_midpoint, 0.16)
        ),
        "dual_minus_offset_midpoint_bootstrap_84": float(
            np.quantile(bootstrap_midpoint, 0.84)
        ),
    }
    matched = {
        "dual_pair_index": matched_dual_index,
        "offset_pair_index": matched_offset_index,
        "match_distance": match_distance,
        "dual_certain": dual_certain,
        "dual_possible": dual_possible,
        "offset_certain": offset_certain,
        "offset_possible": offset_possible,
        "difference_lower": difference_lower,
        "difference_upper": difference_upper,
        "difference_midpoint": difference_midpoint,
    }
    return summary, matched


def _plot_demographics(
    output: Path,
    data: dict[int, dict[str, object]],
    volume_cmpc3: float,
) -> None:
    _plot_settings(6.0)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 5.55),
        gridspec_kw={"wspace": 0.37, "hspace": 0.38},
    )
    axes = axes.ravel()
    output_numbers = sorted(data, key=lambda item: float(data[item]["redshift"]))
    redshift = np.asarray([data[number]["redshift"] for number in output_numbers])

    selections = (
        ("bol43", r"$L_{\rm bol}\geq10^{43}$", "-", COLORS[0], MARKERS[0]),
        ("bol44", r"$L_{\rm bol}\geq10^{44}$", "--", COLORS[1], MARKERS[1]),
        ("hx42", r"$L_{2-10\,{\rm keV}}\geq10^{42}$", ":", COLORS[2], MARKERS[2]),
    )
    for key, label, line_style, color, marker in selections:
        count = np.asarray(
            [data[number]["selection"][key]["statistics"]["dual_pair_count"] for number in output_numbers]
        )
        density = count / volume_cmpc3
        axes[0].errorbar(
            redshift,
            density,
            yerr=np.asarray(
                [
                    data[number]["selection"][key]["statistics"][
                        "dual_pair_number_density_jackknife_error_cmpc3"
                    ]
                    for number in output_numbers
                ]
            ),
            color=color,
            marker=marker,
            mfc="white",
            mec=color,
            ms=4.2,
            lw=1.0,
            ls=line_style,
            capsize=1.5,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$z$")
    axes[0].set_ylabel(r"$n_{\rm dual}$ [cMpc$^{-3}$]")
    axes[0].legend(frameon=False, loc="lower right", handlelength=1.8)
    _panel_label(axes[0], "(a)", x=0.04, horizontal_alignment="left")

    fraction_definitions = (
        ("dual_pair_count", r"$N_{\rm pair}/N_{\rm AGN}$", COLORS[0], MARKERS[0]),
        ("dual_member_count", r"active SMBHs with companion", COLORS[1], MARKERS[1]),
        ("pure_dual_member_count", r"active SMBHs in two-member systems", COLORS[2], MARKERS[2]),
    )
    active_count = np.asarray(
        [data[number]["selection"]["bol43"]["statistics"]["active_agn_count"] for number in output_numbers]
    )
    for field, label, color, marker in fraction_definitions:
        numerator = np.asarray(
            [data[number]["selection"]["bol43"]["statistics"][field] for number in output_numbers]
        )
        fraction = numerator / active_count
        if field == "dual_pair_count":
            fraction_error = np.asarray(
                [
                    data[number]["selection"]["bol43"]["statistics"][
                        "dual_pair_fraction_jackknife_error"
                    ]
                    for number in output_numbers
                ]
            )
        else:
            fraction_error = np.sqrt(numerator) / active_count
        axes[1].errorbar(
            redshift,
            fraction,
            yerr=fraction_error,
            color=color,
            marker=marker,
            mfc="white",
            mec=color,
            ms=4.2,
            lw=1.0,
            capsize=1.5,
            label=label,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$z$")
    axes[1].set_ylabel(r"fraction of active SMBHs")
    axes[1].legend(frameon=False, loc="lower right", handlelength=1.5)
    _panel_label(axes[1], "(b)", x=0.04, horizontal_alignment="left")

    mass_threshold = np.asarray([1.0e4, 1.0e6, 1.0e7])
    for number, color, marker in zip(sorted(data), COLORS, MARKERS):
        fractions = []
        errors = []
        for key in ("m4", "m6", "m7"):
            statistics = data[number]["mass_selection"][key]["statistics"]
            numerator = statistics["dual_pair_count"]
            denominator = statistics["active_agn_count"]
            fractions.append(numerator / denominator)
            errors.append(statistics["dual_pair_fraction_jackknife_error"])
        axes[2].errorbar(
            mass_threshold,
            fractions,
            yerr=errors,
            color=color,
            marker=marker,
            mfc="white",
            mec=color,
            ms=4.2,
            lw=1.0,
            capsize=1.5,
            label=rf"$z={float(data[number]['redshift']):.2f}$",
        )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$M_{\rm BH,min}$ [$M_\odot$]")
    axes[2].set_ylabel(r"$N_{\rm pair}/N_{\rm AGN}$")
    axes[2].legend(frameon=False, loc="lower left", handlelength=1.4)
    _panel_label(axes[2], "(c)")

    log_edges = np.linspace(np.log10(0.5), np.log10(30.0), 10)
    center = 10.0 ** (0.5 * (log_edges[:-1] + log_edges[1:]))
    for number, color, marker in zip(sorted(data)[:2], COLORS, MARKERS):
        pairs = data[number]["mass_selection"]["m6"]["pairs"]
        for selected, line_style, population_label in (
            (pairs["is_dual"], "-", "dual AGN"),
            (pairs["is_offset"], "--", "single AGN"),
        ):
            density, error = _density_per_log_separation(
                pairs["separation_pkpc"][selected], int(pairs["active_count"]), log_edges
            )
            visible = density > 0.0
            axes[3].errorbar(
                center[visible],
                density[visible],
                yerr=error[visible],
                color=color,
                marker=marker,
                mfc="white",
                mec=color,
                ms=3.6,
                lw=0.9,
                ls=line_style,
                capsize=1.2,
                label=rf"$z={float(data[number]['redshift']):.2f}$ {population_label}",
            )
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].set_xlim(0.5, 30.0)
    axes[3].set_xlabel(r"$r_{\rm 3D}$ [pkpc]")
    axes[3].set_ylabel(r"$dP/d\log_{10}r$")
    axes[3].legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.90),
        borderaxespad=0.0,
        handlelength=1.4,
    )
    _panel_label(axes[3], "(d)", x=0.04, horizontal_alignment="left")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)


def _plot_precursors(
    output: Path,
    data: dict[int, dict[str, object]],
    capture_curves: dict[tuple[int, str], dict[str, np.ndarray]],
    matched_comparisons: dict[int, dict[str, object]],
    rng: np.random.Generator,
) -> None:
    _plot_settings(6.0)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 5.55),
        gridspec_kw={"wspace": 0.37, "hspace": 0.38},
    )
    axes = axes.ravel()
    log_edges = np.linspace(np.log10(0.5), np.log10(30.0), 9)
    center = 10.0 ** (0.5 * (log_edges[:-1] + log_edges[1:]))

    curve_outputs = sorted({number for number, _ in capture_curves})
    for number, color, marker in zip(curve_outputs, COLORS, MARKERS):
        pairs = data[number]["selection"]["bol43"]["pairs"]
        selected = pairs["is_dual"] & (pairs["pair_system_multiplicity"] == 2)
        coordinate = np.log10(pairs["separation_pkpc"][selected])
        for field, line_style, velocity_limit in (
            ("projected_selection_probability_dv300", "--", 300),
            ("projected_selection_probability_dv600", "-", 600),
        ):
            count, mean, lower, upper = _binned_mean_interval(
                coordinate,
                pairs[field][selected],
                log_edges,
                rng,
            )
            visible = count >= 5
            axes[0].errorbar(
                center[visible],
                mean[visible],
                yerr=np.vstack((mean[visible] - lower[visible], upper[visible] - mean[visible])),
                color=color,
                marker=marker,
                mfc="white",
                mec=color,
                ms=3.6,
                lw=0.9,
                ls=line_style,
                capsize=1.2,
                label=rf"$z={float(data[number]['redshift']):.2f}$, {velocity_limit}",
            )
    axes[0].set_xscale("log")
    axes[0].set_xlim(0.5, 30.0)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xlabel(r"$r_{\rm 3D}$ [pkpc]")
    axes[0].set_ylabel(r"probability of passing projected selection")
    axes[0].legend(frameon=False, loc="lower right", handlelength=1.5)
    _panel_label(axes[0], "(a)", x=0.04, horizontal_alignment="left")

    ratio_edges = np.linspace(-3.0, 0.0, 13)
    ratio_center = 0.5 * (ratio_edges[:-1] + ratio_edges[1:])
    for number, color, marker in zip(curve_outputs, COLORS, MARKERS):
        pairs = data[number]["mass_selection"]["m6"]["pairs"]
        mass_ratio = pairs["mass_2_msun"] / pairs["mass_1_msun"]
        for selected, line_style, population_label in (
            (
                pairs["is_dual"] & (pairs["pair_system_multiplicity"] == 2),
                "-",
                "dual AGN",
            ),
            (
                pairs["is_offset"] & (pairs["pair_system_multiplicity"] == 2),
                "--",
                "single AGN",
            ),
        ):
            count, _ = np.histogram(np.log10(mass_ratio[selected]), bins=ratio_edges)
            density = count / (np.sum(count) * np.diff(ratio_edges))
            visible = count > 0
            axes[1].plot(
                ratio_center[visible],
                density[visible],
                color=color,
                marker=marker,
                mfc="white",
                mec=color,
                ms=3.6,
                lw=0.9,
                ls=line_style,
                label=rf"$z={float(data[number]['redshift']):.2f}$ {population_label}",
            )
    axes[1].set_xlabel(r"$\log_{10}q_{\rm BH}$")
    axes[1].set_ylabel(r"$p(\log_{10}q_{\rm BH})$")
    axes[1].legend(frameon=False, loc="upper left", handlelength=1.5)
    _panel_label(axes[1], "(b)", y=0.08)

    for number, color in zip(curve_outputs, COLORS):
        for population, line_style, population_label in (
            ("mass_limited_dual", "-", "dual AGN"),
            ("mass_limited_offset", "--", "single AGN"),
        ):
            curve = capture_curves[(number, population)]
            time = curve["time_gyr"]
            axes[2].fill_between(
                time,
                curve["lower_q16"],
                curve["upper_q84"],
                color=color,
                alpha=0.09,
                lw=0.0,
            )
            axes[2].fill_between(
                time,
                curve["certain"],
                curve["possible"],
                color=color,
                alpha=0.20,
                lw=0.0,
            )
            axes[2].plot(
                time,
                0.5 * (curve["certain"] + curve["possible"]),
                color=color,
                lw=1.1,
                ls=line_style,
                label=rf"$z={float(data[number]['redshift']):.2f}$ {population_label}",
            )
    axes[2].set_xlim(0.0, 3.0)
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xlabel(r"time after selection of active pair [Gyr]")
    axes[2].set_ylabel(r"cumulative binary-capture fraction")
    axes[2].legend(frameon=False, loc="lower right")
    _panel_label(axes[2], "(c)")

    matched_outputs = sorted(matched_comparisons)
    matched_redshift = np.asarray(
        [float(data[number]["redshift"]) for number in matched_outputs]
    )
    midpoint = np.asarray(
        [
            matched_comparisons[number]["dual_minus_offset_midpoint"]
            for number in matched_outputs
        ]
    )
    bootstrap_lower = np.asarray(
        [
            matched_comparisons[number]["dual_minus_offset_midpoint_bootstrap_16"]
            for number in matched_outputs
        ]
    )
    bootstrap_upper = np.asarray(
        [
            matched_comparisons[number]["dual_minus_offset_midpoint_bootstrap_84"]
            for number in matched_outputs
        ]
    )
    censoring_lower = np.asarray(
        [
            matched_comparisons[number][
                "dual_minus_offset_capture_fraction_lower_bound"
            ]
            for number in matched_outputs
        ]
    )
    censoring_upper = np.asarray(
        [
            matched_comparisons[number][
                "dual_minus_offset_capture_fraction_upper_bound"
            ]
            for number in matched_outputs
        ]
    )
    axes[3].vlines(
        matched_redshift,
        censoring_lower,
        censoring_upper,
        color=COLORS[3],
        lw=3.2,
        alpha=0.35,
        label="censoring bounds",
    )
    axes[3].errorbar(
        matched_redshift,
        midpoint,
        yerr=np.vstack((midpoint - bootstrap_lower, bootstrap_upper - midpoint)),
        color=COLORS[3],
        marker=MARKERS[3],
        mfc="white",
        mec=COLORS[3],
        ms=4.5,
        lw=1.0,
        capsize=1.5,
        label="midpoint and 68% interval",
    )
    axes[3].axhline(0.0, color="black", lw=0.8, ls=":")
    axes[3].set_xlabel(r"$z$")
    axes[3].set_ylabel(r"matched $f_{\rm cap,dual}-f_{\rm cap,single}$")
    axes[3].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        borderaxespad=0.0,
        handlelength=1.5,
    )
    _panel_label(axes[3], "(d)", x=0.04, horizontal_alignment="left")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)


def _plot_eddington_luminosity(
    output: Path,
    mass_msun: np.ndarray,
    bolometric_luminosity_erg_s: np.ndarray,
) -> None:
    _plot_settings(10.0)
    eddington_luminosity = 1.26e38 * mass_msun
    log_eddington = np.log10(eddington_luminosity)
    log_bolometric = np.log10(bolometric_luminosity_erg_s)
    figure, axis = plt.subplots(figsize=(3.35, 3.35))
    density = axis.hexbin(
        log_eddington,
        log_bolometric,
        gridsize=42,
        extent=(43.0, 47.7, 43.0, 47.7),
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0.15,
    )
    coordinate = np.linspace(43.0, 47.7, 100)
    for ratio, color, line_style in (
        (1.0, "black", "-"),
        (0.1, COLORS[0], "--"),
        (0.01, COLORS[1], ":"),
    ):
        axis.plot(
            coordinate,
            coordinate + np.log10(ratio),
            color=color,
            ls=line_style,
            lw=1.0,
            label=rf"$L_{{\rm bol}}/L_{{\rm Edd}}={ratio:g}$",
        )
    axis.set_xlim(43.0, 47.7)
    axis.set_ylim(43.0, 47.7)
    axis.set_xlabel(r"$\log_{10}L_{\rm Edd}$ [erg s$^{-1}$]")
    axis.set_ylabel(r"$\log_{10}L_{\rm bol}$ [erg s$^{-1}$]")
    axis.legend(frameon=False, loc="upper left", handlelength=1.7)
    colorbar = figure.colorbar(density, ax=axis, pad=0.025)
    colorbar.set_label("AGN count per hexagon")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)


def _write_pair_catalog(path: Path, data: dict[int, dict[str, object]], mass_key: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "output_number",
                "redshift",
                "pair_class",
                "primary_sink_id",
                "secondary_sink_id",
                "separation_pkpc",
                "primary_mass_msun",
                "secondary_mass_msun",
                "mass_ratio",
                "primary_lbol_erg_s",
                "secondary_lbol_erg_s",
                "primary_lhx_erg_s",
                "secondary_lhx_erg_s",
                "primary_eddington_ratio",
                "secondary_eddington_ratio",
                "dual_system_multiplicity",
                "pair_system_multiplicity",
                "relative_speed_kms",
                "projected_selection_probability_dv300",
                "projected_selection_probability_dv600",
                "active_sightline_fraction_nhi_ge_1e23",
                "active_sightline_fraction_nhi_ge_1e24",
                "assigned_capture_output",
                "capture_delay_lower_gyr",
                "capture_delay_upper_gyr",
            )
        )
        for output_number in sorted(data):
            if mass_key is None:
                pairs = data[output_number]["selection"]["bol43"]["pairs"]
                selected = pairs["is_dual"]
            else:
                pairs = data[output_number]["mass_selection"][mass_key]["pairs"]
                selected = pairs["is_dual"] | pairs["is_offset"]
            for pair_number in np.flatnonzero(selected):
                pair_class = "dual" if pairs["is_dual"][pair_number] else "offset"
                writer.writerow(
                    (
                        output_number,
                        data[output_number]["redshift"],
                        pair_class,
                        int(pairs["id_1"][pair_number]),
                        int(pairs["id_2"][pair_number]),
                        pairs["separation_pkpc"][pair_number],
                        pairs["mass_1_msun"][pair_number],
                        pairs["mass_2_msun"][pair_number],
                        pairs["mass_2_msun"][pair_number] / pairs["mass_1_msun"][pair_number],
                        pairs["lbol_1_erg_s"][pair_number],
                        pairs["lbol_2_erg_s"][pair_number],
                        pairs["lhx_1_erg_s"][pair_number],
                        pairs["lhx_2_erg_s"][pair_number],
                        pairs["eddington_ratio_1"][pair_number],
                        pairs["eddington_ratio_2"][pair_number],
                        int(pairs["dual_system_multiplicity"][pair_number]),
                        int(pairs["pair_system_multiplicity"][pair_number]),
                        pairs["relative_speed_kms"][pair_number],
                        pairs["projected_selection_probability_dv300"][pair_number],
                        pairs["projected_selection_probability_dv600"][pair_number],
                        pairs.get("active_sightline_fraction_nhi_ge_1e23", np.full(selected.size, np.nan))[pair_number],
                        pairs.get("active_sightline_fraction_nhi_ge_1e24", np.full(selected.size, np.nan))[pair_number],
                        int(pairs["assigned_capture_output"][pair_number]),
                        pairs["capture_delay_lower_gyr"][pair_number],
                        pairs["capture_delay_upper_gyr"][pair_number],
                    )
                )


def _write_matched_catalog(
    path: Path,
    data: dict[int, dict[str, object]],
    matched_data: dict[int, dict[str, np.ndarray]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "output_number",
                "redshift",
                "dual_primary_sink_id",
                "dual_secondary_sink_id",
                "offset_primary_sink_id",
                "offset_secondary_sink_id",
                "standardized_match_distance",
                "dual_certain_by_1_gyr",
                "dual_possible_by_1_gyr",
                "offset_certain_by_1_gyr",
                "offset_possible_by_1_gyr",
            )
        )
        for output_number in sorted(matched_data):
            pairs = data[output_number]["mass_selection"]["m6"]["pairs"]
            matched = matched_data[output_number]
            dual_index = matched["dual_pair_index"]
            offset_index = matched["offset_pair_index"]
            for row in range(dual_index.size):
                dual_pair = dual_index[row]
                offset_pair = offset_index[row]
                writer.writerow(
                    (
                        output_number,
                        data[output_number]["redshift"],
                        int(pairs["id_1"][dual_pair]),
                        int(pairs["id_2"][dual_pair]),
                        int(pairs["id_1"][offset_pair]),
                        int(pairs["id_2"][offset_pair]),
                        matched["match_distance"][row],
                        int(matched["dual_certain"][row]),
                        int(matched["dual_possible"][row]),
                        int(matched["offset_certain"][row]),
                        int(matched["offset_possible"][row]),
                    )
                )


def analyze(
    agn_directory: Path,
    outputs: tuple[int, ...],
    history_path: Path,
    capture_catalog_path: Path,
    output_directory: Path,
    dimensionless_hubble: float,
    omega_matter: float,
    box_size_cmpc_over_h: float,
    volume_cmpc3: float,
    sightline_count: int,
    bootstrap_realizations: int,
) -> None:
    history = _read_history(history_path)
    output_to_time = dict(zip(history["output"].tolist(), history["time"].tolist()))
    previous_output = {
        int(history["output"][index]): int(history["output"][index - 1])
        for index in range(1, history["output"].size)
    }
    maximum_sink_id = 1_688_677
    capture_key, capture_output = _read_capture_links(capture_catalog_path, maximum_sink_id)
    cosmology = FlatLambdaCDM(
        H0=100.0 * dimensionless_hubble,
        Om0=omega_matter,
        Tcmb0=2.7255,
    )
    sightlines = fibonacci_sightlines(sightline_count)
    rng = np.random.default_rng(20260809)
    data: dict[int, dict[str, object]] = {}
    eddington_sample: dict[str, object] | None = None
    eddington_redshift_distance = np.inf
    summary: dict[str, object] = {
        "selection": {
            "minimum_separation_pkpc": 0.5,
            "maximum_separation_pkpc": 30.0,
            "fiducial_lbol_threshold_erg_s": 1.0e43,
            "comparison_lbol_threshold_erg_s": 1.0e44,
            "hard_xray_threshold_erg_s": 1.0e42,
            "offset_comparison_minimum_mass_msun": 1.0e6,
            "sightline_count": sightline_count,
            "line_of_sight_velocity_limits_kms": [300.0, 600.0],
            "fraction_definitions": [
                "N_dual_pair/N_active_AGN",
                "N_unique_dual_member/N_active_AGN",
                "N_pure_dual_member/N_active_AGN",
            ],
            "host_association": (
                "The local MkAGN records contain zero host masses and the legacy "
                "galaxy catalogs point to unavailable scratch storage. The selected "
                "pairs are spatially associated active SMBHs rather than confirmed "
                "distinct-host dual AGNs."
            ),
            "capture_link": (
                "surviving SMBH assigned by the legacy mkmerging.c distance and "
                "mass criteria, not a direct record of the partner selected by RAMSES"
            ),
            "capture_time": "interval between the last resolved and assigned capture outputs",
            "spatial_jackknife_regions": 8,
        },
        "volume_cmpc3": volume_cmpc3,
        "snapshots": {},
    }

    selection_specs = {
        "bol43": ("Lbol", 1.0e43),
        "bol44": ("Lbol", 1.0e44),
        "hx42": ("LhX", 1.0e42),
    }
    mass_specs = {"m4": 1.0e4, "m6": 1.0e6, "m7": 1.0e7}
    for output_number in outputs:
        path = agn_directory / f"agn.{output_number:05d}.dat"
        redshift, _, records = read_mkagn_snapshot(path)
        mass_msun = np.asarray(records["mass"], dtype=np.float64) / dimensionless_hubble
        bolometric_luminosity = np.asarray(records["Lbol"], dtype=np.float64)
        active_for_eddington = (
            np.isfinite(mass_msun)
            & (mass_msun > 0.0)
            & np.isfinite(bolometric_luminosity)
            & (bolometric_luminosity >= 1.0e43)
        )
        redshift_distance = abs(redshift - 0.625)
        if redshift_distance < eddington_redshift_distance:
            eddington_sample = {
                "output_number": output_number,
                "redshift": redshift,
                "mass_msun": mass_msun[active_for_eddington].copy(),
                "bolometric_luminosity_erg_s": bolometric_luminosity[
                    active_for_eddington
                ].copy(),
            }
            eddington_redshift_distance = redshift_distance
        snapshot_time = output_to_time[output_number]
        snapshot: dict[str, object] = {
            "redshift": redshift,
            "selection": {},
            "mass_selection": {},
        }
        for key, (luminosity_field, threshold) in selection_specs.items():
            pairs = find_agn_pair_population(
                records,
                redshift,
                dimensionless_hubble,
                luminosity_threshold_erg_s=threshold,
                luminosity_field=luminosity_field,
                minimum_mass_msun=1.0e4,
                box_size_cmpc_over_h=box_size_cmpc_over_h,
            )
            pairs = _add_dual_system_information(pairs)
            pairs = _attach_capture_history(
                pairs,
                output_number,
                snapshot_time,
                capture_key,
                capture_output,
                output_to_time,
                previous_output,
                maximum_sink_id,
            )
            pairs = _add_projection_information(
                pairs,
                redshift,
                dimensionless_hubble,
                cosmology,
                sightlines,
                box_size_cmpc_over_h,
            )
            snapshot["selection"][key] = {
                "pairs": pairs,
                "statistics": _selection_statistics(
                    pairs, volume_cmpc3, box_size_cmpc_over_h
                ),
            }
        for key, minimum_mass in mass_specs.items():
            pairs = find_agn_pair_population(
                records,
                redshift,
                dimensionless_hubble,
                luminosity_threshold_erg_s=1.0e43,
                luminosity_field="Lbol",
                minimum_mass_msun=minimum_mass,
                box_size_cmpc_over_h=box_size_cmpc_over_h,
            )
            pairs = _add_dual_system_information(pairs)
            pairs = _attach_capture_history(
                pairs,
                output_number,
                snapshot_time,
                capture_key,
                capture_output,
                output_to_time,
                previous_output,
                maximum_sink_id,
            )
            pairs = _add_projection_information(
                pairs,
                redshift,
                dimensionless_hubble,
                cosmology,
                sightlines,
                box_size_cmpc_over_h,
            )
            snapshot["mass_selection"][key] = {
                "pairs": pairs,
                "statistics": _selection_statistics(
                    pairs, volume_cmpc3, box_size_cmpc_over_h
                ),
            }

        nhi_path = agn_directory / f"agn.{output_number:05d}.NHI.dat"
        if nhi_path.exists():
            nhi_id, nhi_value = _read_nhi_sightlines(nhi_path)
            for selection in snapshot["selection"].values():
                _add_obscuration_information(selection["pairs"], nhi_id, nhi_value)
            for selection in snapshot["mass_selection"].values():
                _add_obscuration_information(selection["pairs"], nhi_id, nhi_value)
        else:
            for group in (snapshot["selection"], snapshot["mass_selection"]):
                for selection in group.values():
                    pair_count = selection["pairs"]["id_1"].size
                    selection["pairs"]["active_sightline_fraction_nhi_ge_1e23"] = np.full(pair_count, np.nan)
                    selection["pairs"]["active_sightline_fraction_nhi_ge_1e24"] = np.full(pair_count, np.nan)
                    selection["pairs"]["nhi_available"] = np.zeros(pair_count, dtype=bool)

        data[output_number] = snapshot
        summary["snapshots"][str(output_number)] = {
            "redshift": redshift,
            "selections": {
                key: value["statistics"] for key, value in snapshot["selection"].items()
            },
            "mass_selections": {
                key: value["statistics"] for key, value in snapshot["mass_selection"].items()
            },
            "fiducial_pair_diagnostics": _population_diagnostics(
                snapshot["selection"]["bol43"]["pairs"]
            ),
            "mass_limited_pair_diagnostics": _population_diagnostics(
                snapshot["mass_selection"]["m6"]["pairs"]
            ),
        }
        fiducial = snapshot["selection"]["bol43"]["statistics"]
        print(
            f"Output {output_number:05d} at z={redshift:.3f}: "
            f"{fiducial['dual_pair_count']:,} dual and "
            f"{snapshot['mass_selection']['m6']['statistics']['offset_pair_count']:,} "
            "mass-limited offset pairs",
            flush=True,
        )

    if eddington_sample is None or eddington_redshift_distance > 0.01:
        raise ValueError("No requested AGN snapshot lies near redshift 0.625")
    eddington_luminosity = 1.26e38 * eddington_sample["mass_msun"]
    eddington_ratio = (
        eddington_sample["bolometric_luminosity_erg_s"] / eddington_luminosity
    )
    summary["eddington_luminosity_comparison"] = {
        "output_number": int(eddington_sample["output_number"]),
        "redshift": float(eddington_sample["redshift"]),
        "lbol_threshold_erg_s": 1.0e43,
        "agn_count": int(eddington_ratio.size),
        "eddington_ratio_quantiles": {
            name: float(np.quantile(eddington_ratio, level))
            for name, level in (("q16", 0.16), ("q50", 0.50), ("q84", 0.84))
        },
        "eddington_ratio_ge_0p1_fraction": float(np.mean(eddington_ratio >= 0.1)),
        "eddington_ratio_ge_1_fraction": float(np.mean(eddington_ratio >= 1.0)),
    }

    capture_curves: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    matched_comparisons: dict[int, dict[str, object]] = {}
    matched_data: dict[int, dict[str, np.ndarray]] = {}
    capture_rows: list[tuple[object, ...]] = []
    final_time = float(history["time"][-1])
    for output_number in outputs:
        followup = final_time - output_to_time[output_number]
        if followup <= 0.0:
            continue
        fiducial_pairs = data[output_number]["selection"]["bol43"]["pairs"]
        mass_limited_pairs = data[output_number]["mass_selection"]["m6"]["pairs"]
        populations = {
            "pure_dual": (
                fiducial_pairs,
                fiducial_pairs["is_dual"]
                & (fiducial_pairs["pair_system_multiplicity"] == 2),
            ),
            "mass_limited_dual": (
                mass_limited_pairs,
                mass_limited_pairs["is_dual"]
                & (mass_limited_pairs["pair_system_multiplicity"] == 2),
            ),
            "mass_limited_offset": (
                mass_limited_pairs,
                mass_limited_pairs["is_offset"]
                & (mass_limited_pairs["pair_system_multiplicity"] == 2),
            ),
        }
        population_summary = {}
        for population_name, (pairs, selected) in populations.items():
            event_lower = pairs["capture_delay_lower_gyr"][selected]
            event_upper = pairs["capture_delay_upper_gyr"][selected]
            time_grid = np.linspace(0.0, followup, 101)
            certain, possible = interval_censored_cumulative_bounds(
                event_lower, event_upper, time_grid, followup
            )
            lower_q16, lower_q84, upper_q16, upper_q84 = _bootstrap_capture_bounds(
                event_lower,
                event_upper,
                time_grid,
                followup,
                rng,
                bootstrap_realizations,
            )
            capture_curves[(output_number, population_name)] = {
                "time_gyr": time_grid,
                "certain": certain,
                "possible": possible,
                "lower_q16": lower_q16,
                "lower_q84": lower_q84,
                "upper_q16": upper_q16,
                "upper_q84": upper_q84,
            }
            for values in zip(
                time_grid,
                certain,
                possible,
                lower_q16,
                lower_q84,
                upper_q16,
                upper_q84,
            ):
                capture_rows.append(
                    (
                        output_number,
                        data[output_number]["redshift"],
                        population_name,
                        followup,
                        *values,
                    )
                )
            evaluation_time = min(1.0, followup)
            population_summary[population_name] = {
                "pair_count": int(np.count_nonzero(selected)),
                "followup_gyr": followup,
                "direct_capture_count": int(np.count_nonzero(np.isfinite(event_upper))),
                "certain_by_1_gyr": float(
                    np.mean(np.isfinite(event_upper) & (event_upper <= evaluation_time))
                ),
                "possible_by_1_gyr": float(
                    np.mean(np.isfinite(event_lower) & (event_lower <= evaluation_time))
                ),
            }
        summary["snapshots"][str(output_number)]["capture_by_population"] = population_summary
        matched_summary, matched = _matched_capture_comparison(
            mass_limited_pairs,
            followup,
            rng,
            bootstrap_realizations,
        )
        matched_comparisons[output_number] = matched_summary
        matched_data[output_number] = matched
        summary["snapshots"][str(output_number)][
            "matched_dual_offset_capture_comparison"
        ] = matched_summary

    output_directory.mkdir(parents=True, exist_ok=True)
    _write_pair_catalog(output_directory / "hr5_dual_agn_pairs.csv", data, None)
    _write_pair_catalog(
        output_directory / "hr5_dual_offset_pairs_mbh_ge_1e6.csv", data, "m6"
    )
    _write_matched_catalog(
        output_directory / "hr5_dual_offset_matched_pairs.csv", data, matched_data
    )
    with (output_directory / "hr5_dual_agn_capture_cdf.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "output_number",
                "redshift",
                "population",
                "followup_gyr",
                "time_since_selection_gyr",
                "certain_capture_fraction",
                "possible_capture_fraction",
                "certain_bootstrap_16",
                "certain_bootstrap_84",
                "possible_bootstrap_16",
                "possible_bootstrap_84",
            )
        )
        writer.writerows(capture_rows)
    (output_directory / "hr5_dual_agn_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot_demographics(
        output_directory / "hr5_dual_agn_demographics.pdf", data, volume_cmpc3
    )
    _plot_precursors(
        output_directory / "hr5_dual_agn_precursors.pdf",
        data,
        capture_curves,
        matched_comparisons,
        rng,
    )
    _plot_eddington_luminosity(
        output_directory / "hr5_eddington_luminosity_z0p625.pdf",
        eddington_sample["mass_msun"],
        eddington_sample["bolometric_luminosity_erg_s"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agn-directory", type=Path, default=DEFAULT_AGN_DIRECTORY)
    parser.add_argument("--outputs", nargs="+", type=int, default=(89, 117, 296))
    parser.add_argument("--history", type=Path, default=Path("results/hr5/hr5_sink_history.csv"))
    parser.add_argument("--capture-catalog", type=Path, default=Path("results/hr5/hr5_capture_catalog.csv"))
    parser.add_argument("--output-directory", type=Path, default=Path("results/hr5/dual_agn"))
    parser.add_argument("--dimensionless-hubble", type=float, default=0.684)
    parser.add_argument("--omega-matter", type=float, default=0.3)
    parser.add_argument("--box-size-cmpc-over-h", type=float, default=717.229040)
    parser.add_argument("--volume-cmpc3", type=float, default=FIDUCIAL_VOLUME_CMPC3)
    parser.add_argument("--sightline-count", type=int, default=128)
    parser.add_argument("--bootstrap-realizations", type=int, default=500)
    args = parser.parse_args()
    analyze(
        args.agn_directory,
        tuple(args.outputs),
        args.history,
        args.capture_catalog,
        args.output_directory,
        args.dimensionless_hubble,
        args.omega_matter,
        args.box_size_cmpc_over_h,
        args.volume_cmpc3,
        args.sightline_count,
        args.bootstrap_realizations,
    )


if __name__ == "__main__":
    main()
