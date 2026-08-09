#!/usr/bin/env python3
"""Calculate a conditional PTA reference signal from the HR5 capture catalogue."""

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
    circular_gw_background_contributions,
    read_tree_header,
)


DEFAULT_TREE = Path(
    "/home/kjhan/BACKUP/GalFinder/"
    "SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/BinarySMBH/"
    "Sink_Merging_Tree.dat.Updated"
)
VOLUME_CMPC3 = 1.087e7
REFERENCE_FREQUENCY_HZ = 1.0 / (365.25 * 86400.0)
REFERENCE_DELAY_GYR = 0.5

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY_BLUE = "#56B4E9"
GRAY = "#4D4D4D"


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


def _panel_label(axis: plt.Axes, label: str) -> None:
    panel_text = axis.text(
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
    panel_text.set_path_effects(
        [path_effects.withStroke(linewidth=1.6, foreground="white")]
    )


def _read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No HR5 history rows were found in {path}")
    return {
        "output_number": np.asarray([row["output_number"] for row in rows], dtype=np.int64),
        "cosmic_time_gyr": np.asarray(
            [row["cosmic_time_gyr"] for row in rows], dtype=np.float64
        ),
    }


def _read_events(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    columns = ("assigned_capture_output", "chirp_mass_last_resolved_msun")
    missing = sorted(set(columns) - set(header))
    if missing:
        raise ValueError(f"The HR5 capture catalogue is missing columns: {missing}")
    output_number, chirp_mass = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=tuple(header.index(name) for name in columns),
        dtype=np.float64,
        unpack=True,
    )
    return output_number.astype(np.int64), chirp_mass


def _fractional_contribution(
    values: np.ndarray,
    contributions: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bin_number = np.digitize(values, edges) - 1
    summed = np.asarray(
        [np.sum(contributions[bin_number == index]) for index in range(edges.size - 1)]
    )
    return np.asarray(
        [np.count_nonzero(bin_number == index) for index in range(edges.size - 1)]
    ), summed / np.sum(summed)


def _write_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    tree_path: Path,
    history_path: Path,
    catalog_path: Path,
    output_dir: Path,
    volume_cmpc3: float,
    maximum_delay_gyr: float,
    delay_step_gyr: float,
) -> None:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    history = _read_history(history_path)
    capture_output, chirp_mass = _read_events(catalog_path)
    output_to_index = {
        int(output): index for index, output in enumerate(history["output_number"])
    }
    capture_index = np.asarray(
        [output_to_index[int(output)] for output in capture_output], dtype=np.int64
    )
    capture_time = history["cosmic_time_gyr"][capture_index]
    valid_mass = np.isfinite(chirp_mass) & (chirp_mass > 0.0)

    header = read_tree_header(tree_path)
    cosmology = FlatLambdaCDM(
        H0=float(header["h0"]),
        Om0=float(header["omega_m"]),
        Tcmb0=2.7255,
    )
    present_age = float(cosmology.age(0.0).value)
    redshift_grid = np.expm1(np.linspace(0.0, np.log1p(20.0), 50000))
    age_grid = np.asarray(cosmology.age(redshift_grid).value)

    delays = np.arange(0.0, maximum_delay_gyr + 0.5 * delay_step_gyr, delay_step_gyr)
    mass_thresholds = (1.0e4, 1.0e6, 1.0e8)
    amplitude = np.zeros(delays.size)
    amplitude_sigma = np.zeros(delays.size)
    uncensored_count = np.zeros(delays.size, dtype=np.int64)
    censored_fraction = np.zeros((len(mass_thresholds), delays.size))
    rows: list[dict[str, object]] = []
    reference_redshift: np.ndarray | None = None
    reference_contribution: np.ndarray | None = None
    reference_mass: np.ndarray | None = None

    for delay_index, delay in enumerate(delays):
        delayed_time = capture_time + delay
        censored = delayed_time > present_age
        event_redshift = np.interp(
            np.minimum(delayed_time, present_age),
            age_grid[::-1],
            redshift_grid[::-1],
        )
        selected = valid_mass & ~censored
        contribution = circular_gw_background_contributions(
            chirp_mass[selected],
            event_redshift[selected],
            volume_cmpc3,
            REFERENCE_FREQUENCY_HZ,
        )
        strain_squared = np.sum(contribution)
        amplitude[delay_index] = np.sqrt(strain_squared)
        amplitude_sigma[delay_index] = (
            0.5 * np.sqrt(np.sum(contribution**2) / strain_squared)
            if strain_squared > 0.0
            else 0.0
        )
        uncensored_count[delay_index] = np.count_nonzero(selected)
        row: dict[str, object] = {
            "delay_gyr": delay,
            "amplitude_at_one_per_year": amplitude[delay_index],
            "poisson_sigma_amplitude": amplitude_sigma[delay_index],
            "uncensored_event_count": uncensored_count[delay_index],
        }
        for threshold_index, threshold in enumerate(mass_thresholds):
            threshold_sample = valid_mass & (chirp_mass >= threshold)
            threshold_count = np.count_nonzero(threshold_sample)
            fraction = (
                np.count_nonzero(threshold_sample & censored) / threshold_count
                if threshold_count
                else np.nan
            )
            censored_fraction[threshold_index, delay_index] = fraction
            row[f"censored_fraction_mchirp_ge_{threshold:.0e}"] = fraction
        rows.append(row)
        if np.isclose(delay, REFERENCE_DELAY_GYR, atol=0.5 * delay_step_gyr):
            reference_redshift = event_redshift[selected].copy()
            reference_contribution = contribution.copy()
            reference_mass = chirp_mass[selected].copy()

    if reference_redshift is None or reference_contribution is None or reference_mass is None:
        raise ValueError("The delay grid must include the 0.5 Gyr reference delay")

    mass_edges = np.array([0.0, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e9, np.inf])
    mass_count, mass_fraction = _fractional_contribution(
        reference_mass, reference_contribution, mass_edges
    )
    redshift_edges = np.array([0.0, 1.0, 2.0, 3.0, 5.0, np.inf])
    redshift_count, redshift_fraction = _fractional_contribution(
        reference_redshift, reference_contribution, redshift_edges
    )

    figure, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    axis = axes[0, 0]
    axis.fill_between(
        delays,
        amplitude - amplitude_sigma,
        amplitude + amplitude_sigma,
        color=BLUE,
        alpha=0.18,
        linewidth=0.0,
    )
    axis.plot(delays, amplitude, color=BLUE, lw=1.35)
    axis.axvspan(0.0, 4.0, color=GRAY, alpha=0.08, zorder=-10)
    axis.set_xlim(0.0, maximum_delay_gyr)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel(r"fixed reference delay $\Delta t_{\rm ref}$ [Gyr]")
    axis.set_ylabel(r"$A_{\rm ref}(1\,\mathrm{yr}^{-1})$")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    _panel_label(axis, "(a)")

    axis = axes[0, 1]
    mass_labels = (
        r"$<4$", r"$4$--$5$", r"$5$--$6$", r"$6$--$7$",
        r"$7$--$8$", r"$8$--$9$", r"$\geq9$",
    )
    mass_colors = (GRAY, SKY_BLUE, BLUE, GREEN, ORANGE, VERMILLION, PURPLE)
    axis.bar(np.arange(mass_fraction.size), mass_fraction, color=mass_colors, width=0.78)
    axis.set_xticks(np.arange(mass_fraction.size), mass_labels)
    axis.set_ylim(0.0, 0.76)
    axis.set_xlabel(r"$\log_{10}(\mathcal{M}_{\rm c}/M_\odot)$ interval")
    axis.set_ylabel(r"fraction of $A_{\rm ref}^{2}$")
    _panel_label(axis, "(b)")

    axis = axes[1, 0]
    redshift_labels = (r"$0$--$1$", r"$1$--$2$", r"$2$--$3$", r"$3$--$5$", r"$\geq5$")
    axis.bar(
        np.arange(redshift_fraction.size),
        redshift_fraction,
        color=(BLUE, SKY_BLUE, GREEN, ORANGE, VERMILLION),
        width=0.78,
    )
    axis.set_xticks(np.arange(redshift_fraction.size), redshift_labels)
    axis.set_ylim(0.0, 0.76)
    axis.set_xlabel("reference-event redshift interval")
    axis.set_ylabel(r"fraction of $A_{\rm ref}^{2}$")
    _panel_label(axis, "(c)")

    axis = axes[1, 1]
    for threshold, fraction, color, line_style in zip(
        mass_thresholds,
        censored_fraction,
        (BLUE, VERMILLION, GREEN),
        ("-", "--", "-."),
    ):
        axis.plot(
            delays,
            fraction,
            color=color,
            ls=line_style,
            lw=1.25,
            label=rf"$\mathcal{{M}}_{{\rm c}}\geq10^{{{int(np.log10(threshold))}}}\,M_\odot$",
        )
    axis.set_xlim(0.0, maximum_delay_gyr)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(r"fixed reference delay $\Delta t_{\rm ref}$ [Gyr]")
    axis.set_ylabel("cosmic-time-censored fraction")
    axis.legend(frameon=False, loc="upper left", handlelength=2.0, borderaxespad=0.3)
    _panel_label(axis, "(d)")

    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.095, top=0.985, wspace=0.32, hspace=0.34)
    figure.savefig(
        output_dir / "hr5_conditional_pta_reference.pdf",
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(figure)

    threshold_fields = tuple(
        f"censored_fraction_mchirp_ge_{threshold:.0e}" for threshold in mass_thresholds
    )
    _write_rows(
        output_dir / "hr5_conditional_pta_delay.csv",
        (
            "delay_gyr",
            "amplitude_at_one_per_year",
            "poisson_sigma_amplitude",
            "uncensored_event_count",
        ) + threshold_fields,
        rows,
    )
    _write_rows(
        output_dir / "hr5_conditional_pta_mass_contribution.csv",
        ("mass_lower_msun", "mass_upper_msun", "event_count", "fraction_of_amplitude_squared"),
        [
            {
                "mass_lower_msun": lower,
                "mass_upper_msun": upper,
                "event_count": count,
                "fraction_of_amplitude_squared": fraction,
            }
            for lower, upper, count, fraction in zip(
                mass_edges[:-1], mass_edges[1:], mass_count, mass_fraction
            )
        ],
    )
    _write_rows(
        output_dir / "hr5_conditional_pta_redshift_contribution.csv",
        ("redshift_lower", "redshift_upper", "event_count", "fraction_of_amplitude_squared"),
        [
            {
                "redshift_lower": lower,
                "redshift_upper": upper,
                "event_count": count,
                "fraction_of_amplitude_squared": fraction,
            }
            for lower, upper, count, fraction in zip(
                redshift_edges[:-1], redshift_edges[1:], redshift_count, redshift_fraction
            )
        ],
    )
    summary = {
        "catalog": str(catalog_path),
        "history": str(history_path),
        "volume_cmpc3": volume_cmpc3,
        "cosmology": {"H0_km_s_Mpc": float(header["h0"]), "Omega_m": float(header["omega_m"])},
        "reference_frequency_hz": REFERENCE_FREQUENCY_HZ,
        "reference_delay_gyr": REFERENCE_DELAY_GYR,
        "valid_chirp_mass_count": int(np.count_nonzero(valid_mass)),
        "assumptions": [
            "every uncensored numerical capture becomes a physical coalescence after the fixed delay",
            "binary orbits are circular and their frequency evolution is driven only by gravitational radiation",
            "chirp masses remain at their last-resolved HR5 values",
            "events after the present cosmic age are censored",
        ],
    }
    (output_dir / "hr5_conditional_pta_reference.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote conditional PTA reference results to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--history", type=Path, default=Path("results/hr5/hr5_sink_history.csv"))
    parser.add_argument("--catalog", type=Path, default=Path("results/hr5/hr5_capture_catalog.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/hr5/pta_reference"))
    parser.add_argument("--volume-cmpc3", type=float, default=VOLUME_CMPC3)
    parser.add_argument("--maximum-delay-gyr", type=float, default=12.0)
    parser.add_argument("--delay-step-gyr", type=float, default=0.25)
    args = parser.parse_args()
    analyze(
        args.tree,
        args.history,
        args.catalog,
        args.output_dir,
        args.volume_cmpc3,
        args.maximum_delay_gyr,
        args.delay_step_gyr,
    )


if __name__ == "__main__":
    main()
