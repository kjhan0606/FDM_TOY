#!/usr/bin/env python3
"""Draw the HR5 fixed-delay and physical-delay comparison schematic."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, Polygon


BLUE = "#0072B2"
VERMILION = "#D55E00"
GRAY = "#777777"


def _polar(radius: float, angle_deg: float) -> tuple[float, float]:
    angle = np.deg2rad(angle_deg)
    return radius * np.cos(angle), radius * np.sin(angle)


def _draw_delay_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("(a) Delay assignment", loc="left", fontweight="bold")

    t_capture = 0.16
    t_present = 0.91
    ax.annotate(
        "",
        xy=(0.98, 0.10),
        xytext=(0.04, 0.10),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
    )
    ax.text(0.98, 0.045, "cosmic time", ha="right", va="center")
    ax.plot([t_capture, t_capture], [0.08, 0.93], ls="--", lw=0.9, color=GRAY)
    ax.plot([t_present, t_present], [0.08, 0.93], ls="--", lw=0.9, color=GRAY)
    ax.text(t_capture, 0.02, "$t_{\\rm cap}$\nHR5 capture", ha="center", va="top")
    ax.text(t_present, 0.02, "$t_0$", ha="center", va="top")

    y_reference = 0.76
    reference_end = 0.52
    ax.text(t_capture - 0.025, y_reference, "fixed delay", ha="right", va="center")
    ax.plot(t_capture, y_reference, "o", ms=5.5, mfc="white", mec="black", zorder=4)
    ax.annotate(
        "",
        xy=(reference_end, y_reference),
        xytext=(t_capture + 0.015, y_reference),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.4},
    )
    ax.plot(reference_end, y_reference, "*", ms=9, color="black", zorder=4)
    ax.add_patch(
        FancyArrowPatch(
            (t_capture, y_reference + 0.105),
            (reference_end, y_reference + 0.105),
            arrowstyle="<->",
            mutation_scale=8,
            lw=0.9,
            color="black",
        )
    )
    ax.text(
        0.5 * (t_capture + reference_end),
        y_reference + 0.13,
        "$\\Delta t_{\\rm ref}$",
        ha="center",
        va="bottom",
    )

    y_physical = 0.45
    numerical_end = 0.35
    fdm_end = 0.67
    physical_end = 0.84
    ax.text(t_capture - 0.025, y_physical, "physical delay", ha="right", va="center")
    ax.plot(t_capture, y_physical, "o", ms=5.5, mfc="white", mec="black", zorder=4)
    ax.plot([t_capture + 0.01, numerical_end], [y_physical, y_physical], lw=3.0, color=GRAY)
    ax.plot([numerical_end, fdm_end], [y_physical, y_physical], lw=3.0, color=BLUE)
    ax.annotate(
        "",
        xy=(physical_end, y_physical),
        xytext=(fdm_end, y_physical),
        arrowprops={"arrowstyle": "->", "color": VERMILION, "lw": 2.5},
    )
    ax.plot(physical_end, y_physical, "*", ms=9, color="black", zorder=4)
    ax.text(
        0.5 * (t_capture + numerical_end),
        y_physical + 0.065,
        "$\\tau_{\\rm num\\rightarrow1pc}$",
        ha="center",
        va="bottom",
        color=GRAY,
    )
    ax.text(
        0.5 * (numerical_end + fdm_end),
        y_physical + 0.065,
        "$\\tau_{\\rm FDM}^{1\\rightarrow0.01pc}$",
        ha="center",
        va="bottom",
        color=BLUE,
    )
    ax.text(
        0.5 * (fdm_end + physical_end),
        y_physical + 0.065,
        "$\\tau_{\\rm GW}$",
        ha="center",
        va="bottom",
        color=VERMILION,
    )

    y_censored = 0.23
    ax.plot(t_capture, y_censored, "o", ms=5.5, mfc="white", mec=GRAY, zorder=4)
    ax.annotate(
        "",
        xy=(0.98, y_censored),
        xytext=(t_capture + 0.01, y_censored),
        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.0, "ls": "--"},
    )
    ax.plot(t_present, y_censored, marker="x", ms=7, mew=1.2, color=GRAY)
    ax.text(
        0.61,
        y_censored - 0.055,
        "$t_{\\rm coal}^{\\rm FDM}>t_0$   censored",
        ha="center",
        va="top",
        color=GRAY,
    )


def _draw_lightcone_panel(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-3.7, 3.7)
    ax.set_ylim(-0.4, 5.7)
    ax.axis("off")
    ax.set_title("(b) Past-light-cone population", loc="left", fontweight="bold")

    theta_lo = 52.0
    theta_hi = 128.0
    r_inner = 4.65
    r_outer = 5.35
    r_boundary = 5.62
    angles = np.linspace(theta_lo, theta_hi, 160)
    inner = np.array([_polar(r_inner, angle) for angle in angles])
    outer = np.array([_polar(r_outer, angle) for angle in angles[::-1]])
    shell = np.vstack([inner, outer])
    ax.add_patch(Polygon(shell, closed=True, facecolor="0.90", edgecolor="none"))

    for angle in (theta_lo, theta_hi):
        x_end, y_end = _polar(r_boundary, angle)
        ax.plot([0.0, x_end], [0.0, y_end], color="black", lw=1.1)
    for radius in (r_inner, r_outer):
        arc_xy = np.array([_polar(radius, angle) for angle in angles])
        ax.plot(arc_xy[:, 0], arc_xy[:, 1], color="0.45", lw=0.8)

    ax.add_patch(Arc((0.0, 0.0), 1.1, 1.1, angle=0, theta1=theta_lo, theta2=theta_hi, lw=0.9))
    ax.text(0.0, 0.68, "$\\Omega$", ha="center", va="center")
    ax.text(0.0, -0.12, "observer", ha="center", va="top")
    x_r, y_r = _polar(r_inner, theta_lo)
    x_dr, y_dr = _polar(r_outer, theta_lo)
    ax.text(x_r + 0.08, y_r - 0.08, "$r$", rotation=theta_lo, ha="left", va="center")
    ax.text(x_dr + 0.08, y_dr - 0.02, "$r+dr$", rotation=theta_lo, ha="left", va="center")
    ax.text(0.0, 5.52, "coalescence shell", ha="center", va="center")

    systems = ((113.0, 4.95, 16.0), (90.0, 5.03, -18.0), (68.0, 4.92, 20.0))
    for angle, radius, offset_angle in systems:
        x_source, y_source = _polar(radius, angle)
        for wave_radius, alpha in ((0.24, 0.55), (0.45, 0.38)):
            ax.add_patch(
                plt.Circle(
                    (x_source, y_source),
                    wave_radius,
                    fill=False,
                    lw=0.7,
                    color="0.35",
                    alpha=alpha,
                )
            )
        ax.plot(x_source, y_source, "*", ms=9, color="black", zorder=5)
        dx, dy = _polar(0.31, offset_angle)
        ax.plot(
            x_source + dx,
            y_source + dy,
            "o",
            ms=5.2,
            mfc="white",
            mec="black",
            zorder=5,
        )

    legend_handles = [
        Line2D([], [], marker="o", ms=5.5, mfc="white", mec="black", ls="none", label="fixed-delay HR5"),
        Line2D([], [], marker="*", ms=8.5, color="black", ls="none", label="physical FDM"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.25),
        frameon=False,
        ncol=2,
        handletextpad=0.45,
        columnspacing=1.2,
    )


def make_figure(output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), gridspec_kw={"wspace": 0.12})
    _draw_delay_panel(axes[0])
    _draw_lightcone_panel(axes[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={"Title": "HR5 and FDM coalescence population comparison"},
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hr5_lightcone_comparison.pdf"),
        help="Output PDF path",
    )
    args = parser.parse_args()
    make_figure(args.output)


if __name__ == "__main__":
    main()
