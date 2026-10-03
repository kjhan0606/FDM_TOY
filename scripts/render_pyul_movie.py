#!/usr/bin/env python3
"""Render a scientific movie from a live-wave PyUL_NBody calculation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from fdm_smbh_delay.pyul import ordered_output_paths, output_index, pyul_unit_system


def _density_frame_paths(run: Path) -> tuple[list[Path], str]:
    plane_paths = sorted(
        (run / "Outputs" / "2Density").glob("R2D_#*.npy"), key=output_index
    )
    if plane_paths:
        return plane_paths, "central_density_plane"
    wave_paths = sorted(
        (run / "Outputs" / "3Wfn").glob("P3D_#*.npy"), key=output_index
    )
    if wave_paths:
        return wave_paths, "central_plane_from_3d_wavefunction"
    raise FileNotFoundError(
        "the run contains neither Outputs/2Density nor Outputs/3Wfn frames"
    )


def _load_density_plane(path: Path, source: str) -> np.ndarray:
    saved = np.load(path, mmap_mode="r")
    if source == "central_density_plane":
        if saved.ndim != 2 or saved.shape[0] != saved.shape[1]:
            raise ValueError(f"invalid central density plane in {path}")
        return np.asarray(saved, dtype=float)
    if saved.ndim != 3 or len(set(saved.shape)) != 1:
        raise ValueError(f"invalid 3D wavefunction in {path}")
    central_wave = np.asarray(saved[:, :, saved.shape[2] // 2])
    return np.abs(central_wave) ** 2


def _select_evenly(paths: list[Path], maximum_frames: int | None) -> list[Path]:
    if maximum_frames is None or len(paths) <= maximum_frames:
        return paths
    if maximum_frames < 2:
        raise ValueError("--max-frames must be at least two")
    indices = np.rint(np.linspace(0, len(paths) - 1, maximum_frames)).astype(int)
    return [paths[index] for index in np.unique(indices)]


def _density_limits(
    paths: list[Path], source: str, density_conversion: float
) -> tuple[float, float]:
    samples = []
    for path in paths:
        plane = _load_density_plane(path, source) * density_conversion
        stride = max(1, plane.shape[0] // 128)
        sample = plane[::stride, ::stride]
        samples.append(sample[np.isfinite(sample) & (sample > 0.0)])
    positive = np.concatenate(samples)
    if positive.size == 0:
        raise ValueError("movie frames contain no positive finite FDM density")
    lower, upper = np.quantile(positive, [0.002, 0.998])
    lower = max(float(lower), np.finfo(float).tiny)
    upper = float(upper)
    if not upper > lower:
        upper = float(np.max(positive))
    if not upper > lower:
        raise ValueError("movie density has no measurable dynamic range")
    return lower, upper


def _read_diagnostics(run: Path) -> np.ndarray:
    path = run / "conservation_timeseries.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run scripts/analyze_pyul_wave_run.py first"
        )
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(data)


def _binary_positions(
    run: Path, length_conversion: float
) -> tuple[np.ndarray, np.ndarray]:
    paths = ordered_output_paths(run / "Outputs" / "NBody", "NTM_#*.npy")
    indices = np.asarray([output_index(path) for path in paths], dtype=int)
    positions = []
    for path in paths:
        state = np.load(path)
        if state.size != 12:
            raise ValueError("movie renderer requires exactly two SMBHs")
        positions.append(state.reshape(2, 6)[:, :3] * length_conversion)
    return indices, np.asarray(positions)


def _writer(output: Path, frames_per_second: int):
    if output.suffix.lower() == ".gif":
        return animation.PillowWriter(fps=frames_per_second)
    if output.suffix.lower() != ".mp4":
        raise ValueError("movie output must end in .mp4 or .gif")
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            "MP4 output requires the optional imageio-ffmpeg package"
        ) from error
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    return animation.FFMpegWriter(
        fps=frames_per_second,
        codec="libx264",
        bitrate=5000,
        extra_args=["-pix_fmt", "yuv420p"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--poster", type=Path)
    parser.add_argument(
        "--zoom-width-pc",
        type=float,
        help="physical width of the central density inset",
    )
    args = parser.parse_args()
    if args.fps <= 0 or args.dpi <= 0:
        raise ValueError("--fps and --dpi must be positive")

    run = args.run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    metadata = json.loads(
        (run / "fdm_adapter_metadata.json").read_text(encoding="utf-8")
    )
    units = pyul_unit_system(metadata)
    diagnostics = _read_diagnostics(run)
    all_density_paths, density_source = _density_frame_paths(run)
    density_paths = _select_evenly(all_density_paths, args.max_frames)
    frame_indices = np.asarray([output_index(path) for path in density_paths])
    if np.any(frame_indices >= diagnostics.size):
        raise ValueError("density frame indices exceed the diagnostic time series")

    nbody_indices, all_positions = _binary_positions(run, units.length_pc)
    if not np.array_equal(nbody_indices, np.arange(nbody_indices.size)):
        raise ValueError("NBody output indices must form a consecutive sequence")
    if np.any(frame_indices >= all_positions.shape[0]):
        raise ValueError("density frame indices exceed the NBody history")

    density_conversion = units.density_msun_pc3
    density_min, density_max = _density_limits(
        density_paths, density_source, density_conversion
    )
    log_density_min = np.log10(density_min)
    log_density_max = np.log10(density_max)
    box_size = float(metadata["box_size_pc"])
    cell_size = float(metadata["cell_size_pc"])
    default_zoom_width = 1.6 * float(diagnostics["separation_pc"][0])
    zoom_width = (
        default_zoom_width
        if args.zoom_width_pc is None
        else float(args.zoom_width_pc)
    )
    if not 4.0 * cell_size <= zoom_width < box_size:
        raise ValueError(
            "--zoom-width-pc must span at least four cells and remain below "
            "the simulation-box width"
        )
    zoom_half_width = 0.5 * zoom_width
    times = diagnostics["time_myr"]
    separation = diagnostics["separation_pc"]

    energy_names = (
        "binary_orbital_energy",
        "wave_intrinsic_energy",
        "wave_bh_interaction_grid",
        "combined_energy",
    )
    energy_changes = {
        name: np.asarray(diagnostics[name] - diagnostics[name][0], dtype=float)
        for name in energy_names
    }
    exchange_scale = max(
        np.max(np.abs(energy_changes[name]))
        for name in energy_names[:-1]
    )
    if exchange_scale <= np.finfo(float).tiny:
        raise ValueError("diagnostics contain no measurable energy exchange")

    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
        }
    )
    figure = plt.figure(figsize=(10.8, 6.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.45, 1.0))
    density_axis = figure.add_subplot(grid[:, 0])
    separation_axis = figure.add_subplot(grid[0, 1])
    energy_axis = figure.add_subplot(grid[1, 1])

    first_density = (
        _load_density_plane(density_paths[0], density_source) * density_conversion
    )
    density_image = density_axis.imshow(
        np.log10(np.maximum(first_density.T, density_min)),
        origin="lower",
        extent=(-box_size / 2.0, box_size / 2.0, -box_size / 2.0, box_size / 2.0),
        cmap="magma",
        vmin=log_density_min,
        vmax=log_density_max,
        interpolation="nearest",
        rasterized=True,
    )
    colorbar = figure.colorbar(density_image, ax=density_axis, pad=0.015)
    colorbar.set_label(r"$\log_{10}(\rho_{\rm FDM}/M_\odot\,{\rm pc}^{-3})$")
    density_axis.set_xlabel(r"$x$ [pc]")
    density_axis.set_ylabel(r"$y$ [pc]")
    density_axis.text(
        0.02, 0.975, "(a)", transform=density_axis.transAxes,
        va="top", ha="left", color="white"
    )
    time_text = density_axis.text(
        0.52, 0.975, "", transform=density_axis.transAxes,
        va="top", ha="center", color="white"
    )
    trail_colours = ("#54d2d2", "#ffb14e")
    trails = [
        density_axis.plot([], [], color=colour, linewidth=0.9, alpha=0.8)[0]
        for colour in trail_colours
    ]
    bh_markers = [
        density_axis.plot(
            [], [], marker="o", linestyle="none", markersize=6.5,
            markerfacecolor=colour, markeredgecolor="white", markeredgewidth=0.7,
            label=f"SMBH {index + 1}",
        )[0]
        for index, colour in enumerate(trail_colours)
    ]
    density_axis.legend(loc="lower left", frameon=False, labelcolor="white")

    zoom_rectangle = Rectangle(
        (-zoom_half_width, -zoom_half_width),
        zoom_width,
        zoom_width,
        fill=False,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.8,
    )
    density_axis.add_patch(zoom_rectangle)
    inset_axis = density_axis.inset_axes([0.59, 0.57, 0.38, 0.38])
    inset_image = inset_axis.imshow(
        np.log10(np.maximum(first_density.T, density_min)),
        origin="lower",
        extent=(-box_size / 2.0, box_size / 2.0, -box_size / 2.0, box_size / 2.0),
        cmap="magma",
        vmin=log_density_min,
        vmax=log_density_max,
        interpolation="nearest",
        rasterized=True,
    )
    inset_axis.set_xlim(-zoom_half_width, zoom_half_width)
    inset_axis.set_ylim(-zoom_half_width, zoom_half_width)
    inset_axis.tick_params(
        axis="both", which="both", direction="in", labelsize=5.5,
        colors="white", length=2.0, pad=1.5
    )
    for spine in inset_axis.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.8)
    inset_axis.text(
        0.04,
        0.95,
        rf"central ${zoom_width:.2f}\,{{\rm pc}}$",
        transform=inset_axis.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=6.0,
    )
    inset_trails = [
        inset_axis.plot([], [], color=colour, linewidth=1.0, alpha=0.9)[0]
        for colour in trail_colours
    ]
    inset_markers = [
        inset_axis.plot(
            [], [], marker="o", linestyle="none", markersize=5.2,
            markerfacecolor=colour, markeredgecolor="white", markeredgewidth=0.6,
        )[0]
        for colour in trail_colours
    ]

    separation_axis.plot(times, separation, color="0.25", linewidth=1.0)
    separation_axis.axhline(
        2.0 * cell_size,
        color="#b23a48",
        linewidth=0.9,
        linestyle="--",
        label=r"$2\Delta x$",
    )
    separation_point = separation_axis.plot(
        [], [], marker="o", markersize=4.5, color="#0077b6", linestyle="none"
    )[0]
    separation_axis.set_xlim(float(times[0]), float(times[-1]))
    separation_axis.set_ylim(0.0, 1.08 * float(np.max(separation)))
    separation_axis.set_xlabel(r"$t$ [Myr]")
    separation_axis.set_ylabel(r"$D$ [pc]")
    separation_axis.legend(loc="lower left", frameon=False)
    separation_axis.text(0.02, 0.94, "(b)", transform=separation_axis.transAxes)

    energy_styles = {
        "binary_orbital_energy": (r"$\Delta E_{\rm orb}$", "#0077b6"),
        "wave_intrinsic_energy": (r"$\Delta E_{\Psi}$", "#d1495b"),
        "wave_bh_interaction_grid": (r"$\Delta W_{\Psi\bullet}$", "#2a9d8f"),
        "combined_energy": (r"$\Delta H$", "#222222"),
    }
    energy_points = {}
    for name, (label, colour) in energy_styles.items():
        normalized = energy_changes[name] / exchange_scale
        energy_axis.plot(times, normalized, color=colour, linewidth=0.9, label=label)
        energy_points[name] = energy_axis.plot(
            [], [], marker="o", markersize=3.8, color=colour, linestyle="none"
        )[0]
    energy_axis.axhline(0.0, color="0.7", linewidth=0.6)
    energy_axis.set_xlim(float(times[0]), float(times[-1]))
    all_normalized = np.concatenate(
        [energy_changes[name] / exchange_scale for name in energy_names]
    )
    energy_range = max(float(np.max(np.abs(all_normalized))), 1.0e-3)
    energy_axis.set_ylim(-1.08 * energy_range, 1.08 * energy_range)
    energy_axis.set_xlabel(r"$t$ [Myr]")
    energy_axis.set_ylabel(r"$\Delta E/E_{\rm exch,max}$")
    energy_axis.legend(loc="best", ncol=2, frameon=False)
    energy_axis.text(0.02, 0.94, "(c)", transform=energy_axis.transAxes)

    def update(frame_number: int):
        saved_index = int(frame_indices[frame_number])
        plane = (
            _load_density_plane(density_paths[frame_number], density_source)
            * density_conversion
        )
        displayed_density = np.log10(np.maximum(plane.T, density_min))
        density_image.set_data(displayed_density)
        inset_image.set_data(displayed_density)
        for member in range(2):
            history = all_positions[: saved_index + 1, member]
            trails[member].set_data(history[:, 0], history[:, 1])
            bh_markers[member].set_data(
                [history[-1, 0]], [history[-1, 1]]
            )
            inset_trails[member].set_data(history[:, 0], history[:, 1])
            inset_markers[member].set_data(
                [history[-1, 0]], [history[-1, 1]]
            )
        time_text.set_text(rf"$t={times[saved_index]:.4f}\,{{\rm Myr}}$")
        separation_point.set_data(
            [times[saved_index]], [separation[saved_index]]
        )
        for name in energy_names:
            energy_points[name].set_data(
                [times[saved_index]],
                [energy_changes[name][saved_index] / exchange_scale],
            )
        return (
            density_image,
            inset_image,
            time_text,
            separation_point,
            *trails,
            *bh_markers,
            *inset_trails,
            *inset_markers,
            *energy_points.values(),
        )

    movie = animation.FuncAnimation(
        figure, update, frames=len(density_paths), interval=1000.0 / args.fps,
        blit=False, repeat=False
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output, writer=_writer(output, args.fps), dpi=args.dpi)
    if args.poster is not None:
        update(len(density_paths) - 1)
        poster = args.poster.expanduser().resolve()
        poster.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(poster, dpi=args.dpi)
    plt.close(figure)

    summary = {
        "run": str(run),
        "output": str(output),
        "density_source": density_source,
        "frame_count": len(density_paths),
        "first_saved_index": int(frame_indices[0]),
        "last_saved_index": int(frame_indices[-1]),
        "frames_per_second": args.fps,
        "density_limits_msun_pc3": [density_min, density_max],
        "central_zoom_width_pc": zoom_width,
        "energy_normalization_msun_pc2_myr2": exchange_scale,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
