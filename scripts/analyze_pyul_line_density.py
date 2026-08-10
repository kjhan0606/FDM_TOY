#!/usr/bin/env python3
"""Measure the evolving FDM density along a line through the soliton centre."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fdm_smbh_delay.pyul import ordered_output_paths, output_index, pyul_unit_system
from fdm_smbh_delay.wave_response import windowed_dominant_frequency


def _symmetric_sample(
    density: np.ndarray, coordinate: np.ndarray, radius: float
) -> np.ndarray:
    positive = int(np.argmin(np.abs(coordinate - radius)))
    negative = int(np.argmin(np.abs(coordinate + radius)))
    return 0.5 * (density[:, positive] + density[:, negative])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.expanduser().resolve()
    metadata = json.loads(
        (run / "fdm_adapter_metadata.json").read_text(encoding="utf-8")
    )
    config = json.loads((run / "config.uldm").read_text(encoding="utf-8"))
    units = pyul_unit_system(metadata)
    paths = ordered_output_paths(run / "Outputs" / "1Density", "R1D_#*.npy")
    indices = np.asarray([output_index(path) for path in paths], dtype=int)
    save_number = int(
        metadata.get("save_number", config["Save Options"]["Number"])
    )
    time = float(metadata["duration_myr"]) * indices / save_number
    density = np.asarray([np.load(path) for path in paths]) * units.density_msun_pc3
    resolution = int(metadata["resolution"])
    if density.shape != (len(paths), resolution):
        raise ValueError("line-density outputs do not match the grid resolution")
    cell_size_pc = float(metadata["box_size_pc"]) / resolution
    coordinate = (np.arange(resolution) - resolution // 2) * cell_size_pc
    core_radius = float(metadata["core_radius_reference_pc"])
    central_density = density[:, resolution // 2]
    inner = np.abs(coordinate) <= 4.0 * core_radius
    reference = density[0, inner]
    reference_norm = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    fractional_line_change = np.linalg.norm(
        density[:, inner] - reference[None, :], axis=1
    ) / reference_norm

    maximum_offset = min(resolution // 2 - 1, int(4.0 * core_radius / cell_size_pc))
    offsets = np.arange(1, maximum_offset + 1)
    positive = density[:, resolution // 2 + offsets]
    negative = density[:, resolution // 2 - offsets]
    symmetry_scale = np.maximum(
        np.linalg.norm(0.5 * (positive + negative), axis=1),
        np.finfo(float).tiny,
    )
    reflection_asymmetry = np.linalg.norm(positive - negative, axis=1) / symmetry_scale

    table = np.column_stack(
        (
            time,
            central_density,
            _symmetric_sample(density, coordinate, core_radius),
            _symmetric_sample(density, coordinate, 2.0 * core_radius),
            _symmetric_sample(density, coordinate, 4.0 * core_radius),
            fractional_line_change,
            reflection_asymmetry,
        )
    )
    np.savetxt(
        run / "line_density_diagnostics.csv",
        table,
        delimiter=",",
        header=(
            "time_myr,central_density_msun_pc3,mean_density_at_rc_msun_pc3,"
            "mean_density_at_2rc_msun_pc3,mean_density_at_4rc_msun_pc3,"
            "fractional_l2_change_within_4rc,reflection_asymmetry_within_4rc"
        ),
        comments="",
    )
    summary = {
        "status": "line_density_diagnosed",
        "samples": len(paths),
        "duration_myr": float(time[-1]),
        "line_direction": "simulation y axis through the soliton centre",
        "initial_central_density_msun_pc3": float(central_density[0]),
        "final_central_density_msun_pc3": float(central_density[-1]),
        "maximum_central_density_msun_pc3": float(np.max(central_density)),
        "central_density_fractional_change": float(
            central_density[-1] / central_density[0] - 1.0
        ),
        "maximum_fractional_l2_change_within_4rc": float(
            np.max(fractional_line_change)
        ),
        "maximum_reflection_asymmetry_within_4rc": float(
            np.max(reflection_asymmetry)
        ),
        "interpretation": (
            "one-dimensional density diagnostic; not a spherical profile or "
            "an energy-deposition measurement"
        ),
    }
    conservation_summary_path = run / "conservation_summary.json"
    if conservation_summary_path.is_file():
        conservation_summary = json.loads(
            conservation_summary_path.read_text(encoding="utf-8")
        )
        resolved_duration = conservation_summary.get(
            "initial_spatially_resolved_duration_myr"
        )
        resolved = (
            np.ones(time.size, dtype=bool)
            if resolved_duration is None
            else time <= float(resolved_duration)
        )
        if np.count_nonzero(resolved) >= 8:
            peak = windowed_dominant_frequency(
                time[resolved], central_density[resolved]
            )
            summary["initial_resolved_central_density_frequency"] = {
                "samples": int(np.count_nonzero(resolved)),
                "duration_myr": float(time[resolved][-1] - time[resolved][0]),
                "dominant_frequency_myr_inverse": peak.frequency_inverse_time,
                "dominant_period_myr": peak.period_time,
                "frequency_resolution_myr_inverse": (
                    peak.frequency_resolution_inverse_time
                ),
                "peak_power_fraction": peak.peak_power_fraction,
                "method": "linear detrending followed by a Hann-windowed FFT",
                "interpretation": (
                    "one-dimensional central-density oscillation; not an "
                    "identified soliton eigenmode"
                ),
            }
    (run / "line_density_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
