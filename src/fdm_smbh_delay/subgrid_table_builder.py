"""Build accepted subgrid rows from matched-separation convergence results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .exchange_scaling import (
    exchange_scales,
    schrodinger_poisson_similarity_parameter,
)
from .subgrid_calibration import (
    ACCEPTED_STATUS,
    MAXIMUM_ACCEPTED_ENERGY_ERROR_OVER_TRANSFER,
    MAXIMUM_ACCEPTED_SPATIAL_SYSTEMATIC_FRACTION,
    MINIMUM_ACCEPTED_COMPLETE_ORBITS,
    MINIMUM_ACCEPTED_CORE_RADIUS_CELLS,
    SubgridCalibrationRow,
)


@dataclass(frozen=True)
class CalibrationSource:
    profile_id: str
    convergence_summary: Path


@dataclass(frozen=True)
class SourceBuildResult:
    profile_id: str
    source_case_id: str
    convergence_summary: str
    source_sha256: str
    accepted_rows: tuple[SubgridCalibrationRow, ...]
    rejected_bins: tuple[dict, ...]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _physical_definition(run: Path) -> dict:
    metadata = _read_json(run / "fdm_adapter_metadata.json")
    config = _read_json(run / "config.uldm")
    particles = config["Matter Particles"]["Condition"]
    solitons = config["ULDM Solitons"]["Condition"]
    if len(particles) != 2 or len(solitons) != 1:
        raise ValueError(f"{run} must contain two SMBHs and one soliton")
    mass1, mass2 = (float(particle[0]) for particle in particles)
    soliton_mass = float(solitons[0][0])
    core_radius = float(metadata["core_radius_reference_pc"])
    return {
        "case_id": str(metadata["case_id"]),
        "mass1_msun": mass1,
        "mass2_msun": mass2,
        "soliton_mass_msun": soliton_mass,
        "core_radius_pc": core_radius,
        "particle_mass_ev": float(metadata["particle_mass_ev"]),
        "cell_size_pc": float(metadata["box_size_pc"])
        / int(metadata["resolution"]),
        "resolution": int(metadata["resolution"]),
    }


def _same_physical_case(reference: dict, comparison: dict) -> bool:
    if reference["case_id"] != comparison["case_id"]:
        return False
    fields_to_compare = (
        "mass1_msun",
        "mass2_msun",
        "soliton_mass_msun",
        "core_radius_pc",
        "particle_mass_ev",
    )
    return all(
        np.isclose(
            reference[field],
            comparison[field],
            rtol=1.0e-12,
            atol=0.0,
        )
        for field in fields_to_compare
    )


def _minimum_core_cells(run: Path, start: float, end: float) -> float:
    path = run / "wave_response_timeseries.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"matched calibration requires sparse wave response: {path}"
        )
    response = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
    required = {"time_myr", "measured_half_density_radius_pc"}
    missing = required - set(response.dtype.names or ())
    if missing:
        raise ValueError(f"{path} lacks {', '.join(sorted(missing))}")
    times = np.asarray(response["time_myr"], dtype=float)
    radii = np.asarray(
        response["measured_half_density_radius_pc"], dtype=float
    )
    if (
        times.size < 2
        or np.any(~np.isfinite(times))
        or np.any(~np.isfinite(radii))
        or np.any(np.diff(times) <= 0.0)
        or np.any(radii <= 0.0)
        or start < times[0] - 1.0e-12
        or end > times[-1] + 1.0e-12
        or end <= start
    ):
        raise ValueError(f"{path} cannot sample the requested time interval")
    inside = (times >= start) & (times <= end)
    sampled = np.concatenate(
        (
            np.asarray([np.interp(start, times, radii)]),
            radii[inside],
            np.asarray([np.interp(end, times, radii)]),
        )
    )
    definition = _physical_definition(run)
    return float(np.min(sampled) / definition["cell_size_pc"])


def _matched_run(bin_row: dict, label: str) -> dict:
    matches = [row for row in bin_row["runs"] if row["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"matched bin does not contain one {label} row")
    return matches[0]


def build_source_rows(
    source: CalibrationSource,
    *,
    maximum_spatial_systematic_fraction: float = (
        MAXIMUM_ACCEPTED_SPATIAL_SYSTEMATIC_FRACTION
    ),
    maximum_energy_error_over_transfer: float = (
        MAXIMUM_ACCEPTED_ENERGY_ERROR_OVER_TRANSFER
    ),
    minimum_complete_orbits_per_bin: int = MINIMUM_ACCEPTED_COMPLETE_ORBITS,
    minimum_core_radius_cells: float = MINIMUM_ACCEPTED_CORE_RADIUS_CELLS,
) -> SourceBuildResult:
    """Accept bins that pass spatial, Hamiltonian, and core sampling gates."""

    limits = np.asarray(
        [
            maximum_spatial_systematic_fraction,
            maximum_energy_error_over_transfer,
            minimum_core_radius_cells,
        ],
        dtype=float,
    )
    if (
        np.any(~np.isfinite(limits))
        or maximum_spatial_systematic_fraction < 0.0
        or maximum_spatial_systematic_fraction
        > MAXIMUM_ACCEPTED_SPATIAL_SYSTEMATIC_FRACTION
        or maximum_energy_error_over_transfer <= 0.0
        or maximum_energy_error_over_transfer
        > MAXIMUM_ACCEPTED_ENERGY_ERROR_OVER_TRANSFER
        or minimum_core_radius_cells < MINIMUM_ACCEPTED_CORE_RADIUS_CELLS
        or minimum_complete_orbits_per_bin < MINIMUM_ACCEPTED_COMPLETE_ORBITS
    ):
        raise ValueError("subgrid acceptance limits are invalid")

    path = source.convergence_summary.expanduser().resolve()
    summary = _read_json(path)
    if summary.get("status") != "common_resolved_interval_compared":
        raise ValueError(f"convergence summary is incomplete: {path}")
    matched = summary.get("matched_separation")
    if not isinstance(matched, dict) or not matched.get("bins"):
        raise ValueError(f"matched-separation bins are absent: {path}")
    if (
        int(matched["minimum_complete_orbits_per_run_per_bin"])
        < minimum_complete_orbits_per_bin
    ):
        raise ValueError("convergence summary used an insufficient orbit gate")
    run_rows = summary["runs"]
    if len(run_rows) != 2:
        raise ValueError("subgrid table sources require one resolution pair")
    reference_label = summary["reference_label"]
    reference_summary = next(
        row for row in run_rows if row["label"] == reference_label
    )
    comparison_summary = next(
        row for row in run_rows if row["label"] != reference_label
    )
    reference_run = Path(reference_summary["run"])
    comparison_run = Path(comparison_summary["run"])
    reference = _physical_definition(reference_run)
    comparison = _physical_definition(comparison_run)
    if not _same_physical_case(reference, comparison):
        raise ValueError("resolution pair does not describe the same physical case")
    if reference["resolution"] <= comparison["resolution"]:
        raise ValueError("the convergence reference must have finer resolution")

    scales = exchange_scales(
        mass1_msun=reference["mass1_msun"],
        mass2_msun=reference["mass2_msun"],
        soliton_mass_msun=reference["soliton_mass_msun"],
        core_radius_pc=reference["core_radius_pc"],
    )
    similarity = schrodinger_poisson_similarity_parameter(
        particle_mass_ev=reference["particle_mass_ev"],
        soliton_mass_msun=reference["soliton_mass_msun"],
        core_radius_pc=reference["core_radius_pc"],
    )
    binary_fraction = (
        reference["mass1_msun"] + reference["mass2_msun"]
    ) / reference["soliton_mass_msun"]
    global_reasons = []
    for row in (reference_summary, comparison_summary):
        if (
            float(row["maximum_energy_error_over_transfer"])
            > maximum_energy_error_over_transfer
        ):
            global_reasons.append(
                f"{row['label']} exceeds the Hamiltonian error limit"
            )

    accepted = []
    rejected = []
    for bin_row in matched["bins"]:
        reference_bin = _matched_run(bin_row, reference_summary["label"])
        comparison_bin = _matched_run(bin_row, comparison_summary["label"])
        reasons = list(global_reasons)
        for run_bin in (reference_bin, comparison_bin):
            if int(run_bin["complete_orbits"]) < minimum_complete_orbits_per_bin:
                reasons.append(
                    f"{run_bin['label']} has fewer than the required orbits"
                )
        reference_core_cells = _minimum_core_cells(
            reference_run,
            float(reference_bin["minimum_time_myr"]),
            float(reference_bin["maximum_time_myr"]),
        )
        comparison_core_cells = _minimum_core_cells(
            comparison_run,
            float(comparison_bin["minimum_time_myr"]),
            float(comparison_bin["maximum_time_myr"]),
        )
        if reference_core_cells < minimum_core_radius_cells:
            reasons.append("reference half-density radius is underresolved")
        if comparison_core_cells < minimum_core_radius_cells:
            reasons.append("comparison half-density radius is underresolved")

        differences = comparison_bin[
            "fractional_rate_difference_from_reference"
        ]
        systematic = {}
        for field in (
            "orbital_power",
            "orbital_torque",
            "wave_total_energy_rate",
        ):
            difference = differences.get(field)
            if difference is None or not np.isfinite(float(difference)):
                reasons.append(f"{field} has no finite spatial comparison")
                systematic[field] = float("nan")
            else:
                systematic[field] = abs(float(difference))
                if systematic[field] > maximum_spatial_systematic_fraction:
                    reasons.append(f"{field} exceeds the spatial systematic limit")

        if reasons:
            rejected.append(
                {
                    "separation_bin_index": int(bin_row["bin"]),
                    "reasons": sorted(set(reasons)),
                    "spatial_systematic_fraction": systematic,
                    "reference_minimum_half_density_radius_over_cell_size": (
                        reference_core_cells
                    ),
                    "comparison_minimum_half_density_radius_over_cell_size": (
                        comparison_core_cells
                    ),
                }
            )
            continue

        rates = reference_bin["rates"]
        core_radius = reference["core_radius_pc"]
        accepted.append(
            SubgridCalibrationRow(
                profile_id=source.profile_id,
                source_case_id=reference["case_id"],
                schrodinger_poisson_similarity_parameter=similarity,
                binary_to_soliton_mass=binary_fraction,
                separation_bin_index=int(bin_row["bin"]),
                lower_separation_over_core_radius=float(
                    bin_row["lower_separation_pc"] / core_radius
                ),
                upper_separation_over_core_radius=float(
                    bin_row["upper_separation_pc"] / core_radius
                ),
                reference_mean_separation_over_core_radius=float(
                    reference_bin["mean_separation_pc"] / core_radius
                ),
                dimensionless_orbital_power=float(
                    rates["orbital_power"]["estimate"]
                    / scales.orbital_power_msun_pc2_myr3
                ),
                dimensionless_orbital_torque=float(
                    rates["orbital_torque"]["estimate"]
                    / scales.orbital_torque_msun_pc2_myr2
                ),
                dimensionless_wave_total_energy_rate=float(
                    rates["wave_total_energy_rate"]["estimate"]
                    / scales.orbital_power_msun_pc2_myr3
                ),
                orbital_power_spatial_systematic_fraction=systematic[
                    "orbital_power"
                ],
                orbital_torque_spatial_systematic_fraction=systematic[
                    "orbital_torque"
                ],
                wave_total_spatial_systematic_fraction=systematic[
                    "wave_total_energy_rate"
                ],
                reference_resolution=reference["resolution"],
                comparison_resolution=comparison["resolution"],
                reference_complete_orbits=int(
                    reference_bin["complete_orbits"]
                ),
                comparison_complete_orbits=int(
                    comparison_bin["complete_orbits"]
                ),
                reference_minimum_half_density_radius_over_cell_size=(
                    reference_core_cells
                ),
                comparison_minimum_half_density_radius_over_cell_size=(
                    comparison_core_cells
                ),
                convergence_status=ACCEPTED_STATUS,
            )
        )
    return SourceBuildResult(
        profile_id=source.profile_id,
        source_case_id=reference["case_id"],
        convergence_summary=str(path),
        source_sha256=_sha256(path),
        accepted_rows=tuple(accepted),
        rejected_bins=tuple(rejected),
    )


def write_calibration_table(
    sources: Iterable[CalibrationSource],
    *,
    output: Path,
    maximum_spatial_systematic_fraction: float = (
        MAXIMUM_ACCEPTED_SPATIAL_SYSTEMATIC_FRACTION
    ),
    maximum_energy_error_over_transfer: float = (
        MAXIMUM_ACCEPTED_ENERGY_ERROR_OVER_TRANSFER
    ),
    minimum_complete_orbits_per_bin: int = MINIMUM_ACCEPTED_COMPLETE_ORBITS,
    minimum_core_radius_cells: float = MINIMUM_ACCEPTED_CORE_RADIUS_CELLS,
) -> dict:
    """Publish a CSV followed by a checksum-bearing commit sidecar.

    The two files cannot be replaced atomically as a pair.  The summary is
    therefore the commit marker: runtime release loading verifies its CSV
    checksum and rejects a table if an interruption left the pair mismatched.
    Rerunning this function repairs either an incomplete or stale pair.
    """

    results = [
        build_source_rows(
            source,
            maximum_spatial_systematic_fraction=(
                maximum_spatial_systematic_fraction
            ),
            maximum_energy_error_over_transfer=(
                maximum_energy_error_over_transfer
            ),
            minimum_complete_orbits_per_bin=minimum_complete_orbits_per_bin,
            minimum_core_radius_cells=minimum_core_radius_cells,
        )
        for source in sources
    ]
    rows = sorted(
        (row for result in results for row in result.accepted_rows),
        key=lambda row: (
            row.profile_id,
            row.binary_to_soliton_mass,
            row.reference_mean_separation_over_core_radius,
        ),
    )
    if not rows:
        raise ValueError("no matched-separation bin passes the subgrid gates")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    names = [field.name for field in fields(SubgridCalibrationRow)]
    temporary = output.with_name(f".{output.name}.tmp")
    import csv

    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
        stream.flush()
        os.fsync(stream.fileno())
    table_sha256 = _sha256(temporary)
    os.replace(temporary, output)
    summary = {
        "status": "accepted_subgrid_calibration_table",
        "schema_version": 1,
        "rows": len(rows),
        "profiles": sorted({row.profile_id for row in rows}),
        "table": {
            "file": output.name,
            "sha256": table_sha256,
            "rows": len(rows),
        },
        "acceptance": {
            "maximum_spatial_systematic_fraction": (
                maximum_spatial_systematic_fraction
            ),
            "maximum_energy_error_over_transfer": (
                maximum_energy_error_over_transfer
            ),
            "minimum_complete_orbits_per_bin": minimum_complete_orbits_per_bin,
            "minimum_core_radius_cells": minimum_core_radius_cells,
            "extrapolation": "prohibited",
        },
        "sources": [
            {
                "profile_id": result.profile_id,
                "source_case_id": result.source_case_id,
                "convergence_summary": result.convergence_summary,
                "source_sha256": result.source_sha256,
                "accepted_bins": len(result.accepted_rows),
                "rejected_bins": list(result.rejected_bins),
            }
            for result in results
        ],
    }
    summary_path = output.with_suffix(".summary.json")
    summary_temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    with summary_temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(summary_temporary, summary_path)
    return summary
