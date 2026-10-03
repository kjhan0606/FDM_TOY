"""Compare q/e resolution candidates with same-cell-size doubled-box controls.

Passing this comparison is necessary, not sufficient, for a production table:
the registered physical-bin design and release provenance remain independent.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree

from .convergence import (
    _COORDINATE_RATIO_BOOTSTRAP_SAMPLES,
    _TARGET_ORBIT_RATE_BOOTSTRAP_BLOCK_LENGTH,
    _initial_resolved_orbit_indices,
    load_convergence_run,
    summarize_convergence,
)
from .qe_followup_design import (
    read_verified_qe_followup_design,
    verify_qe_design_comparison_runs,
)
from .subgrid_table_builder import (
    CalibrationSource,
    SourceBuildResult,
    _physical_definition,
    _sha256,
    _same_physical_case,
    build_source_rows,
)
from .subgrid_calibration import is_qe_extension_case


_RATES = ("orbital_power", "orbital_torque", "wave_total_energy_rate")
_MAPPING_COORDINATES = (
    "mean_semimajor_axis_osculating_pc",
    "minimum_orbit_mean_semimajor_axis_pc",
    "maximum_orbit_mean_semimajor_axis_pc",
)
_ECCENTRICITY_COORDINATES = (
    "mean_eccentricity_osculating",
    "minimum_orbit_mean_eccentricity",
    "maximum_orbit_mean_eccentricity",
)
_RAW_INPUTS = (
    "fdm_adapter_metadata.json",
    "config.uldm",
    "torch_run_summary.json",
    "wave_response_summary.json",
    "conservation_summary.json",
    "orbit_averaged_exchange_summary.json",
    "conservation_timeseries.csv",
    "orbit_averaged_exchange.csv",
    "wave_response_timeseries.csv",
)
_MAX_DIAGNOSTIC_BYTES = 16 * 1024 * 1024


def _read(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("status") != "common_resolved_interval_compared":
        raise ValueError(f"box-control convergence summary is incomplete: {path}")
    return result


def _runs(summary: dict) -> tuple[dict, dict]:
    rows = summary.get("runs")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("box-control comparison requires exactly two runs")
    references = [row for row in rows if row.get("label") == summary.get("reference_label")]
    if len(references) != 1:
        raise ValueError("box-control reference label is ambiguous")
    return references[0], next(row for row in rows if row is not references[0])


def _fixed_bins(summary: dict) -> dict[int, dict]:
    matched = summary.get("matched_separation")
    if not isinstance(matched, dict) or matched.get("bin_edge_policy") != "fixed_physical_edges":
        raise ValueError("box control requires fixed physical separation bins")
    edges = matched.get("separation_bin_edges_pc")
    if (not isinstance(edges, list) or len(edges) < 2
            or summary.get("requested_separation_bin_edges_pc") != edges
            or matched.get("requested_bins") != len(edges) - 1
            or not all(np.isfinite(value) for value in edges)
            or any(left <= 0 or right <= left for left, right in zip(edges[:-1], edges[1:]))):
        raise ValueError("box-control fixed bin edges are invalid")
    bins = matched.get("bins")
    if not isinstance(bins, list) or matched.get("retained_bins") != len(bins):
        raise ValueError("box-control retained-bin count is invalid")
    by_index = {}
    for row in bins:
        index = row.get("bin")
        if (type(index) is not int or index < 0 or index >= len(edges) - 1
                or index in by_index
                or not np.isclose(row.get("lower_separation_pc"), edges[index], rtol=0, atol=1e-12)
                or not np.isclose(row.get("upper_separation_pc"), edges[index + 1], rtol=0, atol=1e-12)):
            raise ValueError("box-control bin is inconsistent with fixed edges")
        run_rows = row.get("runs")
        if not isinstance(run_rows, list) or len(run_rows) != 2:
            raise ValueError("box-control bin requires two run rows")
        for run_row in run_rows:
            try:
                mean_r = float(run_row["mean_separation_pc"])
                mean, lower, upper = (
                    float(run_row[field]) for field in _MAPPING_COORDINATES
                )
                mean_e, lower_e, upper_e = (
                    float(run_row[field]) for field in _ECCENTRICITY_COORDINATES
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "box-control bin lacks measured osculating coordinates"
                ) from error
            if (
                not np.isfinite(mean_r)
                or mean_r <= 0.0
                or mean_r < edges[index] - 1e-12
                or mean_r > edges[index + 1] + 1e-12
                or not all(np.isfinite(value) for value in (mean, lower, upper))
                or lower <= 0.0
                or mean <= 0.0
                or (mean < lower and not np.isclose(mean, lower, rtol=0, atol=1e-12))
                or (mean > upper and not np.isclose(mean, upper, rtol=0, atol=1e-12))
                or not all(np.isfinite(value) for value in (mean_e, lower_e, upper_e))
                or not 0.0 <= lower_e <= upper_e < 1.0
                or not 0.0 <= mean_e < 1.0
                or (mean_e < lower_e and not np.isclose(mean_e, lower_e, rtol=0, atol=1e-12))
                or (mean_e > upper_e and not np.isclose(mean_e, upper_e, rtol=0, atol=1e-12))
            ):
                raise ValueError(
                    "box-control bin has invalid osculating coordinates"
                )
            interval = run_row.get("coordinate_ratio_bootstrap")
            if not isinstance(interval, dict):
                raise ValueError("box-control bin lacks coordinate-ratio bootstrap")
            try:
                estimate = float(interval["estimate"])
                lower_95 = float(interval["lower_95"])
                upper_95 = float(interval["upper_95"])
                block_length = interval["bootstrap_block_length_orbits"]
                samples = interval["bootstrap_samples"]
                independent = interval["minimum_independent_blocks"]
                orbits = run_row["complete_orbits"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "box-control coordinate-ratio bootstrap is incomplete"
                ) from error
            expected_ratio = mean_r / (mean * (1.0 + 0.5 * mean_e**2))
            if (
                not all(np.isfinite(value) for value in (estimate, lower_95, upper_95))
                or lower_95 <= 0.0
                or upper_95 < lower_95
                or not np.isclose(estimate, expected_ratio, rtol=1e-10, atol=1e-12)
                or type(orbits) is not int
                or type(block_length) is not int
                or type(samples) is not int
                or type(independent) is not int
                or orbits < 2
                or block_length != min(
                    _TARGET_ORBIT_RATE_BOOTSTRAP_BLOCK_LENGTH, orbits // 2
                )
                or samples != _COORDINATE_RATIO_BOOTSTRAP_SAMPLES
                or independent != orbits // block_length
            ):
                raise ValueError("box-control coordinate-ratio bootstrap is invalid")
        by_index[index] = row
    return by_index


def _run_bin(bin_row: dict, label: str) -> dict:
    rows = [row for row in bin_row["runs"] if row.get("label") == label]
    if len(rows) != 1:
        raise ValueError("box-control bin lacks its shared run")
    return rows[0]


def _mapping_observation(run_bin: dict) -> dict:
    """Keep measured bin coordinates separate from the rate-release row."""

    mean_r = float(run_bin["mean_separation_pc"])
    mean_a = float(run_bin["mean_semimajor_axis_osculating_pc"])
    mean_e = float(run_bin["mean_eccentricity_osculating"])
    return {
        "complete_orbits": run_bin["complete_orbits"],
        "mean_separation_pc": mean_r,
        **{field: run_bin[field] for field in _MAPPING_COORDINATES},
        **{field: run_bin[field] for field in _ECCENTRICITY_COORDINATES},
        "measured_over_kepler_mean_ratio": mean_r / (
            mean_a * (1.0 + 0.5 * mean_e**2)
        ),
        "coordinate_ratio_bootstrap": dict(run_bin["coordinate_ratio_bootstrap"]),
    }


def _necessary_mapping_coordinate_overlap(observations: dict) -> dict:
    """Bound a necessary rectangular overlap; never assert joint 2D support."""

    runs = tuple(observations[role] for role in ("fine", "coarse", "doubled_box"))
    lower_axis = max(run["minimum_orbit_mean_semimajor_axis_pc"] for run in runs)
    upper_axis = min(run["maximum_orbit_mean_semimajor_axis_pc"] for run in runs)
    lower_e = max(run["minimum_orbit_mean_eccentricity"] for run in runs)
    upper_e = min(run["maximum_orbit_mean_eccentricity"] for run in runs)
    return {
        "status": (
            "rectangular_overlap_necessary_only"
            if lower_axis <= upper_axis and lower_e <= upper_e
            else "no_common_coordinate_rectangle_censored"
        ),
        "lower_semimajor_axis_pc": lower_axis,
        "upper_semimajor_axis_pc": upper_axis,
        "lower_eccentricity": lower_e,
        "upper_eccentricity": upper_e,
        "joint_a_e_support_verified": False,
        "runtime_mapping_admitted": False,
    }


def _joint_orbit_coordinate_proximity(
    observations: dict[str, np.ndarray],
) -> dict:
    """Find the nearest actual three-run (a,e) orbit triplet, without release."""

    roles = ("fine", "coarse", "doubled_box")
    selected: dict[str, np.ndarray] = {}
    coordinates: dict[str, np.ndarray] = {}
    for role in roles:
        orbit = observations[role]
        if orbit.size == 0:
            return {
                "status": "no_joint_orbit_sample_censored",
                "runtime_mapping_admitted": False,
            }
        required = {
            "mean_separation_pc", "mean_semimajor_axis_osculating_pc",
            "mean_eccentricity_osculating",
        }
        if not required <= set(orbit.dtype.names or ()):
            raise ValueError("joint-coordinate diagnostic lacks orbit columns")
        separation = np.asarray(orbit["mean_separation_pc"], dtype=float)
        axis = np.asarray(orbit["mean_semimajor_axis_osculating_pc"], dtype=float)
        eccentricity = np.asarray(orbit["mean_eccentricity_osculating"], dtype=float)
        if (
            np.any(~np.isfinite(separation)) or np.any(separation <= 0.0)
            or np.any(~np.isfinite(axis)) or np.any(axis <= 0.0)
            or np.any(~np.isfinite(eccentricity))
            or np.any((eccentricity < 0.0) | (eccentricity >= 1.0))
        ):
            raise ValueError("joint-coordinate diagnostic has invalid orbit coordinates")
        selected[role] = np.column_stack((separation, axis, eccentricity))
        coordinates[role] = np.column_stack((np.log(axis), eccentricity))

    fine = coordinates["fine"]
    coarse_distance, coarse_index = cKDTree(coordinates["coarse"]).query(
        fine, k=1, p=np.inf
    )
    box_distance, box_index = cKDTree(coordinates["doubled_box"]).query(
        fine, k=1, p=np.inf
    )
    closest_fine = int(np.argmin(np.maximum(coarse_distance, box_distance)))
    indices = {
        "fine": closest_fine,
        "coarse": int(coarse_index[closest_fine]),
        "doubled_box": int(box_index[closest_fine]),
    }
    triplet = {}
    for role in roles:
        separation, axis, eccentricity = selected[role][indices[role]]
        triplet[role] = {
            "mean_separation_pc": float(separation),
            "mean_semimajor_axis_osculating_pc": float(axis),
            "mean_eccentricity_osculating": float(eccentricity),
            "measured_over_kepler_mean_ratio": float(
                separation / (axis * (1.0 + 0.5 * eccentricity**2))
            ),
        }
    return {
        "status": "closest_joint_orbit_triplet_diagnostic_only",
        "complete_orbits_by_role": {
            role: int(selected[role].shape[0]) for role in roles
        },
        "minimum_max_log_axis_or_e_mismatch": float(
            max(coarse_distance[closest_fine], box_distance[closest_fine])
        ),
        "coarse_log_axis_or_e_mismatch": float(coarse_distance[closest_fine]),
        "doubled_box_log_axis_or_e_mismatch": float(box_distance[closest_fine]),
        "closest_triplet": triplet,
        "joint_a_e_support_verified": False,
        "runtime_mapping_admitted": False,
    }


def _joint_orbit_hull_diagnostic(observations: dict[str, np.ndarray]) -> dict:
    """Measure three-run 2D orbit overlap without certifying a runtime map.

    Convex hulls can bridge unsampled holes. Their intersection and the
    barycentric separation differences are diagnostics, not release gates.
    """

    roles = ("fine", "coarse", "doubled_box")
    coordinates = {}
    separations = {}
    hulls = {}
    for role in roles:
        orbit = observations[role]
        required = {
            "mean_separation_pc", "mean_semimajor_axis_osculating_pc",
            "mean_eccentricity_osculating",
        }
        if not required <= set(orbit.dtype.names or ()):
            raise ValueError("joint-hull diagnostic lacks orbit columns")
        separation = np.asarray(orbit["mean_separation_pc"], dtype=float)
        axis = np.asarray(orbit["mean_semimajor_axis_osculating_pc"], dtype=float)
        eccentricity = np.asarray(orbit["mean_eccentricity_osculating"], dtype=float)
        if (
            np.any(~np.isfinite(separation)) or np.any(separation <= 0.0)
            or np.any(~np.isfinite(axis)) or np.any(axis <= 0.0)
            or np.any(~np.isfinite(eccentricity))
            or np.any((eccentricity < 0.0) | (eccentricity >= 1.0))
        ):
            raise ValueError("joint-hull diagnostic has invalid orbit coordinates")
        coordinates[role] = np.column_stack((np.log(axis), eccentricity))
        separations[role] = separation
        unique, inverse = np.unique(
            coordinates[role], axis=0, return_inverse=True,
        )
        for group in range(len(unique)):
            repeated = separation[inverse == group]
            if repeated.size > 1 and not np.allclose(
                repeated, repeated[0], rtol=1.0e-12, atol=0.0,
            ):
                return {
                    "status": "non_single_valued_orbit_coordinates_censored",
                    "conflicting_role": role,
                    "conflicting_revisit_relative_spread": float(
                        (np.max(repeated) - np.min(repeated)) / np.min(repeated)
                    ),
                    "joint_a_e_support_verified": False,
                    "runtime_mapping_admitted": False,
                }
        if len(unique) < 3:
            return {
                "status": "insufficient_2d_coordinate_support_censored",
                "complete_orbits_by_role": {
                    item: int(observations[item].size) for item in roles
                },
                "joint_a_e_support_verified": False,
                "runtime_mapping_admitted": False,
            }
        try:
            hulls[role] = Delaunay(coordinates[role])
        except QhullError:
            return {
                "status": "degenerate_2d_coordinate_support_censored",
                "complete_orbits_by_role": {
                    item: int(observations[item].size) for item in roles
                },
                "joint_a_e_support_verified": False,
                "runtime_mapping_admitted": False,
            }

    fine_points = coordinates["fine"]
    simplex = {
        role: hulls[role].find_simplex(fine_points)
        for role in roles
    }
    common = np.flatnonzero(
        (simplex["coarse"] >= 0) & (simplex["doubled_box"] >= 0)
    )
    if common.size == 0:
        return {
            "status": "no_common_2d_hull_sample_censored",
            "fine_orbits_tested": int(fine_points.shape[0]),
            "joint_a_e_support_verified": False,
            "runtime_mapping_admitted": False,
        }

    def interpolate(role: str) -> tuple[np.ndarray, float]:
        triangulation = hulls[role]
        cells = simplex[role][common]
        transform = triangulation.transform[cells]
        delta = fine_points[common] - transform[:, 2]
        first = np.einsum("nij,nj->ni", transform[:, :2], delta)
        weights = np.column_stack((first, 1.0 - first.sum(axis=1)))
        vertices = triangulation.simplices[cells]
        predicted = np.sum(weights * separations[role][vertices], axis=1)
        triangle_points = coordinates[role][vertices]
        edges = (
            np.linalg.norm(triangle_points[:, 0] - triangle_points[:, 1], axis=1),
            np.linalg.norm(triangle_points[:, 1] - triangle_points[:, 2], axis=1),
            np.linalg.norm(triangle_points[:, 2] - triangle_points[:, 0], axis=1),
        )
        return predicted, float(max(np.max(edge) for edge in edges))

    fine_separation = separations["fine"][common]
    coarse_prediction, coarse_edge = interpolate("coarse")
    box_prediction, box_edge = interpolate("doubled_box")
    coarse_difference = coarse_prediction / fine_separation - 1.0
    box_difference = box_prediction / fine_separation - 1.0
    return {
        "status": "joint_2d_hull_overlap_diagnostic_only",
        "fine_orbits_tested": int(fine_points.shape[0]),
        "fine_orbits_inside_both_other_hulls": int(common.size),
        "sampled_log_axis_range": [
            float(np.min(fine_points[common, 0])),
            float(np.max(fine_points[common, 0])),
        ],
        "sampled_eccentricity_range": [
            float(np.min(fine_points[common, 1])),
            float(np.max(fine_points[common, 1])),
        ],
        "maximum_absolute_coarse_minus_fine_fraction": float(
            np.max(np.abs(coarse_difference))
        ),
        "maximum_absolute_doubled_box_minus_fine_fraction": float(
            np.max(np.abs(box_difference))
        ),
        "maximum_coarse_simplex_edge_log_axis_or_e": coarse_edge,
        "maximum_doubled_box_simplex_edge_log_axis_or_e": box_edge,
        "convex_hull_may_bridge_unsampled_holes": True,
        "joint_a_e_support_verified": False,
        "runtime_mapping_admitted": False,
    }


def _resolved_orbits_in_fixed_bin(
    run: dict, lower_pc: float, upper_pc: float, *, include_upper: bool
) -> np.ndarray:
    orbit = run["orbit_series"]
    if orbit is None:
        raise ValueError("joint-coordinate diagnostic requires orbit series")
    valid = _initial_resolved_orbit_indices(
        orbit,
        float(run["conservation"]["initial_spatially_resolved_duration_myr"]),
    )
    separation = orbit["mean_separation_pc"][valid]
    inside = separation >= lower_pc - 1.0e-14
    inside &= (
        separation <= upper_pc + 1.0e-14
        if include_upper else separation < upper_pc
    )
    return orbit[valid[inside]]


def _mapping_ratio_diagnostic(observations: dict) -> dict:
    """Describe, without accepting, differences at unmatched bin-mean states."""

    fine = observations["fine"]
    coarse = observations["coarse"]
    doubled = observations["doubled_box"]
    denominator = fine["measured_over_kepler_mean_ratio"]
    return {
        "status": "unmatched_state_diagnostic_only_no_mapping_admitted",
        "coarse_minus_fine_fraction_of_fine_ratio": (
            coarse["measured_over_kepler_mean_ratio"] / denominator - 1.0
        ),
        "doubled_box_minus_fine_fraction_of_fine_ratio": (
            doubled["measured_over_kepler_mean_ratio"] / denominator - 1.0
        ),
        "coarse_minus_fine_relative_mean_axis": (
            coarse["mean_semimajor_axis_osculating_pc"]
            / fine["mean_semimajor_axis_osculating_pc"] - 1.0
        ),
        "doubled_box_minus_fine_relative_mean_axis": (
            doubled["mean_semimajor_axis_osculating_pc"]
            / fine["mean_semimajor_axis_osculating_pc"] - 1.0
        ),
        "coarse_minus_fine_mean_eccentricity": (
            coarse["mean_eccentricity_osculating"]
            - fine["mean_eccentricity_osculating"]
        ),
        "doubled_box_minus_fine_mean_eccentricity": (
            doubled["mean_eccentricity_osculating"]
            - fine["mean_eccentricity_osculating"]
        ),
        "same_a_e_state_verified": False,
        "runtime_mapping_admitted": False,
    }


def _numerical_settings(
    run: Path, summary_row: dict, definition: dict,
) -> tuple[float, int, str | None, str | None, str | None, str | None, str | None, str | None, str | None, str | None]:
    metadata = json.loads((run / "fdm_adapter_metadata.json").read_text())
    config = json.loads((run / "config.uldm").read_text())
    step = float(metadata.get("time_step_factor", config["Temporal Step Factor"]))
    rk = int(metadata.get("nbody_rk4_substeps_per_wave_step", int(config["RK Steps"]) // 4))
    backend = metadata.get("backend")
    kinetic_phase_layout = metadata.get("kinetic_phase_layout")
    wave_buffer_lifetime = metadata.get("wave_buffer_lifetime")
    wave_density_layout = metadata.get("wave_density_layout")
    compact_potential_layout = metadata.get("compact_potential_layout")
    potential_phase_layout = metadata.get("potential_phase_layout")
    total_potential_lifetime = metadata.get("total_potential_lifetime")
    saved_energy_density_lifetime = metadata.get("saved_energy_density_lifetime")
    if (not np.isfinite(step) or step <= 0 or rk < 1
            or (backend is not None and backend not in ("pytorch_cpu", "pytorch_cuda"))
            or (kinetic_phase_layout is not None
                and kinetic_phase_layout != "separable_axis_v1")
            or (wave_buffer_lifetime is not None
                and wave_buffer_lifetime != "release_previous_state_before_fft_v1")
            or (wave_density_layout is not None
                and wave_density_layout != "real_imag_addcmul_v1")
            or (compact_potential_layout is not None
                and compact_potential_layout != "x_slab32_inplace_rsqrt_v1")
            or (potential_phase_layout is not None
                and potential_phase_layout != "complex_real_imag_inplace_trig_v1")
            or (total_potential_lifetime is not None
                and total_potential_lifetime != "recompute_before_first_kick_release_before_save_v1")
            or (saved_energy_density_lifetime is not None
                and saved_energy_density_lifetime != "release_before_kinetic_fft_rebuild_for_output_v1")
            or summary_row.get("time_step_factor") != step
            or summary_row.get("nbody_rk4_substeps_per_wave_step") != rk
            or summary_row.get("resolution") != definition["resolution"]
            or not np.isclose(summary_row.get("cell_size_pc"), definition["cell_size_pc"], rtol=1e-12, atol=0)):
        raise ValueError("box-control numerical settings disagree with run inputs")
    return (step, rk, backend, kinetic_phase_layout, wave_buffer_lifetime,
            wave_density_layout, compact_potential_layout, potential_phase_layout,
            total_potential_lifetime, saved_energy_density_lifetime)


def _initial_conditions(run: Path, definition: dict) -> tuple[str, str]:
    config = json.loads((run / "config.uldm").read_text())
    particles = config["Matter Particles"]
    solitons = config["ULDM Solitons"]
    if (not np.isclose(float(particles["Plummer Radius"]),
                       definition["plummer_radius_pc"], rtol=1e-12, atol=0)
            or len(particles["Condition"]) != 2
            or len(solitons["Condition"]) != 1):
        raise ValueError("box-control initial configuration is invalid")
    return (
        json.dumps(particles["Condition"], sort_keys=True),
        json.dumps(solitons["Condition"], sort_keys=True),
    )


def _check_rate_fractions(bin_row: dict, reference_label: str, comparison_label: str) -> None:
    reference = _run_bin(bin_row, reference_label)
    comparison = _run_bin(bin_row, comparison_label)
    for field in _RATES:
        ref = float(reference["rates"][field]["estimate"])
        value = float(comparison["rates"][field]["estimate"])
        recorded = comparison["fractional_rate_difference_from_reference"].get(field)
        if (not np.isfinite(ref) or not np.isfinite(value)
                or not np.isfinite(recorded)
                or abs(ref) <= np.finfo(float).tiny
                or not np.isclose(abs(recorded), abs(value - ref) / abs(ref),
                                  rtol=1e-10, atol=1e-12)):
            raise ValueError(f"box-control rate difference is inconsistent: {field}")


def _verify_registered_comparison_binding(summary: dict) -> dict | None:
    binding = summary.get("qe_design_binding")
    if binding is None:
        return None
    if not isinstance(binding, dict) or set(binding) != {
        "path", "physical_cases_path", "run_manifest_path", "design_sha256",
        "file_sha256", "comparison_kind", "status",
    } or binding["status"] != "registered_design_bound_not_a_calibration_release":
        raise ValueError("q/e registered comparison binding is invalid")
    design, file_sha256 = read_verified_qe_followup_design(
        Path(binding["path"]),
        physical_cases=Path(binding["physical_cases_path"]),
        run_manifest=Path(binding["run_manifest_path"]),
    )
    if (file_sha256 != binding["file_sha256"]
            or design["design_sha256"] != binding["design_sha256"]
            or summary["matched_separation"]["separation_bin_edges_pc"]
            != design["separation_bin_edges_pc"]
            or summary["matched_separation"]["minimum_complete_orbits_per_run_per_bin"]
            != design["minimum_orbits_per_bin"]):
        raise ValueError("q/e registered comparison differs from its fixed design")
    rows = summary["runs"]
    if rows[0]["label"] != summary["reference_label"]:
        raise ValueError("q/e registered reference label is out of order")
    kind = verify_qe_design_comparison_runs(
        design, file_sha256,
        tuple(Path(row["run"]).expanduser().resolve() for row in rows),
    )
    if kind != binding["comparison_kind"]:
        raise ValueError("q/e registered comparison kind differs from run roles")
    return binding


def verify_fixed_comparison_summary(path: Path) -> dict:
    """Recompute a fixed-bin comparison from bounded named raw diagnostics."""

    path = path.expanduser().resolve()
    if path.stat().st_size > _MAX_DIAGNOSTIC_BYTES:
        raise ValueError(f"box-control comparison exceeds size limit: {path}")
    initial_summary_sha256 = _sha256(path)
    saved = _read(path)
    _fixed_bins(saved)
    rows = saved.get("runs")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("box-control comparison requires exactly two runs")
    source_inputs = []
    loaded = []
    for row in rows:
        run = Path(row["run"]).expanduser().resolve()
        label = row["label"]
        hashes = {}
        for name in _RAW_INPUTS:
            input_path = run / name
            if (not input_path.is_file()
                    or input_path.stat().st_size > _MAX_DIAGNOSTIC_BYTES):
                raise ValueError(f"box-control raw diagnostic is absent or too large: {input_path}")
            hashes[name] = _sha256(input_path)
        for name, expected in (
            ("torch_run_summary.json", "complete"),
            ("wave_response_summary.json", "diagnosed"),
        ):
            run_status = json.loads((run / name).read_text(encoding="utf-8"))
            if run_status.get("status") != expected:
                raise ValueError(f"box-control raw run is incomplete: {run}/{name}")
            if ("run" in run_status
                    and Path(run_status["run"]).resolve() != run):
                raise ValueError(f"box-control raw run identity differs: {run}/{name}")
        loaded_run = load_convergence_run(label, run)
        source_inputs.append({
            "label": label,
            "run": str(run),
            "sha256": hashes,
            "orbit_artifact_provenance": loaded_run["orbit_artifact_provenance"],
        })
        loaded.append(loaded_run)
    matched = saved["matched_separation"]
    recomputed = summarize_convergence(
        loaded,
        separation_bins=matched["requested_bins"],
        minimum_orbits_per_separation_bin=(
            matched["minimum_complete_orbits_per_run_per_bin"]
        ),
        separation_bin_edges_pc=tuple(matched["separation_bin_edges_pc"]),
    )
    binding = _verify_registered_comparison_binding(saved)
    if binding is not None:
        recomputed["qe_design_binding"] = binding
    if recomputed != saved:
        raise ValueError(f"box-control comparison is not reproducible from raw diagnostics: {path}")
    for row, loaded_run in zip(source_inputs, loaded, strict=True):
        for name in _RAW_INPUTS:
            if _sha256(Path(loaded_run["run"]) / name) != row["sha256"][name]:
                raise ValueError(f"box-control raw diagnostic changed during audit: {loaded_run['run']}/{name}")
    if _sha256(path) != initial_summary_sha256:
        raise ValueError("box-control comparison changed during audit")
    return {"comparison_sha256": initial_summary_sha256, "raw_inputs": source_inputs}


def assess_qe_box_control(
    resolution_pair: CalibrationSource,
    doubled_box: CalibrationSource,
    *,
    verify_raw: bool = True,
) -> dict:
    """Return candidate/censored bin decisions without admitting table rows."""

    if resolution_pair.profile_id != doubled_box.profile_id:
        raise ValueError("box-control profile IDs differ")
    pair_path = resolution_pair.convergence_summary.expanduser().resolve()
    box_path = doubled_box.convergence_summary.expanduser().resolve()
    pair = _read(pair_path)
    box = _read(box_path)
    pair_binding = pair.get("qe_design_binding")
    box_binding = box.get("qe_design_binding")
    if (pair_binding is None) != (box_binding is None):
        raise ValueError("q/e controls mix registered and unregistered designs")
    if pair_binding is not None:
        if (not isinstance(pair_binding, dict) or not isinstance(box_binding, dict)
                or pair_binding.get("comparison_kind") != "resolution_pair"
                or box_binding.get("comparison_kind") != "doubled_box"
                or any(pair_binding.get(field) != box_binding.get(field)
                       for field in ("path", "physical_cases_path",
                                     "run_manifest_path", "design_sha256",
                                     "file_sha256"))):
            raise ValueError("q/e controls use different registered designs")
    raw_verification = None
    if verify_raw:
        raw_verification = {
            "resolution_pair": verify_fixed_comparison_summary(pair_path),
            "doubled_box": verify_fixed_comparison_summary(box_path),
        }
    pair_bins = _fixed_bins(pair)
    box_bins = _fixed_bins(box)
    pair_edges = pair["matched_separation"]["separation_bin_edges_pc"]
    box_edges = box["matched_separation"]["separation_bin_edges_pc"]
    if pair_edges != box_edges:
        raise ValueError("resolution and box controls use different physical bins")
    pair_ref, pair_other = _runs(pair)
    box_ref, box_other = _runs(box)
    fine_run = Path(pair_ref["run"]).resolve()
    if fine_run != Path(box_other["run"]).resolve():
        raise ValueError("box control does not reuse the resolution-pair fine run")
    fine = _physical_definition(fine_run)
    coarse = _physical_definition(Path(pair_other["run"]).resolve())
    larger_run = Path(box_ref["run"]).resolve()
    larger = _physical_definition(larger_run)
    if (not is_qe_extension_case(fine["case_id"])
            or not _same_physical_case(fine, coarse)
            or not _same_physical_case(fine, larger)
            or fine["resolution"] <= coarse["resolution"]):
        raise ValueError("box-control runs do not share one q/e physical case")
    if (larger["resolution"] != 2 * fine["resolution"]
            or not np.isclose(larger["cell_size_pc"], fine["cell_size_pc"], rtol=1e-12, atol=0)
            or not np.isclose(larger["plummer_radius_pc"], fine["plummer_radius_pc"], rtol=1e-12, atol=0)):
        raise ValueError("box control must double box size at fixed cell size and softening")
    if not (
        _initial_conditions(fine_run, fine)
        == _initial_conditions(Path(pair_other["run"]).resolve(), coarse)
        == _initial_conditions(larger_run, larger)
    ):
        raise ValueError("box-control initial SMBH and soliton conditions differ")
    settings = [
        _numerical_settings(fine_run, pair_ref, fine),
        _numerical_settings(Path(pair_other["run"]).resolve(), pair_other, coarse),
        _numerical_settings(larger_run, box_ref, larger),
        _numerical_settings(fine_run, box_other, fine),
    ]
    if any(setting != settings[0] for setting in settings[1:]):
        raise ValueError("box-control numerical settings differ")
    pair_result = build_source_rows(resolution_pair) if pair_bins else None
    box_result = build_source_rows(doubled_box) if box_bins else None
    candidate_indices = (
        {row.separation_bin_index for row in pair_result.accepted_rows}
        if pair_result is not None else set()
    )
    controlled_indices = (
        {row.separation_bin_index for row in box_result.accepted_rows}
        if box_result is not None else set()
    )
    decisions = []
    for index in sorted(candidate_indices):
        _check_rate_fractions(pair_bins[index], pair_ref["label"], pair_other["label"])
        if index in box_bins:
            _check_rate_fractions(box_bins[index], box_ref["label"], box_other["label"])
            pair_shared = _run_bin(pair_bins[index], pair_ref["label"])
            box_shared = _run_bin(box_bins[index], box_other["label"])
            for field in (
                "complete_orbits", "mean_separation_pc",
                *_ECCENTRICITY_COORDINATES, *_MAPPING_COORDINATES,
            ):
                if not np.isclose(pair_shared[field], box_shared[field], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"shared fine-run bin differs across comparisons: {index}/{field}")
            for field in _RATES:
                if not np.isclose(pair_shared["rates"][field]["estimate"],
                                  box_shared["rates"][field]["estimate"], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"shared fine-run rate differs across comparisons: {index}/{field}")
            if (
                pair_shared["coordinate_ratio_bootstrap"]
                != box_shared["coordinate_ratio_bootstrap"]
            ):
                raise ValueError(f"shared fine-run coordinate bootstrap differs: {index}")
        decisions.append({
            "separation_bin_index": index,
            "status": ("candidate_passes_box_control" if index in controlled_indices
                       else "censored_box_control_missing_or_rejected"),
        })
    return {
        "status": ("box_control_candidates_not_released" if any(
            row["status"] == "candidate_passes_box_control" for row in decisions
        ) else "no_box_controlled_candidate_bins_censored"),
        "profile_id": resolution_pair.profile_id,
        "case_id": fine["case_id"],
        "resolution_pair_sha256": _sha256(pair_path),
        "doubled_box_sha256": _sha256(box_path),
        "candidate_bins": sorted(candidate_indices),
        "box_controlled_candidate_bins": sorted(candidate_indices & controlled_indices),
        "bin_decisions": decisions,
        "raw_diagnostics_verified": verify_raw,
        "raw_verification": raw_verification,
        "qe_design_binding": pair_binding,
        "production_calibration_row_admitted": False,
    }


def prepare_qe_calibration_candidate(
    resolution_pair: CalibrationSource,
    doubled_box: CalibrationSource,
) -> dict:
    """Package only raw-verified, box-supported rows without releasing them."""

    assessment = assess_qe_box_control(
        resolution_pair, doubled_box, verify_raw=True
    )
    if assessment.get("raw_diagnostics_verified") is not True:
        raise ValueError("q/e candidate requires raw-diagnostic verification")
    if (assessment.get("production_calibration_row_admitted") is not False
            or assessment.get("profile_id") != resolution_pair.profile_id):
        raise ValueError("q/e candidate assessment identity is invalid")
    pair_path = resolution_pair.convergence_summary.expanduser().resolve()
    pair_bins = _fixed_bins(_read(pair_path))
    source = build_source_rows(resolution_pair) if pair_bins else None
    if _sha256(pair_path) != assessment["resolution_pair_sha256"]:
        raise ValueError("q/e resolution summary changed after box assessment")
    box_path = doubled_box.convergence_summary.expanduser().resolve()
    if _sha256(box_path) != assessment["doubled_box_sha256"]:
        raise ValueError("q/e box summary changed after box assessment")
    for role, summary_path in (("resolution_pair", pair_path),
                               ("doubled_box", box_path)):
        comparison = assessment["raw_verification"][role]
        if comparison["comparison_sha256"] != _sha256(summary_path):
            raise ValueError("q/e raw verification summary identity is invalid")
        expected_runs = {
            (row["label"], str(Path(row["run"]).resolve()))
            for row in _read(summary_path)["runs"]
        }
        actual_runs = {
            (row["label"], str(Path(row["run"]).resolve()))
            for row in comparison["raw_inputs"]
        }
        if actual_runs != expected_runs or len(comparison["raw_inputs"]) != 2:
            raise ValueError("q/e raw verification run identity is invalid")
        for run in comparison["raw_inputs"]:
            if run.get("orbit_artifact_provenance", {}).get("status") != (
                "verified_orbit_artifact_provenance_v1"
            ):
                raise ValueError(
                    "q/e calibration release requires versioned verified orbit "
                    "artifact provenance"
                )
            if set(run["sha256"]) != set(_RAW_INPUTS):
                raise ValueError("q/e raw verification input set is incomplete")
            for name, expected_sha256 in run["sha256"].items():
                path = Path(run["run"]) / name
                if (not path.is_file()
                        or path.stat().st_size > _MAX_DIAGNOSTIC_BYTES
                        or _sha256(path) != expected_sha256):
                    raise ValueError(f"q/e diagnostic changed after box assessment: {path}")
    controlled = set(assessment["box_controlled_candidate_bins"])
    accepted = (
        {row.separation_bin_index: row for row in source.accepted_rows}
        if source is not None else {}
    )
    if source is not None and source.source_case_id != assessment["case_id"]:
        raise ValueError("q/e candidate case identity is invalid")
    if not controlled <= accepted.keys():
        raise ValueError("q/e box-controlled bins disagree with resolution acceptance")
    rows = [asdict(accepted[index]) for index in sorted(controlled)]
    pair_summary = _read(pair_path)
    box_summary = _read(box_path)
    pair_ref, pair_coarse = _runs(pair_summary)
    box_ref, _ = _runs(box_summary)
    box_bins = _fixed_bins(box_summary)
    mapping_observations = [
        {
            "separation_bin_index": index,
            "fine": _mapping_observation(
                _run_bin(pair_bins[index], pair_ref["label"])
            ),
            "coarse": _mapping_observation(
                _run_bin(pair_bins[index], pair_coarse["label"])
            ),
            "doubled_box": _mapping_observation(
                _run_bin(box_bins[index], box_ref["label"])
            ),
        }
        for index in sorted(controlled)
    ]
    if mapping_observations:
        pair_edges = pair_summary["matched_separation"]["separation_bin_edges_pc"]
        loaded_mapping_runs = {
            "fine": load_convergence_run(
                pair_ref["label"], Path(pair_ref["run"]).resolve()
            ),
            "coarse": load_convergence_run(
                pair_coarse["label"], Path(pair_coarse["run"]).resolve()
            ),
            "doubled_box": load_convergence_run(
                box_ref["label"], Path(box_ref["run"]).resolve()
            ),
        }
        mapping_source_hashes = {
            str(Path(row["run"]).resolve()): row["sha256"]
            for comparison in assessment["raw_verification"].values()
            for row in comparison["raw_inputs"]
        }
    for observation in mapping_observations:
        index = observation["separation_bin_index"]
        orbit_samples = {
            role: _resolved_orbits_in_fixed_bin(
                run, pair_edges[index], pair_edges[index + 1],
                include_upper=index == len(pair_edges) - 2,
            )
            for role, run in loaded_mapping_runs.items()
        }
        observation["closest_joint_orbit_triplet_not_released"] = (
            _joint_orbit_coordinate_proximity(orbit_samples)
        )
        observation["joint_orbit_hull_diagnostic_not_released"] = (
            _joint_orbit_hull_diagnostic(orbit_samples)
        )
        observation["necessary_coordinate_overlap"] = (
            _necessary_mapping_coordinate_overlap(observation)
        )
        observation["unmatched_ratio_diagnostic"] = (
            _mapping_ratio_diagnostic(observation)
        )
    if mapping_observations:
        for run in loaded_mapping_runs.values():
            run_path = Path(run["run"]).resolve()
            expected = mapping_source_hashes[str(run_path)]
            for name in ("orbit_averaged_exchange.csv", "conservation_summary.json"):
                if _sha256(run_path / name) != expected[name]:
                    raise ValueError(
                        f"q/e joint-coordinate source changed during audit: {run_path}/{name}"
                    )
    if (_sha256(pair_path) != assessment["resolution_pair_sha256"]
            or _sha256(box_path) != assessment["doubled_box_sha256"]):
        raise ValueError("q/e comparison changed while packaging mapping observations")
    return {
        "schema_version": 1,
        "status": (
            "qe_box_controlled_candidate_not_released" if rows
            else "qe_no_box_controlled_bins_censored"
        ),
        "profile_id": resolution_pair.profile_id,
        "case_id": assessment["case_id"],
        "candidate_row_count": len(rows),
        "candidate_rows": rows,
        "mapping_observations_not_released": mapping_observations,
        "resolution_rejected_bins": (
            list(source.rejected_bins) if source is not None else []
        ),
        "box_assessment": assessment,
        "production_calibration_row_admitted": False,
    }


def build_box_controlled_qe_source(
    resolution_pair: CalibrationSource,
    doubled_box: CalibrationSource,
) -> tuple[SourceBuildResult, dict]:
    """Select only raw-verified, doubled-box-supported q/e bins for a future release.

    This does not publish a table or validate the separate (a,e)-to-mean-r
    mapping required by the bound-binary runtime.
    """

    package = prepare_qe_calibration_candidate(resolution_pair, doubled_box)
    if (
        package["status"] != "qe_box_controlled_candidate_not_released"
        or package["production_calibration_row_admitted"] is not False
        or package["candidate_row_count"] < 1
        or package["box_assessment"]["raw_diagnostics_verified"] is not True
    ):
        raise ValueError("q/e source has no raw-verified doubled-box-supported bin")
    source = build_source_rows(resolution_pair)
    if (
        source.profile_id != package["profile_id"]
        or source.source_case_id != package["case_id"]
        or source.source_sha256
        != package["box_assessment"]["resolution_pair_sha256"]
        or _sha256(doubled_box.convergence_summary.expanduser().resolve())
        != package["box_assessment"]["doubled_box_sha256"]
    ):
        raise ValueError("q/e source changed after doubled-box verification")
    candidates = package["candidate_rows"]
    indices = [row["separation_bin_index"] for row in candidates]
    if (
        len(indices) != len(set(indices))
        or len(indices) != package["candidate_row_count"]
    ):
        raise ValueError("q/e box-controlled candidate bins are duplicated")
    accepted = {row.separation_bin_index: row for row in source.accepted_rows}
    if len(accepted) != len(source.accepted_rows) or any(
        type(index) is not int
        or index not in accepted
        or asdict(accepted[index]) != row
        for index, row in zip(indices, candidates, strict=True)
    ):
        raise ValueError("q/e box-controlled rows disagree with resolution acceptance")
    selected = tuple(accepted[index] for index in sorted(indices))
    box_rejected = tuple(
        {
            "separation_bin_index": index,
            "reasons": ["doubled-box control is missing or rejected"],
        }
        for index in sorted(accepted.keys() - set(indices))
    )
    return replace(
        source,
        accepted_rows=selected,
        rejected_bins=source.rejected_bins + box_rejected,
    ), package
