"""Compare q/e resolution candidates with same-cell-size doubled-box controls.

Passing this comparison is necessary, not sufficient, for a production table:
the registered physical-bin design and release provenance remain independent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .convergence import load_convergence_run, summarize_convergence
from .subgrid_table_builder import (
    CalibrationSource,
    _physical_definition,
    _sha256,
    _same_physical_case,
    build_source_rows,
)
from .subgrid_calibration import is_qe_extension_case


_RATES = ("orbital_power", "orbital_torque", "wave_total_energy_rate")
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
        by_index[index] = row
    return by_index


def _run_bin(bin_row: dict, label: str) -> dict:
    rows = [row for row in bin_row["runs"] if row.get("label") == label]
    if len(rows) != 1:
        raise ValueError("box-control bin lacks its shared run")
    return rows[0]


def _numerical_settings(run: Path, summary_row: dict, definition: dict) -> tuple[float, int]:
    metadata = json.loads((run / "fdm_adapter_metadata.json").read_text())
    config = json.loads((run / "config.uldm").read_text())
    step = float(metadata.get("time_step_factor", config["Temporal Step Factor"]))
    rk = int(metadata.get("nbody_rk4_substeps_per_wave_step", int(config["RK Steps"]) // 4))
    if (not np.isfinite(step) or step <= 0 or rk < 1
            or summary_row.get("time_step_factor") != step
            or summary_row.get("nbody_rk4_substeps_per_wave_step") != rk
            or summary_row.get("resolution") != definition["resolution"]
            or not np.isclose(summary_row.get("cell_size_pc"), definition["cell_size_pc"], rtol=1e-12, atol=0)):
        raise ValueError("box-control numerical settings disagree with run inputs")
    return step, rk


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
        source_inputs.append({"label": label, "run": str(run), "sha256": hashes})
        loaded.append(load_convergence_run(label, run))
    matched = saved["matched_separation"]
    recomputed = summarize_convergence(
        loaded,
        separation_bins=matched["requested_bins"],
        minimum_orbits_per_separation_bin=(
            matched["minimum_complete_orbits_per_run_per_bin"]
        ),
        separation_bin_edges_pc=tuple(matched["separation_bin_edges_pc"]),
    )
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
        _numerical_settings(larger_run, box_ref, larger),
        _numerical_settings(fine_run, box_other, fine),
    ]
    if settings[0] != settings[1] or settings[0] != settings[2]:
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
            for field in ("complete_orbits", "mean_separation_pc", "mean_eccentricity_osculating"):
                if not np.isclose(pair_shared[field], box_shared[field], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"shared fine-run bin differs across comparisons: {index}/{field}")
            for field in _RATES:
                if not np.isclose(pair_shared["rates"][field]["estimate"],
                                  box_shared["rates"][field]["estimate"], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"shared fine-run rate differs across comparisons: {index}/{field}")
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
        "production_calibration_row_admitted": False,
    }
