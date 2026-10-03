"""Prospective fixed-bin q/e follow-up design, without launching a solver."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from .calibration import estimated_uniform_grid_memory_gib
from .subgrid_calibration import MINIMUM_ACCEPTED_COMPLETE_ORBITS, is_qe_extension_case


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_case(path: Path, case_id: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        matches = [row for row in csv.DictReader(stream) if row.get("case_id") == case_id]
    if len(matches) != 1:
        raise ValueError("q/e design requires exactly one physical case row")
    return matches[0]


def _run_rows(path: Path, case_id: str) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("case_id") == case_id]
    by_resolution = {}
    for row in rows:
        resolution = int(row["effective_grid_cells"])
        if resolution < 1 or resolution in by_resolution:
            raise ValueError("q/e design has duplicate or invalid run resolution")
        by_resolution[resolution] = row
    return by_resolution


def build_qe_followup_design(
    *,
    case_id: str,
    physical_cases: Path,
    run_manifest: Path,
    coarse_resolution: int,
    fine_resolution: int,
    separation_bin_edges_pc: tuple[float, ...],
    duration_myr: float,
    minimum_orbits_per_bin: int = MINIMUM_ACCEPTED_COMPLETE_ORBITS,
    reference_gpu_memory_gib: float = 80.0,
) -> dict:
    """Bind a proposed new run triplet to source rows and immutable bin edges.

    This is a pre-run design only.  The box control is *planned* at twice the
    fine run's box length and grid resolution, not scheduled or certified.
    """

    if not is_qe_extension_case(case_id):
        raise ValueError("follow-up design requires a q/e case")
    if (
        type(coarse_resolution) is not int or type(fine_resolution) is not int
        or coarse_resolution < 1 or fine_resolution <= coarse_resolution
        or type(minimum_orbits_per_bin) is not int
        or minimum_orbits_per_bin < MINIMUM_ACCEPTED_COMPLETE_ORBITS
        or not math.isfinite(duration_myr) or duration_myr <= 0.0
        or not math.isfinite(reference_gpu_memory_gib)
        or reference_gpu_memory_gib <= 0.0
    ):
        raise ValueError("q/e follow-up numerical controls are invalid")
    edges = tuple(separation_bin_edges_pc)
    if (
        len(edges) < 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0.0 for value in edges)
        or any(right <= left for left, right in zip(edges[:-1], edges[1:]))
    ):
        raise ValueError("q/e fixed physical separation edges are invalid")
    case = _unique_case(physical_cases, case_id)
    runs = _run_rows(run_manifest, case_id)
    if coarse_resolution not in runs or fine_resolution not in runs:
        raise ValueError("q/e resolution pair is absent from the run manifest")
    coarse = runs[coarse_resolution]
    fine = runs[fine_resolution]
    coarse_box = float(coarse["box_size_pc"])
    fine_box = float(fine["box_size_pc"])
    coarse_cell = float(coarse["finest_cell_size_pc"])
    fine_cell = float(fine["finest_cell_size_pc"])
    period = float(case["kepler_period_myr"])
    initial_separation = float(case["initial_separation_pc"])
    core_radius = float(case["core_radius_pc"])
    mass_ratio = float(case["mass_ratio_q"])
    eccentricity = float(case["eccentricity"])
    if (
        any(not math.isfinite(value) or value <= 0.0 for value in
            (coarse_box, fine_box, coarse_cell, fine_cell, period,
             initial_separation, core_radius))
        or not math.isfinite(mass_ratio) or not 0.0 < mass_ratio <= 1.0
        or not math.isfinite(eccentricity) or not 0.0 <= eccentricity < 1.0
        or not math.isclose(coarse_box, fine_box, rel_tol=1.0e-12)
        or not math.isclose(coarse_box / coarse_resolution, coarse_cell,
                            rel_tol=1.0e-12)
        or not math.isclose(fine_box / fine_resolution, fine_cell,
                            rel_tol=1.0e-12)
        or fine_cell >= coarse_cell
    ):
        raise ValueError("q/e resolution-pair geometry is inconsistent")
    minimum_orbits = (len(edges) - 1) * minimum_orbits_per_bin
    if duration_myr < period * minimum_orbits:
        raise ValueError("q/e design duration is shorter than the necessary orbit budget")
    box_resolution = 2 * fine_resolution
    box_memory = estimated_uniform_grid_memory_gib(box_resolution)
    payload = {
        "schema_version": 1,
        "status": "qe_prospective_fixed_bin_design_not_a_run_or_release",
        "case_id": case_id,
        "physical_cases_sha256": _sha256(physical_cases),
        "run_manifest_sha256": _sha256(run_manifest),
        "mass_ratio_q": mass_ratio,
        "initial_eccentricity": eccentricity,
        "initial_separation_pc": initial_separation,
        "core_radius_pc": core_radius,
        "separation_bin_edges_pc": list(edges),
        "minimum_orbits_per_bin": minimum_orbits_per_bin,
        "necessary_nominal_orbits": minimum_orbits,
        "initial_kepler_period_myr": period,
        "planned_duration_myr": duration_myr,
        "resolution_pair": {
            "coarse": {"run_id": coarse["run_id"], "resolution": coarse_resolution,
                       "box_size_pc": coarse_box, "cell_size_pc": coarse_cell},
            "fine": {"run_id": fine["run_id"], "resolution": fine_resolution,
                     "box_size_pc": fine_box, "cell_size_pc": fine_cell},
        },
        "doubled_box_control": {
            "resolution": box_resolution,
            "box_size_pc": 2.0 * fine_box,
            "cell_size_pc": fine_cell,
            "uniform_grid_memory_estimate_gib": box_memory,
            "exceeds_reference_gpu_memory": box_memory > reference_gpu_memory_gib,
        },
        "reference_gpu_memory_gib": reference_gpu_memory_gib,
        "production_calibration_row_admitted": False,
        "interpretation": (
            "A prospective numerical design only. Resolved orbit coverage, "
            "conservation, convergence, doubled-box agreement, and resource "
            "feasibility must be demonstrated by new runs."
        ),
    }
    payload["design_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def read_verified_qe_followup_design(
    path: Path, *, physical_cases: Path, run_manifest: Path,
) -> tuple[dict, str]:
    """Rebuild every derived field from the bound CSV inputs before use."""

    resolved = path.expanduser().resolve()
    if resolved.stat().st_size > 64 * 1024:
        raise ValueError("q/e design exceeds the bounded input size")
    before = _sha256(resolved)
    saved = json.loads(resolved.read_text(encoding="utf-8"))
    pair = saved["resolution_pair"]
    expected = build_qe_followup_design(
        case_id=saved["case_id"],
        physical_cases=physical_cases,
        run_manifest=run_manifest,
        coarse_resolution=pair["coarse"]["resolution"],
        fine_resolution=pair["fine"]["resolution"],
        separation_bin_edges_pc=tuple(saved["separation_bin_edges_pc"]),
        duration_myr=saved["planned_duration_myr"],
        minimum_orbits_per_bin=saved["minimum_orbits_per_bin"],
        reference_gpu_memory_gib=saved["reference_gpu_memory_gib"],
    )
    if saved != expected or before != _sha256(resolved):
        raise ValueError("q/e design differs from its bound physical inputs")
    return saved, before


def verify_qe_design_run_request(
    design: dict,
    *,
    case_id: str,
    role: str,
    resolution: int,
    box_size_pc: float,
    duration_myr: float,
) -> None:
    """Prevent a labelled run from silently changing the registered design."""

    roles = {
        "coarse": design["resolution_pair"]["coarse"],
        "fine": design["resolution_pair"]["fine"],
        "doubled_box_control": design["doubled_box_control"],
    }
    if role not in roles:
        raise ValueError("q/e design run role is invalid")
    if (
        role == "doubled_box_control"
        and design["doubled_box_control"]["exceeds_reference_gpu_memory"]
    ):
        raise ValueError(
            "q/e doubled-box estimate exceeds the declared GPU memory; "
            "a different reviewed resource design is required"
        )
    expected = roles[role]
    if (
        case_id != design["case_id"]
        or type(resolution) is not int
        or resolution != expected["resolution"]
        or not math.isfinite(box_size_pc)
        or not math.isclose(box_size_pc, expected["box_size_pc"],
                            rel_tol=1.0e-12, abs_tol=0.0)
        or not math.isfinite(duration_myr)
        or duration_myr < design["planned_duration_myr"]
    ):
        raise ValueError("q/e run request disagrees with the prospective design")


def verify_qe_design_comparison_runs(
    design: dict,
    design_file_sha256: str,
    runs: tuple[Path, Path],
) -> str:
    """Require a resolution pair or doubled-box pair from the same design."""

    roles = []
    for run in runs:
        metadata = json.loads((run / "fdm_adapter_metadata.json").read_text(
            encoding="utf-8"
        ))
        binding = metadata.get("qe_design_binding")
        if (
            not isinstance(binding, dict)
            or binding.get("status") != "qe_prospective_design_bound_not_a_calibration_release"
            or binding.get("design_sha256") != design["design_sha256"]
            or binding.get("file_sha256") != design_file_sha256
        ):
            raise ValueError("q/e comparison run lacks the registered design binding")
        role = binding.get("role")
        verify_qe_design_run_request(
            design, case_id=metadata["case_id"], role=role,
            resolution=metadata["resolution"],
            box_size_pc=metadata["box_size_pc"],
            duration_myr=metadata["duration_myr"],
        )
        roles.append(role)
    if len(set(runs)) != 2 or roles not in (
        ["fine", "coarse"], ["doubled_box_control", "fine"]
    ):
        raise ValueError("q/e comparison must use the registered reference and comparison roles")
    return "resolution_pair" if roles == ["fine", "coarse"] else "doubled_box"
