"""Provenance-preserving evaluation of a pure-FDM outer zoom ensemble.

This module consumes only completed outer-result JSON records.  It does not
submit work, interpolate an incomplete realization, or convert an outer-stage
timing diagnostic into a kpc-to-pc coalescence delay.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .pure_fdm_outer_results import (
    PureFDMOuterConvergenceResult,
    PureFDMOuterPhaseEnsemble,
    PureFDMOuterRunResult,
    assess_pure_fdm_outer_phase_ensemble,
    compare_pure_fdm_outer_resolution_pair,
    read_pure_fdm_outer_result,
)
from .zoom_calibration import GalaxyMergerZoomCase, ZoomGrid


PURE_FDM_OUTER_RESULT_INDEX_SCHEMA_VERSION = 1
PURE_FDM_OUTER_ENSEMBLE_EVALUATION_SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PureFDMOuterResultIndex:
    """Paths keyed by the exact cases of one outer zoom manifest."""

    source_path: Path
    source_sha256: str
    manifest_sha256: str
    result_paths: tuple[tuple[str, Path], ...]

    def path_for(self, case_id: str) -> Path | None:
        return dict(self.result_paths).get(case_id)


@dataclass(frozen=True)
class PureFDMOuterResolutionAssessment:
    """One phase realization's finest adjacent resolution comparison."""

    physics_id: str
    replicate: int
    fine_case_id: str
    coarse_case_id: str
    status: str
    fine_result: PureFDMOuterRunResult | None
    coarse_result: PureFDMOuterRunResult | None
    convergence: PureFDMOuterConvergenceResult | None
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "physics_id": self.physics_id,
            "replicate": self.replicate,
            "fine_case_id": self.fine_case_id,
            "coarse_case_id": self.coarse_case_id,
            "status": self.status,
            "fine_result": _run_source_dict(self.fine_result),
            "coarse_result": _run_source_dict(self.coarse_result),
            "common_core_delay_systematic_fraction": (
                None
                if self.convergence is None
                else self.convergence.common_core_delay_systematic_fraction
            ),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class PureFDMOuterPhysicsAssessment:
    """All phase realizations for one immutable physical point."""

    physics_id: str
    resolution_assessments: tuple[PureFDMOuterResolutionAssessment, ...]
    ensemble: PureFDMOuterPhaseEnsemble | None
    reasons: tuple[str, ...]

    @property
    def status(self) -> str:
        if self.ensemble is not None:
            return self.ensemble.status
        return "censored"

    @property
    def eligible_nested_outer_case_ids(self) -> tuple[str, ...]:
        if self.ensemble is None or not self.ensemble.ready_for_overlap_evaluation:
            return ()
        return tuple(
            assessment.fine_case_id for assessment in self.resolution_assessments
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "physics_id": self.physics_id,
            "status": self.status,
            "resolution_assessments": [
                assessment.as_dict() for assessment in self.resolution_assessments
            ],
            "common_core_delay_mean_myr": (
                None if self.ensemble is None else self.ensemble.common_core_delay_mean_myr
            ),
            "common_core_delay_standard_deviation_myr": (
                None
                if self.ensemble is None
                else self.ensemble.common_core_delay_standard_deviation_myr
            ),
            "eligible_nested_outer_case_ids": list(self.eligible_nested_outer_case_ids),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class PureFDMOuterEnsembleEvaluation:
    """Evaluation record that is intentionally not a delay calibration."""

    grid: ZoomGrid
    result_index: PureFDMOuterResultIndex
    physics_assessments: tuple[PureFDMOuterPhysicsAssessment, ...]

    def as_dict(self) -> dict[str, Any]:
        accepted = [
            item
            for item in self.physics_assessments
            if item.status == "accepted_outer_ensemble_uncalibrated"
        ]
        return {
            "schema_version": PURE_FDM_OUTER_ENSEMBLE_EVALUATION_SCHEMA_VERSION,
            "status": "pure_fdm_outer_ensemble_evaluated",
            "interpretation": (
                "outer convergence and phase diagnostics only; no physical "
                "kpc-to-pc delay is accepted by this record"
            ),
            "manifest_sha256": self.grid.manifest_sha256,
            "result_index": {
                "path": str(self.result_index.source_path),
                "sha256": self.result_index.source_sha256,
            },
            "physics_assessment_count": len(self.physics_assessments),
            "accepted_outer_ensemble_count": len(accepted),
            "eligible_nested_outer_case_ids": [
                case_id
                for item in accepted
                for case_id in item.eligible_nested_outer_case_ids
            ],
            "physics_assessments": [
                item.as_dict() for item in self.physics_assessments
            ],
        }


def _run_source_dict(result: PureFDMOuterRunResult | None) -> dict[str, str] | None:
    if result is None:
        return None
    return {"path": str(result.source_path), "sha256": result.source_sha256}


def load_pure_fdm_outer_result_index(
    path: str | Path,
    *,
    grid: ZoomGrid,
) -> PureFDMOuterResultIndex:
    """Load an index without reading any result that has not been registered.

    Relative result paths are deliberately interpreted relative to the index,
    making an evaluator record portable between a manual Lageunha workspace and
    an archival result directory.
    """

    resolved = Path(path).expanduser().resolve()
    try:
        record = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read pure-FDM outer result index: {error}") from error
    if not isinstance(record, Mapping):
        raise ValueError("pure-FDM outer result index must be a JSON object")
    if record.get("schema_version") != PURE_FDM_OUTER_RESULT_INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported pure-FDM outer result-index schema")
    if record.get("manifest_sha256") != grid.manifest_sha256:
        raise ValueError("pure-FDM outer result index manifest does not match the grid")
    raw_paths = record.get("results")
    if not isinstance(raw_paths, Mapping):
        raise ValueError("pure-FDM outer result index requires a results mapping")
    expected_case_ids = {case.case_id for case in grid.cases}
    unknown = set(raw_paths) - expected_case_ids
    if unknown:
        raise ValueError(f"pure-FDM outer result index has unknown cases: {sorted(unknown)}")
    paths: list[tuple[str, Path]] = []
    for case_id, raw_path in raw_paths.items():
        if not isinstance(case_id, str) or not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("pure-FDM outer result index paths must be non-empty strings")
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = resolved.parent / candidate
        paths.append((case_id, candidate.resolve()))
    return PureFDMOuterResultIndex(
        source_path=resolved,
        source_sha256=_file_sha256(resolved),
        manifest_sha256=grid.manifest_sha256,
        result_paths=tuple(sorted(paths)),
    )


def _read_registered_result(
    case: GalaxyMergerZoomCase,
    result_index: PureFDMOuterResultIndex,
    manifest_sha256: str,
) -> tuple[PureFDMOuterRunResult | None, str | None]:
    path = result_index.path_for(case.case_id)
    if path is None:
        return None, "result is not registered"
    try:
        return (
            read_pure_fdm_outer_result(
                path, case=case, manifest_sha256=manifest_sha256
            ),
            None,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return None, f"result is unreadable or invalid: {error}"


def _evaluate_resolution_pair(
    fine_case: GalaxyMergerZoomCase,
    coarse_case: GalaxyMergerZoomCase,
    result_index: PureFDMOuterResultIndex,
    manifest_sha256: str,
) -> PureFDMOuterResolutionAssessment:
    fine_result, fine_error = _read_registered_result(
        fine_case, result_index, manifest_sha256
    )
    coarse_result, coarse_error = _read_registered_result(
        coarse_case, result_index, manifest_sha256
    )
    reasons = tuple(
        message
        for message in (
            None if fine_error is None else f"fine {fine_error}",
            None if coarse_error is None else f"coarse {coarse_error}",
        )
        if message is not None
    )
    if reasons:
        return PureFDMOuterResolutionAssessment(
            physics_id=fine_case.physics.physics_id,
            replicate=fine_case.replicate,
            fine_case_id=fine_case.case_id,
            coarse_case_id=coarse_case.case_id,
            status="censored",
            fine_result=fine_result,
            coarse_result=coarse_result,
            convergence=None,
            reasons=reasons,
        )
    assert fine_result is not None and coarse_result is not None
    convergence = compare_pure_fdm_outer_resolution_pair(fine_result, coarse_result)
    return PureFDMOuterResolutionAssessment(
        physics_id=fine_case.physics.physics_id,
        replicate=fine_case.replicate,
        fine_case_id=fine_case.case_id,
        coarse_case_id=coarse_case.case_id,
        status=convergence.status,
        fine_result=fine_result,
        coarse_result=coarse_result,
        convergence=convergence,
        reasons=convergence.reasons,
    )


def evaluate_pure_fdm_outer_result_index(
    grid: ZoomGrid,
    result_index: PureFDMOuterResultIndex,
) -> PureFDMOuterEnsembleEvaluation:
    """Evaluate the finest adjacent resolution pair for every phase.

    A missing or invalid result is a censored realization.  It is retained in
    the output rather than aborting a multi-month campaign or being imputed as
    a zero-delay merger.  For a refinement ladder with more than two levels,
    the finest two levels define the pair eligible for a nested handoff.
    """

    if result_index.manifest_sha256 != grid.manifest_sha256:
        raise ValueError("outer result index belongs to a different manifest")
    grouped: dict[tuple[str, int], list[GalaxyMergerZoomCase]] = {}
    for case in grid.cases:
        grouped.setdefault((case.physics.physics_id, case.replicate), []).append(case)

    by_physics: dict[str, list[PureFDMOuterResolutionAssessment]] = {}
    for (physics_id, _), cases in sorted(grouped.items()):
        ordered = sorted(cases, key=lambda item: item.numerics.finest_cell_size_pc)
        if len(ordered) < 2:
            raise ValueError("outer result grid lacks a resolution pair")
        assessment = _evaluate_resolution_pair(
            ordered[0], ordered[1], result_index, grid.manifest_sha256
        )
        by_physics.setdefault(physics_id, []).append(assessment)

    physics_assessments: list[PureFDMOuterPhysicsAssessment] = []
    for physics_id, assessments in sorted(by_physics.items()):
        ordered = tuple(sorted(assessments, key=lambda item: item.replicate))
        missing = [item for item in ordered if item.convergence is None]
        if missing:
            reasons = tuple(
                f"replicate {item.replicate}: {reason}"
                for item in missing
                for reason in item.reasons
            )
            ensemble = None
        else:
            ensemble = assess_pure_fdm_outer_phase_ensemble(
                tuple(item.convergence for item in ordered if item.convergence is not None)
            )
            reasons = ensemble.reasons
        physics_assessments.append(
            PureFDMOuterPhysicsAssessment(
                physics_id=physics_id,
                resolution_assessments=ordered,
                ensemble=ensemble,
                reasons=reasons,
            )
        )
    return PureFDMOuterEnsembleEvaluation(
        grid=grid,
        result_index=result_index,
        physics_assessments=tuple(physics_assessments),
    )
