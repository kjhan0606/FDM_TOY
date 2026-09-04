"""Build non-submitting nested-zoom registration records from outer evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .pure_fdm_outer_evaluation import (
    PureFDMOuterEnsembleEvaluation,
    PureFDMOuterPhysicsAssessment,
)
from .pure_fdm_zoom import NestedZoomCheckpointContract


PURE_FDM_NESTED_REGISTRATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PureFDMNestedZoomRegistration:
    """One checkpoint-bound nested zoom eligible for operator registration."""

    physics_id: str
    replicate: int
    outer_case_id: str
    outer_result_path: str
    outer_result_sha256: str
    checkpoint: NestedZoomCheckpointContract

    def as_dict(self) -> dict[str, Any]:
        return {
            "physics_id": self.physics_id,
            "replicate": self.replicate,
            "outer_case_id": self.outer_case_id,
            "outer_result": {
                "path": self.outer_result_path,
                "sha256": self.outer_result_sha256,
            },
            "checkpoint": self.checkpoint.as_dict(),
            "execution": "operator registration only; no job submission",
        }


@dataclass(frozen=True)
class PureFDMNestedPhysicsRegistration:
    """Registration decision for all phase replicates of one physics point."""

    physics_id: str
    status: str
    registrations: tuple[PureFDMNestedZoomRegistration, ...]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "physics_id": self.physics_id,
            "status": self.status,
            "registrations": [item.as_dict() for item in self.registrations],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class PureFDMNestedRegistrationManifest:
    """A scientific gate, not an execution manifest or a delay result."""

    outer_manifest_sha256: str
    result_index_path: str
    result_index_sha256: str
    physics_registrations: tuple[PureFDMNestedPhysicsRegistration, ...]

    @property
    def ready_registration_count(self) -> int:
        return sum(
            len(item.registrations) for item in self.physics_registrations
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PURE_FDM_NESTED_REGISTRATION_SCHEMA_VERSION,
            "status": (
                "pure_fdm_nested_zoom_registration_ready"
                if self.ready_registration_count
                else "no_pure_fdm_nested_zoom_registration_ready"
            ),
            "interpretation": (
                "checkpoint-bound nested-zoom registrations only; this record "
                "does not submit a job or accept a physical coalescence delay"
            ),
            "outer_manifest_sha256": self.outer_manifest_sha256,
            "result_index": {
                "path": self.result_index_path,
                "sha256": self.result_index_sha256,
            },
            "ready_registration_count": self.ready_registration_count,
            "physics_registrations": [
                item.as_dict() for item in self.physics_registrations
            ],
        }


def _checkpoint_matches_outer_physics(
    assessment: PureFDMOuterPhysicsAssessment,
    index: int,
) -> tuple[PureFDMNestedZoomRegistration | None, str | None]:
    resolution = assessment.resolution_assessments[index]
    result = resolution.fine_result
    if result is None or result.nested_checkpoint is None:
        return None, "fine result has no complete checkpoint contract"
    checkpoint = result.nested_checkpoint
    outer_physics = result.case.physics
    values = (
        (checkpoint.fdm_particle_mass_ev, outer_physics.fdm_particle_mass_ev, "fdm particle mass"),
        (checkpoint.soliton_mass_msun, outer_physics.fdm_soliton_mass_msun, "soliton mass"),
        (checkpoint.core_radius_pc, outer_physics.fdm_core_radius_pc, "core radius"),
    )
    for checkpoint_value, physics_value, label in values:
        assert physics_value is not None
        if not math.isclose(
            checkpoint_value,
            physics_value,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            return None, f"checkpoint {label} does not match outer physics"
    if checkpoint.outer_case_id != resolution.fine_case_id:
        return None, "checkpoint outer case does not match finest accepted case"
    if checkpoint.outer_manifest_sha256 != result.manifest_sha256:
        return None, "checkpoint manifest does not match fine result"
    if not checkpoint.pure_fdm_dark_sector:
        return None, "checkpoint contains a non-FDM dark sector"
    return (
        PureFDMNestedZoomRegistration(
            physics_id=assessment.physics_id,
            replicate=resolution.replicate,
            outer_case_id=resolution.fine_case_id,
            outer_result_path=str(result.source_path),
            outer_result_sha256=result.source_sha256,
            checkpoint=checkpoint,
        ),
        None,
    )


def _register_physics(
    assessment: PureFDMOuterPhysicsAssessment,
) -> PureFDMNestedPhysicsRegistration:
    if assessment.status != "accepted_outer_ensemble_uncalibrated":
        reasons = assessment.reasons or (
            "outer phase ensemble did not pass the uncalibrated acceptance gate",
        )
        return PureFDMNestedPhysicsRegistration(
            assessment.physics_id, "censored", (), reasons
        )
    registrations: list[PureFDMNestedZoomRegistration] = []
    reasons: list[str] = []
    for index, resolution in enumerate(assessment.resolution_assessments):
        if resolution.status != "accepted_outer_uncalibrated":
            reasons.append(f"replicate {resolution.replicate} is not resolution accepted")
            continue
        registration, reason = _checkpoint_matches_outer_physics(assessment, index)
        if reason is not None:
            reasons.append(f"replicate {resolution.replicate}: {reason}")
        else:
            assert registration is not None
            registrations.append(registration)
    if reasons or len(registrations) != len(assessment.resolution_assessments):
        return PureFDMNestedPhysicsRegistration(
            assessment.physics_id, "censored", (), tuple(reasons)
        )
    baryon_statuses = {
        (item.checkpoint.stellar_status, item.checkpoint.gas_status)
        for item in registrations
    }
    if len(baryon_statuses) != 1:
        return PureFDMNestedPhysicsRegistration(
            assessment.physics_id,
            "censored",
            (),
            ("phase replicates disagree about stellar/gas channel availability",),
        )
    return PureFDMNestedPhysicsRegistration(
        assessment.physics_id,
        "ready_for_operator_registration",
        tuple(registrations),
        (),
    )


def build_pure_fdm_nested_registration_manifest(
    evaluation: PureFDMOuterEnsembleEvaluation,
) -> PureFDMNestedRegistrationManifest:
    """Materialize only fully accepted outer checkpoints for nested registration."""

    return PureFDMNestedRegistrationManifest(
        outer_manifest_sha256=evaluation.grid.manifest_sha256,
        result_index_path=str(evaluation.result_index.source_path),
        result_index_sha256=evaluation.result_index.source_sha256,
        physics_registrations=tuple(
            _register_physics(item) for item in evaluation.physics_assessments
        ),
    )
