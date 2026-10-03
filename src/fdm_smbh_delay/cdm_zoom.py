"""Provenance gate for a non-compacting, resolved CDM SMBH zoom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dm_run_provenance import DarkMatterRunProvenance


@dataclass(frozen=True)
class NonCompactingCDMZoomAssessment:
    """Configuration provenance only; never a delay or coalescence result."""

    provenance: DarkMatterRunProvenance
    status: str
    reasons: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.status == "cdm_noncompacting_zoom_provenance_ready"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "interpretation": (
                "configuration provenance for a non-compacting CDM zoom only; "
                "this does not establish a resolved binary track, a calibrated "
                "environmental delay, or physical SMBH coalescence"
            ),
            "run_provenance": self.provenance.as_dict(),
            "reasons": list(self.reasons),
        }


def assess_noncompacting_cdm_zoom_run(
    provenance: DarkMatterRunProvenance,
) -> NonCompactingCDMZoomAssessment:
    """Require the native zero-radius setting with no finite-radii compaction."""

    reasons: list[str] = []
    if provenance.dark_matter_model != "cdm":
        reasons.append("non-compacting zoom provenance requires dark_matter_model=cdm")
    if not provenance.smbh_capture_ledger_enabled:
        reasons.append("non-compacting zoom requires an enabled SMBH capture ledger")
    if not provenance.build_git_hash or provenance.build_git_hash == "unknown":
        reasons.append("non-compacting zoom requires a concrete build_git_hash")
    if provenance.parameter("smbh_merge_radius_cells") != 0.0:
        reasons.append("non-compacting zoom requires smbh_merge_radius_cells=0")
    if provenance.parameter("smbh_compaction_mode") != "no_finite_radius_rmerge_zero":
        reasons.append("non-compacting zoom requires no_finite_radius_rmerge_zero mode")
    return NonCompactingCDMZoomAssessment(
        provenance=provenance,
        status=(
            "cdm_noncompacting_zoom_provenance_ready"
            if not reasons
            else "cdm_noncompacting_zoom_provenance_not_ready"
        ),
        reasons=tuple(reasons),
    )
