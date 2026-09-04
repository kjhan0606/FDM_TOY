"""Preflight contracts for staged pure-FDM outer and nested zooms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any

from .zoom_calibration import GalaxyMergerZoomCase, ZoomGrid


_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_BARYON_STATUSES = {"available", "absent"}


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class DeferredNestedZoomRequest:
    """An inner zoom that cannot start before an exact outer checkpoint exists."""

    outer_case_id: str
    outer_physics_id: str
    outer_replicate: int
    outer_levelmax: int
    required_maximum_cell_size_pc: float = 0.25
    required_minimum_transition_radius_cells: float = 4.0

    def __post_init__(self) -> None:
        if not self.outer_case_id or not self.outer_physics_id:
            raise ValueError("nested zoom request requires outer case identity")
        if self.outer_replicate < 0 or self.outer_levelmax < 1:
            raise ValueError("nested zoom request has invalid outer numerics")
        if self.required_maximum_cell_size_pc <= 0.0:
            raise ValueError("nested zoom cell-size limit must be positive")
        if self.required_minimum_transition_radius_cells < 4.0:
            raise ValueError("nested zoom must retain the four-cell transition gate")


@dataclass(frozen=True)
class NestedZoomCheckpointContract:
    """Evidence required to register one nested inner pure-FDM zoom."""

    outer_manifest_sha256: str
    outer_case_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    capture_event_uid: str
    force_ledger_sha256: str
    wave_ledger_sha256: str
    fdm_particle_mass_ev: float
    soliton_mass_msun: float
    core_radius_pc: float
    eta_sp: float
    checkpoint_separation_pc: float
    target_separation_pc: float
    finest_cell_size_pc: float
    minimum_softening_pc: float
    maximum_wake_extent_pc: float
    de_broglie_wavelength_pc: float
    hjm_wave_seam_clearance_pc: float
    boundary_clearance_pc: float
    stellar_status: str = "absent"
    gas_status: str = "absent"
    pure_fdm_dark_sector: bool = True

    def __post_init__(self) -> None:
        for name in (
            "outer_manifest_sha256",
            "checkpoint_sha256",
            "force_ledger_sha256",
            "wave_ledger_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in ("outer_case_id", "checkpoint_path", "capture_event_uid"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        values = {
            "fdm_particle_mass_ev": self.fdm_particle_mass_ev,
            "soliton_mass_msun": self.soliton_mass_msun,
            "core_radius_pc": self.core_radius_pc,
            "eta_sp": self.eta_sp,
            "checkpoint_separation_pc": self.checkpoint_separation_pc,
            "target_separation_pc": self.target_separation_pc,
            "finest_cell_size_pc": self.finest_cell_size_pc,
            "minimum_softening_pc": self.minimum_softening_pc,
            "maximum_wake_extent_pc": self.maximum_wake_extent_pc,
            "de_broglie_wavelength_pc": self.de_broglie_wavelength_pc,
            "hjm_wave_seam_clearance_pc": self.hjm_wave_seam_clearance_pc,
            "boundary_clearance_pc": self.boundary_clearance_pc,
        }
        for name, value in values.items():
            object.__setattr__(self, name, _positive(value, name))
        if self.target_separation_pc >= self.checkpoint_separation_pc:
            raise ValueError("nested target separation must lie below the checkpoint separation")
        if self.finest_cell_size_pc > 0.25:
            raise ValueError("nested pure-FDM zoom requires finest_cell_size_pc <= 0.25")
        if self.target_separation_pc / self.finest_cell_size_pc < 4.0:
            raise ValueError("nested target violates the four-cell transition gate")
        if self.minimum_softening_pc > self.finest_cell_size_pc:
            raise ValueError("nested softening cannot exceed the finest cell size")
        protected_extent = max(
            self.maximum_wake_extent_pc, self.de_broglie_wavelength_pc
        )
        if self.hjm_wave_seam_clearance_pc <= protected_extent:
            raise ValueError("HJM/wave seam lies inside the protected wake/coherence extent")
        if self.boundary_clearance_pc <= self.maximum_wake_extent_pc:
            raise ValueError("nested boundary lies inside the resolved wake extent")
        if self.stellar_status not in _BARYON_STATUSES:
            raise ValueError("stellar_status must be available or absent")
        if self.gas_status not in _BARYON_STATUSES:
            raise ValueError("gas_status must be available or absent")
        if self.pure_fdm_dark_sector is not True:
            raise ValueError("nested pure-FDM zoom cannot contain a CDM dark sector")

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}


def bind_nested_zoom_checkpoint(
    request: DeferredNestedZoomRequest,
    contract: NestedZoomCheckpointContract,
) -> NestedZoomCheckpointContract:
    """Bind a deferred request only to its exact outer case and resolution gate."""

    if request.outer_case_id != contract.outer_case_id:
        raise ValueError("nested checkpoint contract belongs to a different outer case")
    if contract.finest_cell_size_pc > request.required_maximum_cell_size_pc:
        raise ValueError("nested checkpoint violates the deferred cell-size gate")
    return contract


@dataclass(frozen=True)
class PureFDMOuterZoomPreflight:
    """Validated outer manifest plus deferred, checkpoint-bound inner requests."""

    manifest_sha256: str
    outer_case_count: int
    baryon_mode: str
    nested_requests: tuple[DeferredNestedZoomRequest, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "pure_fdm_outer_zoom_preflight",
            "manifest_sha256": self.manifest_sha256,
            "outer_case_count": self.outer_case_count,
            "baryon_mode": self.baryon_mode,
            "execution": {
                "cpu": "manual Lageunha execution only when required",
                "gpu": "Slurm submission only when required",
                "outer_stage": "no 1-pc completion claim",
                "nested_stage": "requires exact outer checkpoint and provenance",
            },
            "nested_requests": [asdict(request) for request in self.nested_requests],
        }


def _baryon_mode(cases: tuple[GalaxyMergerZoomCase, ...]) -> str:
    has_baryons = any(
        case.physics.host_stellar_mass_msun > 0.0 or case.physics.gas_fraction > 0.0
        for case in cases
    )
    has_baryon_free = any(
        case.physics.host_stellar_mass_msun == 0.0 and case.physics.gas_fraction == 0.0
        for case in cases
    )
    if has_baryons and has_baryon_free:
        return "mixed_manifest"
    return "baryonic" if has_baryons else "baryon_free"


def preflight_pure_fdm_outer_zoom(grid: ZoomGrid) -> PureFDMOuterZoomPreflight:
    """Reject CDM/missing replicate manifests before a run can be registered.

    The result contains one nested request for every finest outer resolution.
    It deliberately does not generate an inner run: its source checkpoint,
    source hash, wave seam, and force ledger do not exist until the outer run
    has completed successfully.
    """

    if not grid.cases:
        raise ValueError("pure-FDM outer manifest is empty")
    grouped: dict[tuple[str, int], list[GalaxyMergerZoomCase]] = {}
    for case in grid.cases:
        physics = case.physics
        if physics.dark_matter_model != "fdm":
            raise ValueError("pure-FDM outer manifest contains a CDM case")
        grouped.setdefault((physics.physics_id, case.replicate), []).append(case)

    nested_requests: list[DeferredNestedZoomRequest] = []
    for (physics_id, replicate), group in grouped.items():
        levels = {case.numerics.levelmax for case in group}
        if len(levels) < 2:
            raise ValueError("pure-FDM outer physics point lacks a resolution pair")
        finest = min(group, key=lambda case: case.numerics.finest_cell_size_pc)
        nested_requests.append(
            DeferredNestedZoomRequest(
                outer_case_id=finest.case_id,
                outer_physics_id=physics_id,
                outer_replicate=replicate,
                outer_levelmax=finest.numerics.levelmax,
            )
        )
    replicate_counts: dict[str, set[int]] = {}
    for case in grid.cases:
        replicate_counts.setdefault(case.physics.physics_id, set()).add(case.replicate)
    if any(len(replicates) < 2 for replicates in replicate_counts.values()):
        raise ValueError("pure-FDM outer physics point lacks phase replicates")

    return PureFDMOuterZoomPreflight(
        manifest_sha256=grid.manifest_sha256,
        outer_case_count=len(grid.cases),
        baryon_mode=_baryon_mode(grid.cases),
        nested_requests=tuple(
            sorted(nested_requests, key=lambda request: request.outer_case_id)
        ),
    )
