"""Read and convergence-test a staged pure-FDM outer-merger result."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .fdm_outer_halo import FDMOuterHaloClosure
from .fdm_outer_wave_ledger import FDMOuterWaveLedger
from .pure_fdm_zoom import NestedZoomCheckpointContract
from .zoom_calibration import GalaxyMergerZoomCase


PURE_FDM_OUTER_RESULT_SCHEMA_VERSION = 1
_STAGE_NAMES = ("outer_start", "common_core", "nested_checkpoint")
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_diagnostic(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite non-negative number") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


@dataclass(frozen=True)
class PureFDMOuterStage:
    status: str
    elapsed_since_capture_myr: float | None
    separation_pc: float | None

    def __post_init__(self) -> None:
        if self.status not in {"complete", "timeout", "censored", "invalid", "not_reached"}:
            raise ValueError("unsupported pure-FDM outer-stage status")
        for name in ("elapsed_since_capture_myr", "separation_pc"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative when supplied")
        if self.status == "complete" and (
            self.elapsed_since_capture_myr is None or self.separation_pc is None
        ):
            raise ValueError("complete outer stages require time and separation")


@dataclass(frozen=True)
class PureFDMOuterRunResult:
    """One outer live/hybrid run, not yet a physical delay calibration."""

    case: GalaxyMergerZoomCase
    manifest_sha256: str
    source_path: Path
    source_sha256: str
    capture_event_uid: str
    stages: tuple[tuple[str, PureFDMOuterStage], ...]
    maximum_relative_energy_error: float
    maximum_relative_angular_momentum_error: float
    minimum_de_broglie_resolution_cells: float
    minimum_wake_resolution_cells: float
    outer_closure: FDMOuterHaloClosure
    outer_wave_ledger: FDMOuterWaveLedger
    nested_checkpoint: NestedZoomCheckpointContract | None

    def __post_init__(self) -> None:
        if self.case.physics.dark_matter_model != "fdm":
            raise ValueError("pure-FDM outer result cannot contain a CDM case")
        _sha256(self.manifest_sha256, "manifest_sha256")
        _sha256(self.source_sha256, "source_sha256")
        if not isinstance(self.capture_event_uid, str) or not self.capture_event_uid.strip():
            raise ValueError("capture_event_uid is required")
        names = tuple(name for name, _ in self.stages)
        if set(names) != set(_STAGE_NAMES) or len(names) != len(_STAGE_NAMES):
            raise ValueError("outer result must contain every named stage exactly once")
        stage_map = dict(self.stages)
        start = stage_map["outer_start"]
        if start.status != "complete" or start.elapsed_since_capture_myr != 0.0:
            raise ValueError("outer_start must be a complete zero-time boundary")
        complete_times = [
            stage.elapsed_since_capture_myr
            for _, stage in self.stages
            if stage.status == "complete"
        ]
        if complete_times != sorted(complete_times):
            raise ValueError("outer-stage times are not physically ordered")
        complete_separations = [
            stage.separation_pc
            for _, stage in self.stages
            if stage.status == "complete"
        ]
        if complete_separations != sorted(complete_separations, reverse=True):
            raise ValueError("outer-stage separations are not monotonically decreasing")
        values = np.asarray(
            [
                self.maximum_relative_energy_error,
                self.maximum_relative_angular_momentum_error,
                self.minimum_de_broglie_resolution_cells,
                self.minimum_wake_resolution_cells,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("outer result diagnostics are invalid")
        nested_stage = stage_map["nested_checkpoint"]
        if nested_stage.status == "complete":
            if self.nested_checkpoint is None:
                raise ValueError("complete nested checkpoint stage lacks its contract")
            if self.nested_checkpoint.outer_case_id != self.case.case_id:
                raise ValueError("nested checkpoint belongs to a different outer case")
            if self.nested_checkpoint.outer_manifest_sha256 != self.manifest_sha256:
                raise ValueError("nested checkpoint manifest does not match outer result")
            if (
                self.nested_checkpoint.wave_ledger_sha256
                != self.outer_wave_ledger.source_sha256
            ):
                raise ValueError("nested checkpoint wave ledger does not match outer evidence")
            if (
                self.nested_checkpoint.force_ledger_sha256
                != self.outer_wave_ledger.force_ledger_sha256
            ):
                raise ValueError("nested checkpoint force ledger does not match outer evidence")
            if not math.isclose(
                float(nested_stage.separation_pc),
                self.nested_checkpoint.checkpoint_separation_pc,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            ):
                raise ValueError("nested checkpoint separation disagrees with its contract")
        elif self.nested_checkpoint is not None:
            raise ValueError("incomplete nested stage cannot carry a checkpoint contract")

    def stage(self, name: str) -> PureFDMOuterStage:
        try:
            return dict(self.stages)[name]
        except KeyError as error:
            raise KeyError(name) from error


@dataclass(frozen=True)
class PureFDMOuterConvergenceResult:
    status: str
    reference: PureFDMOuterRunResult
    comparison: PureFDMOuterRunResult
    common_core_delay_systematic_fraction: float | None
    reasons: tuple[str, ...]

    @property
    def accepted_for_inner_registration(self) -> bool:
        return self.status == "accepted_outer_uncalibrated"


@dataclass(frozen=True)
class PureFDMOuterPhaseEnsemble:
    """Phase-to-phase outer-merger variation after each pair has converged."""

    status: str
    members: tuple[PureFDMOuterConvergenceResult, ...]
    common_core_delay_mean_myr: float | None
    common_core_delay_standard_deviation_myr: float | None
    reasons: tuple[str, ...]

    @property
    def ready_for_overlap_evaluation(self) -> bool:
        return self.status == "accepted_outer_ensemble_uncalibrated"


def read_pure_fdm_outer_result(
    path: str | Path,
    *,
    case: GalaxyMergerZoomCase,
    manifest_sha256: str,
) -> PureFDMOuterRunResult:
    """Read a source result while preserving manifest and checkpoint identity."""

    resolved = Path(path).expanduser().resolve()
    try:
        record = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read pure-FDM outer result: {error}") from error
    if not isinstance(record, Mapping) or record.get("schema_version") != PURE_FDM_OUTER_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported pure-FDM outer-result schema")
    if (
        record.get("case_id") != case.case_id
        or record.get("case") != case.as_dict()
        or _sha256(record.get("manifest_sha256"), "manifest_sha256")
        != _sha256(manifest_sha256, "manifest_sha256")
    ):
        raise ValueError("outer result provenance does not match its manifest case")
    stage_records = record.get("stages")
    if not isinstance(stage_records, Mapping) or set(stage_records) != set(_STAGE_NAMES):
        raise ValueError("outer result must provide every named stage")
    stages: list[tuple[str, PureFDMOuterStage]] = []
    for name in _STAGE_NAMES:
        stage_record = stage_records[name]
        if not isinstance(stage_record, Mapping):
            raise ValueError(f"outer stage {name} must be an object")
        try:
            stage = PureFDMOuterStage(
                status=stage_record.get("status"),
                elapsed_since_capture_myr=stage_record.get("elapsed_since_capture_myr"),
                separation_pc=stage_record.get("separation_pc"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"outer stage {name} is invalid: {error}") from error
        stages.append((name, stage))
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("outer result diagnostics are absent")
    nested_record = record.get("nested_checkpoint")
    if nested_record is not None:
        if not isinstance(nested_record, Mapping) or nested_record.get("schema_version") != 1:
            raise ValueError("nested checkpoint contract schema is invalid")
        nested_payload = dict(nested_record)
        nested_payload.pop("schema_version")
        try:
            nested_contract = NestedZoomCheckpointContract(**nested_payload)
        except (TypeError, ValueError) as error:
            raise ValueError(f"nested checkpoint contract is invalid: {error}") from error
    else:
        nested_contract = None
    try:
        outer_closure = FDMOuterHaloClosure.from_dict(record.get("outer_closure"))
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(f"outer FDM closure is invalid: {error}") from error
    try:
        outer_wave_ledger = FDMOuterWaveLedger.from_dict(record.get("outer_wave_ledger"))
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(f"outer FDM wave ledger is invalid: {error}") from error
    try:
        return PureFDMOuterRunResult(
            case=case,
            manifest_sha256=_sha256(manifest_sha256, "manifest_sha256"),
            source_path=resolved,
            source_sha256=_file_sha256(resolved),
            capture_event_uid=record.get("capture_event_uid", ""),
            stages=tuple(stages),
            maximum_relative_energy_error=_finite_diagnostic(
                diagnostics.get("maximum_relative_energy_error"),
                "maximum_relative_energy_error",
            ),
            maximum_relative_angular_momentum_error=_finite_diagnostic(
                diagnostics.get("maximum_relative_angular_momentum_error"),
                "maximum_relative_angular_momentum_error",
            ),
            minimum_de_broglie_resolution_cells=_finite_diagnostic(
                diagnostics.get("minimum_de_broglie_resolution_cells"),
                "minimum_de_broglie_resolution_cells",
            ),
            minimum_wake_resolution_cells=_finite_diagnostic(
                diagnostics.get("minimum_wake_resolution_cells"),
                "minimum_wake_resolution_cells",
            ),
            outer_closure=outer_closure,
            outer_wave_ledger=outer_wave_ledger,
            nested_checkpoint=nested_contract,
        )
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(f"pure-FDM outer result is invalid: {error}") from error


def compare_pure_fdm_outer_resolution_pair(
    reference: PureFDMOuterRunResult,
    comparison: PureFDMOuterRunResult,
    *,
    maximum_common_core_delay_systematic_fraction: float = 0.20,
    maximum_conservation_error: float = 1.0e-3,
    minimum_resolution_cells: float = 4.0,
) -> PureFDMOuterConvergenceResult:
    """Converge the outer stage without turning it into a completed delay."""

    controls = np.asarray(
        [
            maximum_common_core_delay_systematic_fraction,
            maximum_conservation_error,
            minimum_resolution_cells,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(controls)) or np.any(controls < 0.0):
        raise ValueError("outer convergence gates are invalid")
    if reference.case.physics != comparison.case.physics or reference.case.replicate != comparison.case.replicate:
        raise ValueError("outer resolution pair does not share one physical realization")
    if reference.case.numerics.finest_cell_size_pc >= comparison.case.numerics.finest_cell_size_pc:
        raise ValueError("outer reference must have a finer cell size than comparison")

    reasons: list[str] = []
    for run, label in ((reference, "reference"), (comparison, "comparison")):
        if run.maximum_relative_energy_error > maximum_conservation_error:
            reasons.append(f"{label} energy error exceeds the gate")
        if run.maximum_relative_angular_momentum_error > maximum_conservation_error:
            reasons.append(f"{label} angular-momentum error exceeds the gate")
        if run.minimum_de_broglie_resolution_cells < minimum_resolution_cells:
            reasons.append(f"{label} de Broglie scale is underresolved")
        if run.minimum_wake_resolution_cells < minimum_resolution_cells:
            reasons.append(f"{label} wake is underresolved")
        if run.stage("common_core").status != "complete":
            reasons.append(f"{label} did not reach a common core")
        if run.stage("nested_checkpoint").status != "complete":
            reasons.append(f"{label} did not produce a nested checkpoint")

    fine_time = reference.stage("common_core").elapsed_since_capture_myr
    coarse_time = comparison.stage("common_core").elapsed_since_capture_myr
    if fine_time is None or coarse_time is None:
        systematic = None
        reasons.append("common-core delay cannot be compared")
    else:
        systematic = abs(coarse_time - fine_time) / max(abs(fine_time), np.finfo(float).tiny)
        if systematic > maximum_common_core_delay_systematic_fraction:
            reasons.append("common-core delay resolution systematic exceeds the gate")
    return PureFDMOuterConvergenceResult(
        status="accepted_outer_uncalibrated" if not reasons else "rejected",
        reference=reference,
        comparison=comparison,
        common_core_delay_systematic_fraction=(
            None if systematic is None else float(systematic)
        ),
        reasons=tuple(reasons),
    )


def assess_pure_fdm_outer_phase_ensemble(
    members: tuple[PureFDMOuterConvergenceResult, ...],
) -> PureFDMOuterPhaseEnsemble:
    """Keep phase variation explicit; it is not a resolution systematic.

    An accepted ensemble is still uncalibrated.  Its mean and standard
    deviation are descriptive inputs for a later drift-plus-diffusion fit,
    never a completed kpc-to-pc delay.
    """

    if len(members) < 2:
        raise ValueError("pure-FDM phase ensemble requires at least two members")
    reference_physics = members[0].reference.case.physics
    replicates: set[int] = set()
    reasons: list[str] = []
    times: list[float] = []
    for member in members:
        run = member.reference
        if run.case.physics != reference_physics:
            raise ValueError("phase ensemble mixes physical points")
        if run.case.replicate in replicates:
            raise ValueError("phase ensemble contains a duplicate replicate")
        replicates.add(run.case.replicate)
        if not member.accepted_for_inner_registration:
            reasons.append(f"replicate {run.case.replicate} fails its resolution pair")
            continue
        delay = run.stage("common_core").elapsed_since_capture_myr
        if delay is None:
            reasons.append(f"replicate {run.case.replicate} lacks common-core timing")
            continue
        times.append(float(delay))
    if reasons or len(times) != len(members):
        return PureFDMOuterPhaseEnsemble(
            "censored", members, None, None, tuple(reasons)
        )
    values = np.asarray(times, dtype=float)
    return PureFDMOuterPhaseEnsemble(
        "accepted_outer_ensemble_uncalibrated",
        members,
        float(np.mean(values)),
        float(np.std(values, ddof=1)),
        (),
    )
