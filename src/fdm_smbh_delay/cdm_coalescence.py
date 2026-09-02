"""Provenance-gated CDM capture-to-coalescence delay composition.

This module intentionally does not consume HR5 disappearance histories or a
population-level fixed delay.  A completed CDM environmental interval must be
an explicitly measured, same-model rate integration whose independent phase
ensemble was accepted first.  Until those records exist, the result remains
missing or censored rather than becoming a physical coalescence prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .delay_budget import DelaySegment, TrueMergeEstimate, compose_delay_segments
from .gw import peters_time_myr


CDM_DELAY_STAGE_SCHEMA_VERSION = 3
CDM_RATE_TRACK_SCHEMA_VERSION = 3
CDM_DELAY_STAGES = ("capture_to_hard_binary", "hard_binary_to_gw_regime")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_STATUSES = {"complete", "timeout", "censored", "missing", "invalid"}
_CALIBRATION_METHOD = "resolved_cdm_rate_integration"
_RATE_ESTIMATOR_METHOD = "orbit_covered_blocked_log_separation_linear_regression_v2"
_ORBITAL_PERIOD_METHOD = "instantaneous_two_body_kepler_estimate"
# These are deliberately conservative quality gates.  A positive fitted
# slope is not enough to claim a secular hardening measurement: the fitted
# trend must be separated from zero and explain a non-trivial fraction of the
# block variance.  Otherwise the interval remains censored for later review.
_MINIMUM_ORBITS_PER_BLOCK = 1.0
_MAX_RELATIVE_SLOPE_STANDARD_ERROR = 0.5
_MIN_REGRESSION_R_SQUARED = 0.5
_RAW_ORBIT_TRACK_FIELDS = {
    "schema_version",
    "status",
    "interpretation",
    "dark_matter_model",
    "physics_id",
    "capture_event_uid",
    "capture_binding",
    "runtime_identity",
    "primary_sink_id",
    "secondary_sink_id",
    "position_coordinate",
    "time_coordinate",
    "samples",
    "source_outputs",
}


class _CDMRateTrackCensored(ValueError):
    """Internal signal for a measured but unusable secular-rate candidate."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value.lower()


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and non-negative") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return parsed


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _positive(value: Any, label: str) -> float:
    parsed = _nonnegative(value, label)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return parsed


def _minimum_orbits(value: Any, label: str = "minimum_orbits_per_block") -> float:
    parsed = _positive(value, label)
    if parsed < _MINIMUM_ORBITS_PER_BLOCK:
        raise ValueError(f"{label} must be at least one complete orbit")
    return parsed


def _finite_unit_interval(value: Any, label: str) -> float:
    parsed = _finite(value, label)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return parsed


def _resolve(reference: str, base: Path) -> Path:
    path = Path(reference).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _verify_cdm_phase_runtime_identity(
    result: Any,
    *,
    expected_manifest_sha256: str,
    expected_runtime_identity: tuple[Path, str],
) -> None:
    """Semantically revalidate the runtime identity of a rate-track anchor.

    A schema-v5 result carries a runtime-identity path and digest, but a
    digest alone is not a provenance claim.  The CDM stage therefore opens
    the saved non-compacting runtime identity, checks that its contract is the
    exact phase case/capture, and requires the result's diagnosed rate ledger
    to name a run-provenance sidecar from that identity's output set.  This
    keeps an unrelated result from borrowing the identity named by the raw
    orbit track.
    """

    expected_path, expected_sha256 = expected_runtime_identity
    expected_path = Path(expected_path).expanduser().resolve()
    if result.runtime_identity_path != expected_path:
        raise ValueError("CDM phase result runtime identity differs from the rate-track anchor")
    if result.runtime_identity_sha256 != expected_sha256:
        raise ValueError("CDM phase result runtime identity SHA-256 differs from the rate-track anchor")

    from .cdm_zoom_runtime_identity import (
        read_verified_cdm_noncompacting_zoom_runtime_identity,
    )

    try:
        runtime = read_verified_cdm_noncompacting_zoom_runtime_identity(expected_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"CDM phase result runtime identity is not verified: {error}") from error
    if runtime.source_sha256 != expected_sha256:
        raise ValueError("CDM phase result runtime identity SHA-256 differs from its current record")
    contract = runtime.contract
    if contract.case.as_dict() != result.case.as_dict():
        raise ValueError("CDM phase result runtime identity case differs from its result")
    if contract.plan.grid.manifest_sha256 != expected_manifest_sha256:
        raise ValueError("CDM phase result runtime identity manifest differs from its phase")
    capture_binding = contract.capture_binding
    if (
        capture_binding.get("capture_event_uid") != result.capture_event_uid
        or capture_binding.get("capture_event_sha256") != result.capture_event_sha256
    ):
        raise ValueError("CDM phase result runtime identity capture differs from its result")

    try:
        ledger_record = json.loads(result.rate_ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"CDM phase result rate ledger cannot be read: {error}") from error
    run_source = ledger_record.get("run_provenance")
    if not isinstance(run_source, Mapping) or set(run_source) != {"path", "sha256"}:
        raise ValueError("CDM phase result rate ledger run provenance is invalid")
    run_reference = _nonempty(
        run_source.get("path"), "CDM phase result rate-ledger run provenance path"
    )
    run_path = _resolve(run_reference, result.rate_ledger_path.parent)
    if run_reference != str(run_path):
        raise ValueError("CDM phase result rate-ledger run provenance path is not canonical")
    run_sha256 = _sha256(
        run_source.get("sha256"), "CDM phase result rate-ledger run provenance SHA-256"
    )
    try:
        if _file_sha256(run_path) != run_sha256:
            raise ValueError("CDM phase result rate-ledger run provenance SHA-256 differs")
    except OSError as error:
        raise ValueError(f"cannot read CDM phase result rate-ledger run provenance: {error}") from error
    output_provenances: set[tuple[Path, str]] = set()
    for output in runtime.outputs:
        source = output.get("dm_run_provenance")
        if not isinstance(source, Mapping) or set(source) != {"path", "sha256", "build_git_hash"}:
            raise ValueError("CDM runtime identity output run provenance is invalid")
        output_path = Path(source["path"]).expanduser().resolve()
        output_sha256 = _sha256(source.get("sha256"), "CDM runtime output run-provenance SHA-256")
        output_provenances.add((output_path, output_sha256))
    if (run_path, run_sha256) not in output_provenances:
        raise ValueError(
            "CDM phase result rate-ledger run provenance is outside its runtime output set"
        )


@dataclass(frozen=True)
class CDMDelayStageSummary:
    """One environmental CDM interval, or an explicit non-complete outcome."""

    source_path: Path
    source_sha256: str
    stage: str
    segment: DelaySegment
    start_separation_pc: float | None
    end_separation_pc: float | None
    physics_id: str | None
    phase_ensemble_path: Path | None
    phase_ensemble_sha256: str | None
    rate_track_path: Path | None
    rate_track_sha256: str | None
    delay_uncertainty_myr: float | None
    capture_event_uid: str | None
    capture_event_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "summary_path": str(self.source_path),
            "summary_sha256": self.source_sha256,
            "segment": {
                "name": self.segment.name,
                "status": self.segment.status,
                "delay_myr": self.segment.delay_myr,
                "elapsed_lower_bound_myr": self.segment.elapsed_lower_bound_myr,
                "reason": self.segment.reason,
                "source_case_id": self.segment.source_case_id,
                "source_sha256": self.segment.source_sha256,
            },
            "delay_uncertainty_myr": self.delay_uncertainty_myr,
            "capture_event": None
            if self.capture_event_uid is None
            else {
                "event_uid": self.capture_event_uid,
                "event_sha256": self.capture_event_sha256,
            },
            "interval_pc": None
            if self.start_separation_pc is None
            else {
                "start_separation_pc": self.start_separation_pc,
                "end_separation_pc": self.end_separation_pc,
            },
            "physics_id": self.physics_id,
            "phase_ensemble": None
            if self.phase_ensemble_path is None
            else {
                "path": str(self.phase_ensemble_path),
                "sha256": self.phase_ensemble_sha256,
            },
            "rate_track": None
            if self.rate_track_path is None
            else {
                "path": str(self.rate_track_path),
                "sha256": self.rate_track_sha256,
            },
        }


@dataclass(frozen=True)
class CDMResolvedRateTrack:
    """Measured same-model logarithmic separation-rate samples for one stage."""

    source_path: Path
    source_sha256: str
    stage: str
    physics_id: str
    capture_event_uid: str
    capture_event_sha256: str
    runtime_identity_path: Path
    runtime_identity_sha256: str
    separations_pc: tuple[float, ...]
    logarithmic_decay_rates_per_myr: tuple[float, ...]
    rate_uncertainties_per_myr: tuple[float, ...]
    regression_r_squared: tuple[float, ...]
    orbit_counts: tuple[float, ...]
    discarded_sample_count: int

    @property
    def start_separation_pc(self) -> float:
        return self.separations_pc[0]

    @property
    def end_separation_pc(self) -> float:
        return self.separations_pc[-1]

    @property
    def integrated_delay_myr(self) -> float:
        return _integrate_rate_values(
            self.separations_pc,
            self.logarithmic_decay_rates_per_myr,
        )

    @property
    def integrated_delay_uncertainty_myr(self) -> float:
        """First-order independent propagation of block slope uncertainties."""
        return _propagate_rate_uncertainty(
            self.separations_pc,
            self.logarithmic_decay_rates_per_myr,
            self.rate_uncertainties_per_myr,
        )


def _integrate_rate_values(
    separations_pc: tuple[float, ...] | list[float],
    rates_per_myr: tuple[float, ...] | list[float],
) -> float:
    """Integrate d ln(a) / |d ln(a)/dt| over measured support."""

    if len(separations_pc) < 2 or len(separations_pc) != len(rates_per_myr):
        raise ValueError("CDM rate integration requires at least two paired points")
    delay = 0.0
    for left_a, right_a, left_rate, right_rate in zip(
        separations_pc[:-1],
        separations_pc[1:],
        rates_per_myr[:-1],
        rates_per_myr[1:],
    ):
        if left_rate <= 0.0 or right_rate <= 0.0:
            raise ValueError("CDM rate integration requires positive decay rates")
        width = math.log(left_a / right_a)
        if math.isclose(left_rate, right_rate, rel_tol=1.0e-12, abs_tol=0.0):
            delay += width / left_rate
        else:
            delay += width * math.log(right_rate / left_rate) / (right_rate - left_rate)
    if not math.isfinite(delay) or delay < 0.0:
        raise ValueError("CDM integrated delay is invalid")
    return delay


def _propagate_rate_uncertainty(
    separations_pc: tuple[float, ...] | list[float],
    rates_per_myr: tuple[float, ...] | list[float],
    uncertainties_per_myr: tuple[float, ...] | list[float],
) -> float:
    if len(rates_per_myr) != len(uncertainties_per_myr):
        raise ValueError("CDM rate uncertainty length differs from rate values")
    variance = 0.0
    for index, sigma in enumerate(uncertainties_per_myr):
        if sigma <= 0.0:
            continue
        rate = rates_per_myr[index]
        delta = min(sigma, 0.5 * rate)
        lower = list(rates_per_myr)
        upper = list(rates_per_myr)
        lower[index] = rate - delta
        upper[index] = rate + delta
        derivative = (
            _integrate_rate_values(separations_pc, upper)
            - _integrate_rate_values(separations_pc, lower)
        ) / (2.0 * delta)
        variance += (derivative * sigma) ** 2
    result = math.sqrt(max(variance, 0.0))
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("integrated CDM delay uncertainty is invalid")
    return result


def _accepted_phase_ensemble(
    path: Path,
    expected_sha256: str,
    expected_physics_id: str,
    *,
    expected_capture_event_uid: str | None = None,
    expected_capture_event_sha256: str | None = None,
    expected_runtime_identity: tuple[Path, str] | None = None,
) -> tuple[str, str]:
    """Reassess a saved phase ensemble before using it for a CDM delay.

    The phase JSON is an assessment cache, not an authority by itself.  Each
    member stores the exact result/case/input/ledger identities; this reader
    reconstructs those results through the model-specific strict reader,
    reruns the resolution comparison, and finally reruns the phase-ensemble
    gate.  A hand-written status with empty or stale members therefore cannot
    authorize a completed CDM stage.
    """

    try:
        actual_sha256 = _file_sha256(path)
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM phase ensemble: {error}") from error
    if actual_sha256 != expected_sha256:
        raise ValueError("CDM phase ensemble SHA-256 differs from stage summary")
    if not isinstance(record, Mapping):
        raise ValueError("CDM phase ensemble must be a JSON object")
    expected_fields = {
        "schema_version",
        "status",
        "interpretation",
        "dark_matter_model",
        "physics_id",
        "zoom_manifest_sha256",
        "zoom_specification_path",
        "zoom_specification_sha256",
        "capture_event",
        "replicates",
        "members",
        "reasons",
    }
    if set(record) != expected_fields or record.get("schema_version") != 2:
        raise ValueError("CDM phase ensemble schema_version is unsupported")
    if record.get("dark_matter_model") != "cdm":
        raise ValueError("CDM delay stages cannot use a non-CDM phase ensemble")
    if record.get("physics_id") != expected_physics_id:
        raise ValueError("CDM phase ensemble physics_id differs from stage summary")
    if record.get("status") != "accepted_model_specific_phase_ensemble":
        raise ValueError("CDM phase ensemble was not accepted")
    manifest_sha256 = _sha256(
        record.get("zoom_manifest_sha256"), "CDM phase ensemble zoom_manifest_sha256"
    )
    specification_reference = _nonempty(
        record.get("zoom_specification_path"),
        "CDM phase ensemble zoom_specification_path",
    )
    specification_path = _resolve(specification_reference, path.parent)
    if specification_reference != str(specification_path):
        raise ValueError("CDM phase ensemble zoom specification path is not canonical")
    specification_sha256 = _sha256(
        record.get("zoom_specification_sha256"),
        "CDM phase ensemble zoom_specification_sha256",
    )
    try:
        if _file_sha256(specification_path) != specification_sha256:
            raise ValueError("CDM phase ensemble zoom specification SHA-256 differs")
    except OSError as error:
        raise ValueError(f"cannot read CDM phase ensemble zoom specification: {error}") from error
    capture_event = record.get("capture_event")
    if not isinstance(capture_event, Mapping) or set(capture_event) != {
        "event_uid",
        "event_sha256",
    }:
        raise ValueError("CDM phase ensemble capture event is invalid")
    capture_event_uid = _nonempty(
        capture_event.get("event_uid"), "CDM phase ensemble capture_event_uid"
    )
    capture_event_sha256 = _sha256(
        capture_event.get("event_sha256"),
        "CDM phase ensemble capture_event_sha256",
    )
    members = record.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("CDM phase ensemble must contain at least two members")

    # The result descriptors include the full serialized case.  Reconstructing
    # it here avoids trusting a caller-supplied case label while still allowing
    # the phase record to be moved together with its source result files.
    from .model_specific_physics import (
        ModelSpecificResolutionAssessment,
        assess_model_specific_phase_ensemble,
        compare_model_specific_resolution_pair,
        read_resolved_model_physics_result,
    )
    from .zoom_calibration import (
        GalaxyMergerZoomCase,
        _numerics_from_mapping,
        _physics_from_mapping,
        load_zoom_grid,
    )

    try:
        grid = load_zoom_grid(specification_path)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"CDM phase ensemble zoom specification is invalid: {error}") from error
    if grid.manifest_sha256 != manifest_sha256:
        raise ValueError(
            "CDM phase ensemble manifest SHA-256 differs from its zoom specification"
        )
    expected_runtime_key: tuple[Path, str] | None = None
    if expected_runtime_identity is not None:
        expected_runtime_key = (
            Path(expected_runtime_identity[0]).expanduser().resolve(),
            _sha256(
                expected_runtime_identity[1],
                "expected CDM runtime identity SHA-256",
            ),
        )

    def _case_from_record(value: Any) -> GalaxyMergerZoomCase:
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "case_id",
            "physics_id",
            "replicate",
            "physics",
            "numerics",
        }:
            raise ValueError("CDM phase result case identity is invalid")
        replicate = value.get("replicate")
        if isinstance(replicate, bool) or not isinstance(replicate, int):
            raise ValueError("CDM phase result replicate is invalid")
        try:
            case = GalaxyMergerZoomCase(
                _physics_from_mapping(dict(value["physics"])),
                _numerics_from_mapping(dict(value["numerics"])),
                replicate=replicate,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"CDM phase result case cannot be reconstructed: {error}") from error
        if case.as_dict() != dict(value):
            raise ValueError("CDM phase result case identity does not round-trip")
        matching_cases = [candidate for candidate in grid.cases if candidate.case_id == case.case_id]
        if len(matching_cases) != 1 or matching_cases[0].as_dict() != case.as_dict():
            raise ValueError("CDM phase result case is not an exact member of its zoom specification")
        return case

    descriptor_fields = {
        "case_id",
        "case",
        "zoom_manifest_sha256",
        "source",
        "runtime_identity",
        "capture_event_uid",
        "capture_event_sha256",
        "physics_input",
        "rate_ledger",
    }

    def _read_descriptor(value: Any):
        if not isinstance(value, Mapping) or set(value) != descriptor_fields:
            raise ValueError("CDM phase result descriptor fields are invalid")
        case = _case_from_record(value.get("case"))
        if value.get("case_id") != case.case_id:
            raise ValueError("CDM phase result descriptor case_id differs from case")
        descriptor_manifest = _sha256(
            value.get("zoom_manifest_sha256"),
            "CDM phase result descriptor zoom_manifest_sha256",
        )
        if descriptor_manifest != manifest_sha256:
            raise ValueError("CDM phase members use different zoom manifests")
        source_record = value.get("source")
        if not isinstance(source_record, Mapping) or set(source_record) != {
            "path",
            "sha256",
        }:
            raise ValueError("CDM phase result source identity is invalid")
        source_path = _resolve(
            _nonempty(source_record.get("path"), "CDM phase result source path"),
            path.parent,
        )
        if source_record.get("path") != str(source_path):
            raise ValueError("CDM phase result source path is not canonical")
        source_sha256 = _sha256(
            source_record.get("sha256"), "CDM phase result source SHA-256"
        )
        try:
            result = read_resolved_model_physics_result(
                source_path,
                case=case,
                zoom_manifest_sha256=manifest_sha256,
            )
        except (OSError, ValueError) as error:
            raise ValueError(f"CDM phase result cannot be revalidated: {error}") from error
        if result.source_sha256 != source_sha256:
            raise ValueError("CDM phase result source SHA-256 differs")
        runtime_identity_record = value.get("runtime_identity")
        if not isinstance(runtime_identity_record, Mapping) or set(runtime_identity_record) != {
            "path",
            "sha256",
        }:
            raise ValueError("CDM phase result runtime identity is invalid")
        runtime_identity_path = _resolve(
            _nonempty(
                runtime_identity_record.get("path"),
                "CDM phase result runtime identity path",
            ),
            path.parent,
        )
        if runtime_identity_record.get("path") != str(runtime_identity_path):
            raise ValueError("CDM phase result runtime identity path is not canonical")
        runtime_identity_sha256 = _sha256(
            runtime_identity_record.get("sha256"),
            "CDM phase result runtime identity SHA-256",
        )
        try:
            if _file_sha256(runtime_identity_path) != runtime_identity_sha256:
                raise ValueError("CDM phase result runtime identity SHA-256 differs")
        except OSError as error:
            raise ValueError(
                f"cannot read CDM phase result runtime identity: {error}"
            ) from error
        if ModelSpecificResolutionAssessment._result_descriptor(result) != dict(value):
            raise ValueError("CDM phase result descriptor differs from its current result")
        if expected_runtime_key is not None and (
            result.runtime_identity_path,
            result.runtime_identity_sha256,
        ) == expected_runtime_key:
            _verify_cdm_phase_runtime_identity(
                result,
                expected_manifest_sha256=manifest_sha256,
                expected_runtime_identity=expected_runtime_key,
            )
        return result

    reassessed_members = []
    member_runtime_identities: set[tuple[Path, str]] = set()
    for index, member in enumerate(members):
        if not isinstance(member, Mapping) or set(member) != {
            "schema_version",
            "status",
            "interpretation",
            "dark_matter_model",
            "reference",
            "comparison",
            "matched_rate_points",
            "maximum_power_fractional_difference",
            "maximum_torque_fractional_difference",
            "maximum_eccentricity_difference",
            "reasons",
        }:
            raise ValueError(f"CDM phase member {index} fields are invalid")
        reference = _read_descriptor(member.get("reference"))
        comparison = _read_descriptor(member.get("comparison"))
        try:
            assessment = compare_model_specific_resolution_pair(reference, comparison)
        except (OSError, ValueError) as error:
            raise ValueError(f"CDM phase member {index} cannot be reassessed: {error}") from error
        if assessment.as_dict() != dict(member):
            raise ValueError("CDM phase member differs from its reassessed resolution pair")
        member_runtime_identities.update(
            (
                result.runtime_identity_path,
                result.runtime_identity_sha256,
            )
            for result in (reference, comparison)
        )
        reassessed_members.append(assessment)
    try:
        reassessed = assess_model_specific_phase_ensemble(tuple(reassessed_members))
    except (OSError, ValueError) as error:
        raise ValueError(f"CDM phase ensemble cannot be reassessed: {error}") from error
    if reassessed.as_dict(zoom_specification_path=specification_path) != dict(record):
        raise ValueError("CDM phase ensemble differs from its reassessed members")
    if capture_event_uid != reassessed.members[0].reference.capture_event_uid:
        raise ValueError("CDM phase ensemble capture event UID is not bound to its members")
    if capture_event_sha256 != reassessed.members[0].reference.capture_event_sha256:
        raise ValueError("CDM phase ensemble capture event SHA-256 is not bound to its members")
    if expected_capture_event_uid is not None and capture_event_uid != expected_capture_event_uid:
        raise ValueError("CDM phase ensemble capture event UID differs from the CDM stage")
    if expected_capture_event_sha256 is not None and capture_event_sha256 != _sha256(
        expected_capture_event_sha256, "expected CDM capture event SHA-256"
    ):
        raise ValueError("CDM phase ensemble capture event SHA-256 differs from the CDM stage")
    if expected_runtime_key is not None:
        if expected_runtime_key not in member_runtime_identities:
            raise ValueError(
                "CDM rate-track runtime identity is not one of the accepted phase members"
            )
    return capture_event_uid, capture_event_sha256


def _read_attested_raw_orbit_track(path: Path, expected_physics_id: str) -> dict[str, Any]:
    """Re-extract a lagRamses track before accepting any derived rate.

    A rate table is only a physical CDM result when its raw samples still
    correspond to the verified runtime identity and to the current
    ``COMPLETE``/info/sink/sidecar files.  Re-running the metadata-only
    extractor here closes the otherwise easy path of hand-writing samples and
    copying a capture-event digest into an otherwise plausible JSON file.
    """

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM raw orbit track: {error}") from error
    if not isinstance(record, Mapping):
        raise ValueError("CDM raw orbit track must be a JSON object")
    if set(record) != _RAW_ORBIT_TRACK_FIELDS or (
        record.get("schema_version") != 1
        or record.get("status") != "raw_relative_orbit_track"
        or record.get("dark_matter_model") != "cdm"
        or record.get("physics_id") != expected_physics_id
    ):
        raise ValueError("CDM raw orbit track identity is invalid")
    runtime_identity = record.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping) or set(runtime_identity) != {
        "path",
        "sha256",
        "contract_path",
        "contract_sha256",
    }:
        raise ValueError("CDM raw orbit track runtime identity is invalid")
    runtime_path = _resolve(
        _nonempty(runtime_identity.get("path"), "raw runtime identity path"),
        path.parent,
    )
    runtime_sha256 = _sha256(
        runtime_identity.get("sha256"), "raw runtime identity SHA-256"
    )
    contract_path = _resolve(
        _nonempty(runtime_identity.get("contract_path"), "raw runtime contract path"),
        path.parent,
    )
    contract_sha256 = _sha256(
        runtime_identity.get("contract_sha256"), "raw runtime contract SHA-256"
    )
    if runtime_identity.get("path") != str(runtime_path) or runtime_identity.get(
        "contract_path"
    ) != str(contract_path):
        raise ValueError("CDM raw orbit track runtime identity paths are not canonical")
    try:
        if _file_sha256(runtime_path) != runtime_sha256:
            raise ValueError("CDM raw runtime identity SHA-256 differs")
        if _file_sha256(contract_path) != contract_sha256:
            raise ValueError("CDM raw runtime contract SHA-256 differs")
    except OSError as error:
        raise ValueError(f"cannot read CDM raw runtime identity: {error}") from error
    try:
        from .lagramses_cdm_orbit import extract_lagramses_cdm_pair_orbit_track

        fresh = extract_lagramses_cdm_pair_orbit_track(runtime_path).as_dict()
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot re-extract CDM raw orbit provenance: {error}") from error
    if dict(record) != fresh:
        raise ValueError(
            "CDM raw orbit track differs from its current lagRamses extraction"
        )
    return dict(record)


def _raw_runtime_identity_artifact(
    record: Mapping[str, Any],
    *,
    base: Path,
) -> tuple[Path, str]:
    """Return the current runtime identity named by an attested raw track."""

    identity = record.get("runtime_identity")
    if not isinstance(identity, Mapping) or set(identity) != {
        "path",
        "sha256",
        "contract_path",
        "contract_sha256",
    }:
        raise ValueError("CDM raw orbit runtime identity artifact is invalid")
    identity_path = _resolve(
        _nonempty(identity.get("path"), "CDM raw orbit runtime identity path"),
        base,
    )
    if identity.get("path") != str(identity_path):
        raise ValueError("CDM raw orbit runtime identity path is not canonical")
    identity_sha256 = _sha256(
        identity.get("sha256"), "CDM raw orbit runtime identity SHA-256"
    )
    try:
        if _file_sha256(identity_path) != identity_sha256:
            raise ValueError("CDM raw orbit runtime identity SHA-256 differs")
    except OSError as error:
        raise ValueError(f"cannot read CDM raw orbit runtime identity: {error}") from error
    return identity_path, identity_sha256


def _raw_orbit_samples(path: Path, expected_physics_id: str) -> list[dict[str, float]]:
    record = _read_attested_raw_orbit_track(path, expected_physics_id)
    samples = record.get("samples")
    if not isinstance(samples, list):
        raise ValueError("CDM raw orbit track samples are invalid")
    parsed: list[dict[str, float]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"CDM raw orbit sample {index} is invalid")
        if sample.get("orbital_period_myr") is None:
            raise _CDMRateTrackCensored(
                "missing_orbital_period_for_secular_rate"
            )
        if sample.get("orbital_period_method") != _ORBITAL_PERIOD_METHOD:
            raise _CDMRateTrackCensored(
                "unsupported_orbital_period_method_for_secular_rate"
            )
        parsed.append(
            {
                "time_myr": _nonnegative(
                    sample.get("time_myr"), f"raw orbit sample {index} time_myr"
                ),
                "separation_pc": _positive(
                    sample.get("separation_pc"), f"raw orbit sample {index} separation_pc"
                ),
                "orbital_period_myr": _positive(
                    sample.get("orbital_period_myr"),
                    f"raw orbit sample {index} orbital_period_myr",
                ),
            }
        )
    if any(
        left["time_myr"] >= right["time_myr"]
        for left, right in zip(parsed[:-1], parsed[1:])
    ):
        raise ValueError("CDM raw orbit sample times must be strictly increasing")
    return parsed


def _block_rate_points(
    samples: list[dict[str, float]],
    *,
    samples_per_block: int,
    minimum_orbits_per_block: float = 1.0,
) -> tuple[list[dict[str, float]], int]:
    if isinstance(samples_per_block, bool) or not isinstance(samples_per_block, int):
        raise ValueError("samples_per_block must be an integer")
    if samples_per_block < 5:
        raise ValueError("samples_per_block must be at least five")
    minimum_orbits_per_block = _minimum_orbits(minimum_orbits_per_block)
    blocks: list[list[dict[str, float]]] = []
    start = 0
    while start + samples_per_block <= len(samples):
        end = start + samples_per_block
        while end <= len(samples):
            periods = [item["orbital_period_myr"] for item in samples[start:end]]
            median_period = float(np.median(periods)) if periods else 0.0
            elapsed = samples[end - 1]["time_myr"] - samples[start]["time_myr"]
            if median_period > 0.0 and elapsed / median_period >= minimum_orbits_per_block:
                break
            end += 1
        if end > len(samples):
            break
        blocks.append(samples[start:end])
        start = end
    if len(blocks) < 3:
        raise _CDMRateTrackCensored(
            "insufficient_complete_orbit_blocks_for_secular_rate"
        )
    discarded_sample_count = len(samples) - start
    points: list[dict[str, float]] = []
    for block_index, block in enumerate(blocks):
        count = len(block)
        mean_time = sum(item["time_myr"] for item in block) / count
        logarithms = [math.log(item["separation_pc"]) for item in block]
        mean_log_separation = sum(logarithms) / count
        denominator = sum((item["time_myr"] - mean_time) ** 2 for item in block)
        if denominator <= 0.0:
            raise ValueError("raw orbit regression block has no time span")
        slope = sum(
            (item["time_myr"] - mean_time) * (log_separation - mean_log_separation)
            for item, log_separation in zip(block, logarithms)
        ) / denominator
        if not math.isfinite(slope) or slope >= 0.0:
            raise _CDMRateTrackCensored(
                f"nonnegative_secular_slope_in_block_{block_index}"
            )
        residuals = [
            log_separation
            - (mean_log_separation + slope * (item["time_myr"] - mean_time))
            for item, log_separation in zip(block, logarithms)
        ]
        residual_sum_squares = sum(residual * residual for residual in residuals)
        slope_standard_error = math.sqrt(
            residual_sum_squares / max(count - 2, 1) / denominator
        )
        total_sum_squares = sum(
            (logarithm - mean_log_separation) ** 2 for logarithm in logarithms
        )
        r_squared = (
            1.0 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0.0
            else 1.0
        )
        r_squared = min(max(r_squared, 0.0), 1.0)
        if slope_standard_error >= _MAX_RELATIVE_SLOPE_STANDARD_ERROR * abs(slope):
            raise _CDMRateTrackCensored(
                f"slope_uncertainty_too_large_in_block_{block_index}"
            )
        if r_squared < _MIN_REGRESSION_R_SQUARED:
            raise _CDMRateTrackCensored(
                f"low_regression_r_squared_in_block_{block_index}"
            )
        median_period = float(
            np.median([item["orbital_period_myr"] for item in block])
        )
        orbit_count = (
            block[-1]["time_myr"] - block[0]["time_myr"]
        ) / median_period
        points.append(
            {
                "separation_pc": math.exp(mean_log_separation),
                "dln_separation_dt_per_myr": slope,
                "slope_standard_error_per_myr": slope_standard_error,
                "r_squared": r_squared,
                "orbit_count": orbit_count,
            }
        )
    if any(
        left["separation_pc"] <= right["separation_pc"]
        for left, right in zip(points[:-1], points[1:])
    ):
        raise _CDMRateTrackCensored(
            "nonmonotonic_orbit_averaged_separation"
        )
    return points, discarded_sample_count


def derive_cdm_secular_rate_track(
    raw_orbit_track_path: str | Path,
    *,
    stage: str,
    samples_per_block: int,
    minimum_orbits_per_block: float = 1.0,
) -> dict[str, Any]:
    """Make a reproducible candidate rate table from orbit-covered blocks.

    ``samples_per_block`` is a minimum sampling requirement, not an orbit
    average by itself.  Every retained block must span at least
    ``minimum_orbits_per_block`` instantaneous Kepler-period sampling proxies.
    A stalled or
    insufficiently covered track is returned as an explicit censored record.
    """

    if stage not in CDM_DELAY_STAGES:
        raise ValueError("stage is not a CDM delay stage")
    if isinstance(samples_per_block, bool) or not isinstance(samples_per_block, int):
        raise ValueError("samples_per_block must be an integer")
    if samples_per_block < 5:
        raise ValueError("samples_per_block must be at least five")
    minimum_orbits_per_block = _minimum_orbits(minimum_orbits_per_block)
    source = Path(raw_orbit_track_path).expanduser().resolve()
    try:
        raw_record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM raw orbit track: {error}") from error
    if not isinstance(raw_record, Mapping):
        raise ValueError("CDM raw orbit track must be a JSON object")
    physics_id = _nonempty(raw_record.get("physics_id"), "CDM raw orbit physics_id")
    capture_event_uid = _nonempty(
        raw_record.get("capture_event_uid"), "CDM raw orbit capture_event_uid"
    )
    capture_binding = raw_record.get("capture_binding")
    if not isinstance(capture_binding, Mapping):
        raise ValueError("CDM raw orbit capture_binding is invalid")
    capture_event_sha256 = _sha256(
        capture_binding.get("capture_event_sha256"),
        "CDM raw orbit capture_event_sha256",
    )
    runtime_identity_path, runtime_identity_sha256 = _raw_runtime_identity_artifact(
        raw_record,
        base=source.parent,
    )
    try:
        samples = _raw_orbit_samples(source, physics_id)
        points, discarded_sample_count = _block_rate_points(
            samples,
            samples_per_block=samples_per_block,
            minimum_orbits_per_block=minimum_orbits_per_block,
        )
    except _CDMRateTrackCensored as error:
        return {
            "schema_version": CDM_RATE_TRACK_SCHEMA_VERSION,
            "status": "censored",
            "dark_matter_model": "cdm",
            "stage": stage,
            "physics_id": physics_id,
            "capture_event_uid": capture_event_uid,
            "capture_event_sha256": capture_event_sha256,
            "estimator": {
                "method": _RATE_ESTIMATOR_METHOD,
                "raw_orbit_track_path": str(source),
                "raw_orbit_track_sha256": _file_sha256(source),
                "runtime_identity": {
                    "path": str(runtime_identity_path),
                    "sha256": runtime_identity_sha256,
                },
                "samples_per_block": samples_per_block,
                "minimum_orbits_per_block": minimum_orbits_per_block,
                "discarded_sample_count": 0,
            },
            "rate_points": [],
            "integrated_delay_uncertainty_myr": None,
            "reason": error.reason,
        }
    separations = tuple(point["separation_pc"] for point in points)
    rates = tuple(-point["dln_separation_dt_per_myr"] for point in points)
    uncertainties = tuple(point["slope_standard_error_per_myr"] for point in points)
    return {
        "schema_version": CDM_RATE_TRACK_SCHEMA_VERSION,
        "status": "complete",
        "dark_matter_model": "cdm",
        "stage": stage,
        "physics_id": physics_id,
        "capture_event_uid": capture_event_uid,
        "capture_event_sha256": capture_event_sha256,
        "estimator": {
            "method": _RATE_ESTIMATOR_METHOD,
            "raw_orbit_track_path": str(source),
            "raw_orbit_track_sha256": _file_sha256(source),
            "runtime_identity": {
                "path": str(runtime_identity_path),
                "sha256": runtime_identity_sha256,
            },
            "samples_per_block": samples_per_block,
            "minimum_orbits_per_block": minimum_orbits_per_block,
            "discarded_sample_count": discarded_sample_count,
        },
        "rate_points": points,
        "integrated_delay_uncertainty_myr": _propagate_rate_uncertainty(
            separations, rates, uncertainties
        ),
        "reason": None,
    }


def read_cdm_resolved_rate_track(
    path: str | Path,
    *,
    expected_stage: str,
    expected_physics_id: str,
) -> CDMResolvedRateTrack:
    """Read a reproducible CDM decay-rate track without extrapolating support."""

    if expected_stage not in CDM_DELAY_STAGES:
        raise ValueError("expected_stage is not a CDM delay stage")
    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM resolved rate track: {error}") from error
    expected_fields = {
        "schema_version",
        "status",
        "dark_matter_model",
        "stage",
        "physics_id",
        "capture_event_uid",
        "capture_event_sha256",
        "estimator",
        "rate_points",
        "integrated_delay_uncertainty_myr",
        "reason",
    }
    if not isinstance(record, Mapping) or set(record) != expected_fields:
        raise ValueError("CDM resolved rate-track fields are invalid")
    if record.get("schema_version") != CDM_RATE_TRACK_SCHEMA_VERSION:
        raise ValueError("CDM resolved rate track schema_version is unsupported")
    if record.get("status") != "complete":
        reason = record.get("reason")
        raise ValueError(
            "CDM resolved rate track is not complete"
            + (f": {reason}" if isinstance(reason, str) and reason else "")
        )
    if record.get("dark_matter_model") != "cdm":
        raise ValueError("CDM resolved rate track requires dark_matter_model=cdm")
    if record.get("stage") != expected_stage:
        raise ValueError("CDM resolved rate track stage differs from the delay stage")
    if record.get("physics_id") != expected_physics_id:
        raise ValueError("CDM resolved rate track physics_id differs from the phase ensemble")
    capture_event_uid = _nonempty(
        record.get("capture_event_uid"), "CDM rate-track capture_event_uid"
    )
    capture_event_sha256 = _sha256(
        record.get("capture_event_sha256"), "CDM rate-track capture_event_sha256"
    )
    if record.get("reason") is not None:
        raise ValueError("complete CDM resolved rate tracks cannot carry a reason")
    declared_uncertainty = _nonnegative(
        record.get("integrated_delay_uncertainty_myr"),
        "integrated_delay_uncertainty_myr",
    )
    estimator = record.get("estimator")
    if not isinstance(estimator, Mapping) or set(estimator) != {
        "method",
        "raw_orbit_track_path",
        "raw_orbit_track_sha256",
        "runtime_identity",
        "samples_per_block",
        "minimum_orbits_per_block",
        "discarded_sample_count",
    } or estimator.get("method") != _RATE_ESTIMATOR_METHOD:
        raise ValueError("CDM resolved rate-track estimator is invalid")
    reference = _nonempty(estimator.get("raw_orbit_track_path"), "raw_orbit_track_path")
    raw_path = _resolve(reference, source.parent)
    expected_raw_sha256 = _sha256(
        estimator.get("raw_orbit_track_sha256"),
        "raw_orbit_track_sha256",
    )
    if _file_sha256(raw_path) != expected_raw_sha256:
        raise ValueError("CDM raw orbit track SHA-256 differs from rate track")
    block_size = estimator.get("samples_per_block")
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError("samples_per_block must be an integer")
    minimum_orbits_per_block = _minimum_orbits(
        estimator.get("minimum_orbits_per_block")
    )
    discarded_sample_count = estimator.get("discarded_sample_count")
    if (
        isinstance(discarded_sample_count, bool)
        or not isinstance(discarded_sample_count, int)
        or discarded_sample_count < 0
    ):
        raise ValueError("discarded_sample_count must be a non-negative integer")
    try:
        raw_samples = _raw_orbit_samples(raw_path, expected_physics_id)
    except _CDMRateTrackCensored as error:
        raise ValueError(f"CDM raw orbit track is censored: {error.reason}") from error
    try:
        raw_record = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM raw orbit track identity: {error}") from error
    if not isinstance(raw_record, Mapping):
        raise ValueError("CDM raw orbit track identity is invalid")
    runtime_identity_path, runtime_identity_sha256 = _raw_runtime_identity_artifact(
        raw_record,
        base=raw_path.parent,
    )
    estimator_identity = estimator.get("runtime_identity")
    if not isinstance(estimator_identity, Mapping) or set(estimator_identity) != {
        "path",
        "sha256",
    }:
        raise ValueError("CDM rate-track runtime identity is invalid")
    if estimator_identity.get("path") != str(runtime_identity_path) or _sha256(
        estimator_identity.get("sha256"), "CDM rate-track runtime identity SHA-256"
    ) != runtime_identity_sha256:
        raise ValueError("CDM rate-track runtime identity differs from raw orbit track")
    if _nonempty(raw_record.get("capture_event_uid"), "raw capture_event_uid") != capture_event_uid:
        raise ValueError("CDM rate-track capture event UID differs from raw orbit track")
    raw_binding = raw_record.get("capture_binding")
    if not isinstance(raw_binding, Mapping) or _sha256(
        raw_binding.get("capture_event_sha256"), "raw capture_event_sha256"
    ) != capture_event_sha256:
        raise ValueError("CDM rate-track capture event SHA-256 differs from raw orbit track")
    try:
        expected_points, expected_discarded = _block_rate_points(
            raw_samples,
            samples_per_block=block_size,
            minimum_orbits_per_block=minimum_orbits_per_block,
        )
    except _CDMRateTrackCensored as error:
        raise ValueError(f"CDM raw orbit track is censored: {error.reason}") from error
    if discarded_sample_count != expected_discarded:
        raise ValueError("CDM discarded sample count differs from raw orbit track")
    points = record.get("rate_points")
    if not isinstance(points, list) or len(points) != len(expected_points):
        raise ValueError("CDM resolved rate track points differ from its raw orbit track")
    separations: list[float] = []
    rates: list[float] = []
    uncertainties: list[float] = []
    r_squared_values: list[float] = []
    orbit_counts: list[float] = []
    for index, (point, expected) in enumerate(zip(points, expected_points)):
        if not isinstance(point, Mapping) or set(point) != {
            "separation_pc",
            "dln_separation_dt_per_myr",
            "slope_standard_error_per_myr",
            "r_squared",
            "orbit_count",
        }:
            raise ValueError(f"CDM resolved rate point {index} fields are invalid")
        separation = _positive(point.get("separation_pc"), f"rate point {index} separation_pc")
        raw_rate = point.get("dln_separation_dt_per_myr")
        if isinstance(raw_rate, bool):
            raise ValueError("CDM resolved rate track must have negative dln_separation_dt_per_myr")
        try:
            signed_rate = float(raw_rate)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "CDM resolved rate track must have negative dln_separation_dt_per_myr"
            ) from error
        if not math.isfinite(signed_rate) or signed_rate >= 0.0:
            raise ValueError("CDM resolved rate track must have negative dln_separation_dt_per_myr")
        if not math.isclose(separation, expected["separation_pc"], rel_tol=1.0e-12):
            raise ValueError("CDM rate separation differs from its raw orbit regression")
        if not math.isclose(signed_rate, expected["dln_separation_dt_per_myr"], rel_tol=1.0e-12):
            raise ValueError("CDM rate differs from its raw orbit regression")
        uncertainty = _nonnegative(
            point.get("slope_standard_error_per_myr"),
            f"rate point {index} slope_standard_error_per_myr",
        )
        if not math.isclose(
            uncertainty,
            expected["slope_standard_error_per_myr"],
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError("CDM rate uncertainty differs from its raw orbit regression")
        r_squared = _finite_unit_interval(point.get("r_squared"), f"rate point {index} r_squared")
        if not math.isclose(r_squared, expected["r_squared"], rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("CDM regression R-squared differs from its raw orbit regression")
        orbit_count = _positive(point.get("orbit_count"), f"rate point {index} orbit_count")
        if not math.isclose(orbit_count, expected["orbit_count"], rel_tol=1.0e-12):
            raise ValueError("CDM orbit coverage differs from its raw orbit regression")
        separations.append(separation)
        rates.append(-signed_rate)
        uncertainties.append(uncertainty)
        r_squared_values.append(r_squared)
        orbit_counts.append(orbit_count)
    calculated_uncertainty = _propagate_rate_uncertainty(
        separations,
        rates,
        uncertainties,
    )
    if not math.isclose(
        declared_uncertainty,
        calculated_uncertainty,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError("CDM integrated delay uncertainty differs from rate points")
    return CDMResolvedRateTrack(
        source_path=source,
        source_sha256=_file_sha256(source),
        stage=expected_stage,
        physics_id=expected_physics_id,
        capture_event_uid=capture_event_uid,
        capture_event_sha256=capture_event_sha256,
        runtime_identity_path=runtime_identity_path,
        runtime_identity_sha256=runtime_identity_sha256,
        separations_pc=tuple(separations),
        logarithmic_decay_rates_per_myr=tuple(rates),
        rate_uncertainties_per_myr=tuple(uncertainties),
        regression_r_squared=tuple(r_squared_values),
        orbit_counts=tuple(orbit_counts),
        discarded_sample_count=discarded_sample_count,
    )


def _read_completed_calibration(
    record: Any,
    *,
    base: Path,
    stage: str,
) -> tuple[str, str, str, Path, str, CDMResolvedRateTrack]:
    if not isinstance(record, Mapping) or set(record) != {
        "method",
        "phase_ensemble_path",
        "phase_ensemble_sha256",
        "physics_id",
        "capture_event_uid",
        "capture_event_sha256",
        "rate_track_path",
        "rate_track_sha256",
    }:
        raise ValueError("completed CDM stage calibration fields are invalid")
    if record.get("method") != _CALIBRATION_METHOD:
        raise ValueError(
            "completed CDM stage must use resolved_cdm_rate_integration; "
            "fixed or HR5 population delays are not accepted"
        )
    physics_id = _nonempty(record.get("physics_id"), "CDM physics_id")
    capture_event_uid = _nonempty(
        record.get("capture_event_uid"), "CDM capture_event_uid"
    )
    capture_event_sha256 = _sha256(
        record.get("capture_event_sha256"), "CDM capture_event_sha256"
    )
    reference = _nonempty(record.get("phase_ensemble_path"), "phase_ensemble_path")
    digest = _sha256(record.get("phase_ensemble_sha256"), "phase_ensemble_sha256")
    ensemble_path = _resolve(reference, base)
    rate_reference = _nonempty(record.get("rate_track_path"), "rate_track_path")
    rate_digest = _sha256(record.get("rate_track_sha256"), "rate_track_sha256")
    rate_path = _resolve(rate_reference, base)
    track = read_cdm_resolved_rate_track(
        rate_path,
        expected_stage=stage,
        expected_physics_id=physics_id,
    )
    if track.source_sha256 != rate_digest:
        raise ValueError("CDM resolved rate-track SHA-256 differs from stage summary")
    _accepted_phase_ensemble(
        ensemble_path,
        digest,
        physics_id,
        expected_capture_event_uid=capture_event_uid,
        expected_capture_event_sha256=capture_event_sha256,
        expected_runtime_identity=(
            track.runtime_identity_path,
            track.runtime_identity_sha256,
        ),
    )
    if (
        track.capture_event_uid != capture_event_uid
        or track.capture_event_sha256 != capture_event_sha256
    ):
        raise ValueError("CDM stage capture event differs from its resolved rate track")
    return (
        physics_id,
        capture_event_uid,
        capture_event_sha256,
        ensemble_path,
        digest,
        track,
    )


def read_cdm_delay_stage_summary(
    path: str | Path,
    *,
    expected_stage: str,
) -> CDMDelayStageSummary:
    """Read one strict CDM interval record without treating it as a run command."""

    if expected_stage not in CDM_DELAY_STAGES:
        raise ValueError("expected_stage is not a CDM delay stage")
    source = Path(path).expanduser().resolve()
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM delay stage summary: {error}") from error
    expected_fields = {
        "schema_version",
        "status",
        "dark_matter_model",
        "stage",
        "delay_myr",
        "delay_uncertainty_myr",
        "elapsed_lower_bound_myr",
        "reason",
        "interval_pc",
        "calibration",
    }
    if not isinstance(record, Mapping) or set(record) != expected_fields:
        raise ValueError("CDM delay stage summary fields are invalid")
    if record.get("schema_version") != CDM_DELAY_STAGE_SCHEMA_VERSION:
        raise ValueError("CDM delay stage schema_version is unsupported")
    if record.get("dark_matter_model") != "cdm":
        raise ValueError("CDM delay stages require dark_matter_model=cdm")
    if record.get("stage") != expected_stage:
        raise ValueError("CDM delay stage does not match the requested interval")
    status = record.get("status")
    if status not in _STATUSES:
        raise ValueError("CDM delay stage status is unsupported")
    elapsed = _nonnegative(record.get("elapsed_lower_bound_myr"), "elapsed_lower_bound_myr")
    reason_value = record.get("reason")
    if reason_value is not None and (
        not isinstance(reason_value, str) or not reason_value.strip()
    ):
        raise ValueError("CDM delay stage reason must be non-empty text when supplied")
    reason = None if reason_value is None else reason_value.strip()
    if status != "complete":
        if (
            record.get("delay_myr") is not None
            or record.get("delay_uncertainty_myr") is not None
            or record.get("interval_pc") is not None
        ):
            raise ValueError("non-complete CDM stage cannot carry a delay or interval")
        if record.get("calibration") is not None:
            raise ValueError("non-complete CDM stage cannot claim an accepted calibration")
        if status in {"timeout", "censored", "invalid"} and reason is None:
            raise ValueError("non-complete CDM stage requires a reason")
        return CDMDelayStageSummary(
            source_path=source,
            source_sha256=_file_sha256(source),
            stage=expected_stage,
            segment=DelaySegment(
                expected_stage,
                status,
                None,
                elapsed,
                reason=reason,
            ),
            start_separation_pc=None,
            end_separation_pc=None,
            physics_id=None,
            phase_ensemble_path=None,
            phase_ensemble_sha256=None,
            rate_track_path=None,
            rate_track_sha256=None,
            delay_uncertainty_myr=None,
            capture_event_uid=None,
            capture_event_sha256=None,
        )

    delay = _nonnegative(record.get("delay_myr"), "delay_myr")
    delay_uncertainty = _nonnegative(
        record.get("delay_uncertainty_myr"), "delay_uncertainty_myr"
    )
    interval = record.get("interval_pc")
    if not isinstance(interval, Mapping) or set(interval) != {
        "start_separation_pc",
        "end_separation_pc",
    }:
        raise ValueError("completed CDM stage interval_pc fields are invalid")
    start = _positive(interval.get("start_separation_pc"), "start_separation_pc")
    end = _positive(interval.get("end_separation_pc"), "end_separation_pc")
    if start <= end:
        raise ValueError("completed CDM stage must evolve from larger to smaller separation")
    (
        physics_id,
        capture_event_uid,
        capture_event_sha256,
        ensemble_path,
        ensemble_sha256,
        rate_track,
    ) = _read_completed_calibration(
        record.get("calibration"),
        base=source.parent,
        stage=expected_stage,
    )
    if not math.isclose(
        start,
        rate_track.start_separation_pc,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ) or not math.isclose(
        end,
        rate_track.end_separation_pc,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise ValueError("CDM stage interval differs from its resolved rate-track support")
    if not math.isclose(
        delay,
        rate_track.integrated_delay_myr,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise ValueError("CDM stage delay differs from its resolved rate-track integral")
    if not math.isclose(
        delay_uncertainty,
        rate_track.integrated_delay_uncertainty_myr,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError("CDM stage delay uncertainty differs from its resolved rate-track")
    source_sha256 = _file_sha256(source)
    return CDMDelayStageSummary(
        source_path=source,
        source_sha256=source_sha256,
        stage=expected_stage,
        segment=DelaySegment(
            expected_stage,
            "complete",
            delay,
            elapsed,
            reason=reason or "accepted CDM resolved-rate integration",
            source_case_id=physics_id,
            source_sha256=source_sha256,
        ),
        start_separation_pc=start,
        end_separation_pc=end,
        physics_id=physics_id,
        phase_ensemble_path=ensemble_path,
        phase_ensemble_sha256=ensemble_sha256,
        rate_track_path=rate_track.source_path,
        rate_track_sha256=rate_track.source_sha256,
        delay_uncertainty_myr=delay_uncertainty,
        capture_event_uid=capture_event_uid,
        capture_event_sha256=capture_event_sha256,
    )


def build_cdm_delay_stage_record(
    rate_track_path: str | Path,
    phase_ensemble_path: str | Path,
) -> dict[str, Any]:
    """Integrate one measured CDM rate track into a self-verifying stage record.

    The rate track supplies only its resolved separation support.  This helper
    never extends the first or final interval, interpolates to another physics
    point, or replaces a rejected/missing phase ensemble with a population
    delay.
    """

    phase_path = Path(phase_ensemble_path).expanduser().resolve()
    try:
        phase_record = json.loads(phase_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM phase ensemble: {error}") from error
    if not isinstance(phase_record, Mapping):
        raise ValueError("CDM phase ensemble must be a JSON object")
    physics_id = _nonempty(phase_record.get("physics_id"), "CDM phase ensemble physics_id")
    phase_sha256 = _file_sha256(phase_path)

    raw_track_path = Path(rate_track_path).expanduser().resolve()
    try:
        raw_track = json.loads(raw_track_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CDM resolved rate track: {error}") from error
    if not isinstance(raw_track, Mapping):
        raise ValueError("CDM resolved rate track must be a JSON object")
    stage = raw_track.get("stage")
    if stage not in CDM_DELAY_STAGES:
        raise ValueError("CDM resolved rate track stage is unsupported")
    if raw_track.get("status") == "censored":
        reason = raw_track.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("censored CDM rate track requires a reason")
        return {
            "schema_version": CDM_DELAY_STAGE_SCHEMA_VERSION,
            "status": "censored",
            "dark_matter_model": "cdm",
            "stage": stage,
            "delay_myr": None,
            "delay_uncertainty_myr": None,
            "elapsed_lower_bound_myr": 0.0,
            "reason": f"CDM resolved rate track: {reason.strip()}",
            "interval_pc": None,
            "calibration": None,
        }
    track = read_cdm_resolved_rate_track(
        raw_track_path,
        expected_stage=stage,
        expected_physics_id=physics_id,
    )
    _accepted_phase_ensemble(
        phase_path,
        phase_sha256,
        physics_id,
        expected_capture_event_uid=track.capture_event_uid,
        expected_capture_event_sha256=track.capture_event_sha256,
        expected_runtime_identity=(
            track.runtime_identity_path,
            track.runtime_identity_sha256,
        ),
    )
    return {
        "schema_version": CDM_DELAY_STAGE_SCHEMA_VERSION,
        "status": "complete",
        "dark_matter_model": "cdm",
        "stage": track.stage,
        "delay_myr": track.integrated_delay_myr,
        "delay_uncertainty_myr": track.integrated_delay_uncertainty_myr,
        "elapsed_lower_bound_myr": 0.0,
        "reason": "accepted CDM resolved logarithmic-separation-rate integration",
        "interval_pc": {
            "start_separation_pc": track.start_separation_pc,
            "end_separation_pc": track.end_separation_pc,
        },
        "calibration": {
            "method": _CALIBRATION_METHOD,
            "phase_ensemble_path": str(phase_path),
            "phase_ensemble_sha256": phase_sha256,
            "physics_id": physics_id,
            "capture_event_uid": track.capture_event_uid,
            "capture_event_sha256": track.capture_event_sha256,
            "rate_track_path": str(track.source_path),
            "rate_track_sha256": track.source_sha256,
        },
    }


def peters_gravitational_wave_segment(
    *,
    mass1_msun: float,
    mass2_msun: float,
    start_separation_pc: float,
    eccentricity: float = 0.0,
) -> DelaySegment:
    """Make the final GW-only segment from explicitly supplied orbital data."""

    delay = peters_time_myr(
        mass1_msun,
        mass2_msun,
        start_separation_pc,
        eccentricity,
    )
    return DelaySegment(
        "gravitational_wave",
        "complete",
        delay,
        reason="orbit-averaged Peters gravitational-wave evolution",
        source_case_id="peters_orbit_averaged",
    )


def _invalidated(segment: DelaySegment, reason: str) -> DelaySegment:
    lower_bound = segment.elapsed_lower_bound_myr
    if segment.status == "complete" and segment.delay_myr is not None:
        lower_bound += segment.delay_myr
    return DelaySegment(
        segment.name,
        "invalid",
        None,
        lower_bound,
        reason=reason,
        source_case_id=segment.source_case_id,
        source_sha256=segment.source_sha256,
    )


def compose_cdm_coalescence_time(
    sink_time_myr: float,
    capture_to_hard: CDMDelayStageSummary,
    hard_to_gw: CDMDelayStageSummary,
    gravitational_wave: DelaySegment,
    *,
    gravitational_wave_start_separation_pc: float,
) -> TrueMergeEstimate:
    """Compose separate CDM stages only when their physical seams agree."""

    if gravitational_wave.name != "gravitational_wave" or gravitational_wave.status != "complete":
        raise ValueError("the CDM composer requires a completed gravitational-wave segment")
    gw_start = _positive(
        gravitational_wave_start_separation_pc,
        "gravitational_wave_start_separation_pc",
    )
    outer = capture_to_hard.segment
    inner = hard_to_gw.segment
    gw = gravitational_wave
    if outer.status == "complete" and inner.status == "complete":
        assert capture_to_hard.end_separation_pc is not None
        assert hard_to_gw.start_separation_pc is not None
        if capture_to_hard.physics_id != hard_to_gw.physics_id:
            inner = _invalidated(
                inner,
                "CDM environmental stages use different physics_id values",
            )
        elif (
            capture_to_hard.capture_event_uid != hard_to_gw.capture_event_uid
            or capture_to_hard.capture_event_sha256 != hard_to_gw.capture_event_sha256
        ):
            inner = _invalidated(
                inner,
                "CDM environmental stages use different capture events",
            )
        elif not math.isclose(
            capture_to_hard.end_separation_pc,
            hard_to_gw.start_separation_pc,
            rel_tol=1.0e-10,
            abs_tol=0.0,
        ):
            inner = _invalidated(
                inner,
                "CDM capture-to-hard and hardening intervals do not share one handoff separation",
            )
    if inner.status == "complete":
        assert hard_to_gw.end_separation_pc is not None
        if not math.isclose(
            hard_to_gw.end_separation_pc,
            gw_start,
            rel_tol=1.0e-10,
            abs_tol=0.0,
        ):
            gw = _invalidated(
                gw,
                "CDM hardening and Peters intervals do not share one handoff separation",
            )
    return compose_delay_segments(sink_time_myr, (outer, inner, gw))
