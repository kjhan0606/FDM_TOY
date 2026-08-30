"""Convert one complete pure-compatible capture record into dual-SMBH seed data.

This boundary deliberately derives only the two sink records.  It never
manufactures an FDM soliton fit, infers a source CDM fraction, or assigns a sink spin
from the gas angular-momentum diagnostic in the capture ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .capture_ledger import CaptureEvent, CaptureMember
from .dual_soliton_seed import DualSMBHSinkSeed, PureFDMDualSolitonSeed


CAPTURE_FDM_SEED_FRAME_SCHEMA_VERSION = 1


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _vector(value: Any, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-vector")
    return tuple(float(item) for item in array)


def _sink_dict(sink: DualSMBHSinkSeed) -> dict[str, Any]:
    return {
        "sink_mass_code": sink.sink_mass_code,
        "smbh_mass_code": sink.smbh_mass_code,
        "center_box": list(sink.center_box),
        "velocity_code": list(sink.velocity_code),
        "angular_momentum_code": list(sink.angular_momentum_code),
        "dark_matter_fraction": sink.dark_matter_fraction,
    }


@dataclass(frozen=True)
class CaptureFDMSeedFrame:
    """Explicit local frame and units for a non-cosmological FDM re-zoom."""

    box_length_pc: float
    mass_unit_msun: float
    velocity_unit_pc_myr: float
    origin_position_pc: tuple[float, float, float]
    frame_velocity_pc_myr: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in ("box_length_pc", "mass_unit_msun", "velocity_unit_pc_myr"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=True))
        object.__setattr__(
            self, "origin_position_pc", _vector(self.origin_position_pc, "origin_position_pc")
        )
        object.__setattr__(
            self,
            "frame_velocity_pc_myr",
            _vector(self.frame_velocity_pc_myr, "frame_velocity_pc_myr"),
        )

    def position_box(self, position_pc: np.ndarray) -> tuple[float, float, float]:
        position = (
            np.asarray(position_pc, dtype=float) - self.origin_position_pc
        ) / self.box_length_pc + 0.5
        if np.any(~np.isfinite(position)) or np.any(position < 0.0) or np.any(position >= 1.0):
            raise ValueError("capture member lies outside the explicit local FDM zoom box")
        return tuple(float(value) for value in position)

    def velocity_code(self, velocity_pc_myr: np.ndarray) -> tuple[float, float, float]:
        velocity = (
            np.asarray(velocity_pc_myr, dtype=float) - self.frame_velocity_pc_myr
        ) / self.velocity_unit_pc_myr
        return _vector(velocity, "local sink velocity_code")

    def as_dict(self) -> dict[str, Any]:
        return {
            "box_length_pc": self.box_length_pc,
            "mass_unit_msun": self.mass_unit_msun,
            "velocity_unit_pc_myr": self.velocity_unit_pc_myr,
            "origin_position_pc": list(self.origin_position_pc),
            "frame_velocity_pc_myr": list(self.frame_velocity_pc_myr),
        }


@dataclass(frozen=True)
class CaptureFDMSeedAssignment:
    """Explicit member ordering and sink angular momenta for the local seed."""

    primary_member_id: int
    secondary_member_id: int
    angular_momentum_code: tuple[tuple[float, float, float], tuple[float, float, float]]

    def __post_init__(self) -> None:
        if (
            isinstance(self.primary_member_id, bool)
            or isinstance(self.secondary_member_id, bool)
            or not isinstance(self.primary_member_id, int)
            or not isinstance(self.secondary_member_id, int)
            or self.primary_member_id == self.secondary_member_id
        ):
            raise ValueError("primary_member_id and secondary_member_id must be distinct integers")
        if not isinstance(self.angular_momentum_code, (tuple, list)) or len(self.angular_momentum_code) != 2:
            raise ValueError("angular_momentum_code must contain primary and secondary vectors")
        object.__setattr__(
            self,
            "angular_momentum_code",
            tuple(
                _vector(value, "sink angular_momentum_code")
                for value in self.angular_momentum_code
            ),
        )

    @property
    def member_ids(self) -> tuple[int, int]:
        return (self.primary_member_id, self.secondary_member_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_member_id": self.primary_member_id,
            "secondary_member_id": self.secondary_member_id,
            "angular_momentum_code": [list(value) for value in self.angular_momentum_code],
        }


@dataclass(frozen=True)
class CaptureSMBHMassProjection:
    """Authoritative SMBH masses to use with a capture-ledger binary.

    The pre-compaction ledger records the numerical sink mass and its
    kinematics, but this lagRamses branch does not record a separately
    resolved SMBH mass.  A seed therefore requires a separately versioned
    mass projection; it must never reinterpret an accreted-mass diagnostic as
    an SMBH mass or infer a source CDM fraction.
    """

    event_uid: str
    member_ids: tuple[int, int]
    smbh_masses_msun: tuple[float, float]
    source_case_id: str
    source_sha256: str
    source_path: str
    target_dark_matter_model: str = "fdm"
    target_sink_dark_matter_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.event_uid, str) or not self.event_uid.strip():
            raise ValueError("mass projection event_uid is required")
        if (
            not isinstance(self.member_ids, (tuple, list))
            or len(self.member_ids) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in self.member_ids)
            or self.member_ids[0] == self.member_ids[1]
        ):
            raise ValueError("mass projection member_ids must contain two distinct integers")
        object.__setattr__(self, "member_ids", tuple(self.member_ids))
        if not isinstance(self.smbh_masses_msun, (tuple, list)) or len(self.smbh_masses_msun) != 2:
            raise ValueError("mass projection requires two SMBH masses")
        object.__setattr__(
            self,
            "smbh_masses_msun",
            tuple(
                _finite(value, "projected SMBH mass", positive=True)
                for value in self.smbh_masses_msun
            ),
        )
        for name in ("source_case_id", "source_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"mass projection {name} is required")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in self.source_sha256)
        ):
            raise ValueError("mass projection source_sha256 must be a 64-character hexadecimal digest")
        if self.target_dark_matter_model != "fdm":
            raise ValueError("capture-to-seed target must set dark_matter_model=fdm")
        fraction = _finite(
            self.target_sink_dark_matter_fraction,
            "target_sink_dark_matter_fraction",
        )
        if fraction != 0.0:
            raise ValueError("pure-FDM target sinks must have zero CDM mass fraction")
        object.__setattr__(self, "target_sink_dark_matter_fraction", fraction)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_uid": self.event_uid,
            "member_ids": list(self.member_ids),
            "smbh_masses_msun": list(self.smbh_masses_msun),
            "source_case_id": self.source_case_id,
            "source_sha256": self.source_sha256,
            "source_path": self.source_path,
            "target_dark_matter_model": self.target_dark_matter_model,
            "target_sink_dark_matter_fraction": self.target_sink_dark_matter_fraction,
        }


def verify_mass_projection_source(
    projection: CaptureSMBHMassProjection,
    *,
    reference_directory: str | Path,
) -> Path:
    """Verify the exact local catalog file named by a mass projection."""

    source_path = Path(projection.source_path).expanduser()
    if not source_path.is_absolute():
        source_path = Path(reference_directory).expanduser() / source_path
    source_path = source_path.resolve()
    try:
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"cannot read mass projection source: {error}") from error
    if digest.lower() != projection.source_sha256.lower():
        raise ValueError("mass projection source SHA-256 does not match source_path")
    return source_path


@dataclass(frozen=True)
class CaptureFDMSeedFrameSpecification:
    """Serializable inputs needed to derive only the dual-SMBH sink pair."""

    event_uid: str
    frame: CaptureFDMSeedFrame
    assignment: CaptureFDMSeedAssignment
    mass_projection: CaptureSMBHMassProjection

    def __post_init__(self) -> None:
        if not isinstance(self.event_uid, str) or not self.event_uid.strip():
            raise ValueError("event_uid is required")
        if self.mass_projection.event_uid != self.event_uid:
            raise ValueError("mass projection event_uid must match the frame specification")
        if self.mass_projection.member_ids != self.assignment.member_ids:
            raise ValueError("mass projection member_ids must match the seed assignment")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CAPTURE_FDM_SEED_FRAME_SCHEMA_VERSION,
            "event_uid": self.event_uid,
            "frame": self.frame.as_dict(),
            "assignment": self.assignment.as_dict(),
            "mass_projection": self.mass_projection.as_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "CaptureFDMSeedFrameSpecification":
        if (
            not isinstance(record, Mapping)
            or record.get("schema_version") != CAPTURE_FDM_SEED_FRAME_SCHEMA_VERSION
        ):
            raise ValueError("unsupported capture-to-FDM seed frame schema")
        if set(record) != {
            "schema_version",
            "event_uid",
            "frame",
            "assignment",
            "mass_projection",
        }:
            raise ValueError("capture-to-FDM seed frame fields are invalid")
        frame = record.get("frame")
        assignment = record.get("assignment")
        projection = record.get("mass_projection")
        if (
            not isinstance(frame, Mapping)
            or not isinstance(assignment, Mapping)
            or not isinstance(projection, Mapping)
        ):
            raise ValueError(
                "capture-to-FDM seed frame requires frame, assignment, and mass_projection objects"
            )
        try:
            return cls(
                event_uid=record.get("event_uid"),
                frame=CaptureFDMSeedFrame(**frame),
                assignment=CaptureFDMSeedAssignment(**assignment),
                mass_projection=CaptureSMBHMassProjection(**projection),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"capture-to-FDM seed frame is invalid: {error}") from error


@dataclass(frozen=True)
class CaptureDerivedDualSMBHSinkPair:
    """Capture-provenance-bound sink rows, not a complete FDM soliton seed."""

    event_uid: str
    event_sha256: str
    capture_ledger_path: Path
    member_ids: tuple[int, int]
    frame: CaptureFDMSeedFrame
    mass_projection: CaptureSMBHMassProjection
    sinks: tuple[DualSMBHSinkSeed, DualSMBHSinkSeed]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "capture_binary_dual_smbh_sink_pair_derived",
            "interpretation": (
                "capture kinematics and total sink masses are transferred from the "
                "ledger; SMBH masses come only from the explicit mass projection; "
                "the zero target CDM fraction defines the new pure-FDM setup and is "
                "not inferred about the source run; FDM soliton parameters, "
                "relaxation, and outer-merger calibration remain required"
            ),
            "capture_event": {
                "event_uid": self.event_uid,
                "event_sha256": self.event_sha256,
                "ledger_path": str(self.capture_ledger_path),
                "member_ids": list(self.member_ids),
            },
            "frame": self.frame.as_dict(),
            "mass_projection": self.mass_projection.as_dict(),
            "sinks": [_sink_dict(sink) for sink in self.sinks],
            "requirements": {
                "dark_matter_model": "fdm",
                "target_sink_dark_matter_fraction": 0.0,
                "soliton_components_required": 2,
                "analytic_fdm_drag": False,
                "capture_seed_identity_must_be_verified": True,
            },
        }


def _member_by_id(event: CaptureEvent, member_id: int) -> CaptureMember:
    for member in event.members:
        if member.sink_id == member_id:
            return member
    raise ValueError(f"capture member ID {member_id} is absent from event {event.event_uid}")


def _validate_binary_event(
    event: CaptureEvent, assignment: CaptureFDMSeedAssignment
) -> None:
    state = event.binary_orbital_state
    if event.classification != "BINARY" or state is None:
        raise ValueError("only a complete unambiguous BINARY capture can seed a dual-SMBH zoom")
    if set(assignment.member_ids) != set(state.member_ids):
        raise ValueError("seed assignment member IDs do not match the capture binary")
    pair = event.pairs[0]
    if not pair.within_numerical_merge_radius or not pair.source_two_body_bound or not state.bound:
        raise ValueError("capture binary does not pass the numerical and two-body-bound seed gate")


def _seed_sink_from_member(
    member: CaptureMember,
    *,
    frame: CaptureFDMSeedFrame,
    smbh_mass_msun: float,
    angular_momentum_code: tuple[float, float, float],
) -> DualSMBHSinkSeed:
    return DualSMBHSinkSeed(
        sink_mass_code=member.mass_msun / frame.mass_unit_msun,
        smbh_mass_code=smbh_mass_msun / frame.mass_unit_msun,
        center_box=frame.position_box(member.position_pc),
        velocity_code=frame.velocity_code(member.velocity_pc_myr),
        angular_momentum_code=angular_momentum_code,
        dark_matter_fraction=0.0,
    )


def _validate_mass_projection(
    event: CaptureEvent,
    assignment: CaptureFDMSeedAssignment,
    projection: CaptureSMBHMassProjection,
) -> None:
    if projection.event_uid != event.event_uid:
        raise ValueError("mass projection event_uid does not match the capture event")
    if projection.member_ids != assignment.member_ids:
        raise ValueError("mass projection member_ids do not match the seed assignment")


def derive_dual_smbh_sink_pair_from_capture(
    event: CaptureEvent,
    *,
    frame: CaptureFDMSeedFrame,
    assignment: CaptureFDMSeedAssignment,
    mass_projection: CaptureSMBHMassProjection,
) -> CaptureDerivedDualSMBHSinkPair:
    """Transfer one bound capture pair using an explicit SMBH mass projection."""

    _validate_binary_event(event, assignment)
    _validate_mass_projection(event, assignment, mass_projection)
    primary = _member_by_id(event, assignment.primary_member_id)
    secondary = _member_by_id(event, assignment.secondary_member_id)
    sinks = (
        _seed_sink_from_member(
            primary,
            frame=frame,
            smbh_mass_msun=mass_projection.smbh_masses_msun[0],
            angular_momentum_code=assignment.angular_momentum_code[0],
        ),
        _seed_sink_from_member(
            secondary,
            frame=frame,
            smbh_mass_msun=mass_projection.smbh_masses_msun[1],
            angular_momentum_code=assignment.angular_momentum_code[1],
        ),
    )
    return CaptureDerivedDualSMBHSinkPair(
        event_uid=event.event_uid,
        event_sha256=event.event_sha256,
        capture_ledger_path=event.source_path,
        member_ids=assignment.member_ids,
        frame=frame,
        mass_projection=mass_projection,
        sinks=sinks,
    )


def verify_pure_fdm_seed_matches_capture_sink_pair(
    seed: PureFDMDualSolitonSeed,
    pair: CaptureDerivedDualSMBHSinkPair,
) -> dict[str, Any]:
    """Verify a completed two-soliton seed preserved the derived SMBH rows."""

    reasons: list[str] = []
    for index, (actual, expected) in enumerate(zip(seed.sinks, pair.sinks), start=1):
        for name in ("sink_mass_code", "smbh_mass_code", "dark_matter_fraction"):
            if not math.isclose(
                getattr(actual, name), getattr(expected, name), rel_tol=1.0e-12, abs_tol=1.0e-14
            ):
                reasons.append(f"seed sink {index} {name} differs from the capture-derived row")
        for name in ("center_box", "velocity_code", "angular_momentum_code"):
            if not np.allclose(
                getattr(actual, name), getattr(expected, name), rtol=1.0e-12, atol=1.0e-14
            ):
                reasons.append(f"seed sink {index} {name} differs from the capture-derived row")
    return {
        "schema_version": 1,
        "status": (
            "capture_seed_identity_verified"
            if not reasons
            else "capture_seed_identity_not_verified"
        ),
        "capture_event": {
            "event_uid": pair.event_uid,
            "event_sha256": pair.event_sha256,
            "ledger_path": str(pair.capture_ledger_path),
            "member_ids": list(pair.member_ids),
        },
        "seed_case_id": seed.case_id,
        "reasons": reasons,
    }
