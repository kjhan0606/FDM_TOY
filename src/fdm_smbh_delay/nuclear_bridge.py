"""Explicit lagRamses-to-nuclear-inspiral handoff contract.

The bridge is deliberately a data contract, not a new hardening law.  A
lagRamses capture record supplies the binary state; an environment snapshot
supplies the local stellar, gaseous, and FDM closure inputs.  Missing channel
data is never interpreted as zero density or zero torque.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .capture_ledger import CaptureEvent
from .lagramses import PairOrbitalState, pair_orbital_state


BRIDGE_SCHEMA_VERSION = 1
_SOURCE_HASH_RE = r"[0-9a-fA-F]{64}"


class BridgeStatus(str, Enum):
    READY = "ready"
    MISSING_ENVIRONMENT = "missing_environment"
    CENSORED = "censored"
    INVALID = "invalid"


class ChannelStatus(str, Enum):
    AVAILABLE = "available"
    ABSENT = "absent"
    MISSING = "missing"
    CENSORED = "censored"
    INVALID = "invalid"


_CHANNELS = ("stellar", "gas", "fdm")
_FDM_MODES = {"analytic_unresolved", "live_resolved"}


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _optional_finite(
    value: Any, name: str, *, minimum: float | None = None
) -> float | None:
    if value is None:
        return None
    return _finite(value, name, minimum=minimum)


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return result


def _optional_vector(value: Any, name: str) -> np.ndarray | None:
    if value is None:
        return None
    return _vector(value, name)


def _source_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(_SOURCE_HASH_RE, value) is None:
        raise ValueError(f"{name} must be exactly 64 hexadecimal characters")
    return value.lower()


@dataclass(frozen=True)
class EnvironmentChannel:
    """One explicitly classified environmental closure at the capture state."""

    channel: str
    status: str
    density_msun_pc3: float | None = None
    enclosed_mass_msun: float | None = None
    bulk_velocity_pc_myr: np.ndarray | None = None
    velocity_dispersion_pc_myr: float | None = None
    sound_speed_pc_myr: float | None = None
    rotation_fraction: float | None = None
    core_radius_pc: float | None = None
    fdm_mode: str | None = None
    resolved_wake: bool | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.channel not in _CHANNELS:
            raise ValueError(f"unsupported environment channel={self.channel!r}")
        try:
            status = ChannelStatus(self.status)
        except ValueError as error:
            raise ValueError(f"unsupported channel status={self.status!r}") from error

        density = _optional_finite(
            self.density_msun_pc3, "density_msun_pc3", minimum=0.0
        )
        enclosed = _optional_finite(
            self.enclosed_mass_msun, "enclosed_mass_msun", minimum=0.0
        )
        bulk = _optional_vector(self.bulk_velocity_pc_myr, "bulk_velocity_pc_myr")
        dispersion = _optional_finite(
            self.velocity_dispersion_pc_myr,
            "velocity_dispersion_pc_myr",
            minimum=0.0,
        )
        sound_speed = _optional_finite(
            self.sound_speed_pc_myr, "sound_speed_pc_myr", minimum=0.0
        )
        rotation = _optional_finite(
            self.rotation_fraction, "rotation_fraction", minimum=0.0
        )
        if rotation is not None and rotation > 1.0:
            raise ValueError("rotation_fraction must lie in [0, 1]")
        core_radius = _optional_finite(
            self.core_radius_pc, "core_radius_pc", minimum=0.0
        )

        if status in {ChannelStatus.MISSING, ChannelStatus.CENSORED, ChannelStatus.INVALID}:
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError(
                    f"{self.channel} non-available channels require a reason"
                )
        if status is ChannelStatus.AVAILABLE:
            if density is None or enclosed is None or bulk is None:
                raise ValueError(
                    f"{self.channel} available channel lacks density, enclosed mass, or bulk velocity"
                )
            if self.channel == "stellar" and (
                dispersion is None or dispersion <= 0.0
            ):
                raise ValueError(
                    "available stellar channel requires positive velocity dispersion"
                )
            if self.channel == "gas" and (
                sound_speed is None or sound_speed <= 0.0 or rotation is None
            ):
                raise ValueError(
                    "available gas channel requires sound speed and rotation fraction"
                )
            if self.channel == "fdm":
                if self.fdm_mode not in _FDM_MODES:
                    raise ValueError(
                        "available FDM channel requires analytic_unresolved or live_resolved mode"
                    )
                expected_wake = self.fdm_mode == "live_resolved"
                if self.resolved_wake is not expected_wake:
                    raise ValueError(
                        "resolved_wake must agree with the FDM mode; analytic drag cannot accompany a live wake"
                    )
        if status is ChannelStatus.ABSENT and (
            density not in (None, 0.0) or enclosed not in (None, 0.0)
        ):
            raise ValueError("an absent channel cannot carry a positive mass density")

        object.__setattr__(self, "status", status.value)
        object.__setattr__(self, "density_msun_pc3", density)
        object.__setattr__(self, "enclosed_mass_msun", enclosed)
        object.__setattr__(self, "bulk_velocity_pc_myr", bulk)
        object.__setattr__(self, "velocity_dispersion_pc_myr", dispersion)
        object.__setattr__(self, "sound_speed_pc_myr", sound_speed)
        object.__setattr__(self, "rotation_fraction", rotation)
        object.__setattr__(self, "core_radius_pc", core_radius)

    @property
    def available(self) -> bool:
        return self.status in {
            ChannelStatus.AVAILABLE.value,
            ChannelStatus.ABSENT.value,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "status": self.status,
            "density_msun_pc3": self.density_msun_pc3,
            "enclosed_mass_msun": self.enclosed_mass_msun,
            "bulk_velocity_pc_myr": (
                None
                if self.bulk_velocity_pc_myr is None
                else self.bulk_velocity_pc_myr.tolist()
            ),
            "velocity_dispersion_pc_myr": self.velocity_dispersion_pc_myr,
            "sound_speed_pc_myr": self.sound_speed_pc_myr,
            "rotation_fraction": self.rotation_fraction,
            "core_radius_pc": self.core_radius_pc,
            "fdm_mode": self.fdm_mode,
            "resolved_wake": self.resolved_wake,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "EnvironmentChannel":
        if not isinstance(record, Mapping):
            raise ValueError("environment channel must be an object")
        return cls(
            channel=record.get("channel", ""),
            status=record.get("status", ""),
            density_msun_pc3=record.get("density_msun_pc3"),
            enclosed_mass_msun=record.get("enclosed_mass_msun"),
            bulk_velocity_pc_myr=record.get("bulk_velocity_pc_myr"),
            velocity_dispersion_pc_myr=record.get("velocity_dispersion_pc_myr"),
            sound_speed_pc_myr=record.get("sound_speed_pc_myr"),
            rotation_fraction=record.get("rotation_fraction"),
            core_radius_pc=record.get("core_radius_pc"),
            fdm_mode=record.get("fdm_mode"),
            resolved_wake=record.get("resolved_wake"),
            reason=record.get("reason"),
        )


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Local environment required by the kpc-to-pc bridge."""

    event_uid: str
    time_myr: float
    redshift: float
    radius_pc: float
    channels: tuple[EnvironmentChannel, ...]
    source_case_id: str
    source_sha256: str
    source_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.event_uid, str) or not self.event_uid.strip():
            raise ValueError("environment event_uid is required")
        _finite(self.time_myr, "time_myr", minimum=0.0)
        _finite(self.redshift, "redshift", minimum=0.0)
        _finite(self.radius_pc, "radius_pc", minimum=0.0)
        if not isinstance(self.source_case_id, str) or not self.source_case_id.strip():
            raise ValueError("source_case_id is required")
        _source_hash(self.source_sha256, "source_sha256")
        if not isinstance(self.source_path, str) or not self.source_path.strip():
            raise ValueError("source_path is required")
        if len(self.channels) != len(_CHANNELS):
            raise ValueError("environment snapshot must contain stellar, gas, and fdm channels")
        names = tuple(channel.channel for channel in self.channels)
        if set(names) != set(_CHANNELS) or len(set(names)) != len(_CHANNELS):
            raise ValueError("environment channels must contain one stellar, gas, and fdm record")

    @property
    def status(self) -> str:
        statuses = {channel.status for channel in self.channels}
        if ChannelStatus.INVALID.value in statuses:
            return BridgeStatus.INVALID.value
        if ChannelStatus.CENSORED.value in statuses:
            return BridgeStatus.CENSORED.value
        if ChannelStatus.MISSING.value in statuses:
            return BridgeStatus.MISSING_ENVIRONMENT.value
        return BridgeStatus.READY.value

    def channel(self, name: str) -> EnvironmentChannel:
        for channel in self.channels:
            if channel.channel == name:
                return channel
        raise KeyError(name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "event_uid": self.event_uid,
            "time_myr": self.time_myr,
            "redshift": self.redshift,
            "radius_pc": self.radius_pc,
            "channels": [channel.as_dict() for channel in self.channels],
            "source_case_id": self.source_case_id,
            "source_sha256": self.source_sha256,
            "source_path": self.source_path,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "EnvironmentSnapshot":
        if not isinstance(record, Mapping):
            raise ValueError("environment snapshot must be an object")
        if record.get("schema_version") != BRIDGE_SCHEMA_VERSION:
            raise ValueError("unsupported bridge environment schema")
        channels = tuple(
            EnvironmentChannel.from_dict(item) for item in record.get("channels", ())
        )
        return cls(
            event_uid=record.get("event_uid", ""),
            time_myr=record.get("time_myr"),
            redshift=record.get("redshift"),
            radius_pc=record.get("radius_pc"),
            channels=channels,
            source_case_id=record.get("source_case_id", ""),
            source_sha256=record.get("source_sha256", ""),
            source_path=record.get("source_path", ""),
        )


@dataclass(frozen=True)
class NuclearBridgeInput:
    """Serializable lagRamses capture state plus its environment snapshot."""

    event_uid: str
    run_id: str
    capture_time_myr: float
    redshift: float
    pair: PairOrbitalState
    environment: EnvironmentSnapshot
    target_semimajor_axis_pc: float
    source_path: str
    source_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.event_uid, str)
            or not self.event_uid.strip()
            or not isinstance(self.run_id, str)
            or not self.run_id.strip()
        ):
            raise ValueError("event_uid and run_id are required")
        _finite(self.capture_time_myr, "capture_time_myr", minimum=0.0)
        _finite(self.redshift, "redshift", minimum=0.0)
        _finite(
            self.target_semimajor_axis_pc,
            "target_semimajor_axis_pc",
            minimum=0.0,
        )
        if self.target_semimajor_axis_pc == 0.0:
            raise ValueError("target_semimajor_axis_pc must be positive")
        if self.environment.event_uid != self.event_uid:
            raise ValueError("environment and capture event_uid values must agree")
        if not isinstance(self.source_path, str) or not self.source_path.strip():
            raise ValueError("source_path is required")
        _source_hash(self.source_sha256, "source_sha256")

    @property
    def status(self) -> str:
        return self.environment.status

    @property
    def ready_for_integration(self) -> bool:
        return self.status == BridgeStatus.READY.value

    @property
    def physical_binding_status(self) -> str:
        if self.pair.semi_major_axis_pc is None:
            return "unbound_or_undefined"
        return "bound" if self.pair.bound else "unbound_or_undefined"

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons = [
            channel.reason
            for channel in self.environment.channels
            if channel.reason is not None
        ]
        if self.pair.semi_major_axis_pc is None:
            reasons.append(
                "the capture state is not a bound Keplerian binary; the bridge must model dual-nucleus rebinding"
            )
        return tuple(reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "event_uid": self.event_uid,
            "run_id": self.run_id,
            "capture_time_myr": self.capture_time_myr,
            "redshift": self.redshift,
            "target_semimajor_axis_pc": self.target_semimajor_axis_pc,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "pair": self.pair.as_dict(),
            "environment": self.environment.as_dict(),
            "status": self.status,
            "ready_for_integration": self.ready_for_integration,
            "physical_binding_status": self.physical_binding_status,
            "reasons": list(self.reasons),
        }

    def write_json(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.as_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        return output

    @classmethod
    def from_capture_event(
        cls,
        event: CaptureEvent,
        *,
        run_id: str,
        capture_time_myr: float,
        environment: EnvironmentSnapshot,
        target_semimajor_axis_pc: float = 1.0,
    ) -> "NuclearBridgeInput":
        pair = event.binary_orbital_state
        if pair is None:
            raise ValueError(
                "only an unambiguous BINARY capture can initialize the nuclear bridge"
            )
        if environment.event_uid != event.event_uid:
            raise ValueError("environment event_uid does not match the capture event")
        return cls(
            event_uid=event.event_uid,
            run_id=run_id,
            capture_time_myr=capture_time_myr,
            redshift=event.redshift,
            pair=pair,
            environment=environment,
            target_semimajor_axis_pc=target_semimajor_axis_pc,
            source_path=str(event.source_path),
            source_sha256=event.event_sha256,
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "NuclearBridgeInput":
        if not isinstance(record, Mapping):
            raise ValueError("bridge input must be an object")
        if record.get("schema_version") != BRIDGE_SCHEMA_VERSION:
            raise ValueError("unsupported nuclear bridge schema")
        pair_record = record.get("pair")
        if not isinstance(pair_record, Mapping):
            raise ValueError("bridge input lacks pair state")
        try:
            member_ids = tuple(int(value) for value in pair_record["member_ids"])
            masses = tuple(float(value) for value in pair_record["masses_msun"])
            relative_position = _vector(
                pair_record["separation_vector_pc"], "separation_vector_pc"
            )
            relative_velocity = _vector(
                pair_record["relative_velocity_pc_myr"], "relative_velocity_pc_myr"
            )
            centre_position = _vector(
                pair_record["centre_of_mass_position_pc"],
                "centre_of_mass_position_pc",
            )
            centre_velocity = _vector(
                pair_record["centre_of_mass_velocity_pc_myr"],
                "centre_of_mass_velocity_pc_myr",
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("bridge pair state is malformed") from error
        if len(member_ids) != 2 or len(masses) != 2:
            raise ValueError("bridge pair state must contain two members")
        if any(not math.isfinite(value) or value <= 0.0 for value in masses):
            raise ValueError("bridge pair masses must be finite and positive")
        total_mass = sum(masses)
        positions = np.asarray(
            [
                centre_position + masses[1] / total_mass * relative_position,
                centre_position - masses[0] / total_mass * relative_position,
            ]
        )
        velocities = np.asarray(
            [
                centre_velocity + masses[1] / total_mass * relative_velocity,
                centre_velocity - masses[0] / total_mass * relative_velocity,
            ]
        )
        pair = pair_orbital_state(
            member_ids=member_ids,
            masses_msun=masses,
            positions_pc=positions,
            velocities_pc_myr=velocities,
        )
        environment = EnvironmentSnapshot.from_dict(record.get("environment", {}))
        return cls(
            event_uid=record.get("event_uid", ""),
            run_id=record.get("run_id", ""),
            capture_time_myr=record.get("capture_time_myr"),
            redshift=record.get("redshift"),
            pair=pair,
            environment=environment,
            target_semimajor_axis_pc=record.get("target_semimajor_axis_pc"),
            source_path=record.get("source_path", ""),
            source_sha256=record.get("source_sha256", ""),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "NuclearBridgeInput":
        try:
            record = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"bridge input is unreadable: {path}") from error
        return cls.from_dict(record)
