"""Assemble a pure-FDM dual-soliton seed without copying capture sink rows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .capture_fdm_seed import CaptureDerivedDualSMBHSinkPair
from .dual_soliton_seed import (
    DualSolitonComponent,
    PureFDMDualSolitonSeed,
)


CAPTURE_SOLITON_CONFIGURATION_SCHEMA_VERSION = 1


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class CaptureSolitonConfiguration:
    """Independently specified wave-core inputs for a capture-derived seed."""

    case_id: str
    dark_matter_model: str
    stellar_status: str
    gas_status: str
    box_length_code: float
    m_axion_ev: float
    profile_c: float
    solitons: tuple[DualSolitonComponent, DualSolitonComponent]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id is required")
        if self.dark_matter_model != "fdm":
            raise ValueError("capture-derived seed requires dark_matter_model=fdm")
        if self.stellar_status not in {"available", "absent"}:
            raise ValueError("stellar_status must be available or absent")
        if self.gas_status not in {"available", "absent"}:
            raise ValueError("gas_status must be available or absent")
        for name in ("box_length_code", "m_axion_ev", "profile_c"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if not isinstance(self.solitons, (tuple, list)) or len(self.solitons) != 2:
            raise ValueError("capture-derived seed requires exactly two explicit solitons")
        if not all(isinstance(item, DualSolitonComponent) for item in self.solitons):
            raise ValueError("solitons must contain two validated soliton components")
        object.__setattr__(self, "solitons", tuple(self.solitons))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CAPTURE_SOLITON_CONFIGURATION_SCHEMA_VERSION,
            "case_id": self.case_id,
            "dark_matter_model": self.dark_matter_model,
            "stellar_status": self.stellar_status,
            "gas_status": self.gas_status,
            "box_length_code": self.box_length_code,
            "m_axion_ev": self.m_axion_ev,
            "profile_c": self.profile_c,
            "solitons": [
                {
                    "rho0_code": item.rho0_code,
                    "core_radius_box": item.core_radius_box,
                    "center_box": list(item.center_box),
                    "velocity_code": list(item.velocity_code),
                    "phase_radians": item.phase_radians,
                }
                for item in self.solitons
            ],
        }


def capture_soliton_configuration_from_mapping(
    record: Mapping[str, Any],
) -> CaptureSolitonConfiguration:
    """Load wave-core inputs while explicitly excluding hand-written sinks."""

    expected = {
        "schema_version",
        "case_id",
        "dark_matter_model",
        "stellar_status",
        "gas_status",
        "box_length_code",
        "m_axion_ev",
        "profile_c",
        "solitons",
    }
    if (
        not isinstance(record, Mapping)
        or record.get("schema_version") != CAPTURE_SOLITON_CONFIGURATION_SCHEMA_VERSION
        or set(record) != expected
    ):
        raise ValueError("capture soliton configuration fields are invalid")
    solitons = record.get("solitons")
    if not isinstance(solitons, list):
        raise ValueError("capture soliton configuration requires a soliton list")
    try:
        return CaptureSolitonConfiguration(
            case_id=record.get("case_id"),
            dark_matter_model=record.get("dark_matter_model"),
            stellar_status=record.get("stellar_status"),
            gas_status=record.get("gas_status"),
            box_length_code=record.get("box_length_code"),
            m_axion_ev=record.get("m_axion_ev"),
            profile_c=record.get("profile_c"),
            solitons=tuple(DualSolitonComponent(**item) for item in solitons),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"capture soliton configuration is invalid: {error}") from error


def assemble_capture_derived_pure_fdm_seed(
    pair: CaptureDerivedDualSMBHSinkPair,
    configuration: CaptureSolitonConfiguration,
) -> PureFDMDualSolitonSeed:
    """Combine explicit wave-core inputs with immutable capture-derived sinks."""

    return PureFDMDualSolitonSeed(
        case_id=configuration.case_id,
        dark_matter_model=configuration.dark_matter_model,
        stellar_status=configuration.stellar_status,
        gas_status=configuration.gas_status,
        box_length_code=configuration.box_length_code,
        m_axion_ev=configuration.m_axion_ev,
        profile_c=configuration.profile_c,
        solitons=configuration.solitons,
        sinks=pair.sinks,
    )


def capture_derived_seed_mapping(seed: PureFDMDualSolitonSeed) -> dict[str, Any]:
    """Return the exact YAML input schema accepted by the seed materializer."""

    record = seed.as_dict()
    record.pop("solver_policy")
    return record
