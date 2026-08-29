"""Materialize a parameterized all-wave dual-soliton lagRamses seed."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import yaml


DUAL_SOLITON_SEED_SCHEMA_VERSION = 1


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
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must have three components")
    return tuple(_finite(item, name) for item in value)  # type: ignore[return-value]


def _fortran(value: float) -> str:
    return f"{value:.16e}".replace("e", "d")


def _canonical_sha256(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DualSolitonComponent:
    rho0_code: float
    core_radius_box: float
    center_box: tuple[float, float, float]
    velocity_code: tuple[float, float, float]
    phase_radians: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "rho0_code", _finite(self.rho0_code, "rho0_code", positive=True))
        radius = _finite(self.core_radius_box, "core_radius_box", positive=True)
        if radius >= 0.5:
            raise ValueError("core_radius_box must be below half a periodic box")
        object.__setattr__(self, "core_radius_box", radius)
        centre = _vector(self.center_box, "center_box")
        if any(value < 0.0 or value >= 1.0 for value in centre):
            raise ValueError("center_box components must lie in [0, 1)")
        object.__setattr__(self, "center_box", centre)
        object.__setattr__(self, "velocity_code", _vector(self.velocity_code, "velocity_code"))
        object.__setattr__(self, "phase_radians", _finite(self.phase_radians, "phase_radians"))


@dataclass(frozen=True)
class DualSMBHSinkSeed:
    sink_mass_code: float
    smbh_mass_code: float
    center_box: tuple[float, float, float]
    velocity_code: tuple[float, float, float]
    angular_momentum_code: tuple[float, float, float]
    dark_matter_fraction: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sink_mass_code", _finite(self.sink_mass_code, "sink_mass_code", positive=True))
        object.__setattr__(self, "smbh_mass_code", _finite(self.smbh_mass_code, "smbh_mass_code", positive=True))
        centre = _vector(self.center_box, "sink center_box")
        if any(value < 0.0 or value >= 1.0 for value in centre):
            raise ValueError("sink center_box components must lie in [0, 1)")
        object.__setattr__(self, "center_box", centre)
        object.__setattr__(self, "velocity_code", _vector(self.velocity_code, "sink velocity_code"))
        object.__setattr__(
            self,
            "angular_momentum_code",
            _vector(self.angular_momentum_code, "angular_momentum_code"),
        )
        fraction = _finite(self.dark_matter_fraction, "dark_matter_fraction")
        if fraction != 0.0:
            raise ValueError("pure-FDM seed sinks cannot carry a CDM mass fraction")
        object.__setattr__(self, "dark_matter_fraction", fraction)


def _periodic_distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    delta = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    delta -= np.rint(delta)
    return float(np.linalg.norm(delta))


@dataclass(frozen=True)
class PureFDMDualSolitonSeed:
    case_id: str
    dark_matter_model: str
    stellar_status: str
    gas_status: str
    box_length_code: float
    m_axion_ev: float
    profile_c: float
    solitons: tuple[DualSolitonComponent, DualSolitonComponent]
    sinks: tuple[DualSMBHSinkSeed, DualSMBHSinkSeed]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id is required")
        if self.dark_matter_model != "fdm":
            raise ValueError("dual-soliton seed requires dark_matter_model=fdm")
        if self.stellar_status not in {"available", "absent"}:
            raise ValueError("stellar_status must be available or absent")
        if self.gas_status not in {"available", "absent"}:
            raise ValueError("gas_status must be available or absent")
        object.__setattr__(self, "box_length_code", _finite(self.box_length_code, "box_length_code", positive=True))
        object.__setattr__(self, "m_axion_ev", _finite(self.m_axion_ev, "m_axion_ev", positive=True))
        object.__setattr__(self, "profile_c", _finite(self.profile_c, "profile_c", positive=True))
        if len(self.solitons) != 2 or len(self.sinks) != 2:
            raise ValueError("dual-soliton seed requires exactly two solitons and two SMBHs")
        for index, (soliton, sink) in enumerate(zip(self.solitons, self.sinks), start=1):
            if _periodic_distance(soliton.center_box, sink.center_box) > soliton.core_radius_box:
                raise ValueError(
                    f"SMBH {index} lies outside its assigned soliton core support"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DUAL_SOLITON_SEED_SCHEMA_VERSION,
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
            "sinks": [
                {
                    "sink_mass_code": item.sink_mass_code,
                    "smbh_mass_code": item.smbh_mass_code,
                    "center_box": list(item.center_box),
                    "velocity_code": list(item.velocity_code),
                    "angular_momentum_code": list(item.angular_momentum_code),
                    "dark_matter_fraction": item.dark_matter_fraction,
                }
                for item in self.sinks
            ],
            "solver_policy": {
                "fdm_use_hjm": False,
                "fdm_outer_ledger": True,
                "analytic_fdm_drag": False,
                "initial_state": "coherent_dual_soliton_seed_requires_relaxation",
            },
        }


def pure_fdm_dual_soliton_seed_from_mapping(
    record: Mapping[str, Any],
) -> PureFDMDualSolitonSeed:
    """Validate one exact seed mapping without filling in physical inputs."""

    if not isinstance(record, Mapping) or record.get("schema_version") != DUAL_SOLITON_SEED_SCHEMA_VERSION:
        raise ValueError("unsupported dual-soliton seed schema")
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
        "sinks",
    }
    unknown = set(record) - expected
    if unknown:
        raise ValueError(f"dual-soliton seed has unknown fields: {sorted(unknown)}")
    solitons = record.get("solitons")
    sinks = record.get("sinks")
    if not isinstance(solitons, list) or not isinstance(sinks, list):
        raise ValueError("dual-soliton seed requires soliton and sink lists")
    try:
        return PureFDMDualSolitonSeed(
            case_id=record.get("case_id"),
            dark_matter_model=record.get("dark_matter_model"),
            stellar_status=record.get("stellar_status"),
            gas_status=record.get("gas_status"),
            box_length_code=record.get("box_length_code"),
            m_axion_ev=record.get("m_axion_ev"),
            profile_c=record.get("profile_c"),
            solitons=tuple(DualSolitonComponent(**item) for item in solitons),  # type: ignore[arg-type]
            sinks=tuple(DualSMBHSinkSeed(**item) for item in sinks),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"dual-soliton seed is invalid: {error}") from error


def load_pure_fdm_dual_soliton_seed(path: str | Path) -> PureFDMDualSolitonSeed:
    """Read an exact two-soliton/two-SMBH seed without inferring missing physics."""

    resolved = Path(path).expanduser().resolve()
    try:
        record = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read dual-soliton seed: {error}") from error
    return pure_fdm_dual_soliton_seed_from_mapping(record)


def _fdm_namelist_fragment(seed: PureFDMDualSolitonSeed) -> str:
    lines = [
        "! Generated by materialize_dual_soliton_ic.py; do not combine with HJM fluid levels.",
        f"! Required AMR_PARAMS boxlen={_fortran(seed.box_length_code)}",
        "&FDM_PARAMS",
        f"  m_axion={_fortran(seed.m_axion_ev)}",
        "  fdm_use_hjm=.false.",
        "  fdm_outer_ledger=.true.",
        "  fdm_dual_soliton_ic=.true.",
        f"  fdm_dual_soliton_profile_c={_fortran(seed.profile_c)}",
    ]
    for index, soliton in enumerate(seed.solitons, start=1):
        lines.extend(
            (
                f"  fdm_dual_soliton_rho0({index})={_fortran(soliton.rho0_code)}",
                f"  fdm_dual_soliton_rc_box({index})={_fortran(soliton.core_radius_box)}",
                f"  fdm_dual_soliton_phase({index})={_fortran(soliton.phase_radians)}",
            )
        )
        for dimension in range(3):
            lines.append(
                f"  fdm_dual_soliton_center_box({index},{dimension + 1})="
                f"{_fortran(soliton.center_box[dimension])}"
            )
            lines.append(
                f"  fdm_dual_soliton_velocity({index},{dimension + 1})="
                f"{_fortran(soliton.velocity_code[dimension])}"
            )
    lines.append("/")
    return "\n".join(lines) + "\n"


def _ic_sink(seed: PureFDMDualSolitonSeed) -> str:
    # lagRamses init_sink uses list-directed reads and does not skip comments.
    lines: list[str] = []
    for sink in seed.sinks:
        offset = tuple(
            seed.box_length_code * (value - 0.5) for value in sink.center_box
        )
        fields = (
            sink.sink_mass_code,
            *offset,
            *sink.velocity_code,
            *sink.angular_momentum_code,
            sink.smbh_mass_code,
            sink.dark_matter_fraction,
        )
        lines.append(" ".join(_fortran(float(value)) for value in fields))
    return "\n".join(lines) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def materialize_pure_fdm_dual_soliton_seed(
    seed: PureFDMDualSolitonSeed,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write a new, non-submitting IC directory for one exact seed manifest."""

    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    namelist = destination / "fdm_dual_soliton_params.nml"
    sink_file = destination / "ic_sink"
    _write_atomic(namelist, _fdm_namelist_fragment(seed))
    _write_atomic(sink_file, _ic_sink(seed))
    payload = seed.as_dict()
    manifest = {
        "schema_version": DUAL_SOLITON_SEED_SCHEMA_VERSION,
        "status": "pure_fdm_dual_soliton_seed_materialized",
        "interpretation": (
            "initial-condition material only; requires relaxation, paired "
            "resolution, and phase-replicate validation before outer calibration"
        ),
        "seed": payload,
        "seed_sha256": _canonical_sha256(payload),
        "files": {
            "fdm_namelist_fragment": {
                "path": str(namelist),
                "sha256": _file_sha256(namelist),
            },
            "ic_sink": {"path": str(sink_file), "sha256": _file_sha256(sink_file)},
        },
    }
    manifest_path = destination / "dual_soliton_seed_manifest.json"
    _write_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
