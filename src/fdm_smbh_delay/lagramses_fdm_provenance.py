"""Read the compact raw FDM provenance record emitted by the lagRamses patch."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any


_V1_REQUIRED_KEYS = {
    "time_code",
    "aexp",
    "nstep_coarse",
    "m_axion_ev",
    "hbar_code",
    "fdm_use_hjm",
    "fdm_first_wave_level",
    "analytic_fdm_drag_enabled",
    "force_accounting",
    "leaf_mass_code",
    "integrated_current_code",
    "leaf_cell_count",
    "complete_current_stencil_cell_count",
    "complete_current_stencil_fraction",
    "psi_snapshot_prefix",
}

_V2_DUAL_SOLITON_KEYS = {
    "fdm_dual_soliton_ic",
    "fdm_dual_soliton_profile_c",
    "fdm_dual_soliton_rho0",
    "fdm_dual_soliton_rc_box",
    "fdm_dual_soliton_center_box_1",
    "fdm_dual_soliton_center_box_2",
    "fdm_dual_soliton_velocity_1",
    "fdm_dual_soliton_velocity_2",
    "fdm_dual_soliton_phase",
}
_V2_REQUIRED_KEYS = _V1_REQUIRED_KEYS | _V2_DUAL_SOLITON_KEYS


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float(value: str, name: str, *, nonnegative: bool = False) -> float:
    try:
        parsed = float(value.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(parsed) or (nonnegative and parsed < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return parsed


def _integer(value: str, name: str, *, nonnegative: bool = False) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} is not an integer") from error
    if nonnegative and parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _logical(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"t", ".true.", "true"}:
        return True
    if normalized in {"f", ".false.", "false"}:
        return False
    raise ValueError(f"{name} is not a Fortran logical")


def _float_vector(
    value: str, name: str, *, length: int, nonnegative: bool = False
) -> tuple[float, ...]:
    tokens = value.split()
    if len(tokens) != length:
        raise ValueError(f"{name} must contain {length} components")
    return tuple(
        _float(component, name, nonnegative=nonnegative) for component in tokens
    )


@dataclass(frozen=True)
class LagRamsesFDMOuterWaveProvenance:
    """A raw summary, intentionally insufficient to be a calibrated response."""

    source_path: Path
    source_sha256: str
    source_schema_version: int
    time_code: float
    aexp: float
    nstep_coarse: int
    m_axion_ev: float
    hbar_code: float
    fdm_use_hjm: bool
    fdm_first_wave_level: int
    analytic_fdm_drag_enabled: bool
    force_accounting: str
    leaf_mass_code: float
    integrated_current_code: tuple[float, float, float]
    leaf_cell_count: float
    complete_current_stencil_cell_count: float
    complete_current_stencil_fraction: float
    psi_snapshot_prefix: str
    fdm_dual_soliton_ic: bool | None
    fdm_dual_soliton_profile_c: float | None
    fdm_dual_soliton_rho0: tuple[float, float] | None
    fdm_dual_soliton_rc_box: tuple[float, float] | None
    fdm_dual_soliton_centres_box: tuple[tuple[float, float, float], ...] | None
    fdm_dual_soliton_velocities: tuple[tuple[float, float, float], ...] | None
    fdm_dual_soliton_phase: tuple[float, float] | None

    def decision(self, *, minimum_current_stencil_fraction: float = 0.95) -> dict[str, Any]:
        """State whether the raw record can enter further postprocessing.

        ``available`` means only that the raw record is internally usable.  It
        does not mean that an outer FDM response or a physical delay is
        calibrated; field/profile indices and the remaining wave observables
        are still required by :class:`FDMOuterWaveLedger`.
        """

        if (
            not math.isfinite(minimum_current_stencil_fraction)
            or not 0.0 <= minimum_current_stencil_fraction <= 1.0
        ):
            raise ValueError("minimum_current_stencil_fraction must lie in [0, 1]")
        if self.analytic_fdm_drag_enabled:
            return {
                "status": "censored",
                "reason": "analytic FDM drag is enabled with resolved wave provenance",
            }
        if self.force_accounting != "resolved_wave_only":
            return {
                "status": "censored",
                "reason": "raw FDM force accounting is not resolved_wave_only",
            }
        if self.complete_current_stencil_fraction < minimum_current_stencil_fraction:
            return {
                "status": "censored",
                "reason": "FDM current stencil coverage lies below the required threshold",
            }
        return {
            "status": "available_raw_provenance",
            "reason": (
                "raw provenance only; field/profile/mode/granule/force-ledger "
                "postprocessing remains required"
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "lagramses_fdm_outer_wave_provenance",
            "source": {"path": str(self.source_path), "sha256": self.source_sha256},
            "source_schema_version": self.source_schema_version,
            "time_code": self.time_code,
            "aexp": self.aexp,
            "nstep_coarse": self.nstep_coarse,
            "m_axion_ev": self.m_axion_ev,
            "hbar_code": self.hbar_code,
            "fdm_use_hjm": self.fdm_use_hjm,
            "fdm_first_wave_level": self.fdm_first_wave_level,
            "analytic_fdm_drag_enabled": self.analytic_fdm_drag_enabled,
            "force_accounting": self.force_accounting,
            "leaf_mass_code": self.leaf_mass_code,
            "integrated_current_code": list(self.integrated_current_code),
            "leaf_cell_count": self.leaf_cell_count,
            "complete_current_stencil_cell_count": self.complete_current_stencil_cell_count,
            "complete_current_stencil_fraction": self.complete_current_stencil_fraction,
            "psi_snapshot_prefix": self.psi_snapshot_prefix,
            "fdm_dual_soliton": (
                None
                if self.fdm_dual_soliton_ic is None
                else {
                    "enabled": self.fdm_dual_soliton_ic,
                    "profile_c": self.fdm_dual_soliton_profile_c,
                    "rho0_code": list(self.fdm_dual_soliton_rho0 or ()),
                    "core_radius_box": list(self.fdm_dual_soliton_rc_box or ()),
                    "centres_box": [
                        list(value) for value in self.fdm_dual_soliton_centres_box or ()
                    ],
                    "velocities_code": [
                        list(value) for value in self.fdm_dual_soliton_velocities or ()
                    ],
                    "phase_radians": list(self.fdm_dual_soliton_phase or ()),
                }
            ),
        }


def read_lagramses_fdm_outer_wave_provenance(
    path: str | Path,
) -> LagRamsesFDMOuterWaveProvenance:
    """Parse one V1/V2 raw record and reject missing, duplicate, or altered keys."""

    resolved = Path(path).expanduser().resolve()
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read lagRamses FDM provenance: {error}") from error
    if not lines:
        raise ValueError("unsupported lagRamses FDM provenance schema")
    header = lines[0].strip()
    if header == "# fdm_outer_wave_provenance_v1":
        source_schema_version = 1
        required_keys = _V1_REQUIRED_KEYS
    elif header == "# fdm_outer_wave_provenance_v2":
        source_schema_version = 2
        required_keys = _V2_REQUIRED_KEYS
    else:
        raise ValueError("unsupported lagRamses FDM provenance schema")
    values: dict[str, str] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError("lagRamses FDM provenance has a malformed line")
        key, value = (part.strip() for part in stripped.split("=", 1))
        if key in values:
            raise ValueError(f"lagRamses FDM provenance duplicates {key}")
        values[key] = value
    missing = required_keys - set(values)
    unknown = set(values) - required_keys
    if missing or unknown:
        message: list[str] = []
        if missing:
            message.append(f"missing keys: {sorted(missing)}")
        if unknown:
            message.append(f"unknown keys: {sorted(unknown)}")
        raise ValueError("lagRamses FDM provenance " + "; ".join(message))
    current = _float_vector(
        values["integrated_current_code"], "integrated_current_code", length=3
    )
    leaf_count = _float(values["leaf_cell_count"], "leaf_cell_count", nonnegative=True)
    stencil_count = _float(
        values["complete_current_stencil_cell_count"],
        "complete_current_stencil_cell_count",
        nonnegative=True,
    )
    fraction = _float(
        values["complete_current_stencil_fraction"],
        "complete_current_stencil_fraction",
        nonnegative=True,
    )
    if fraction > 1.0 or stencil_count > leaf_count:
        raise ValueError("FDM current stencil coverage is inconsistent")
    if leaf_count == 0.0 and fraction != 0.0:
        raise ValueError("empty FDM provenance cannot have current coverage")
    if leaf_count > 0.0 and not math.isclose(
        fraction, stencil_count / leaf_count, rel_tol=1.0e-10, abs_tol=1.0e-12
    ):
        raise ValueError("FDM current stencil fraction does not match its counts")
    snapshot_prefix = values["psi_snapshot_prefix"]
    if not snapshot_prefix or "/" in snapshot_prefix or "\\" in snapshot_prefix:
        raise ValueError("psi_snapshot_prefix must be a basename")
    dual_enabled: bool | None = None
    dual_profile_c: float | None = None
    dual_rho0: tuple[float, float] | None = None
    dual_rc_box: tuple[float, float] | None = None
    dual_centres: tuple[tuple[float, float, float], ...] | None = None
    dual_velocities: tuple[tuple[float, float, float], ...] | None = None
    dual_phase: tuple[float, float] | None = None
    if source_schema_version == 2:
        dual_enabled = _logical(values["fdm_dual_soliton_ic"], "fdm_dual_soliton_ic")
        dual_profile_c = _float(
            values["fdm_dual_soliton_profile_c"],
            "fdm_dual_soliton_profile_c",
            nonnegative=True,
        )
        dual_rho0 = _float_vector(
            values["fdm_dual_soliton_rho0"],
            "fdm_dual_soliton_rho0",
            length=2,
            nonnegative=True,
        )
        dual_rc_box = _float_vector(
            values["fdm_dual_soliton_rc_box"],
            "fdm_dual_soliton_rc_box",
            length=2,
            nonnegative=True,
        )
        dual_centres = tuple(
            _float_vector(
                values[f"fdm_dual_soliton_center_box_{index}"],
                f"fdm_dual_soliton_center_box_{index}",
                length=3,
            )
            for index in (1, 2)
        )
        dual_velocities = tuple(
            _float_vector(
                values[f"fdm_dual_soliton_velocity_{index}"],
                f"fdm_dual_soliton_velocity_{index}",
                length=3,
            )
            for index in (1, 2)
        )
        dual_phase = _float_vector(
            values["fdm_dual_soliton_phase"], "fdm_dual_soliton_phase", length=2
        )
        if dual_enabled and (
            dual_profile_c <= 0.0
            or any(value <= 0.0 for value in dual_rho0)
            or any(value <= 0.0 or value >= 0.5 for value in dual_rc_box)
            or any(
                value < 0.0 or value >= 1.0
                for centre in dual_centres
                for value in centre
            )
        ):
            raise ValueError("enabled dual-soliton provenance has invalid component parameters")

    return LagRamsesFDMOuterWaveProvenance(
        source_path=resolved,
        source_sha256=_file_sha256(resolved),
        source_schema_version=source_schema_version,
        time_code=_float(values["time_code"], "time_code"),
        aexp=_float(values["aexp"], "aexp", nonnegative=True),
        nstep_coarse=_integer(values["nstep_coarse"], "nstep_coarse", nonnegative=True),
        m_axion_ev=_float(values["m_axion_ev"], "m_axion_ev", nonnegative=True),
        hbar_code=_float(values["hbar_code"], "hbar_code", nonnegative=True),
        fdm_use_hjm=_logical(values["fdm_use_hjm"], "fdm_use_hjm"),
        fdm_first_wave_level=_integer(
            values["fdm_first_wave_level"], "fdm_first_wave_level", nonnegative=True
        ),
        analytic_fdm_drag_enabled=_logical(
            values["analytic_fdm_drag_enabled"], "analytic_fdm_drag_enabled"
        ),
        force_accounting=values["force_accounting"],
        leaf_mass_code=_float(values["leaf_mass_code"], "leaf_mass_code", nonnegative=True),
        integrated_current_code=current,
        leaf_cell_count=leaf_count,
        complete_current_stencil_cell_count=stencil_count,
        complete_current_stencil_fraction=fraction,
        psi_snapshot_prefix=snapshot_prefix,
        fdm_dual_soliton_ic=dual_enabled,
        fdm_dual_soliton_profile_c=dual_profile_c,
        fdm_dual_soliton_rho0=dual_rho0,  # type: ignore[arg-type]
        fdm_dual_soliton_rc_box=dual_rc_box,  # type: ignore[arg-type]
        fdm_dual_soliton_centres_box=dual_centres,
        fdm_dual_soliton_velocities=dual_velocities,
        fdm_dual_soliton_phase=dual_phase,  # type: ignore[arg-type]
    )
