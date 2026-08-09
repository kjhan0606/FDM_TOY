"""YAML configuration schema and construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy import units as u

from .soliton import SchiveSoliton
from .units import parse_quantity, parse_vector, require_finite_positive


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _float(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


@dataclass(frozen=True)
class ModelConfig:
    name: str
    alpha_df: float
    drag: bool
    fdm_bulk_velocity_pc_myr: np.ndarray
    velocity_floor_pc_myr: float


@dataclass(frozen=True)
class BinaryConfig:
    mass1_msun: float
    mass2_msun: float
    separation_pc: float | None
    eccentricity: float
    orbit: str
    position1_pc: np.ndarray | None
    position2_pc: np.ndarray | None
    velocity1_pc_myr: np.ndarray | None
    velocity2_pc_myr: np.ndarray | None

    @property
    def has_explicit_state(self) -> bool:
        return self.position1_pc is not None


@dataclass(frozen=True)
class FDMConfig:
    particle_mass_ev: float
    core_radius_pc: float
    profile: str
    mass_definition: str
    soliton_mass_msun: float | None
    central_density_msun_pc3: float | None

    def build_soliton(self) -> SchiveSoliton:
        if self.profile != "schive_fit":
            raise ValueError(f"unsupported profile={self.profile!r}")
        if self.soliton_mass_msun is not None:
            return SchiveSoliton.from_mass(
                self.soliton_mass_msun,
                self.core_radius_pc,
                self.mass_definition,
            )
        assert self.central_density_msun_pc3 is not None
        return SchiveSoliton.from_central_density(
            self.central_density_msun_pc3,
            self.core_radius_pc,
        )


@dataclass(frozen=True)
class IntegrationConfig:
    stop_separation_pc: float
    max_time_myr: float
    output_samples: int
    rtol: float
    atol: float
    max_step_myr: float
    energy_budget_relerr_limit: float


@dataclass(frozen=True)
class CaseConfig:
    model: ModelConfig
    binary: BinaryConfig
    fdm: FDMConfig
    integration: IntegrationConfig
    raw: dict[str, Any]
    source_path: Path | None = None


def _parse_model(raw: dict[str, Any]) -> ModelConfig:
    name = str(raw.get("name", "wave_df_3d"))
    if name != "wave_df_3d":
        raise ValueError("model.name must be 'wave_df_3d'")
    alpha = _float(raw.get("alpha_df", 0.341), "model.alpha_df", minimum=0.0)
    if alpha == 0.0:
        raise ValueError("model.alpha_df must be positive")
    drag = raw.get("drag", True)
    if not isinstance(drag, bool):
        raise ValueError("model.drag must be boolean")
    bulk_raw = raw.get(
        "fdm_bulk_velocity", ["0 km/s", "0 km/s", "0 km/s"]
    )
    bulk = parse_vector(bulk_raw, u.pc / u.Myr, "model.fdm_bulk_velocity")
    floor = parse_quantity(
        raw.get("velocity_floor", "1e-12 km/s"),
        u.pc / u.Myr,
        "model.velocity_floor",
    )
    require_finite_positive(floor, "model.velocity_floor")
    return ModelConfig(name, alpha, drag, bulk, floor)


def _parse_binary(raw: dict[str, Any]) -> BinaryConfig:
    m1 = parse_quantity(raw.get("M1"), u.Msun, "binary.M1")
    m2 = parse_quantity(raw.get("M2"), u.Msun, "binary.M2")
    require_finite_positive(m1, "binary.M1")
    require_finite_positive(m2, "binary.M2")

    explicit_keys = ("r1", "r2", "v1", "v2")
    any_explicit = any(key in raw for key in explicit_keys)
    if any_explicit and not all(key in raw for key in explicit_keys):
        raise ValueError("binary explicit state requires r1, r2, v1, and v2")

    eccentricity = _float(raw.get("eccentricity", 0.0), "binary.eccentricity", minimum=0.0)
    if eccentricity >= 1.0:
        raise ValueError("binary.eccentricity must be < 1 for a bound initializer")

    if any_explicit:
        r1 = parse_vector(raw["r1"], u.pc, "binary.r1")
        r2 = parse_vector(raw["r2"], u.pc, "binary.r2")
        v1 = parse_vector(raw["v1"], u.pc / u.Myr, "binary.v1")
        v2 = parse_vector(raw["v2"], u.pc / u.Myr, "binary.v2")
        separation = float(np.linalg.norm(r1 - r2))
        require_finite_positive(separation, "binary explicit separation")
        return BinaryConfig(m1, m2, separation, eccentricity, "explicit", r1, r2, v1, v2)

    separation = parse_quantity(raw.get("separation"), u.pc, "binary.separation")
    require_finite_positive(separation, "binary.separation")
    orbit = str(raw.get("orbit", "circular"))
    if orbit not in {"circular", "apocentre"}:
        raise ValueError("binary.orbit must be 'circular' or 'apocentre'")
    if orbit == "circular" and eccentricity != 0.0:
        raise ValueError("binary.orbit='circular' requires eccentricity=0")
    return BinaryConfig(m1, m2, separation, eccentricity, orbit, None, None, None, None)


def _parse_fdm(raw: dict[str, Any]) -> FDMConfig:
    particle_mass = parse_quantity(raw.get("particle_mass"), u.eV, "fdm.particle_mass")
    core_radius = parse_quantity(raw.get("core_radius"), u.pc, "fdm.core_radius")
    require_finite_positive(particle_mass, "fdm.particle_mass")
    require_finite_positive(core_radius, "fdm.core_radius")
    profile = str(raw.get("profile", "schive_fit"))

    has_mass = "soliton_mass" in raw
    has_density = "central_density" in raw
    if has_mass == has_density:
        raise ValueError("fdm requires exactly one of soliton_mass or central_density")
    if has_mass:
        soliton_mass = parse_quantity(raw["soliton_mass"], u.Msun, "fdm.soliton_mass")
        require_finite_positive(soliton_mass, "fdm.soliton_mass")
        mass_definition = str(raw.get("mass_definition", ""))
        if mass_definition not in {"total_profile", "within_rc"}:
            raise ValueError(
                "fdm.mass_definition must be 'total_profile' or 'within_rc'"
            )
        central_density = None
    else:
        central_density = parse_quantity(
            raw["central_density"], u.Msun / u.pc**3, "fdm.central_density"
        )
        require_finite_positive(central_density, "fdm.central_density")
        soliton_mass = None
        mass_definition = "central_density"
    return FDMConfig(
        particle_mass,
        core_radius,
        profile,
        mass_definition,
        soliton_mass,
        central_density,
    )


def _parse_integration(raw: dict[str, Any]) -> IntegrationConfig:
    stop = parse_quantity(
        raw.get("stop_separation", "0.01 pc"), u.pc, "integration.stop_separation"
    )
    max_time = parse_quantity(raw.get("max_time"), u.Myr, "integration.max_time")
    require_finite_positive(stop, "integration.stop_separation")
    require_finite_positive(max_time, "integration.max_time")
    output_samples = raw.get("output_samples", 1000)
    if isinstance(output_samples, bool) or not isinstance(output_samples, int) or output_samples < 2:
        raise ValueError("integration.output_samples must be an integer >= 2")
    rtol = _float(raw.get("rtol", 1.0e-9), "integration.rtol", minimum=0.0)
    atol = _float(raw.get("atol", 1.0e-12), "integration.atol", minimum=0.0)
    if rtol == 0.0 or atol == 0.0:
        raise ValueError("integration tolerances must be positive")
    if "max_step" in raw:
        max_step = parse_quantity(raw["max_step"], u.Myr, "integration.max_step")
        require_finite_positive(max_step, "integration.max_step")
    else:
        max_step = float("inf")
    energy_limit = _float(
        raw.get("energy_budget_relerr_limit", 1.0e-6),
        "integration.energy_budget_relerr_limit",
        minimum=0.0,
    )
    return IntegrationConfig(stop, max_time, output_samples, rtol, atol, max_step, energy_limit)


def load_config(path: str | Path) -> CaseConfig:
    """Load and validate a YAML case."""

    source_path = Path(path).expanduser().resolve()
    with source_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    raw = _mapping(loaded, "configuration")
    for section in ("model", "binary", "fdm", "integration"):
        if section not in raw:
            raise ValueError(f"missing required section: {section}")
    return CaseConfig(
        model=_parse_model(_mapping(raw["model"], "model")),
        binary=_parse_binary(_mapping(raw["binary"], "binary")),
        fdm=_parse_fdm(_mapping(raw["fdm"], "fdm")),
        integration=_parse_integration(_mapping(raw["integration"], "integration")),
        raw=raw,
        source_path=source_path,
    )
