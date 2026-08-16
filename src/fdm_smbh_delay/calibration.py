"""Parameter-space definitions for live-wave SMBH calibration runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable, Mapping

import numpy as np

from .constants import G_INTERNAL, KM_S_TO_PC_MYR
from .exchange_scaling import schrodinger_poisson_similarity_parameter
from .literature import (
    boey2025_schive_soliton,
    koo2024_equivalent_schive_soliton,
)
from .orbit import make_orbital_state
from .soliton import SchiveSoliton


@dataclass(frozen=True)
class WaveCalibrationCase:
    case_id: str
    tier: int
    origin: str
    particle_mass_ev: float
    soliton_mass_msun: float
    core_radius_pc: float
    central_density_msun_pc3: float
    schrodinger_poisson_similarity_parameter: float
    mass_ratio_q: float
    eccentricity: float
    binary_to_soliton_mass: float
    semi_major_axis_over_core_radius: float
    mass1_msun: float
    mass2_msun: float
    semi_major_axis_pc: float
    initial_separation_pc: float
    initial_speed1_km_s: float
    initial_speed2_km_s: float
    kepler_period_myr: float
    target_duration_myr: float
    output_cadence_myr: float

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class WaveRunSpecification:
    run_id: str
    case_id: str
    tier: int
    effective_grid_cells: int
    box_size_pc: float
    finest_cell_size_pc: float
    core_cells: float
    plummer_radius_pc: float
    estimated_uniform_grid_memory_gib: float
    analytic_fdm_drag: bool = False
    required_solver: str = "coupled_schrodinger_poisson_with_moving_smbhs"

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class ExchangeCalibrationEligibility:
    secular: bool
    phase_dependent: bool


def exchange_calibration_eligibility(
    *,
    before_first_underresolved_orbit: bool,
    initial_resolved_energy_conservation_passed: bool,
    half_density_radius_spatially_resolved: bool,
    wave_mode_time_offset_over_orbital_period: float,
    maximum_wave_mode_time_offset_over_orbital_period: float = 0.5,
) -> ExchangeCalibrationEligibility:
    """Classify secular and phase-dependent live-wave calibration rows.

    Secular power and torque require a resolved binary, a resolved measured
    core, and an accepted Hamiltonian ledger.  A complex wave-mode coefficient
    additionally requires a nearby three-dimensional state because its phase
    cannot be assigned from a temporally distant snapshot.
    """

    maximum_offset = float(
        maximum_wave_mode_time_offset_over_orbital_period
    )
    if not np.isfinite(maximum_offset) or maximum_offset < 0.0:
        raise ValueError("the maximum wave-mode time offset must be non-negative")
    secular = bool(
        before_first_underresolved_orbit
        and initial_resolved_energy_conservation_passed
        and half_density_radius_spatially_resolved
    )
    offset = float(wave_mode_time_offset_over_orbital_period)
    phase_dependent = bool(
        secular and np.isfinite(offset) and abs(offset) <= maximum_offset
    )
    return ExchangeCalibrationEligibility(
        secular=secular,
        phase_dependent=phase_dependent,
    )


def component_masses(
    *, soliton_mass_msun: float, binary_to_soliton_mass: float, mass_ratio_q: float
) -> tuple[float, float]:
    if soliton_mass_msun <= 0.0 or binary_to_soliton_mass <= 0.0:
        raise ValueError("soliton and binary masses must be positive")
    if not 0.0 < mass_ratio_q <= 1.0:
        raise ValueError("mass_ratio_q must satisfy 0 < q <= 1")
    total_binary_mass = soliton_mass_msun * binary_to_soliton_mass
    mass1 = total_binary_mass / (1.0 + mass_ratio_q)
    return mass1, mass_ratio_q * mass1


def build_wave_case(
    *,
    case_id: str,
    tier: int,
    origin: str,
    particle_mass_ev: float,
    soliton: SchiveSoliton,
    mass_ratio_q: float,
    eccentricity: float,
    binary_to_soliton_mass: float,
    semi_major_axis_over_core_radius: float,
    target_orbits: float = 200.0,
    minimum_duration_myr: float = 0.1,
    maximum_duration_myr: float = 1.0,
) -> WaveCalibrationCase:
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must satisfy 0 <= e < 1")
    if semi_major_axis_over_core_radius <= 0.0:
        raise ValueError("semi-major axis must be positive")
    mass1, mass2 = component_masses(
        soliton_mass_msun=soliton.total_mass_msun,
        binary_to_soliton_mass=binary_to_soliton_mass,
        mass_ratio_q=mass_ratio_q,
    )
    semi_major_axis = semi_major_axis_over_core_radius * soliton.core_radius_pc
    initial_separation = semi_major_axis * (1.0 + eccentricity)
    state = make_orbital_state(
        mass1_msun=mass1,
        mass2_msun=mass2,
        separation_pc=initial_separation,
        eccentricity=eccentricity,
        soliton=soliton,
    )
    kepler_period = 2.0 * np.pi * np.sqrt(
        semi_major_axis**3 / (G_INTERNAL * (mass1 + mass2))
    )
    duration = min(
        maximum_duration_myr,
        max(minimum_duration_myr, target_orbits * kepler_period),
    )
    return WaveCalibrationCase(
        case_id=case_id,
        tier=tier,
        origin=origin,
        particle_mass_ev=particle_mass_ev,
        soliton_mass_msun=soliton.total_mass_msun,
        core_radius_pc=soliton.core_radius_pc,
        central_density_msun_pc3=soliton.central_density_msun_pc3,
        schrodinger_poisson_similarity_parameter=(
            schrodinger_poisson_similarity_parameter(
                particle_mass_ev=particle_mass_ev,
                soliton_mass_msun=soliton.total_mass_msun,
                core_radius_pc=soliton.core_radius_pc,
            )
        ),
        mass_ratio_q=mass_ratio_q,
        eccentricity=eccentricity,
        binary_to_soliton_mass=binary_to_soliton_mass,
        semi_major_axis_over_core_radius=semi_major_axis_over_core_radius,
        mass1_msun=mass1,
        mass2_msun=mass2,
        semi_major_axis_pc=semi_major_axis,
        initial_separation_pc=initial_separation,
        initial_speed1_km_s=float(
            np.linalg.norm(state.velocity1_pc_myr) / KM_S_TO_PC_MYR
        ),
        initial_speed2_km_s=float(
            np.linalg.norm(state.velocity2_pc_myr) / KM_S_TO_PC_MYR
        ),
        kepler_period_myr=float(kepler_period),
        target_duration_myr=float(duration),
        output_cadence_myr=float(kepler_period / 32.0),
    )


def literature_anchor_cases() -> list[WaveCalibrationCase]:
    cases: list[WaveCalibrationCase] = []
    koo_soliton = koo2024_equivalent_schive_soliton()
    for mass_each in (0.6e8, 1.0e8, 1.5e8):
        binary_fraction = 2.0 * mass_each / koo_soliton.total_mass_msun
        cases.append(
            build_wave_case(
                case_id=f"koo_mbh{mass_each / 1.0e8:.1f}e8",
                tier=0,
                origin="koo2024",
                particle_mass_ev=1.0e-21,
                soliton=koo_soliton,
                mass_ratio_q=1.0,
                eccentricity=0.0,
                binary_to_soliton_mass=binary_fraction,
                semi_major_axis_over_core_radius=0.9 / koo_soliton.core_radius_pc,
                minimum_duration_myr=0.1,
                maximum_duration_myr=0.1,
            )
        )
    boey_soliton = boey2025_schive_soliton()
    for percent_each in (2, 5, 10):
        cases.append(
            build_wave_case(
                case_id=f"boey_each{percent_each:02d}pct",
                tier=0,
                origin="boey2025",
                particle_mass_ev=1.0e-21,
                soliton=boey_soliton,
                mass_ratio_q=1.0,
                eccentricity=0.0,
                binary_to_soliton_mass=2.0 * percent_each / 100.0,
                semi_major_axis_over_core_radius=3.0 / boey_soliton.core_radius_pc,
                minimum_duration_myr=0.8,
                maximum_duration_myr=0.8,
            )
        )
    return cases


def structured_parameter_cases(
    *,
    q_values: Iterable[float],
    eccentricities: Iterable[float],
    binary_mass_fractions: Iterable[float],
    semi_major_axis_over_core: Iterable[float],
    fiducial: tuple[float, float, float, float],
) -> list[WaveCalibrationCase]:
    soliton = boey2025_schive_soliton()
    cases: list[WaveCalibrationCase] = []
    for index, values in enumerate(
        product(q_values, eccentricities, binary_mass_fractions, semi_major_axis_over_core)
    ):
        differing_axes = sum(
            not np.isclose(value, reference)
            for value, reference in zip(values, fiducial, strict=True)
        )
        tier = 1 if differing_axes <= 1 else 2
        q, eccentricity, mass_fraction, a_over_core = values
        cases.append(
            build_wave_case(
                case_id=f"grid_{index:03d}",
                tier=tier,
                origin="structured_grid",
                particle_mass_ev=1.0e-21,
                soliton=soliton,
                mass_ratio_q=q,
                eccentricity=eccentricity,
                binary_to_soliton_mass=mass_fraction,
                semi_major_axis_over_core_radius=a_over_core,
            )
        )
    return cases


def designed_parameter_cases(
    definitions: Iterable[Mapping[str, float | int | str]],
) -> list[WaveCalibrationCase]:
    """Build an explicit sparse design for costly q-e-separation extensions."""

    soliton = boey2025_schive_soliton()
    cases = []
    case_ids: set[str] = set()
    for definition in definitions:
        case_id = str(definition["case_id"])
        if not case_id or case_id in case_ids:
            raise ValueError("designed calibration case IDs must be unique")
        case_ids.add(case_id)
        cases.append(
            build_wave_case(
                case_id=case_id,
                tier=int(definition.get("tier", 1)),
                origin=str(definition.get("origin", "designed_qe_extension")),
                particle_mass_ev=float(
                    definition.get("particle_mass_ev", 1.0e-21)
                ),
                soliton=soliton,
                mass_ratio_q=float(definition["mass_ratio_q"]),
                eccentricity=float(definition["eccentricity"]),
                binary_to_soliton_mass=float(
                    definition["binary_to_soliton_mass"]
                ),
                semi_major_axis_over_core_radius=float(
                    definition["semi_major_axis_over_core_radius"]
                ),
                target_orbits=float(definition.get("target_orbits", 12.0)),
                minimum_duration_myr=float(
                    definition.get("minimum_duration_myr", 0.01)
                ),
                maximum_duration_myr=float(
                    definition.get("maximum_duration_myr", 0.2)
                ),
            )
        )
    if not cases:
        raise ValueError("designed calibration grid cannot be empty")
    return cases


def estimated_uniform_grid_memory_gib(
    effective_grid_cells: int, live_arrays: int = 16
) -> float:
    """Conservative memory scale for double-precision real work arrays."""

    if effective_grid_cells < 1 or live_arrays < 1:
        raise ValueError("grid size and live-array count must be positive")
    return float(effective_grid_cells**3 * live_arrays * 8 / 1024**3)


def run_specifications(
    cases: Iterable[WaveCalibrationCase],
    *,
    box_over_core_radius: float,
    resolutions_by_tier: dict[int, Iterable[int]],
) -> list[WaveRunSpecification]:
    if box_over_core_radius <= 0.0:
        raise ValueError("box_over_core_radius must be positive")
    specifications: list[WaveRunSpecification] = []
    for case in cases:
        box_size = box_over_core_radius * case.core_radius_pc
        for resolution in resolutions_by_tier[case.tier]:
            cell_size = box_size / resolution
            specifications.append(
                WaveRunSpecification(
                    run_id=f"{case.case_id}_n{resolution}",
                    case_id=case.case_id,
                    tier=case.tier,
                    effective_grid_cells=int(resolution),
                    box_size_pc=box_size,
                    finest_cell_size_pc=cell_size,
                    core_cells=case.core_radius_pc / cell_size,
                    plummer_radius_pc=max(0.001, 0.5 * cell_size),
                    estimated_uniform_grid_memory_gib=estimated_uniform_grid_memory_gib(
                        int(resolution)
                    ),
                )
            )
    return specifications
