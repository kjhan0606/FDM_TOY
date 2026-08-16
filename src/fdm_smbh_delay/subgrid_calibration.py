"""Interpolation and conservative application of converged subgrid rates."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .exchange_scaling import ExchangeScales, exchange_scales
from .orbital_exchange import (
    FiniteOrbitalExchangeStep,
    advance_keplerian_exchange,
)


ACCEPTED_STATUS = "accepted_with_spatial_systematic"


@dataclass(frozen=True)
class SubgridCalibrationRow:
    """One accepted matched-separation measurement."""

    profile_id: str
    source_case_id: str
    schrodinger_poisson_similarity_parameter: float
    binary_to_soliton_mass: float
    separation_bin_index: int
    lower_separation_over_core_radius: float
    upper_separation_over_core_radius: float
    reference_mean_separation_over_core_radius: float
    dimensionless_orbital_power: float
    dimensionless_orbital_torque: float
    dimensionless_wave_total_energy_rate: float
    orbital_power_spatial_systematic_fraction: float
    orbital_torque_spatial_systematic_fraction: float
    wave_total_spatial_systematic_fraction: float
    reference_resolution: int
    comparison_resolution: int
    reference_complete_orbits: int
    comparison_complete_orbits: int
    reference_minimum_half_density_radius_over_cell_size: float
    comparison_minimum_half_density_radius_over_cell_size: float
    convergence_status: str = ACCEPTED_STATUS

    def __post_init__(self) -> None:
        scalar_values = np.asarray(
            [
                self.schrodinger_poisson_similarity_parameter,
                self.binary_to_soliton_mass,
                self.lower_separation_over_core_radius,
                self.upper_separation_over_core_radius,
                self.reference_mean_separation_over_core_radius,
                self.dimensionless_orbital_power,
                self.dimensionless_orbital_torque,
                self.dimensionless_wave_total_energy_rate,
                self.orbital_power_spatial_systematic_fraction,
                self.orbital_torque_spatial_systematic_fraction,
                self.wave_total_spatial_systematic_fraction,
                self.reference_minimum_half_density_radius_over_cell_size,
                self.comparison_minimum_half_density_radius_over_cell_size,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(scalar_values)):
            raise ValueError("subgrid calibration rows must be finite")
        if not self.profile_id or not self.source_case_id:
            raise ValueError("profile and source case identifiers are required")
        if self.convergence_status != ACCEPTED_STATUS:
            raise ValueError("subgrid rows must pass the spatial convergence gate")
        if (
            self.schrodinger_poisson_similarity_parameter <= 0.0
            or self.binary_to_soliton_mass <= 0.0
            or self.lower_separation_over_core_radius <= 0.0
            or self.upper_separation_over_core_radius
            <= self.lower_separation_over_core_radius
            or not self.lower_separation_over_core_radius
            <= self.reference_mean_separation_over_core_radius
            <= self.upper_separation_over_core_radius
        ):
            raise ValueError("subgrid row axes are invalid")
        if (
            self.separation_bin_index < 0
            or self.reference_resolution <= self.comparison_resolution
            or self.comparison_resolution <= 0
            or self.reference_complete_orbits < 8
            or self.comparison_complete_orbits < 8
            or self.reference_minimum_half_density_radius_over_cell_size < 2.0
            or self.comparison_minimum_half_density_radius_over_cell_size < 2.0
        ):
            raise ValueError("subgrid row numerical metadata are invalid")
        systematic = np.asarray(
            [
                self.orbital_power_spatial_systematic_fraction,
                self.orbital_torque_spatial_systematic_fraction,
                self.wave_total_spatial_systematic_fraction,
            ]
        )
        if np.any(systematic < 0.0):
            raise ValueError("spatial systematic fractions must be non-negative")


@dataclass(frozen=True)
class InterpolatedSubgridRates:
    """Dimensionless exchange rates at one point inside the measured domain."""

    profile_id: str
    schrodinger_poisson_similarity_parameter: float
    binary_to_soliton_mass: float
    separation_over_core_radius: float
    dimensionless_orbital_power: float
    dimensionless_orbital_torque: float
    dimensionless_wave_total_energy_rate: float
    orbital_power_spatial_systematic_fraction: float
    orbital_torque_spatial_systematic_fraction: float
    wave_total_spatial_systematic_fraction: float


@dataclass(frozen=True)
class PhysicalSubgridRates:
    """Interpolated target rates in the internal physical units."""

    dimensionless: InterpolatedSubgridRates
    scales: ExchangeScales
    orbital_power: float
    orbital_torque: float
    wave_total_energy_rate: float
    orbital_power_spatial_systematic: float
    orbital_torque_spatial_systematic: float
    wave_total_spatial_systematic: float


@dataclass(frozen=True)
class ResidualOrbitalRates:
    """Target rates minus work and torque already supplied by the live wave."""

    target_orbital_power: float
    target_orbital_torque: float
    resolved_orbital_power: float
    resolved_orbital_torque: float
    residual_orbital_power: float
    residual_orbital_torque: float


@dataclass(frozen=True)
class CalibratedOrbitalStep:
    """One finite target-rate update with explicit conservation residuals."""

    rates: PhysicalSubgridRates
    exchange: FiniteOrbitalExchangeStep
    energy_closure_error: float
    angular_momentum_closure_error: float
    energy_closure_relative_to_exchange: float
    angular_momentum_closure_relative_to_exchange: float


def _linear(value0: float, value1: float, weight: float) -> float:
    return float((1.0 - weight) * value0 + weight * value1)


def _interpolate_rate_rows(
    lower: SubgridCalibrationRow,
    upper: SubgridCalibrationRow,
    weight: float,
    *,
    binary_mass_fraction: float,
    separation_ratio: float,
) -> InterpolatedSubgridRates:
    if lower.profile_id != upper.profile_id:
        raise ValueError("calibration profiles cannot be mixed")
    similarity = _linear(
        lower.schrodinger_poisson_similarity_parameter,
        upper.schrodinger_poisson_similarity_parameter,
        weight,
    )
    return InterpolatedSubgridRates(
        profile_id=lower.profile_id,
        schrodinger_poisson_similarity_parameter=similarity,
        binary_to_soliton_mass=float(binary_mass_fraction),
        separation_over_core_radius=float(separation_ratio),
        dimensionless_orbital_power=_linear(
            lower.dimensionless_orbital_power,
            upper.dimensionless_orbital_power,
            weight,
        ),
        dimensionless_orbital_torque=_linear(
            lower.dimensionless_orbital_torque,
            upper.dimensionless_orbital_torque,
            weight,
        ),
        dimensionless_wave_total_energy_rate=_linear(
            lower.dimensionless_wave_total_energy_rate,
            upper.dimensionless_wave_total_energy_rate,
            weight,
        ),
        orbital_power_spatial_systematic_fraction=max(
            lower.orbital_power_spatial_systematic_fraction,
            upper.orbital_power_spatial_systematic_fraction,
        ),
        orbital_torque_spatial_systematic_fraction=max(
            lower.orbital_torque_spatial_systematic_fraction,
            upper.orbital_torque_spatial_systematic_fraction,
        ),
        wave_total_spatial_systematic_fraction=max(
            lower.wave_total_spatial_systematic_fraction,
            upper.wave_total_spatial_systematic_fraction,
        ),
    )


class SubgridCalibrationTable:
    """Strict interpolation over accepted profile, mass, and separation rows."""

    def __init__(self, rows: Iterable[SubgridCalibrationRow]) -> None:
        self.rows = tuple(rows)
        if not self.rows:
            raise ValueError("a subgrid calibration table cannot be empty")
        keys = [
            (
                row.profile_id,
                row.binary_to_soliton_mass,
                row.separation_bin_index,
            )
            for row in self.rows
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("subgrid calibration rows must be unique")
        profile_similarity: dict[str, list[float]] = {}
        for row in self.rows:
            profile_similarity.setdefault(row.profile_id, []).append(
                row.schrodinger_poisson_similarity_parameter
            )
        for profile, values in profile_similarity.items():
            if not np.allclose(values, values[0], rtol=1.0e-10, atol=0.0):
                raise ValueError(
                    f"profile {profile} mixes similarity parameters"
                )

    @classmethod
    def from_csv(cls, path: Path) -> "SubgridCalibrationTable":
        resolved = path.expanduser().resolve()
        with resolved.open(newline="", encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
        rows = []
        for record in records:
            rows.append(
                SubgridCalibrationRow(
                    profile_id=record["profile_id"],
                    source_case_id=record["source_case_id"],
                    schrodinger_poisson_similarity_parameter=float(
                        record["schrodinger_poisson_similarity_parameter"]
                    ),
                    binary_to_soliton_mass=float(
                        record["binary_to_soliton_mass"]
                    ),
                    separation_bin_index=int(record["separation_bin_index"]),
                    lower_separation_over_core_radius=float(
                        record["lower_separation_over_core_radius"]
                    ),
                    upper_separation_over_core_radius=float(
                        record["upper_separation_over_core_radius"]
                    ),
                    reference_mean_separation_over_core_radius=float(
                        record["reference_mean_separation_over_core_radius"]
                    ),
                    dimensionless_orbital_power=float(
                        record["dimensionless_orbital_power"]
                    ),
                    dimensionless_orbital_torque=float(
                        record["dimensionless_orbital_torque"]
                    ),
                    dimensionless_wave_total_energy_rate=float(
                        record["dimensionless_wave_total_energy_rate"]
                    ),
                    orbital_power_spatial_systematic_fraction=float(
                        record[
                            "orbital_power_spatial_systematic_fraction"
                        ]
                    ),
                    orbital_torque_spatial_systematic_fraction=float(
                        record[
                            "orbital_torque_spatial_systematic_fraction"
                        ]
                    ),
                    wave_total_spatial_systematic_fraction=float(
                        record["wave_total_spatial_systematic_fraction"]
                    ),
                    reference_resolution=int(record["reference_resolution"]),
                    comparison_resolution=int(record["comparison_resolution"]),
                    reference_complete_orbits=int(
                        record["reference_complete_orbits"]
                    ),
                    comparison_complete_orbits=int(
                        record["comparison_complete_orbits"]
                    ),
                    reference_minimum_half_density_radius_over_cell_size=float(
                        record[
                            "reference_minimum_half_density_radius_over_cell_size"
                        ]
                    ),
                    comparison_minimum_half_density_radius_over_cell_size=float(
                        record[
                            "comparison_minimum_half_density_radius_over_cell_size"
                        ]
                    ),
                    convergence_status=record["convergence_status"],
                )
            )
        return cls(rows)

    def _mass_plane(
        self,
        *,
        profile_id: str,
        binary_mass_fraction: float,
    ) -> tuple[SubgridCalibrationRow, ...]:
        rows = tuple(
            sorted(
                (
                    row
                    for row in self.rows
                    if row.profile_id == profile_id
                    and np.isclose(
                        row.binary_to_soliton_mass,
                        binary_mass_fraction,
                        rtol=1.0e-12,
                        atol=0.0,
                    )
                ),
                key=lambda row: row.reference_mean_separation_over_core_radius,
            )
        )
        if not rows:
            raise ValueError("requested mass plane is absent")
        return rows

    def _interpolate_separation(
        self,
        rows: tuple[SubgridCalibrationRow, ...],
        separation_ratio: float,
    ) -> InterpolatedSubgridRates:
        if (
            separation_ratio < rows[0].lower_separation_over_core_radius
            or separation_ratio > rows[-1].upper_separation_over_core_radius
        ):
            raise ValueError("separation lies outside the calibrated range")
        centres = np.asarray(
            [row.reference_mean_separation_over_core_radius for row in rows]
        )
        if separation_ratio <= centres[0]:
            lower = upper = rows[0]
            weight = 0.0
        elif separation_ratio >= centres[-1]:
            lower = upper = rows[-1]
            weight = 0.0
        else:
            upper_index = int(np.searchsorted(centres, separation_ratio))
            lower = rows[upper_index - 1]
            upper = rows[upper_index]
            if (
                upper.separation_bin_index
                != lower.separation_bin_index + 1
                or upper.lower_separation_over_core_radius
                > lower.upper_separation_over_core_radius + 1.0e-12
            ):
                raise ValueError("separation interpolation crosses an unmeasured gap")
            denominator = (
                upper.reference_mean_separation_over_core_radius
                - lower.reference_mean_separation_over_core_radius
            )
            weight = (
                separation_ratio
                - lower.reference_mean_separation_over_core_radius
            ) / denominator
        return _interpolate_rate_rows(
            lower,
            upper,
            weight,
            binary_mass_fraction=lower.binary_to_soliton_mass,
            separation_ratio=separation_ratio,
        )

    def interpolate(
        self,
        *,
        profile_id: str,
        binary_to_soliton_mass: float,
        separation_over_core_radius: float,
    ) -> InterpolatedSubgridRates:
        values = np.asarray(
            [binary_to_soliton_mass, separation_over_core_radius], dtype=float
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("subgrid interpolation coordinates must be positive")
        masses = sorted(
            {
                row.binary_to_soliton_mass
                for row in self.rows
                if row.profile_id == profile_id
            }
        )
        if not masses:
            raise ValueError(f"unknown calibration profile: {profile_id}")
        exact = next(
            (
                mass
                for mass in masses
                if np.isclose(
                    mass,
                    binary_to_soliton_mass,
                    rtol=1.0e-12,
                    atol=0.0,
                )
            ),
            None,
        )
        if exact is not None:
            return self._interpolate_separation(
                self._mass_plane(
                    profile_id=profile_id,
                    binary_mass_fraction=exact,
                ),
                separation_over_core_radius,
            )
        if binary_to_soliton_mass < masses[0] or binary_to_soliton_mass > masses[-1]:
            raise ValueError("binary mass fraction lies outside the calibrated range")
        upper_index = int(np.searchsorted(masses, binary_to_soliton_mass))
        lower_mass = masses[upper_index - 1]
        upper_mass = masses[upper_index]
        lower_rates = self._interpolate_separation(
            self._mass_plane(
                profile_id=profile_id,
                binary_mass_fraction=lower_mass,
            ),
            separation_over_core_radius,
        )
        upper_rates = self._interpolate_separation(
            self._mass_plane(
                profile_id=profile_id,
                binary_mass_fraction=upper_mass,
            ),
            separation_over_core_radius,
        )
        weight = (binary_to_soliton_mass - lower_mass) / (
            upper_mass - lower_mass
        )
        return InterpolatedSubgridRates(
            profile_id=profile_id,
            schrodinger_poisson_similarity_parameter=_linear(
                lower_rates.schrodinger_poisson_similarity_parameter,
                upper_rates.schrodinger_poisson_similarity_parameter,
                weight,
            ),
            binary_to_soliton_mass=float(binary_to_soliton_mass),
            separation_over_core_radius=float(separation_over_core_radius),
            dimensionless_orbital_power=_linear(
                lower_rates.dimensionless_orbital_power,
                upper_rates.dimensionless_orbital_power,
                weight,
            ),
            dimensionless_orbital_torque=_linear(
                lower_rates.dimensionless_orbital_torque,
                upper_rates.dimensionless_orbital_torque,
                weight,
            ),
            dimensionless_wave_total_energy_rate=_linear(
                lower_rates.dimensionless_wave_total_energy_rate,
                upper_rates.dimensionless_wave_total_energy_rate,
                weight,
            ),
            orbital_power_spatial_systematic_fraction=max(
                lower_rates.orbital_power_spatial_systematic_fraction,
                upper_rates.orbital_power_spatial_systematic_fraction,
            ),
            orbital_torque_spatial_systematic_fraction=max(
                lower_rates.orbital_torque_spatial_systematic_fraction,
                upper_rates.orbital_torque_spatial_systematic_fraction,
            ),
            wave_total_spatial_systematic_fraction=max(
                lower_rates.wave_total_spatial_systematic_fraction,
                upper_rates.wave_total_spatial_systematic_fraction,
            ),
        )


def physical_subgrid_rates(
    table: SubgridCalibrationTable,
    *,
    profile_id: str,
    mass1_msun: float,
    mass2_msun: float,
    soliton_mass_msun: float,
    core_radius_pc: float,
    separation_pc: float,
) -> PhysicalSubgridRates:
    dimensionless = table.interpolate(
        profile_id=profile_id,
        binary_to_soliton_mass=(mass1_msun + mass2_msun)
        / soliton_mass_msun,
        separation_over_core_radius=separation_pc / core_radius_pc,
    )
    scales = exchange_scales(
        mass1_msun=mass1_msun,
        mass2_msun=mass2_msun,
        soliton_mass_msun=soliton_mass_msun,
        core_radius_pc=core_radius_pc,
    )
    power = (
        dimensionless.dimensionless_orbital_power
        * scales.orbital_power_msun_pc2_myr3
    )
    torque = (
        dimensionless.dimensionless_orbital_torque
        * scales.orbital_torque_msun_pc2_myr2
    )
    wave_total = (
        dimensionless.dimensionless_wave_total_energy_rate
        * scales.orbital_power_msun_pc2_myr3
    )
    return PhysicalSubgridRates(
        dimensionless=dimensionless,
        scales=scales,
        orbital_power=float(power),
        orbital_torque=float(torque),
        wave_total_energy_rate=float(wave_total),
        orbital_power_spatial_systematic=float(
            abs(power)
            * dimensionless.orbital_power_spatial_systematic_fraction
        ),
        orbital_torque_spatial_systematic=float(
            abs(torque)
            * dimensionless.orbital_torque_spatial_systematic_fraction
        ),
        wave_total_spatial_systematic=float(
            abs(wave_total)
            * dimensionless.wave_total_spatial_systematic_fraction
        ),
    )


def residual_orbital_rates(
    rates: PhysicalSubgridRates,
    *,
    resolved_orbital_power: float,
    resolved_orbital_torque: float,
) -> ResidualOrbitalRates:
    resolved = np.asarray(
        [resolved_orbital_power, resolved_orbital_torque], dtype=float
    )
    if np.any(~np.isfinite(resolved)):
        raise ValueError("resolved work and torque must be finite")
    return ResidualOrbitalRates(
        target_orbital_power=rates.orbital_power,
        target_orbital_torque=rates.orbital_torque,
        resolved_orbital_power=float(resolved_orbital_power),
        resolved_orbital_torque=float(resolved_orbital_torque),
        residual_orbital_power=float(
            rates.orbital_power - resolved_orbital_power
        ),
        residual_orbital_torque=float(
            rates.orbital_torque - resolved_orbital_torque
        ),
    )


def advance_calibrated_exchange(
    rates: PhysicalSubgridRates,
    *,
    mass1_msun: float,
    mass2_msun: float,
    semimajor_axis_pc: float,
    eccentricity: float,
    time_step_myr: float,
) -> CalibratedOrbitalStep:
    exchange = advance_keplerian_exchange(
        mass1_msun=mass1_msun,
        mass2_msun=mass2_msun,
        semimajor_axis_pc=semimajor_axis_pc,
        eccentricity=eccentricity,
        orbital_power=rates.orbital_power,
        orbital_torque=rates.orbital_torque,
        time_step_myr=time_step_myr,
    )
    energy_closure = (
        exchange.final_orbital_energy
        - exchange.initial_orbital_energy
        + exchange.wave_energy_increment
    )
    angular_momentum_closure = (
        exchange.final_orbital_angular_momentum
        - exchange.initial_orbital_angular_momentum
        + exchange.wave_angular_momentum_increment
    )
    energy_scale = max(
        abs(exchange.wave_energy_increment), np.finfo(float).tiny
    )
    angular_momentum_scale = max(
        abs(exchange.wave_angular_momentum_increment),
        np.finfo(float).tiny,
    )
    return CalibratedOrbitalStep(
        rates=rates,
        exchange=exchange,
        energy_closure_error=float(energy_closure),
        angular_momentum_closure_error=float(angular_momentum_closure),
        energy_closure_relative_to_exchange=float(
            energy_closure / energy_scale
        ),
        angular_momentum_closure_relative_to_exchange=float(
            angular_momentum_closure / angular_momentum_scale
        ),
    )
