"""Interpolation and conservative application of converged subgrid rates."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .exchange_scaling import ExchangeScales, exchange_scales
from .orbital_exchange import (
    FiniteOrbitalExchangeStep,
    advance_keplerian_exchange,
)


ACCEPTED_STATUS = "accepted_with_spatial_systematic"
ACCEPTED_TABLE_STATUS = "accepted_subgrid_calibration_table"
MAXIMUM_ACCEPTED_SPATIAL_SYSTEMATIC_FRACTION = 0.20
MAXIMUM_ACCEPTED_ENERGY_ERROR_OVER_TRANSFER = 0.01
MINIMUM_ACCEPTED_COMPLETE_ORBITS = 8
MINIMUM_ACCEPTED_CORE_RADIUS_CELLS = 2.0
CALIBRATION_INTERPOLATION_SPECIFICATION = {
    "profile_axis": "discrete_no_cross_profile_interpolation",
    "mass_axis": "piecewise_linear_binary_to_soliton_mass",
    "separation_axis": "piecewise_linear_reference_bin_centres",
    "outer_half_bins": "nearest_accepted_bin_value",
    "missing_separation_bins": "crossing_prohibited",
    "spatial_systematics": "maximum_of_all_bracketing_rows",
    "extrapolation": "prohibited",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            or self.reference_complete_orbits < MINIMUM_ACCEPTED_COMPLETE_ORBITS
            or self.comparison_complete_orbits < MINIMUM_ACCEPTED_COMPLETE_ORBITS
            or self.reference_minimum_half_density_radius_over_cell_size
            < MINIMUM_ACCEPTED_CORE_RADIUS_CELLS
            or self.comparison_minimum_half_density_radius_over_cell_size
            < MINIMUM_ACCEPTED_CORE_RADIUS_CELLS
        ):
            raise ValueError("subgrid row numerical metadata are invalid")
        systematic = np.asarray(
            [
                self.orbital_power_spatial_systematic_fraction,
                self.orbital_torque_spatial_systematic_fraction,
                self.wave_total_spatial_systematic_fraction,
            ]
        )
        if np.any(systematic < 0.0) or np.any(
            systematic > MAXIMUM_ACCEPTED_SPATIAL_SYSTEMATIC_FRACTION
        ):
            raise ValueError(
                "spatial systematic fractions lie outside the accepted range"
            )


def summarize_calibrated_domains(
    rows: Iterable[SubgridCalibrationRow],
) -> list[dict]:
    """Return deterministic mass-plane domains and maximum systematics."""

    grouped: dict[tuple[str, float], list[SubgridCalibrationRow]] = {}
    for row in rows:
        grouped.setdefault(
            (row.profile_id, row.binary_to_soliton_mass), []
        ).append(row)
    domains = []
    for (profile_id, mass_fraction), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: row.separation_bin_index)
        domains.append(
            {
                "profile_id": profile_id,
                "schrodinger_poisson_similarity_parameter": ordered[
                    0
                ].schrodinger_poisson_similarity_parameter,
                "binary_to_soliton_mass": mass_fraction,
                "source_case_ids": sorted(
                    {row.source_case_id for row in ordered}
                ),
                "accepted_separation_bin_indices": [
                    row.separation_bin_index for row in ordered
                ],
                "minimum_separation_over_core_radius": min(
                    row.lower_separation_over_core_radius for row in ordered
                ),
                "maximum_separation_over_core_radius": max(
                    row.upper_separation_over_core_radius for row in ordered
                ),
                "maximum_spatial_systematic_fraction": {
                    "orbital_power": max(
                        row.orbital_power_spatial_systematic_fraction
                        for row in ordered
                    ),
                    "orbital_torque": max(
                        row.orbital_torque_spatial_systematic_fraction
                        for row in ordered
                    ),
                    "wave_total_energy_rate": max(
                        row.wave_total_spatial_systematic_fraction
                        for row in ordered
                    ),
                },
            }
        )
    return domains


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
    residual: ResidualOrbitalRates
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

    @classmethod
    def from_release(cls, path: Path) -> "SubgridCalibrationTable":
        """Load only a committed CSV/summary pair with valid provenance.

        ``from_csv`` remains useful for in-memory fixtures and exploratory
        tables. Production use should call this method so a stop between the
        two file replacements cannot silently expose a mixed release.
        """

        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"subgrid release table is absent: {resolved}")
        summary_path = resolved.with_suffix(".summary.json")
        if not summary_path.is_file():
            raise ValueError(
                f"subgrid release commit sidecar is absent: {summary_path}"
            )
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"subgrid release commit sidecar is unreadable: {summary_path}"
            ) from error
        if (
            summary.get("status") != ACCEPTED_TABLE_STATUS
            or summary.get("schema_version") != 2
        ):
            raise ValueError("subgrid release status or schema is invalid")
        table_metadata = summary.get("table")
        if not isinstance(table_metadata, dict):
            raise ValueError("subgrid release table metadata is absent")
        if table_metadata.get("file") != resolved.name:
            raise ValueError("subgrid release table filename does not match")
        expected_sha256 = table_metadata.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or _sha256(resolved) != expected_sha256
        ):
            raise ValueError("subgrid release table checksum does not match")
        release_input_sha256 = summary.get("release_input_sha256")
        if (
            not isinstance(release_input_sha256, str)
            or len(release_input_sha256) != 64
            or table_metadata.get("release_input_sha256")
            != release_input_sha256
        ):
            raise ValueError("subgrid release input identity does not match")
        with resolved.open(newline="", encoding="utf-8") as stream:
            release_records = list(csv.DictReader(stream))
        release_ids = {
            record.get("release_input_sha256") for record in release_records
        }
        if release_ids != {release_input_sha256}:
            raise ValueError("subgrid release CSV input identity does not match")
        table = cls.from_csv(resolved)
        expected_rows = table_metadata.get("rows")
        if (
            not isinstance(expected_rows, int)
            or expected_rows != len(table.rows)
            or summary.get("rows") != len(table.rows)
        ):
            raise ValueError("subgrid release row count does not match")
        profiles = sorted({row.profile_id for row in table.rows})
        if summary.get("profiles") != profiles:
            raise ValueError("subgrid release profile list does not match")
        if summary.get("interpolation") != CALIBRATION_INTERPOLATION_SPECIFICATION:
            raise ValueError("subgrid release interpolation rules do not match")
        acceptance = summary.get("acceptance")
        if not isinstance(acceptance, dict):
            raise ValueError("subgrid release acceptance criteria are absent")
        try:
            maximum_spatial = float(
                acceptance["maximum_spatial_systematic_fraction"]
            )
            maximum_energy_error = float(
                acceptance["maximum_energy_error_over_transfer"]
            )
            minimum_orbits = int(
                acceptance["minimum_complete_orbits_per_bin"]
            )
            minimum_core_cells = float(acceptance["minimum_core_radius_cells"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "subgrid release acceptance criteria are invalid"
            ) from error
        acceptance_values = np.asarray(
            [maximum_spatial, maximum_energy_error, minimum_core_cells],
            dtype=float,
        )
        if (
            np.any(~np.isfinite(acceptance_values))
            or maximum_spatial
            > MAXIMUM_ACCEPTED_SPATIAL_SYSTEMATIC_FRACTION
            or maximum_energy_error
            > MAXIMUM_ACCEPTED_ENERGY_ERROR_OVER_TRANSFER
            or minimum_orbits < MINIMUM_ACCEPTED_COMPLETE_ORBITS
            or minimum_core_cells < MINIMUM_ACCEPTED_CORE_RADIUS_CELLS
            or acceptance.get("extrapolation") != "prohibited"
        ):
            raise ValueError("subgrid release acceptance criteria are unsafe")
        sources = summary.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("subgrid release provenance sources are absent")
        accepted_rows = 0
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("subgrid release provenance source is invalid")
            source_sha256 = source.get("source_sha256")
            source_rows = source.get("accepted_bins")
            input_files = source.get("inputs")
            if (
                not source.get("profile_id")
                or not source.get("source_case_id")
                or not source.get("convergence_summary")
                or not isinstance(source_sha256, str)
                or len(source_sha256) != 64
                or not isinstance(source_rows, int)
                or source_rows < 0
                or not isinstance(source.get("rejected_bins"), list)
                or not isinstance(input_files, list)
                or not input_files
            ):
                raise ValueError("subgrid release provenance source is invalid")
            input_roles = []
            for input_file in input_files:
                if not isinstance(input_file, dict):
                    raise ValueError(
                        "subgrid release provenance input is invalid"
                    )
                input_sha256 = input_file.get("sha256")
                if (
                    not input_file.get("role")
                    or not input_file.get("path")
                    or not isinstance(input_sha256, str)
                    or len(input_sha256) != 64
                ):
                    raise ValueError(
                        "subgrid release provenance input is invalid"
                    )
                input_roles.append(input_file["role"])
            if len(input_roles) != len(set(input_roles)):
                raise ValueError("subgrid release provenance roles are duplicated")
            required_roles = {
                "convergence_summary",
                "reference_metadata",
                "reference_config",
                "reference_wave_response",
                "comparison_metadata",
                "comparison_config",
                "comparison_wave_response",
            }
            if set(input_roles) != required_roles:
                raise ValueError("subgrid release provenance roles are incomplete")
            convergence_sha256 = next(
                input_file["sha256"]
                for input_file in input_files
                if input_file["role"] == "convergence_summary"
            )
            if source_sha256 != convergence_sha256:
                raise ValueError(
                    "subgrid release convergence checksum does not match"
                )
            accepted_rows += source_rows
        if accepted_rows != len(table.rows):
            raise ValueError("subgrid release provenance row count does not close")
        domains = summary.get("calibrated_domains")
        if domains != summarize_calibrated_domains(table.rows):
            raise ValueError("subgrid release calibrated domains do not match")
        return table

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


def find_mass_interpolation_witness(
    table: SubgridCalibrationTable,
    *,
    profile_id: str,
) -> InterpolatedSubgridRates | None:
    """Return one measured-overlap point that exercises mass interpolation.

    A pair of accepted mass planes alone is insufficient: their accepted
    separation intervals must overlap without crossing a rejected bin.  This
    helper supplies a concrete witness for release-level validation.
    """

    masses = sorted(
        {
            row.binary_to_soliton_mass
            for row in table.rows
            if row.profile_id == profile_id
        }
    )
    for lower_mass, upper_mass in zip(masses, masses[1:]):
        lower_rows = table._mass_plane(
            profile_id=profile_id,
            binary_mass_fraction=lower_mass,
        )
        upper_rows = table._mass_plane(
            profile_id=profile_id,
            binary_mass_fraction=upper_mass,
        )
        for lower_row in lower_rows:
            for upper_row in upper_rows:
                overlap_lower = max(
                    lower_row.lower_separation_over_core_radius,
                    upper_row.lower_separation_over_core_radius,
                )
                overlap_upper = min(
                    lower_row.upper_separation_over_core_radius,
                    upper_row.upper_separation_over_core_radius,
                )
                if overlap_lower >= overlap_upper:
                    continue
                try:
                    return table.interpolate(
                        profile_id=profile_id,
                        binary_to_soliton_mass=0.5
                        * (lower_mass + upper_mass),
                        separation_over_core_radius=0.5
                        * (overlap_lower + overlap_upper),
                    )
                except ValueError:
                    continue
    return None


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
    resolved_orbital_power: float = 0.0,
    resolved_orbital_torque: float = 0.0,
) -> CalibratedOrbitalStep:
    """Apply only the calibrated exchange absent from the resolved wake.

    For an unresolved calculation the resolved rates retain their zero
    defaults. A live-wave calculation must supply its measured work and torque;
    these are subtracted from the target before the finite orbital update so
    the same FDM exchange is not applied twice.
    """

    residual = residual_orbital_rates(
        rates,
        resolved_orbital_power=resolved_orbital_power,
        resolved_orbital_torque=resolved_orbital_torque,
    )
    exchange = advance_keplerian_exchange(
        mass1_msun=mass1_msun,
        mass2_msun=mass2_msun,
        semimajor_axis_pc=semimajor_axis_pc,
        eccentricity=eccentricity,
        orbital_power=residual.residual_orbital_power,
        orbital_torque=residual.residual_orbital_torque,
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
        residual=residual,
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
