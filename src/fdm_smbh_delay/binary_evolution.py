"""Coupled orbit-averaged evolution of a bound SMBH binary in ``a`` and ``e``."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass, replace as dataclass_replace
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from .constants import G_INTERNAL
from .delay_budget import DelaySegment
from .gw import peters_orbital_rates, peters_time_myr
from .orbital_exchange import keplerian_exchange_rates

if TYPE_CHECKING:
    from .subgrid_calibration import SubgridCalibrationTable


CHANNELS = ("stellar", "gas", "fdm", "gw")


class UncalibratedBinaryState(ValueError):
    """Raised when a rate provider has no accepted support at ``(a, e)``."""


@dataclass(frozen=True)
class StellarHardeningModel:
    density_msun_pc3: float
    velocity_dispersion_pc_myr: float
    hardening_coefficient: float
    eccentricity_growth_coefficient: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.density_msun_pc3,
                self.velocity_dispersion_pc_myr,
                self.hardening_coefficient,
                self.eccentricity_growth_coefficient,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("stellar hardening parameters must be finite")
        if (
            self.density_msun_pc3 <= 0.0
            or self.velocity_dispersion_pc_myr <= 0.0
            or self.hardening_coefficient < 0.0
        ):
            raise ValueError("stellar hardening density, dispersion, and H are invalid")

    @property
    def inverse_semimajor_axis_rate_per_pc_myr(self) -> float:
        return float(
            self.hardening_coefficient
            * G_INTERNAL
            * self.density_msun_pc3
            / self.velocity_dispersion_pc_myr
        )


@dataclass(frozen=True)
class GasMigrationModel:
    migration_timescale_myr: float
    eccentricity_damping_timescale_myr: float | None = None

    def __post_init__(self) -> None:
        values = [self.migration_timescale_myr]
        if self.eccentricity_damping_timescale_myr is not None:
            values.append(self.eccentricity_damping_timescale_myr)
        if np.any(~np.isfinite(values)) or np.any(np.asarray(values) <= 0.0):
            raise ValueError("gas migration timescales must be finite and positive")


@dataclass(frozen=True)
class FDMExchangeRates:
    orbital_power_msun_pc2_myr3: float
    orbital_torque_msun_pc2_myr2: float
    calibration_id: str

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.orbital_power_msun_pc2_myr3, self.orbital_torque_msun_pc2_myr2]
        )
        if np.any(~np.isfinite(values)) or not self.calibration_id:
            raise ValueError("FDM exchange rates and calibration ID are required")


FDMRateProvider = Callable[[float, float], FDMExchangeRates]


@dataclass(frozen=True)
class MeanSeparationEstimate:
    """A mapping-provided mean and uncertainty interval in physical pc.

    ``mapping_id`` identifies the independently checked mapping artifact; this
    value alone is provenance, not evidence that the artifact passed its gates.
    """

    mean_pc: float
    lower_pc: float
    upper_pc: float
    mapping_id: str

    def __post_init__(self) -> None:
        values = np.asarray([self.mean_pc, self.lower_pc, self.upper_pc], dtype=float)
        if (
            np.any(~np.isfinite(values))
            or any(
                isinstance(value, (bool, np.bool_))
                for value in (self.mean_pc, self.lower_pc, self.upper_pc)
            )
            or self.lower_pc <= 0.0
            or not self.lower_pc <= self.mean_pc <= self.upper_pc
            or not isinstance(self.mapping_id, str)
            or not self.mapping_id.strip()
        ):
            raise ValueError("orbit-mean separation estimate is invalid")


MeanSeparationProvider = Callable[[float, float], MeanSeparationEstimate]
_NO_FDM_PROVIDER_IDENTITY = "absent"


def legacy_circular_fdm_rate_provider(
    table: "SubgridCalibrationTable",
    *,
    profile_id: str,
    mass1_msun: float,
    mass2_msun: float,
    soliton_mass_msun: float,
    core_radius_pc: float,
    particle_mass_ev: float,
    calibrated_mass_ratio: float = 1.0,
    mass_ratio_tolerance: float = 1.0e-12,
    maximum_eccentricity: float = 1.0e-3,
) -> FDMRateProvider:
    """Adapt the current mass/separation table without inventing q-e support.

    The version-2 release table has no mass-ratio or eccentricity axes.  This
    adapter therefore admits only its explicitly supplied anchor and raises
    ``UncalibratedBinaryState`` everywhere else.  The q-e table introduced in
    the next schema replaces this compatibility adapter.
    """

    from .subgrid_calibration import SubgridCalibrationTable

    if (
        not isinstance(table, SubgridCalibrationTable)
        or not table.is_verified_release(2)
    ):
        raise ValueError("legacy FDM rates require a verified schema-v2 release")
    controls = np.asarray(
        [
            mass1_msun,
            mass2_msun,
            soliton_mass_msun,
            core_radius_pc,
            particle_mass_ev,
            calibrated_mass_ratio,
            mass_ratio_tolerance,
            maximum_eccentricity,
        ],
        dtype=float,
    )
    if (
        np.any(~np.isfinite(controls))
        or np.any(controls[:6] <= 0.0)
        or mass_ratio_tolerance < 0.0
        or not 0.0 <= maximum_eccentricity < 1.0
    ):
        raise ValueError("legacy FDM provider controls are invalid")
    actual_mass_ratio = min(mass1_msun, mass2_msun) / max(mass1_msun, mass2_msun)

    def provider(semimajor_axis_pc: float, eccentricity: float) -> FDMExchangeRates:
        if not np.isclose(
            actual_mass_ratio,
            calibrated_mass_ratio,
            rtol=mass_ratio_tolerance,
            atol=0.0,
        ):
            raise UncalibratedBinaryState(
                "mass ratio lies outside the legacy equal-mass FDM anchor"
            )
        if eccentricity > maximum_eccentricity:
            raise UncalibratedBinaryState(
                "eccentricity lies outside the legacy near-circular FDM anchor"
            )
        from .subgrid_calibration import physical_subgrid_rates

        try:
            rates = physical_subgrid_rates(
                table,
                profile_id=profile_id,
                mass1_msun=mass1_msun,
                mass2_msun=mass2_msun,
                soliton_mass_msun=soliton_mass_msun,
                core_radius_pc=core_radius_pc,
                particle_mass_ev=particle_mass_ev,
                separation_pc=semimajor_axis_pc,
            )
        except ValueError as error:
            raise UncalibratedBinaryState(str(error)) from error
        return FDMExchangeRates(
            rates.orbital_power,
            rates.orbital_torque,
            (
                f"legacy-v2:{profile_id}:table={table.release_table_sha256}:"
                f"m1={mass1_msun:.17g}:m2={mass2_msun:.17g}:"
                f"ms={soliton_mass_msun:.17g}:rc={core_radius_pc:.17g}:"
                f"mp={particle_mass_ev:.17g}:q0={calibrated_mass_ratio:.17g}:"
                f"qtol={mass_ratio_tolerance:.17g}:emax={maximum_eccentricity:.17g}"
            ),
        )

    provider._fdm_static_identity = (  # type: ignore[attr-defined]
        f"legacy-v2:{profile_id}:table={table.release_table_sha256}:"
        f"m1={mass1_msun:.17g}:m2={mass2_msun:.17g}:"
        f"ms={soliton_mass_msun:.17g}:rc={core_radius_pc:.17g}:"
        f"mp={particle_mass_ev:.17g}:q0={calibrated_mass_ratio:.17g}:"
        f"qtol={mass_ratio_tolerance:.17g}:emax={maximum_eccentricity:.17g}"
    )
    provider._fdm_bound_masses = (mass1_msun, mass2_msun)  # type: ignore[attr-defined]
    return provider


def calibrated_qe_fdm_rate_provider(
    table: "SubgridCalibrationTable",
    *,
    profile_id: str,
    mass1_msun: float,
    mass2_msun: float,
    soliton_mass_msun: float,
    core_radius_pc: float,
    particle_mass_ev: float,
    mean_separation_provider: MeanSeparationProvider | None = None,
    mean_separation_provider_identity: str | None = None,
) -> FDMRateProvider:
    """Adapt an accepted q/e table without q, e, or a extrapolation.

    The table admits interpolation only when measured ``(q, e)`` planes
    bracket the state and share mass/separation support. Its separation bins
    are measured orbit-mean distances, which can differ materially from the
    Kepler mean in a disturbed soliton. The caller must supply a separately
    validated mapping from secular ``(a,e)`` to orbit-mean separation with
    a bounded uncertainty interval. An absent mapping, an unsupported
    interval, or an unstructured point estimate censors the state rather
    than assuming Kepler motion.
    Restart-capable integrations must also supply
    ``mean_separation_provider_identity`` matching every returned
    ``MeanSeparationEstimate.mapping_id``; callable names do not identify
    captured mapping data reliably.
    Converting a failure to
    ``UncalibratedBinaryState`` makes the orbit integrator return a censored
    calibration gap instead of silently substituting another plane.
    """

    controls = np.asarray(
        [mass1_msun, mass2_msun, soliton_mass_msun, core_radius_pc, particle_mass_ev],
        dtype=float,
    )
    if np.any(~np.isfinite(controls)) or np.any(controls <= 0.0):
        raise ValueError("calibrated FDM provider scales must be positive")
    if mean_separation_provider_identity is not None and (
        not isinstance(mean_separation_provider_identity, str)
        or not mean_separation_provider_identity.strip()
    ):
        raise ValueError("mean-separation provider identity must be a nonempty string")
    if mean_separation_provider is None and mean_separation_provider_identity is not None:
        raise ValueError("mean-separation provider identity requires a provider")
    if mean_separation_provider is not None and mean_separation_provider_identity is None:
        raise ValueError("q/e FDM mapping requires an explicit stable identity")
    from .subgrid_calibration import SubgridCalibrationTable, physical_subgrid_rates

    if (
        not isinstance(table, SubgridCalibrationTable)
        or not table.is_verified_release(5)
    ):
        raise ValueError("q/e FDM rates require a verified schema-v5 release")
    profile_boundaries_pc = tuple(sorted({
            ratio * core_radius_pc
            for row in table.rows
            if row.profile_id == profile_id
            for ratio in (
                row.lower_separation_over_core_radius,
                row.reference_mean_separation_over_core_radius,
                row.upper_separation_over_core_radius,
            )
        }))

    def provider(semimajor_axis_pc: float, eccentricity: float) -> FDMExchangeRates:
        try:
            if mean_separation_provider is None:
                raise UncalibratedBinaryState(
                    "validated orbit-mean separation mapping is unavailable"
                )
            mapped = mean_separation_provider(semimajor_axis_pc, eccentricity)
            if not isinstance(mapped, MeanSeparationEstimate):
                raise ValueError(
                    "orbit-mean separation mapping lacks a validated interval"
                )
            if (
                mean_separation_provider_identity is not None
                and mapped.mapping_id != mean_separation_provider_identity
            ):
                raise ValueError(
                    "orbit-mean separation mapping identity does not match "
                    "the provider provenance"
                )
            mean_separation_pc = mapped.mean_pc
            rates = physical_subgrid_rates(
                table,
                profile_id=profile_id,
                mass1_msun=mass1_msun,
                mass2_msun=mass2_msun,
                soliton_mass_msun=soliton_mass_msun,
                core_radius_pc=core_radius_pc,
                particle_mass_ev=particle_mass_ev,
                separation_pc=mean_separation_pc,
                eccentricity=eccentricity,
            )
            # Interpolation support is piecewise in the accepted row edges and
            # centres. Check every interval segment, not just the mean: a
            # narrow missing bin can otherwise hide between supported ends.
            if mapped.lower_pc != mapped.upper_pc:
                critical = {mapped.lower_pc, mean_separation_pc, mapped.upper_pc}
                critical.update(profile_boundaries_pc[
                    bisect_right(profile_boundaries_pc, mapped.lower_pc):
                    bisect_left(profile_boundaries_pc, mapped.upper_pc)
                ])
                boundaries = sorted(critical)
                probes = boundaries + [
                    (lower + upper) / 2.0
                    for lower, upper in zip(boundaries, boundaries[1:])
                ]
                for separation_pc in probes:
                    if separation_pc == mean_separation_pc:
                        continue
                    physical_subgrid_rates(
                        table,
                        profile_id=profile_id,
                        mass1_msun=mass1_msun,
                        mass2_msun=mass2_msun,
                        soliton_mass_msun=soliton_mass_msun,
                        core_radius_pc=core_radius_pc,
                        particle_mass_ev=particle_mass_ev,
                        separation_pc=separation_pc,
                        eccentricity=eccentricity,
                    )
        except (TypeError, ValueError) as error:
            raise UncalibratedBinaryState(str(error)) from error
        dimensionless = rates.dimensionless
        calibration_id = (
            f"v5:{profile_id}:table={table.release_table_sha256}:"
            f"eta={dimensionless.schrodinger_poisson_similarity_parameter:.12g}:"
            f"q={dimensionless.mass_ratio_q:.12g}:"
            f"m1={mass1_msun:.17g}:m2={mass2_msun:.17g}:"
            f"ms={soliton_mass_msun:.17g}:rc={core_radius_pc:.17g}:"
            f"mp={particle_mass_ev:.17g}:"
            f"rmap={mapped.mapping_id}"
        )
        return FDMExchangeRates(
            rates.orbital_power,
            rates.orbital_torque,
            calibration_id,
        )

    if mean_separation_provider_identity is not None:
        mapping_identity = mean_separation_provider_identity
        provider._fdm_static_identity = (  # type: ignore[attr-defined]
            f"v5:{profile_id}:table={table.release_table_sha256}:"
            f"m1={mass1_msun:.17g}:m2={mass2_msun:.17g}:"
            f"ms={soliton_mass_msun:.17g}:rc={core_radius_pc:.17g}:"
            f"mp={particle_mass_ev:.17g}:mapping={mapping_identity}"
        )
    provider._fdm_bound_masses = (mass1_msun, mass2_msun)  # type: ignore[attr-defined]
    return provider


@dataclass(frozen=True)
class BoundBinaryModel:
    mass1_msun: float
    mass2_msun: float
    stellar: StellarHardeningModel | None = None
    gas: GasMigrationModel | None = None
    fdm_rate_provider: FDMRateProvider | None = None
    fdm_provider_identity: str | None = None

    def __post_init__(self) -> None:
        values = np.asarray([self.mass1_msun, self.mass2_msun], dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("SMBH masses must be finite and positive")
        if self.fdm_provider_identity is not None and (
            not isinstance(self.fdm_provider_identity, str)
            or not self.fdm_provider_identity.strip()
            or self.fdm_provider_identity in {_NO_FDM_PROVIDER_IDENTITY, "none"}
        ):
            raise ValueError("explicit FDM provider identity must be a nonempty string")
        if self.fdm_rate_provider is None and self.fdm_provider_identity is not None:
            raise ValueError("FDM provider identity requires an FDM rate provider")
        stamped_identity = (
            getattr(self.fdm_rate_provider, "_fdm_static_identity", None)
            if self.fdm_rate_provider is not None
            else None
        )
        if (
            isinstance(stamped_identity, str)
            and self.fdm_provider_identity is not None
            and self.fdm_provider_identity != stamped_identity
        ):
            raise ValueError(
                "explicit FDM provider identity conflicts with built-in provenance"
            )
        bound_masses = (
            getattr(self.fdm_rate_provider, "_fdm_bound_masses", None)
            if self.fdm_rate_provider is not None
            else None
        )
        if bound_masses is not None and bound_masses != (
            self.mass1_msun,
            self.mass2_msun,
        ):
            raise ValueError("built-in FDM provider masses do not match the binary model")
        if (
            bound_masses is not None
            and not isinstance(stamped_identity, str)
            and self.fdm_provider_identity is not None
        ):
            raise ValueError(
                "explicit identity cannot replace missing built-in FDM provenance"
            )

    @property
    def total_mass_msun(self) -> float:
        return self.mass1_msun + self.mass2_msun

    @property
    def reduced_mass_msun(self) -> float:
        return self.mass1_msun * self.mass2_msun / self.total_mass_msun


@dataclass(frozen=True)
class ChannelRates:
    semimajor_axis_rate_pc_myr: float
    eccentricity_squared_rate_per_myr: float
    orbital_power_msun_pc2_myr3: float
    orbital_torque_msun_pc2_myr2: float


@dataclass(frozen=True)
class BinaryRateBudget:
    stellar: ChannelRates
    gas: ChannelRates
    fdm: ChannelRates
    gw: ChannelRates
    total_semimajor_axis_rate_pc_myr: float
    total_eccentricity_squared_rate_per_myr: float
    environmental_channels_present: bool = False
    fdm_calibration_id: str | None = None

    @property
    def environmental_semimajor_axis_rate_pc_myr(self) -> float:
        return (
            self.stellar.semimajor_axis_rate_pc_myr
            + self.gas.semimajor_axis_rate_pc_myr
            + self.fdm.semimajor_axis_rate_pc_myr
        )

    def channel(self, name: str) -> ChannelRates:
        if name not in CHANNELS:
            raise KeyError(name)
        return getattr(self, name)


@dataclass(frozen=True)
class BoundBinaryState:
    elapsed_myr: float
    semimajor_axis_pc: float
    eccentricity_squared: float
    extracted_energy_by_channel: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    extracted_angular_momentum_by_channel: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    completed_steps: int = 0
    model_identity: str | None = None
    fdm_calibration_id: str | None = None
    reference_energy_total: float | None = None
    reference_angular_momentum_total: float | None = None

    def __post_init__(self) -> None:
        scalars = np.asarray(
            [self.elapsed_myr, self.semimajor_axis_pc, self.eccentricity_squared],
            dtype=float,
        )
        if (
            np.any(~np.isfinite(scalars))
            or self.elapsed_myr < 0.0
            or self.semimajor_axis_pc <= 0.0
            or not 0.0 <= self.eccentricity_squared < 1.0
            or self.completed_steps < 0
        ):
            raise ValueError("bound-binary state is invalid")
        for values in (
            self.extracted_energy_by_channel,
            self.extracted_angular_momentum_by_channel,
        ):
            array = np.asarray(values, dtype=float)
            if array.shape != (4,) or np.any(~np.isfinite(array)):
                raise ValueError("binary exchange reservoirs must contain four finite values")
        references = (self.reference_energy_total, self.reference_angular_momentum_total)
        if any(value is not None and not np.isfinite(value) for value in references):
            raise ValueError("binary closure references must be finite when supplied")
        if (self.reference_energy_total is None) != (
            self.reference_angular_momentum_total is None
        ):
            raise ValueError("binary closure references must be supplied together")
        if (self.model_identity is None) != (self.reference_energy_total is None):
            raise ValueError(
                "binary model identity and closure references must be supplied together"
            )
        is_restart = (
            self.completed_steps > 0
            or self.elapsed_myr > 0.0
            or any(self.extracted_energy_by_channel)
            or any(self.extracted_angular_momentum_by_channel)
        )
        if is_restart and (
            self.model_identity is None or self.reference_energy_total is None
        ):
            raise ValueError("restart state lacks model identity or closure references")

    @property
    def eccentricity(self) -> float:
        return float(np.sqrt(self.eccentricity_squared))


@dataclass(frozen=True)
class BinaryEvolutionConfig:
    """Integration controls with conservative per-step conservation gates.

    The default 1e-6 relative limits match the established integration tests
    while leaving substantial margin above their usual roundoff-level closure.
    """
    maximum_time_myr: float
    maximum_step_myr: float
    target_semimajor_axis_pc: float
    timestep_fraction: float = 0.01
    maximum_steps: int = 1_000_000
    sample_interval_steps: int = 1
    stop_at_gw_transition: bool = True
    maximum_relative_energy_closure_error: float = 1.0e-6
    maximum_relative_angular_momentum_closure_error: float = 1.0e-6

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.maximum_time_myr,
                self.maximum_step_myr,
                self.target_semimajor_axis_pc,
                self.timestep_fraction,
                self.maximum_relative_energy_closure_error,
                self.maximum_relative_angular_momentum_closure_error,
            ]
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("binary integration controls must be finite and positive")
        if self.timestep_fraction > 0.25:
            raise ValueError("binary timestep_fraction must not exceed 0.25")
        if self.maximum_steps < 1 or self.sample_interval_steps < 1:
            raise ValueError("binary step controls must be positive integers")


@dataclass(frozen=True)
class BinaryEvolutionSample:
    elapsed_myr: float
    semimajor_axis_pc: float
    eccentricity: float
    orbital_energy: float
    orbital_angular_momentum: float
    energy_closure_error: float
    angular_momentum_closure_error: float
    rates: BinaryRateBudget

    @property
    def energy_closure_relative_error(self) -> float:
        """Dimensionless closure residual relative to the orbital energy."""

        return float(
            abs(self.energy_closure_error)
            / max(abs(self.orbital_energy), np.finfo(float).tiny)
        )

    @property
    def angular_momentum_closure_relative_error(self) -> float:
        """Dimensionless closure residual relative to orbital angular momentum."""

        return float(
            abs(self.angular_momentum_closure_error)
            / max(abs(self.orbital_angular_momentum), np.finfo(float).tiny)
        )


@dataclass(frozen=True)
class BinaryEvolutionResult:
    status: str
    final_state: BoundBinaryState
    samples: tuple[BinaryEvolutionSample, ...]
    reason: str
    gw_completion_delay_myr: float | None

    @property
    def environment_fdm_segment(self) -> DelaySegment:
        if self.status in {"gw_transition", "reached_target"}:
            return DelaySegment(
                "environment_fdm_to_gw", "complete", self.final_state.elapsed_myr
            )
        if self.status == "timeout":
            return DelaySegment(
                "environment_fdm_to_gw",
                "timeout",
                None,
                elapsed_lower_bound_myr=self.final_state.elapsed_myr,
                reason=self.reason,
            )
        if self.status == "stalled":
            return DelaySegment(
                "environment_fdm_to_gw",
                "censored",
                None,
                elapsed_lower_bound_myr=self.final_state.elapsed_myr,
                reason=self.reason,
            )
        if self.status == "checkpoint":
            return DelaySegment(
                "environment_fdm_to_gw",
                "missing",
                None,
                elapsed_lower_bound_myr=self.final_state.elapsed_myr,
                reason=self.reason,
            )
        if self.status == "uncalibrated":
            return DelaySegment(
                "environment_fdm_to_gw",
                "censored",
                None,
                elapsed_lower_bound_myr=self.final_state.elapsed_myr,
                reason=self.reason,
            )
        if self.status == "censored":
            return DelaySegment(
                "environment_fdm_to_gw",
                "censored",
                None,
                elapsed_lower_bound_myr=self.final_state.elapsed_myr,
                reason=self.reason,
            )
        return DelaySegment("environment_fdm_to_gw", "invalid", None)

    @property
    def gravitational_wave_segment(self) -> DelaySegment:
        if self.gw_completion_delay_myr is not None:
            return DelaySegment(
                "gravitational_wave", "complete", self.gw_completion_delay_myr
            )
        return DelaySegment("gravitational_wave", "missing", None)


def orbital_invariants(
    model: BoundBinaryModel, semimajor_axis_pc: float, eccentricity_squared: float
) -> tuple[float, float]:
    energy = -G_INTERNAL * model.mass1_msun * model.mass2_msun / (
        2.0 * semimajor_axis_pc
    )
    angular_momentum = model.reduced_mass_msun * np.sqrt(
        G_INTERNAL
        * model.total_mass_msun
        * semimajor_axis_pc
        * (1.0 - eccentricity_squared)
    )
    return float(energy), float(angular_momentum)


def _rates_from_elements(
    model: BoundBinaryModel,
    semimajor_axis_pc: float,
    eccentricity_squared: float,
    semimajor_axis_rate: float,
    eccentricity_squared_rate: float,
) -> ChannelRates:
    energy, angular_momentum = orbital_invariants(
        model, semimajor_axis_pc, eccentricity_squared
    )
    power = (
        G_INTERNAL
        * model.mass1_msun
        * model.mass2_msun
        / (2.0 * semimajor_axis_pc**2)
        * semimajor_axis_rate
    )
    torque = 0.5 * angular_momentum * (
        semimajor_axis_rate / semimajor_axis_pc
        - eccentricity_squared_rate / (1.0 - eccentricity_squared)
    )
    assert energy < 0.0
    return ChannelRates(
        float(semimajor_axis_rate),
        float(eccentricity_squared_rate),
        float(power),
        float(torque),
    )


def _zero_rates() -> ChannelRates:
    return ChannelRates(0.0, 0.0, 0.0, 0.0)


def binary_rate_budget(
    model: BoundBinaryModel,
    *,
    semimajor_axis_pc: float,
    eccentricity_squared: float,
) -> BinaryRateBudget:
    if (
        not np.isfinite(semimajor_axis_pc)
        or semimajor_axis_pc <= 0.0
        or not np.isfinite(eccentricity_squared)
        or not 0.0 <= eccentricity_squared < 1.0
    ):
        raise ValueError("binary rate coordinates are invalid")
    eccentricity = float(np.sqrt(eccentricity_squared))

    stellar = _zero_rates()
    if model.stellar is not None:
        inverse_axis_rate = model.stellar.inverse_semimajor_axis_rate_per_pc_myr
        axis_rate = -inverse_axis_rate * semimajor_axis_pc**2
        eccentricity_squared_rate = (
            2.0
            * model.stellar.eccentricity_growth_coefficient
            * eccentricity_squared
            * inverse_axis_rate
            * semimajor_axis_pc
        )
        stellar = _rates_from_elements(
            model,
            semimajor_axis_pc,
            eccentricity_squared,
            axis_rate,
            eccentricity_squared_rate,
        )

    gas = _zero_rates()
    if model.gas is not None:
        axis_rate = -semimajor_axis_pc / model.gas.migration_timescale_myr
        eccentricity_squared_rate = (
            0.0
            if model.gas.eccentricity_damping_timescale_myr is None
            else -2.0
            * eccentricity_squared
            / model.gas.eccentricity_damping_timescale_myr
        )
        gas = _rates_from_elements(
            model,
            semimajor_axis_pc,
            eccentricity_squared,
            axis_rate,
            eccentricity_squared_rate,
        )

    fdm = _zero_rates()
    fdm_calibration_id = None
    if model.fdm_rate_provider is not None:
        exchange = model.fdm_rate_provider(semimajor_axis_pc, eccentricity)
        fdm_calibration_id = exchange.calibration_id
        converted = keplerian_exchange_rates(
            mass1_msun=model.mass1_msun,
            mass2_msun=model.mass2_msun,
            semimajor_axis_pc=semimajor_axis_pc,
            eccentricity=eccentricity,
            orbital_power=exchange.orbital_power_msun_pc2_myr3,
            orbital_torque=exchange.orbital_torque_msun_pc2_myr2,
        )
        fdm = ChannelRates(
            converted.semimajor_axis_rate_pc_myr,
            converted.eccentricity_squared_rate_per_myr,
            exchange.orbital_power_msun_pc2_myr3,
            exchange.orbital_torque_msun_pc2_myr2,
        )

    peters = peters_orbital_rates(
        model.mass1_msun,
        model.mass2_msun,
        semimajor_axis_pc,
        eccentricity,
    )
    gw = _rates_from_elements(
        model,
        semimajor_axis_pc,
        eccentricity_squared,
        peters.semimajor_axis_rate_pc_myr,
        peters.eccentricity_squared_rate_per_myr,
    )
    channels = (stellar, gas, fdm, gw)
    return BinaryRateBudget(
        stellar=stellar,
        gas=gas,
        fdm=fdm,
        gw=gw,
        total_semimajor_axis_rate_pc_myr=float(
            sum(rate.semimajor_axis_rate_pc_myr for rate in channels)
        ),
        total_eccentricity_squared_rate_per_myr=float(
            sum(rate.eccentricity_squared_rate_per_myr for rate in channels)
        ),
        environmental_channels_present=any(
            channel is not None
            for channel in (model.stellar, model.gas, model.fdm_rate_provider)
        ),
        fdm_calibration_id=fdm_calibration_id,
    )


def gw_dominates_environment(rates: BinaryRateBudget) -> bool:
    """Return true only when a resolved, shrinking environment is overtaken.

    A missing environment or a non-shrinking/expanding environmental rate is
    not evidence that the GW regime has begun.  Treating either case as a
    zero environmental rate would silently turn missing physics into a GW-only
    completion.
    """

    if not rates.environmental_channels_present:
        return False
    environmental_rate = rates.environmental_semimajor_axis_rate_pc_myr
    if not np.isfinite(environmental_rate) or environmental_rate >= 0.0:
        return False
    return bool(-rates.gw.semimajor_axis_rate_pc_myr >= -environmental_rate)


def find_gw_transition_pc(
    model: BoundBinaryModel,
    *,
    eccentricity: float,
    minimum_semimajor_axis_pc: float,
    maximum_semimajor_axis_pc: float,
    samples: int = 256,
) -> float | None:
    """Find the fixed-equality radius where GW shrinkage overtakes environment."""

    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must satisfy 0 <= e < 1")
    if (
        minimum_semimajor_axis_pc <= 0.0
        or maximum_semimajor_axis_pc <= minimum_semimajor_axis_pc
        or samples < 2
    ):
        raise ValueError("GW transition bracket is invalid")

    def balance(axis: float) -> float:
        rates = binary_rate_budget(
            model,
            semimajor_axis_pc=axis,
            eccentricity_squared=eccentricity**2,
        )
        return float(
            -rates.gw.semimajor_axis_rate_pc_myr
            - max(-rates.environmental_semimajor_axis_rate_pc_myr, 0.0)
        )

    axes = np.geomspace(
        minimum_semimajor_axis_pc, maximum_semimajor_axis_pc, samples
    )
    values = np.asarray([balance(axis) for axis in axes])
    exact = np.flatnonzero(values == 0.0)
    if exact.size:
        return float(axes[int(exact[0])])
    for lower, upper, lower_value, upper_value in zip(
        axes[:-1], axes[1:], values[:-1], values[1:]
    ):
        if lower_value * upper_value < 0.0:
            return float(brentq(balance, lower, upper, xtol=1.0e-14, rtol=1.0e-12))
    return None


def _sample(
    state: BoundBinaryState,
    model: BoundBinaryModel,
    reference_energy_total: float,
    reference_angular_momentum_total: float,
) -> BinaryEvolutionSample:
    energy, angular_momentum = orbital_invariants(
        model, state.semimajor_axis_pc, state.eccentricity_squared
    )
    conserved_energy = energy + sum(state.extracted_energy_by_channel)
    conserved_angular_momentum = angular_momentum + sum(
        state.extracted_angular_momentum_by_channel
    )
    return BinaryEvolutionSample(
        elapsed_myr=state.elapsed_myr,
        semimajor_axis_pc=state.semimajor_axis_pc,
        eccentricity=state.eccentricity,
        orbital_energy=energy,
        orbital_angular_momentum=angular_momentum,
        energy_closure_error=float(conserved_energy - reference_energy_total),
        angular_momentum_closure_error=float(
            conserved_angular_momentum - reference_angular_momentum_total
        ),
        rates=binary_rate_budget(
            model,
            semimajor_axis_pc=state.semimajor_axis_pc,
            eccentricity_squared=state.eccentricity_squared,
        ),
    )


def advance_bound_binary_rk4(
    state: BoundBinaryState,
    model: BoundBinaryModel,
    time_step_myr: float,
) -> BoundBinaryState:
    if not np.isfinite(time_step_myr) or time_step_myr <= 0.0:
        raise ValueError("binary time step must be finite and positive")
    if model.fdm_rate_provider is not None and _fdm_provider_identity(model) is None:
        raise ValueError("binary evolution requires a stable FDM provider identity")
    a0 = state.semimajor_axis_pc
    y0 = state.eccentricity_squared
    identity = _model_identity(model)
    if state.model_identity is not None and state.model_identity != identity:
        raise ValueError("restart model identity does not match the binary model")
    rates1 = binary_rate_budget(
        model, semimajor_axis_pc=a0, eccentricity_squared=y0
    )
    if state.model_identity is None:
        initial_energy, initial_angular_momentum = orbital_invariants(model, a0, y0)
        state = dataclass_replace(
            state,
            model_identity=identity,
            fdm_calibration_id=rates1.fdm_calibration_id,
            reference_energy_total=initial_energy
            + sum(state.extracted_energy_by_channel),
            reference_angular_momentum_total=initial_angular_momentum
            + sum(state.extracted_angular_momentum_by_channel),
        )
    elif state.fdm_calibration_id != rates1.fdm_calibration_id:
        raise ValueError("restart FDM calibration identity does not match the rate model")
    k1a = rates1.total_semimajor_axis_rate_pc_myr
    k1y = rates1.total_eccentricity_squared_rate_per_myr
    rates2 = binary_rate_budget(
        model,
        semimajor_axis_pc=a0 + 0.5 * time_step_myr * k1a,
        eccentricity_squared=y0 + 0.5 * time_step_myr * k1y,
    )
    k2a = rates2.total_semimajor_axis_rate_pc_myr
    k2y = rates2.total_eccentricity_squared_rate_per_myr
    rates3 = binary_rate_budget(
        model,
        semimajor_axis_pc=a0 + 0.5 * time_step_myr * k2a,
        eccentricity_squared=y0 + 0.5 * time_step_myr * k2y,
    )
    k3a = rates3.total_semimajor_axis_rate_pc_myr
    k3y = rates3.total_eccentricity_squared_rate_per_myr
    rates4 = binary_rate_budget(
        model,
        semimajor_axis_pc=a0 + time_step_myr * k3a,
        eccentricity_squared=y0 + time_step_myr * k3y,
    )
    k4a = rates4.total_semimajor_axis_rate_pc_myr
    k4y = rates4.total_eccentricity_squared_rate_per_myr
    final_axis = a0 + time_step_myr * (k1a + 2.0 * k2a + 2.0 * k3a + k4a) / 6.0
    final_e2 = y0 + time_step_myr * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
    if final_axis <= 0.0 or not 0.0 <= final_e2 < 1.0:
        raise ValueError("finite binary step left the bound-orbit domain")

    stage_rates = (rates1, rates2, rates3, rates4)
    calibration_ids = {rate.fdm_calibration_id for rate in stage_rates}
    if state.fdm_calibration_id is not None:
        calibration_ids.add(state.fdm_calibration_id)
    if len(calibration_ids) > 1:
        raise UncalibratedBinaryState(
            "FDM calibration identity changed within a binary step"
        )
    powers = np.asarray(
        [
            time_step_myr
            * sum(
                weight * rate.channel(name).orbital_power_msun_pc2_myr3
                for weight, rate in zip((1.0, 2.0, 2.0, 1.0), stage_rates)
            )
            / 6.0
            for name in CHANNELS
        ]
    )
    torques = np.asarray(
        [
            time_step_myr
            * sum(
                weight * rate.channel(name).orbital_torque_msun_pc2_myr2
                for weight, rate in zip((1.0, 2.0, 2.0, 1.0), stage_rates)
            )
            / 6.0
            for name in CHANNELS
        ]
    )
    # ``power`` and ``torque`` are orbital dE/dt and dL/dt.  The reservoirs
    # therefore receive their negatives.  RK4 quadrature is independent of
    # the final element difference, so the reported closure is diagnostic
    # rather than exact by construction.
    energy_increment = -powers
    angular_increment = -torques
    return BoundBinaryState(
        elapsed_myr=state.elapsed_myr + time_step_myr,
        semimajor_axis_pc=float(final_axis),
        eccentricity_squared=float(final_e2),
        extracted_energy_by_channel=tuple(
            np.asarray(state.extracted_energy_by_channel) + energy_increment
        ),
        extracted_angular_momentum_by_channel=tuple(
            np.asarray(state.extracted_angular_momentum_by_channel) + angular_increment
        ),
        completed_steps=state.completed_steps + 1,
        model_identity=state.model_identity,
        fdm_calibration_id=state.fdm_calibration_id,
        reference_energy_total=state.reference_energy_total,
        reference_angular_momentum_total=state.reference_angular_momentum_total,
    )


def _adaptive_step(
    state: BoundBinaryState,
    rates: BinaryRateBudget,
    config: BinaryEvolutionConfig,
) -> float:
    timescales = []
    if rates.total_semimajor_axis_rate_pc_myr != 0.0:
        timescales.append(
            state.semimajor_axis_pc
            / abs(rates.total_semimajor_axis_rate_pc_myr)
        )
    eccentricity_rate = rates.total_eccentricity_squared_rate_per_myr
    if eccentricity_rate > 0.0:
        timescales.append((1.0 - state.eccentricity_squared) / eccentricity_rate)
    elif eccentricity_rate < 0.0 and state.eccentricity_squared > 0.0:
        timescales.append(state.eccentricity_squared / abs(eccentricity_rate))
    remaining = config.maximum_time_myr - state.elapsed_myr
    candidates = [config.maximum_step_myr, remaining]
    candidates.extend(config.timestep_fraction * value for value in timescales)
    return float(min(candidates))


def _samples_with_final(
    samples: list[BinaryEvolutionSample],
    state: BoundBinaryState,
    model: BoundBinaryModel,
    reference_energy_total: float,
    reference_angular_momentum_total: float,
) -> tuple[BinaryEvolutionSample, ...]:
    if not samples or samples[-1].elapsed_myr != state.elapsed_myr:
        samples.append(
            _sample(
                state,
                model,
                reference_energy_total,
                reference_angular_momentum_total,
            )
        )
    return tuple(samples)


def _model_identity(model: BoundBinaryModel) -> str:
    provider_identity = _fdm_provider_identity(model)
    return (
        f"m1={model.mass1_msun:.17g}:m2={model.mass2_msun:.17g}:"
        f"stellar={model.stellar!r}:gas={model.gas!r}:"
        f"fdm={provider_identity if provider_identity is not None else 'unidentified'}"
    )


def _fdm_provider_identity(model: BoundBinaryModel) -> str | None:
    if model.fdm_rate_provider is None:
        return _NO_FDM_PROVIDER_IDENTITY
    identity = getattr(model.fdm_rate_provider, "_fdm_static_identity", None)
    if isinstance(identity, str) and identity:
        return identity
    return model.fdm_provider_identity


def _closure_failure_reason(
    state: BoundBinaryState,
    model: BoundBinaryModel,
    config: BinaryEvolutionConfig,
) -> str | None:
    energy, angular_momentum = orbital_invariants(
        model, state.semimajor_axis_pc, state.eccentricity_squared
    )
    energy_error = abs(
        energy
        + sum(state.extracted_energy_by_channel)
        - state.reference_energy_total
    ) / max(abs(energy), np.finfo(float).tiny)
    angular_error = abs(
        angular_momentum
        + sum(state.extracted_angular_momentum_by_channel)
        - state.reference_angular_momentum_total
    ) / max(abs(angular_momentum), np.finfo(float).tiny)
    if not np.isfinite(energy_error):
        return "energy closure gate failed (non-finite residual)"
    if energy_error > config.maximum_relative_energy_closure_error:
        return f"energy closure gate failed ({energy_error:.6g})"
    if not np.isfinite(angular_error):
        return "angular-momentum closure gate failed (non-finite residual)"
    if angular_error > config.maximum_relative_angular_momentum_closure_error:
        return f"angular-momentum closure gate failed ({angular_error:.6g})"
    return None


def integrate_bound_binary(
    *,
    initial_state: BoundBinaryState,
    model: BoundBinaryModel,
    config: BinaryEvolutionConfig,
    step_budget: int | None = None,
) -> BinaryEvolutionResult:
    """Evolve a hard binary until the event-specific GW transition."""

    if step_budget is not None and step_budget < 1:
        raise ValueError("step_budget must be positive when supplied")
    if initial_state.elapsed_myr > config.maximum_time_myr:
        raise ValueError("initial binary state lies beyond maximum time")
    state = initial_state
    if state.model_identity is not None and _fdm_provider_identity(model) is None:
        raise ValueError("restart requires an explicit stable FDM provider identity")
    identity = _model_identity(model)
    initial_energy, initial_angular_momentum = orbital_invariants(
        model, state.semimajor_axis_pc, state.eccentricity_squared
    )
    if state.model_identity is not None:
        if state.model_identity != identity:
            raise ValueError("restart model identity does not match the binary model")
        closure_failure = _closure_failure_reason(state, model, config)
        if closure_failure is not None:
            return BinaryEvolutionResult("invalid", state, (), closure_failure, None)
    try:
        initial_rates = binary_rate_budget(
            model,
            semimajor_axis_pc=state.semimajor_axis_pc,
            eccentricity_squared=state.eccentricity_squared,
        )
    except UncalibratedBinaryState as error:
        return BinaryEvolutionResult("uncalibrated", state, (), str(error), None)
    if model.fdm_rate_provider is not None and _fdm_provider_identity(model) is None:
        raise ValueError("binary evolution requires a stable FDM provider identity")
    if state.model_identity is None:
        state = dataclass_replace(
            state,
            model_identity=identity,
            fdm_calibration_id=initial_rates.fdm_calibration_id,
            reference_energy_total=initial_energy
            + sum(state.extracted_energy_by_channel),
            reference_angular_momentum_total=initial_angular_momentum
            + sum(state.extracted_angular_momentum_by_channel),
        )
    elif state.fdm_calibration_id != initial_rates.fdm_calibration_id:
        raise ValueError("restart FDM calibration identity does not match the rate model")
    reference_energy_total = state.reference_energy_total
    reference_angular_momentum_total = state.reference_angular_momentum_total
    try:
        samples = [
            _sample(
                state,
                model,
                reference_energy_total,
                reference_angular_momentum_total,
            )
        ]
    except UncalibratedBinaryState as error:
        return BinaryEvolutionResult(
            "uncalibrated", state, (), str(error), None
        )
    steps_this_call = 0
    while state.completed_steps < config.maximum_steps:
        try:
            rates = binary_rate_budget(
                model,
                semimajor_axis_pc=state.semimajor_axis_pc,
                eccentricity_squared=state.eccentricity_squared,
            )
        except UncalibratedBinaryState as error:
            return BinaryEvolutionResult(
                "uncalibrated",
                state,
                tuple(samples),
                str(error),
                None,
            )
        if not rates.environmental_channels_present:
            return BinaryEvolutionResult(
                "censored",
                state,
                _samples_with_final(
                    samples,
                    state,
                    model,
                    reference_energy_total,
                    reference_angular_momentum_total,
                ),
                "no environmental channel is available for the binary stage",
                None,
            )
        if rates.environmental_semimajor_axis_rate_pc_myr >= 0.0:
            status = (
                "stalled"
                if rates.environmental_semimajor_axis_rate_pc_myr > 0.0
                else "censored"
            )
            return BinaryEvolutionResult(
                status,
                state,
                _samples_with_final(
                    samples,
                    state,
                    model,
                    reference_energy_total,
                    reference_angular_momentum_total,
                ),
                "environmental channels are not shrinking the binary",
                None,
            )
        if state.semimajor_axis_pc <= config.target_semimajor_axis_pc:
            gw_delay = peters_time_myr(
                model.mass1_msun,
                model.mass2_msun,
                state.semimajor_axis_pc,
                state.eccentricity,
            )
            return BinaryEvolutionResult(
                "reached_target",
                state,
                _samples_with_final(
                    samples,
                    state,
                    model,
                    reference_energy_total,
                    reference_angular_momentum_total,
                ),
                "binary reached the requested semimajor-axis boundary",
                gw_delay,
            )
        if config.stop_at_gw_transition and gw_dominates_environment(rates):
            gw_delay = peters_time_myr(
                model.mass1_msun,
                model.mass2_msun,
                state.semimajor_axis_pc,
                state.eccentricity,
            )
            return BinaryEvolutionResult(
                "gw_transition",
                state,
                _samples_with_final(
                    samples,
                    state,
                    model,
                    reference_energy_total,
                    reference_angular_momentum_total,
                ),
                "GW semimajor-axis shrinkage overtook environmental shrinkage",
                gw_delay,
            )
        if state.elapsed_myr >= config.maximum_time_myr:
            return BinaryEvolutionResult(
                "timeout",
                state,
                _samples_with_final(
                    samples,
                    state,
                    model,
                    reference_energy_total,
                    reference_angular_momentum_total,
                ),
                "available binary-evolution time was exhausted",
                None,
            )
        if step_budget is not None and steps_this_call >= step_budget:
            return BinaryEvolutionResult(
                "checkpoint",
                state,
                _samples_with_final(
                    samples,
                    state,
                    model,
                    reference_energy_total,
                    reference_angular_momentum_total,
                ),
                "bounded step budget reached; resume from final_state",
                None,
            )
        time_step = _adaptive_step(state, rates, config)
        if not np.isfinite(time_step) or time_step <= 0.0:
            return BinaryEvolutionResult(
                "invalid",
                state,
                _samples_with_final(
                    samples,
                    state,
                    model,
                    reference_energy_total,
                    reference_angular_momentum_total,
                ),
                "adaptive binary time step became invalid",
                None,
            )
        try:
            candidate = advance_bound_binary_rk4(state, model, time_step)
            # RK stages can all be supported while their weighted endpoint
            # lands outside a narrow measured (a,e) domain. Reject that
            # endpoint before it becomes a restartable accepted state, even
            # when no sample is due on this step.
            endpoint_rates = binary_rate_budget(
                model,
                semimajor_axis_pc=candidate.semimajor_axis_pc,
                eccentricity_squared=candidate.eccentricity_squared,
            )
            if endpoint_rates.fdm_calibration_id != state.fdm_calibration_id:
                raise UncalibratedBinaryState(
                    "FDM calibration identity changed at a binary-step endpoint"
                )
        except UncalibratedBinaryState as error:
            return BinaryEvolutionResult(
                "uncalibrated",
                state,
                _samples_with_final(
                    samples,
                    state,
                    model,
                    reference_energy_total,
                    reference_angular_momentum_total,
                ),
                str(error),
                None,
            )
        closure_failure = _closure_failure_reason(candidate, model, config)
        if closure_failure is not None:
            return BinaryEvolutionResult(
                "invalid",
                state,
                _samples_with_final(
                    samples,
                    state,
                    model,
                    reference_energy_total,
                    reference_angular_momentum_total,
                ),
                closure_failure,
                None,
            )
        state = candidate
        steps_this_call += 1
        if state.completed_steps % config.sample_interval_steps == 0:
            samples.append(
                _sample(
                    state,
                    model,
                    reference_energy_total,
                    reference_angular_momentum_total,
                )
            )
    return BinaryEvolutionResult(
        "invalid",
        state,
        _samples_with_final(
            samples,
            state,
            model,
            reference_energy_total,
            reference_angular_momentum_total,
        ),
        "maximum binary step count was exhausted",
        None,
    )
