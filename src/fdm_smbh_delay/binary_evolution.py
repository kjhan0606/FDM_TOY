"""Coupled orbit-averaged evolution of a bound SMBH binary in ``a`` and ``e``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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


def legacy_circular_fdm_rate_provider(
    table: "SubgridCalibrationTable",
    *,
    profile_id: str,
    mass1_msun: float,
    mass2_msun: float,
    soliton_mass_msun: float,
    core_radius_pc: float,
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

    controls = np.asarray(
        [
            mass1_msun,
            mass2_msun,
            soliton_mass_msun,
            core_radius_pc,
            calibrated_mass_ratio,
            mass_ratio_tolerance,
            maximum_eccentricity,
        ],
        dtype=float,
    )
    if (
        np.any(~np.isfinite(controls))
        or np.any(controls[:5] <= 0.0)
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
                separation_pc=semimajor_axis_pc,
            )
        except ValueError as error:
            raise UncalibratedBinaryState(str(error)) from error
        return FDMExchangeRates(
            rates.orbital_power,
            rates.orbital_torque,
            f"legacy-v2:{profile_id}",
        )

    return provider


def calibrated_qe_fdm_rate_provider(
    table: "SubgridCalibrationTable",
    *,
    profile_id: str,
    mass1_msun: float,
    mass2_msun: float,
    soliton_mass_msun: float,
    core_radius_pc: float,
) -> FDMRateProvider:
    """Adapt an accepted schema-v3 table without q, e, or a extrapolation.

    The table admits interpolation only when measured ``(q, e)`` planes
    bracket the state and share mass/separation support. Converting a failure to
    ``UncalibratedBinaryState`` makes the orbit integrator return a censored
    calibration gap instead of silently substituting another plane.
    """

    controls = np.asarray(
        [mass1_msun, mass2_msun, soliton_mass_msun, core_radius_pc],
        dtype=float,
    )
    if np.any(~np.isfinite(controls)) or np.any(controls <= 0.0):
        raise ValueError("calibrated FDM provider scales must be positive")

    def provider(semimajor_axis_pc: float, eccentricity: float) -> FDMExchangeRates:
        from .subgrid_calibration import physical_subgrid_rates

        try:
            rates = physical_subgrid_rates(
                table,
                profile_id=profile_id,
                mass1_msun=mass1_msun,
                mass2_msun=mass2_msun,
                soliton_mass_msun=soliton_mass_msun,
                core_radius_pc=core_radius_pc,
                separation_pc=semimajor_axis_pc,
                eccentricity=eccentricity,
            )
        except ValueError as error:
            raise UncalibratedBinaryState(str(error)) from error
        dimensionless = rates.dimensionless
        calibration_id = (
            f"v3:{profile_id}:q={dimensionless.mass_ratio_q:.12g}:"
            f"e={dimensionless.reference_eccentricity:.12g}"
        )
        return FDMExchangeRates(
            rates.orbital_power,
            rates.orbital_torque,
            calibration_id,
        )

    return provider


@dataclass(frozen=True)
class BoundBinaryModel:
    mass1_msun: float
    mass2_msun: float
    stellar: StellarHardeningModel | None = None
    gas: GasMigrationModel | None = None
    fdm_rate_provider: FDMRateProvider | None = None

    def __post_init__(self) -> None:
        values = np.asarray([self.mass1_msun, self.mass2_msun], dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("SMBH masses must be finite and positive")

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

    @property
    def eccentricity(self) -> float:
        return float(np.sqrt(self.eccentricity_squared))


@dataclass(frozen=True)
class BinaryEvolutionConfig:
    maximum_time_myr: float
    maximum_step_myr: float
    target_semimajor_axis_pc: float
    timestep_fraction: float = 0.01
    maximum_steps: int = 1_000_000
    sample_interval_steps: int = 1
    stop_at_gw_transition: bool = True

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.maximum_time_myr,
                self.maximum_step_myr,
                self.target_semimajor_axis_pc,
                self.timestep_fraction,
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
        if self.status in {"timeout", "stalled"}:
            return DelaySegment(
                "environment_fdm_to_gw",
                "timeout",
                None,
                elapsed_lower_bound_myr=self.final_state.elapsed_myr,
            )
        if self.status in {"checkpoint", "uncalibrated"}:
            return DelaySegment(
                "environment_fdm_to_gw",
                "missing",
                None,
                elapsed_lower_bound_myr=self.final_state.elapsed_myr,
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
    if model.fdm_rate_provider is not None:
        exchange = model.fdm_rate_provider(semimajor_axis_pc, eccentricity)
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
    )


def gw_dominates_environment(rates: BinaryRateBudget) -> bool:
    environmental_shrinkage = max(
        -rates.environmental_semimajor_axis_rate_pc_myr, 0.0
    )
    return bool(-rates.gw.semimajor_axis_rate_pc_myr >= environmental_shrinkage)


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


def _derivative(
    model: BoundBinaryModel, semimajor_axis_pc: float, eccentricity_squared: float
) -> tuple[float, float]:
    rates = binary_rate_budget(
        model,
        semimajor_axis_pc=semimajor_axis_pc,
        eccentricity_squared=eccentricity_squared,
    )
    return (
        rates.total_semimajor_axis_rate_pc_myr,
        rates.total_eccentricity_squared_rate_per_myr,
    )


def _allocate_exact_exchange(
    total_orbital_change: float, channel_rates: np.ndarray
) -> np.ndarray:
    total_rate = float(np.sum(channel_rates))
    if abs(total_rate) <= np.finfo(float).tiny:
        if abs(total_orbital_change) <= np.finfo(float).eps:
            return np.zeros(4)
        raise ValueError("finite orbital change has no conjugate channel rate")
    return -total_orbital_change * channel_rates / total_rate


def advance_bound_binary_rk4(
    state: BoundBinaryState,
    model: BoundBinaryModel,
    time_step_myr: float,
) -> BoundBinaryState:
    if not np.isfinite(time_step_myr) or time_step_myr <= 0.0:
        raise ValueError("binary time step must be finite and positive")
    a0 = state.semimajor_axis_pc
    y0 = state.eccentricity_squared
    k1a, k1y = _derivative(model, a0, y0)
    k2a, k2y = _derivative(
        model, a0 + 0.5 * time_step_myr * k1a, y0 + 0.5 * time_step_myr * k1y
    )
    k3a, k3y = _derivative(
        model, a0 + 0.5 * time_step_myr * k2a, y0 + 0.5 * time_step_myr * k2y
    )
    k4a, k4y = _derivative(
        model, a0 + time_step_myr * k3a, y0 + time_step_myr * k3y
    )
    final_axis = a0 + time_step_myr * (k1a + 2.0 * k2a + 2.0 * k3a + k4a) / 6.0
    final_e2 = y0 + time_step_myr * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
    if final_axis <= 0.0 or not 0.0 <= final_e2 < 1.0:
        raise ValueError("finite binary step left the bound-orbit domain")

    initial_energy, initial_angular_momentum = orbital_invariants(model, a0, y0)
    final_energy, final_angular_momentum = orbital_invariants(
        model, final_axis, final_e2
    )
    midpoint_rates = binary_rate_budget(
        model,
        semimajor_axis_pc=0.5 * (a0 + final_axis),
        eccentricity_squared=0.5 * (y0 + final_e2),
    )
    powers = np.asarray(
        [midpoint_rates.channel(name).orbital_power_msun_pc2_myr3 for name in CHANNELS]
    )
    torques = np.asarray(
        [midpoint_rates.channel(name).orbital_torque_msun_pc2_myr2 for name in CHANNELS]
    )
    energy_increment = _allocate_exact_exchange(final_energy - initial_energy, powers)
    angular_increment = _allocate_exact_exchange(
        final_angular_momentum - initial_angular_momentum, torques
    )
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
    initial_energy, initial_angular_momentum = orbital_invariants(
        model, state.semimajor_axis_pc, state.eccentricity_squared
    )
    reference_energy_total = initial_energy + sum(
        state.extracted_energy_by_channel
    )
    reference_angular_momentum_total = initial_angular_momentum + sum(
        state.extracted_angular_momentum_by_channel
    )
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
            state = advance_bound_binary_rk4(state, model, time_step)
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
