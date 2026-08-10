"""Orbit-averaged element rates from calibrated energy and torque exchange."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import G_INTERNAL


@dataclass(frozen=True)
class OrbitalExchangeRates:
    orbital_energy: float
    orbital_angular_momentum: float
    semimajor_axis_rate_pc_myr: float
    eccentricity_squared_rate_per_myr: float
    eccentricity_rate_per_myr: float | None
    wave_energy_rate: float
    wave_angular_momentum_rate: float


@dataclass(frozen=True)
class KeplerianElements:
    separation: float
    relative_speed: float
    specific_energy: float
    specific_angular_momentum: np.ndarray
    semimajor_axis: float | None
    eccentricity_vector: np.ndarray
    eccentricity: float


@dataclass(frozen=True)
class FiniteOrbitalExchangeStep:
    initial_semimajor_axis_pc: float
    final_semimajor_axis_pc: float
    initial_eccentricity: float
    final_eccentricity: float
    initial_orbital_energy: float
    final_orbital_energy: float
    initial_orbital_angular_momentum: float
    final_orbital_angular_momentum: float
    orbital_phase_increment_rad: float
    wave_energy_increment: float
    wave_angular_momentum_increment: float


def keplerian_elements_from_relative_state(
    *,
    total_mass: float,
    displacement: np.ndarray,
    relative_velocity: np.ndarray,
    gravitational_constant: float = G_INTERNAL,
) -> KeplerianElements:
    """Return point-mass osculating elements in a consistent unit system.

    The elements describe the instantaneous Kepler orbit defined by the
    relative state.  An extended external potential or softened mutual force
    produces reversible oscillations in these diagnostic elements.
    """

    position = np.asarray(displacement, dtype=float)
    velocity = np.asarray(relative_velocity, dtype=float)
    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("relative position and velocity must have shape (3,)")
    if (
        total_mass <= 0.0
        or gravitational_constant <= 0.0
        or np.any(~np.isfinite(position))
        or np.any(~np.isfinite(velocity))
    ):
        raise ValueError("mass, gravity, position, and velocity must be finite")
    separation = float(np.linalg.norm(position))
    if separation <= 0.0:
        raise ValueError("relative separation must be positive")
    relative_speed = float(np.linalg.norm(velocity))
    gravitational_mass = gravitational_constant * total_mass
    angular_momentum = np.cross(position, velocity)
    specific_energy = 0.5 * relative_speed**2 - gravitational_mass / separation
    eccentricity_vector = (
        np.cross(velocity, angular_momentum) / gravitational_mass
        - position / separation
    )
    semimajor_axis = (
        -gravitational_mass / (2.0 * specific_energy)
        if specific_energy < 0.0
        else None
    )
    return KeplerianElements(
        separation=separation,
        relative_speed=relative_speed,
        specific_energy=float(specific_energy),
        specific_angular_momentum=angular_momentum,
        semimajor_axis=(
            None if semimajor_axis is None else float(semimajor_axis)
        ),
        eccentricity_vector=eccentricity_vector,
        eccentricity=float(np.linalg.norm(eccentricity_vector)),
    )


def keplerian_exchange_rates(
    *,
    mass1_msun: float,
    mass2_msun: float,
    semimajor_axis_pc: float,
    eccentricity: float,
    orbital_power: float,
    orbital_torque: float,
    circular_tolerance: float = 1.0e-12,
) -> OrbitalExchangeRates:
    """Convert secular orbital power and torque to rates of ``a`` and ``e``.

    ``orbital_power`` is ``dE_orb/dt`` and ``orbital_torque`` is
    ``d|L_orb|/dt``.  Energy and angular momentum delivered to the wave have
    the opposite signs.  The conversion assumes the Keplerian internal orbit
    of the binary.  A smooth external potential and a resolved FDM wake must
    first be separated from the calibrated secular exchange.

    The rate of ``e**2`` remains regular for a circular orbit.  A scalar rate
    of ``e`` is undefined at exactly zero eccentricity unless the calibrated
    rates preserve circularity within ``circular_tolerance``.
    """

    values = np.asarray(
        [
            mass1_msun,
            mass2_msun,
            semimajor_axis_pc,
            eccentricity,
            orbital_power,
            orbital_torque,
            circular_tolerance,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(values)):
        raise ValueError("all orbital quantities must be finite")
    if mass1_msun <= 0.0 or mass2_msun <= 0.0 or semimajor_axis_pc <= 0.0:
        raise ValueError("masses and semimajor axis must be positive")
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must satisfy 0 <= e < 1")
    if circular_tolerance < 0.0:
        raise ValueError("circular tolerance must be non-negative")

    total_mass = mass1_msun + mass2_msun
    reduced_mass = mass1_msun * mass2_msun / total_mass
    orbital_energy = -G_INTERNAL * mass1_msun * mass2_msun / (
        2.0 * semimajor_axis_pc
    )
    orbital_angular_momentum = reduced_mass * np.sqrt(
        G_INTERNAL
        * total_mass
        * semimajor_axis_pc
        * (1.0 - eccentricity**2)
    )
    semimajor_axis_rate = (
        2.0
        * semimajor_axis_pc**2
        * orbital_power
        / (G_INTERNAL * mass1_msun * mass2_msun)
    )
    eccentricity_squared_rate = (1.0 - eccentricity**2) * (
        semimajor_axis_rate / semimajor_axis_pc
        - 2.0 * orbital_torque / orbital_angular_momentum
    )
    if eccentricity > 0.0:
        eccentricity_rate = eccentricity_squared_rate / (2.0 * eccentricity)
    elif abs(eccentricity_squared_rate) <= circular_tolerance:
        eccentricity_rate = 0.0
    else:
        eccentricity_rate = None
    return OrbitalExchangeRates(
        orbital_energy=float(orbital_energy),
        orbital_angular_momentum=float(orbital_angular_momentum),
        semimajor_axis_rate_pc_myr=float(semimajor_axis_rate),
        eccentricity_squared_rate_per_myr=float(eccentricity_squared_rate),
        eccentricity_rate_per_myr=(
            None if eccentricity_rate is None else float(eccentricity_rate)
        ),
        wave_energy_rate=float(-orbital_power),
        wave_angular_momentum_rate=float(-orbital_torque),
    )


def advance_keplerian_exchange(
    *,
    mass1_msun: float,
    mass2_msun: float,
    semimajor_axis_pc: float,
    eccentricity: float,
    orbital_power: float,
    orbital_torque: float,
    time_step_myr: float,
    eccentricity_squared_tolerance: float = 1.0e-12,
) -> FiniteOrbitalExchangeStep:
    """Advance a bound Kepler orbit by finite energy and angular-momentum changes.

    The calibrated power and torque must already exclude reversible forcing by
    the smooth FDM potential.  The wave increments have the opposite signs.
    """

    if not np.isfinite(time_step_myr) or time_step_myr <= 0.0:
        raise ValueError("time step must be finite and positive")
    if (
        not np.isfinite(eccentricity_squared_tolerance)
        or eccentricity_squared_tolerance < 0.0
    ):
        raise ValueError("eccentricity tolerance must be finite and non-negative")
    rates = keplerian_exchange_rates(
        mass1_msun=mass1_msun,
        mass2_msun=mass2_msun,
        semimajor_axis_pc=semimajor_axis_pc,
        eccentricity=eccentricity,
        orbital_power=orbital_power,
        orbital_torque=orbital_torque,
    )
    final_energy = rates.orbital_energy + orbital_power * time_step_myr
    final_angular_momentum = (
        rates.orbital_angular_momentum + orbital_torque * time_step_myr
    )
    if final_energy >= 0.0:
        raise ValueError("finite exchange produces an unbound orbit")
    if final_angular_momentum <= 0.0:
        raise ValueError("finite exchange removes all orbital angular momentum")

    total_mass = mass1_msun + mass2_msun
    reduced_mass = mass1_msun * mass2_msun / total_mass
    final_semimajor_axis = -G_INTERNAL * mass1_msun * mass2_msun / (
        2.0 * final_energy
    )
    final_eccentricity_squared = 1.0 - final_angular_momentum**2 / (
        reduced_mass**2 * G_INTERNAL * total_mass * final_semimajor_axis
    )
    if (
        final_eccentricity_squared < -eccentricity_squared_tolerance
        or final_eccentricity_squared >= 1.0
    ):
        raise ValueError("finite exchange produces invalid bound-orbit elements")
    final_eccentricity_squared = max(0.0, final_eccentricity_squared)
    final_eccentricity = np.sqrt(final_eccentricity_squared)
    initial_frequency = np.sqrt(
        G_INTERNAL * total_mass / semimajor_axis_pc**3
    )
    final_frequency = np.sqrt(
        G_INTERNAL * total_mass / final_semimajor_axis**3
    )
    return FiniteOrbitalExchangeStep(
        initial_semimajor_axis_pc=float(semimajor_axis_pc),
        final_semimajor_axis_pc=float(final_semimajor_axis),
        initial_eccentricity=float(eccentricity),
        final_eccentricity=float(final_eccentricity),
        initial_orbital_energy=rates.orbital_energy,
        final_orbital_energy=float(final_energy),
        initial_orbital_angular_momentum=rates.orbital_angular_momentum,
        final_orbital_angular_momentum=float(final_angular_momentum),
        orbital_phase_increment_rad=float(
            0.5 * (initial_frequency + final_frequency) * time_step_myr
        ),
        wave_energy_increment=float(-orbital_power * time_step_myr),
        wave_angular_momentum_increment=float(-orbital_torque * time_step_myr),
    )
