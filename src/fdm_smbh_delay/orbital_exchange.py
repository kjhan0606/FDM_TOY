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
