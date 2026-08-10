"""Natural scales for FDM and SMBH energy and angular-momentum exchange."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.constants import G, c, hbar

from .constants import G_INTERNAL


@dataclass(frozen=True)
class ExchangeScales:
    soliton_dynamical_time_myr: float
    orbital_energy_msun_pc2_myr2: float
    orbital_angular_momentum_msun_pc2_myr: float
    orbital_power_msun_pc2_myr3: float
    orbital_torque_msun_pc2_myr2: float


def schrodinger_poisson_similarity_parameter(
    *,
    particle_mass_ev: float,
    soliton_mass_msun: float,
    core_radius_pc: float,
) -> float:
    """Return ``hbar^2/(G m^2 M_s r_c)`` for a physical soliton.

    This dimensionless coefficient multiplies the quantum term relative to
    self-gravity after length, mass, and time are normalized by ``r_c``,
    ``M_s``, and ``sqrt(r_c^3/(G M_s))``. Self-similar ground-state solitons
    share the same value even when their physical boson masses differ.
    """

    values = np.asarray(
        [particle_mass_ev, soliton_mass_msun, core_radius_pc], dtype=float
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(
            "particle mass, soliton mass, and core radius must be positive"
        )
    particle_mass = particle_mass_ev * u.eV / c**2
    return float(
        (
            hbar**2
            / (
                G
                * particle_mass**2
                * (soliton_mass_msun * u.Msun)
                * (core_radius_pc * u.pc)
            )
        )
        .decompose()
        .value
    )


def exchange_scales(
    *,
    mass1_msun: float,
    mass2_msun: float,
    soliton_mass_msun: float,
    core_radius_pc: float,
) -> ExchangeScales:
    """Return scales defined by the soliton dynamical time and core radius."""

    values = np.asarray(
        [mass1_msun, mass2_msun, soliton_mass_msun, core_radius_pc], dtype=float
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("SMBH masses, soliton mass, and core radius must be positive")
    binary_mass = mass1_msun + mass2_msun
    reduced_mass = mass1_msun * mass2_msun / binary_mass
    dynamical_time = np.sqrt(core_radius_pc**3 / (G_INTERNAL * soliton_mass_msun))
    energy = G_INTERNAL * mass1_msun * mass2_msun / core_radius_pc
    angular_momentum = reduced_mass * np.sqrt(
        G_INTERNAL * binary_mass * core_radius_pc
    )
    return ExchangeScales(
        soliton_dynamical_time_myr=float(dynamical_time),
        orbital_energy_msun_pc2_myr2=float(energy),
        orbital_angular_momentum_msun_pc2_myr=float(angular_momentum),
        orbital_power_msun_pc2_myr3=float(energy / dynamical_time),
        orbital_torque_msun_pc2_myr2=float(angular_momentum / dynamical_time),
    )
