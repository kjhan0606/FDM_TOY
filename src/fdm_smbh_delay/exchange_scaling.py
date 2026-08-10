"""Natural scales for FDM and SMBH energy and angular-momentum exchange."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import G_INTERNAL


@dataclass(frozen=True)
class ExchangeScales:
    soliton_dynamical_time_myr: float
    orbital_energy_msun_pc2_myr2: float
    orbital_angular_momentum_msun_pc2_myr: float
    orbital_power_msun_pc2_myr3: float
    orbital_torque_msun_pc2_myr2: float


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
