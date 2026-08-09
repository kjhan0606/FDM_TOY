"""Gravitational-wave completion time below the FDM model's stop radius."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .constants import C_PC_PER_MYR, G_INTERNAL


def peters_circular_time_myr(
    mass1_msun: float, mass2_msun: float, semimajor_axis_pc: float
) -> float:
    """Peters coalescence time for a circular binary."""

    if min(mass1_msun, mass2_msun, semimajor_axis_pc) <= 0.0:
        raise ValueError("masses and semimajor axis must be positive")
    return float(
        5.0
        * C_PC_PER_MYR**5
        * semimajor_axis_pc**4
        / (256.0 * G_INTERNAL**3 * mass1_msun * mass2_msun * (mass1_msun + mass2_msun))
    )


def peters_time_myr(
    mass1_msun: float,
    mass2_msun: float,
    semimajor_axis_pc: float,
    eccentricity: float = 0.0,
) -> float:
    """Exact orbit-averaged Peters time for an initial eccentricity.

    The integral form is used for ``e > 0`` and approaches the circular result
    continuously as ``e -> 0``.
    """

    if min(mass1_msun, mass2_msun, semimajor_axis_pc) <= 0.0:
        raise ValueError("masses and semimajor axis must be positive")
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must satisfy 0 <= e < 1")
    if eccentricity < 1.0e-6:
        return peters_circular_time_myr(mass1_msun, mass2_msun, semimajor_axis_pc)

    beta_gw = (
        64.0
        / 5.0
        * G_INTERNAL**3
        * mass1_msun
        * mass2_msun
        * (mass1_msun + mass2_msun)
        / C_PC_PER_MYR**5
    )
    e0 = eccentricity
    c0 = (
        semimajor_axis_pc
        * (1.0 - e0**2)
        * e0 ** (-12.0 / 19.0)
        * (1.0 + 121.0 / 304.0 * e0**2) ** (-870.0 / 2299.0)
    )

    def integrand(eccentricity_value: float) -> float:
        return (
            eccentricity_value ** (29.0 / 19.0)
            * (1.0 + 121.0 / 304.0 * eccentricity_value**2) ** (1181.0 / 2299.0)
            / (1.0 - eccentricity_value**2) ** 1.5
        )

    integral, _ = quad(integrand, 0.0, e0, epsabs=0.0, epsrel=1.0e-10)
    return float(12.0 / 19.0 * c0**4 / beta_gw * integral)
