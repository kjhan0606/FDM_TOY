"""Published analytic and empirical separation-curve cross-checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import G_INTERNAL


def koo_separation_pc(time_myr: float | np.ndarray, d0_pc: float, q0: float) -> float | np.ndarray:
    """Koo et al. small-separation analytic curve.

    ``q0`` must be expressed in ``pc^-5/2 Myr^-1`` when the other arguments
    use pc and Myr.
    """

    time = np.asarray(time_myr, dtype=float)
    if np.any(time < 0.0) or d0_pc <= 0.0 or q0 <= 0.0:
        raise ValueError("time must be non-negative and d0_pc/q0 positive")
    result = d0_pc * (1.0 + 2.5 * q0 * d0_pc**2.5 * time) ** (-0.4)
    if np.ndim(time_myr) == 0:
        return float(result)
    return result


def koo_time_between_myr(d_initial_pc: float, d_final_pc: float, q0: float) -> float:
    if not 0.0 < d_final_pc < d_initial_pc or q0 <= 0.0:
        raise ValueError("require 0 < d_final_pc < d_initial_pc and q0 > 0")
    return float(
        ((d_initial_pc / d_final_pc) ** 2.5 - 1.0)
        / (2.5 * q0 * d_initial_pc**2.5)
    )


def koo_separation_rate_at_separation_pc_myr(
    separation_pc: float | np.ndarray, q0: float
) -> float | np.ndarray:
    """Return the derivative implied by the Koo et al. separation fit."""

    separation = np.asarray(separation_pc, dtype=float)
    if np.any(separation <= 0.0) or q0 <= 0.0:
        raise ValueError("separation and q0 must be positive")
    result = -q0 * separation**3.5
    if np.ndim(separation_pc) == 0:
        return float(result)
    return result


def koo_kepler_inferred_orbital_power(
    *,
    separation_pc: float | np.ndarray,
    mass1_msun: float,
    mass2_msun: float,
    q0: float,
) -> float | np.ndarray:
    """Map the Koo separation rate to isolated circular two-body power.

    The FDM potential does not appear in this conversion. The result is a
    comparison quantity and is not the energy deposited in the live wave.
    """

    separation = np.asarray(separation_pc, dtype=float)
    if mass1_msun <= 0.0 or mass2_msun <= 0.0:
        raise ValueError("black hole masses must be positive")
    separation_rate = koo_separation_rate_at_separation_pc_myr(separation, q0)
    result = (
        G_INTERNAL
        * mass1_msun
        * mass2_msun
        * separation_rate
        / (2.0 * separation**2)
    )
    if np.ndim(separation_pc) == 0:
        return float(result)
    return result


def koo_q0_pc_m5half_myr(
    *,
    soliton_mass_msun: float,
    black_hole_mass_msun: float,
    particle_mass_ev: float,
    gamma: float = 2.192,
) -> float:
    """Koo et al. (2024), equation (18).

    ``black_hole_mass_msun`` is the mass of either member of their equal-mass
    binary. The effective mass is ``M_s + 2 gamma M_bh``.
    """

    values = (soliton_mass_msun, black_hole_mass_msun, particle_mass_ev, gamma)
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("masses and gamma must be finite and positive")
    effective_mass = soliton_mass_msun + 2.0 * gamma * black_hole_mass_msun
    return float(
        1.324
        * (effective_mass / 1.0e9) ** 4
        * (black_hole_mass_msun / 1.0e8) ** 0.5
        * (particle_mass_ev / 1.0e-21) ** 8
    )


@dataclass(frozen=True)
class BoeyFit:
    mass_ratio_percent: int
    a_pc: float
    b_per_myr: float
    c: float

    def separation_pc(self, time_myr: float | np.ndarray) -> float | np.ndarray:
        time = np.asarray(time_myr, dtype=float)
        if np.any(time < 0.0):
            raise ValueError("time must be non-negative")
        result = self.a_pc * (1.0 + self.b_per_myr * time) ** (-self.c)
        if np.ndim(time_myr) == 0:
            return float(result)
        return result

    def time_between_myr(self, d_initial_pc: float, d_final_pc: float) -> float:
        if not 0.0 < d_final_pc < d_initial_pc:
            raise ValueError("require 0 < d_final_pc < d_initial_pc")
        return float(
            ((self.a_pc / d_final_pc) ** (1.0 / self.c)
             - (self.a_pc / d_initial_pc) ** (1.0 / self.c))
            / self.b_per_myr
        )


BOEY_2025_FITS: dict[int, BoeyFit] = {
    2: BoeyFit(2, 2.94, 3.12, 0.662),
    5: BoeyFit(5, 2.78, 7.44, 0.777),
    10: BoeyFit(10, 2.72, 22.7, 0.733),
}
