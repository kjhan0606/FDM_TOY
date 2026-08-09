"""Published analytic and empirical separation-curve cross-checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
