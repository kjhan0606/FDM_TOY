"""Unit conversions and file ordering for public PyUL_NBody outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PYUL_PARSEC_M = 3.0857e16
PYUL_SOLAR_MASS_KG = 1.989e30
PYUL_YEAR_S = 60.0 * 60.0 * 24.0 * 365.0
PYUL_MYR_S = 1.0e6 * PYUL_YEAR_S


@dataclass(frozen=True)
class PyulUnitSystem:
    length_pc: float
    time_myr: float
    mass_msun: float
    energy_msun_pc2_myr2: float

    @property
    def velocity_pc_myr(self) -> float:
        return self.length_pc / self.time_myr

    @property
    def density_msun_pc3(self) -> float:
        return self.mass_msun / self.length_pc**3

    @property
    def mass_rate_msun_myr(self) -> float:
        return self.mass_msun / self.time_myr

    @property
    def energy_rate_msun_pc2_myr3(self) -> float:
        return self.energy_msun_pc2_myr2 / self.time_myr


def pyul_unit_system(metadata: dict) -> PyulUnitSystem:
    """Build physical conversions from adapter metadata or PyUL constants."""

    keys = (
        "pyul_length_unit_m",
        "pyul_time_unit_s",
        "pyul_mass_unit_kg",
        "pyul_energy_unit_j",
    )
    if all(key in metadata for key in keys):
        length_m, time_s, mass_kg, energy_j = (
            float(metadata[key]) for key in keys
        )
    else:
        axion_mass = metadata["particle_mass_ev"] * 1.78266191e-36
        hbar_si = 1.0545718e-34
        parsec_m = 3.0857e16
        gravitational_si = 6.67e-11
        omega_m = 0.31
        hubble_si = 67.7 / (parsec_m * 1.0e3)
        time_s = (3.0 * hubble_si**2 * omega_m / (8.0 * np.pi)) ** -0.5
        length_m = (
            8.0
            * np.pi
            * hbar_si**2
            / (3.0 * axion_mass**2 * hubble_si**2 * omega_m)
        ) ** 0.25
        mass_kg = (
            (3.0 * hubble_si**2 * omega_m / (8.0 * np.pi)) ** 0.25
            * hbar_si**1.5
            / (axion_mass**1.5 * gravitational_si)
        )
        energy_j = mass_kg * length_m**2 / time_s**2
    energy_internal_j = PYUL_SOLAR_MASS_KG * (PYUL_PARSEC_M / PYUL_MYR_S) ** 2
    return PyulUnitSystem(
        length_pc=length_m / PYUL_PARSEC_M,
        time_myr=time_s / PYUL_MYR_S,
        mass_msun=mass_kg / PYUL_SOLAR_MASS_KG,
        energy_msun_pc2_myr2=energy_j / energy_internal_j,
    )


def ordered_output_paths(directory: Path, pattern: str) -> list[Path]:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {directory / pattern}")
    return paths
