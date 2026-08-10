"""Physical definitions used by the Koo (2024) and Boey (2025) benchmarks."""

from __future__ import annotations

from .soliton import SchiveSoliton


def koo2024_central_density_msun_pc3(
    particle_mass_ev: float, soliton_mass_msun: float
) -> float:
    """Koo et al. (2024), equation (5)."""

    if particle_mass_ev <= 0.0 or soliton_mass_msun <= 0.0:
        raise ValueError("particle and soliton masses must be positive")
    return float(
        7.05e6
        * (particle_mass_ev / 1.0e-21) ** 6
        * (soliton_mass_msun / 1.0e9) ** 4
    )


def equivalent_schive_core_radius_pc(
    *, total_mass_msun: float, central_density_msun_pc3: float
) -> float:
    """Half-density radius of a Schive fit matching total mass and density."""

    if total_mass_msun <= 0.0 or central_density_msun_pc3 <= 0.0:
        raise ValueError("total mass and central density must be positive")
    unit_profile_mass = SchiveSoliton.from_central_density(
        central_density_msun_pc3, 1.0
    ).total_mass_msun
    return float((total_mass_msun / unit_profile_mass) ** (1.0 / 3.0))


def koo2024_equivalent_schive_soliton(
    particle_mass_ev: float = 1.0e-21,
    soliton_mass_msun: float = 1.0e9,
) -> SchiveSoliton:
    """Schive proxy matching Koo's ground-state mass and central density.

    Koo et al. evolve a numerical Schrödinger--Poisson ground state rather than
    the analytic Schive fit. This proxy is used only for the static comparison.
    """

    density = koo2024_central_density_msun_pc3(
        particle_mass_ev, soliton_mass_msun
    )
    core_radius = equivalent_schive_core_radius_pc(
        total_mass_msun=soliton_mass_msun,
        central_density_msun_pc3=density,
    )
    return SchiveSoliton.from_central_density(density, core_radius)


def boey2025_central_density_msun_pc3(
    particle_mass_ev: float, core_radius_pc: float
) -> float:
    """Boey et al. (2025), equation (4), for the half-density radius."""

    if particle_mass_ev <= 0.0 or core_radius_pc <= 0.0:
        raise ValueError("particle mass and core radius must be positive")
    return float(
        1.9
        * (particle_mass_ev / 1.0e-23) ** -2
        * (core_radius_pc / 1.0e3) ** -4
    )


def boey2025_schive_soliton(
    particle_mass_ev: float = 1.0e-21,
    core_radius_pc: float = 2.2,
) -> SchiveSoliton:
    density = boey2025_central_density_msun_pc3(
        particle_mass_ev, core_radius_pc
    )
    return SchiveSoliton.from_central_density(density, core_radius_pc)


def soliton_definition_summary(soliton: SchiveSoliton) -> dict[str, float]:
    mass_within_core = float(soliton.enclosed_mass(soliton.core_radius_pc))
    return {
        "core_radius_pc": soliton.core_radius_pc,
        "central_density_msun_pc3": soliton.central_density_msun_pc3,
        "total_profile_mass_msun": soliton.total_mass_msun,
        "mass_within_core_radius_msun": mass_within_core,
        "fraction_within_core_radius": mass_within_core / soliton.total_mass_msun,
    }
