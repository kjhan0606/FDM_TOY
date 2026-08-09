"""Validity flags for interpreting a toy-model integration."""

from __future__ import annotations

from collections.abc import Iterable


def calibration_flags(m_fdm_ev: float, masses_msun: Iterable[float]) -> list[str]:
    """Flag obvious extrapolation beyond the Koo/Boey calibration envelope."""

    flags: list[str] = []
    if not 5.0e-22 <= m_fdm_ev <= 2.0e-21:
        flags.append("EXTRAPOLATED_M_FDM")
    masses = list(masses_msun)
    if any(mass < 2.0e7 or mass > 1.5e8 for mass in masses):
        flags.append("EXTRAPOLATED_BH_MASS")
    return flags


def integration_flags(
    *,
    status: str,
    max_eta_nl: float,
    min_enclosed_to_bh_mass: float,
    velocity_floor_used: bool,
    max_energy_budget_relerr: float,
    energy_budget_tolerance: float,
    max_injection_to_binding: float,
) -> list[str]:
    flags: list[str] = []
    if max_eta_nl >= 1.0:
        flags.append("NONLINEAR_DF")
    if min_enclosed_to_bh_mass < 1.0:
        flags.append("BACKGROUND_NOT_DOMINANT")
    if velocity_floor_used:
        flags.append("VELOCITY_FLOOR_USED")
    if max_energy_budget_relerr > energy_budget_tolerance:
        flags.append("ENERGY_BUDGET_DRIFT")
    if max_injection_to_binding >= 0.1:
        flags.append("STATIC_SOLITON_BACKREACTION")
    if max_injection_to_binding >= 1.0:
        flags.append("SOLITON_DISRUPTION_POSSIBLE")
    if status == "timeout":
        flags.append("TIMEOUT_CENSORED")
    elif status == "unbound":
        flags.append("UNBOUND_CROSSING")
    elif status == "invalid":
        flags.append("INTEGRATION_INVALID")
    return flags
