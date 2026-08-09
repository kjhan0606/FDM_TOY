"""Direct 3D integration of two SMBHs in a static spherical soliton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from .config import CaseConfig
from .constants import G_INTERNAL
from .soliton import SphericalSoliton
from .validity import calibration_flags, integration_flags
from .wave_drag import DragEvaluation, evaluate_drag


@dataclass(frozen=True)
class InitialState:
    position1_pc: np.ndarray
    position2_pc: np.ndarray
    velocity1_pc_myr: np.ndarray
    velocity2_pc_myr: np.ndarray

    def packed(self) -> np.ndarray:
        return np.concatenate(
            (
                self.position1_pc,
                self.position2_pc,
                self.velocity1_pc_myr,
                self.velocity2_pc_myr,
                # Cumulative lab-frame energy, FDM-rest-frame excitation
                # energy, and three-vector momentum transferred to FDM.
                np.zeros(5),
            )
        )


@dataclass(frozen=True)
class IntegrationResult:
    summary: dict[str, Any]
    timeseries: dict[str, np.ndarray]
    solver_message: str


def make_orbital_state(
    *,
    mass1_msun: float,
    mass2_msun: float,
    separation_pc: float,
    eccentricity: float,
    soliton: SphericalSoliton,
) -> InitialState:
    """Construct a COM-centred binary at apocentre.

    The circular speed includes the differential spherical-soliton force.
    For eccentric cases it is scaled by the Keplerian apocentre factor
    ``sqrt(1-e)``; this is an initializer, not an exact eccentric equilibrium
    in an extended potential.
    """

    total_mass = mass1_msun + mass2_msun
    r1 = np.array([separation_pc * mass2_msun / total_mass, 0.0, 0.0])
    r2 = np.array([-separation_pc * mass1_msun / total_mass, 0.0, 0.0])
    mutual_relative_acceleration = G_INTERNAL * total_mass / separation_pc**2
    soliton_relative_acceleration = -(
        soliton.acceleration(r1) - soliton.acceleration(r2)
    )[0]
    inward_relative_acceleration = mutual_relative_acceleration + soliton_relative_acceleration
    if inward_relative_acceleration <= 0.0:
        raise ValueError("initial state has no inward central acceleration")
    relative_speed = np.sqrt(inward_relative_acceleration * separation_pc)
    relative_speed *= np.sqrt(1.0 - eccentricity)
    v1 = np.array([0.0, relative_speed * mass2_msun / total_mass, 0.0])
    v2 = np.array([0.0, -relative_speed * mass1_msun / total_mass, 0.0])
    return InitialState(r1, r2, v1, v2)


def initial_state_from_config(config: CaseConfig, soliton: SphericalSoliton) -> InitialState:
    binary = config.binary
    if binary.has_explicit_state:
        assert binary.position1_pc is not None
        assert binary.position2_pc is not None
        assert binary.velocity1_pc_myr is not None
        assert binary.velocity2_pc_myr is not None
        return InitialState(
            binary.position1_pc.copy(),
            binary.position2_pc.copy(),
            binary.velocity1_pc_myr.copy(),
            binary.velocity2_pc_myr.copy(),
        )
    assert binary.separation_pc is not None
    return make_orbital_state(
        mass1_msun=binary.mass1_msun,
        mass2_msun=binary.mass2_msun,
        separation_pc=binary.separation_pc,
        eccentricity=binary.eccentricity,
        soliton=soliton,
    )


def _mutual_accelerations(
    mass1_msun: float, mass2_msun: float, r1: np.ndarray, r2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    displacement = r1 - r2
    separation = float(np.linalg.norm(displacement))
    if separation == 0.0:
        raise FloatingPointError("SMBH positions coincide")
    return (
        -G_INTERNAL * mass2_msun * displacement / separation**3,
        G_INTERNAL * mass1_msun * displacement / separation**3,
    )


def _drag_pair(
    config: CaseConfig,
    soliton: SphericalSoliton,
    r1: np.ndarray,
    r2: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> tuple[DragEvaluation, DragEvaluation]:
    separation = float(np.linalg.norm(r1 - r2))
    common = {
        "separation_pc": separation,
        "soliton": soliton,
        "m_fdm_ev": config.fdm.particle_mass_ev,
        "alpha_df": config.model.alpha_df,
        "bulk_velocity_pc_myr": config.model.fdm_bulk_velocity_pc_myr,
        "velocity_floor_pc_myr": config.model.velocity_floor_pc_myr,
    }
    return (
        evaluate_drag(
            mass_msun=config.binary.mass1_msun,
            position_pc=r1,
            velocity_pc_myr=v1,
            **common,
        ),
        evaluate_drag(
            mass_msun=config.binary.mass2_msun,
            position_pc=r2,
            velocity_pc_myr=v2,
            **common,
        ),
    )


def _energies(
    mass1_msun: float,
    mass2_msun: float,
    soliton: SphericalSoliton,
    r1: np.ndarray,
    r2: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> tuple[float, float, float, float]:
    separation = float(np.linalg.norm(r1 - r2))
    kinetic = 0.5 * mass1_msun * float(v1 @ v1) + 0.5 * mass2_msun * float(v2 @ v2)
    binary_potential = -G_INTERNAL * mass1_msun * mass2_msun / separation
    soliton_potential = mass1_msun * soliton.potential(float(np.linalg.norm(r1)))
    soliton_potential += mass2_msun * soliton.potential(float(np.linalg.norm(r2)))
    return kinetic, binary_potential, soliton_potential, kinetic + binary_potential + soliton_potential


def pair_specific_energy(
    mass1_msun: float,
    mass2_msun: float,
    r1: np.ndarray,
    r2: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> float:
    separation = float(np.linalg.norm(r1 - r2))
    relative_speed2 = float((v1 - v2) @ (v1 - v2))
    return 0.5 * relative_speed2 - G_INTERNAL * (mass1_msun + mass2_msun) / separation


def _osculating_eccentricity(
    mass1_msun: float,
    mass2_msun: float,
    r1: np.ndarray,
    r2: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> float:
    displacement = r1 - r2
    velocity = v1 - v2
    separation = float(np.linalg.norm(displacement))
    h = np.cross(displacement, velocity)
    e_vector = np.cross(velocity, h) / (G_INTERNAL * (mass1_msun + mass2_msun))
    e_vector -= displacement / separation
    return float(np.linalg.norm(e_vector))


def integrate_case(config: CaseConfig) -> IntegrationResult:
    """Integrate one validated case and return summary plus sampled history."""

    soliton = config.fdm.build_soliton()
    initial = initial_state_from_config(config, soliton)
    m1 = config.binary.mass1_msun
    m2 = config.binary.mass2_msun
    stop = config.integration.stop_separation_pc
    initial_separation = float(np.linalg.norm(initial.position1_pc - initial.position2_pc))
    if initial_separation <= stop:
        raise ValueError("initial separation must exceed integration.stop_separation")

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        r1, r2 = state[0:3], state[3:6]
        v1, v2 = state[6:9], state[9:12]
        mutual1, mutual2 = _mutual_accelerations(m1, m2, r1, r2)
        acceleration1 = mutual1 + soliton.acceleration(r1)
        acceleration2 = mutual2 + soliton.acceleration(r2)
        drag1, drag2 = _drag_pair(config, soliton, r1, r2, v1, v2)
        power_to_fdm_lab = 0.0
        power_to_fdm_rest = 0.0
        momentum_rate_to_fdm = np.zeros(3, dtype=float)
        if config.model.drag:
            acceleration1 = acceleration1 + drag1.acceleration_pc_myr2
            acceleration2 = acceleration2 + drag2.acceleration_pc_myr2
            power_to_fdm_lab = -float(
                drag1.force_msun_pc_myr2 @ v1
                + drag2.force_msun_pc_myr2 @ v2
            )
            power_to_fdm_rest = -float(
                drag1.force_msun_pc_myr2 @ drag1.relative_velocity_pc_myr
                + drag2.force_msun_pc_myr2 @ drag2.relative_velocity_pc_myr
            )
            momentum_rate_to_fdm = -(
                drag1.force_msun_pc_myr2 + drag2.force_msun_pc_myr2
            )
        return np.concatenate(
            (
                v1,
                v2,
                acceleration1,
                acceleration2,
                [power_to_fdm_lab, power_to_fdm_rest],
                momentum_rate_to_fdm,
            )
        )

    def target_event(_time: float, state: np.ndarray) -> float:
        return float(np.linalg.norm(state[0:3] - state[3:6]) - stop)

    target_event.terminal = True  # type: ignore[attr-defined]
    target_event.direction = -1.0  # type: ignore[attr-defined]

    solution = solve_ivp(
        rhs,
        (0.0, config.integration.max_time_myr),
        initial.packed(),
        method="DOP853",
        rtol=config.integration.rtol,
        atol=config.integration.atol,
        max_step=config.integration.max_step_myr,
        dense_output=True,
        events=target_event,
    )

    reached = bool(solution.t_events[0].size)
    final_time = float(solution.t_events[0][0]) if reached else float(solution.t[-1])
    if not solution.success or solution.sol is None:
        status = "invalid"
    elif reached:
        event_state = solution.sol(final_time)
        bound = pair_specific_energy(
            m1, m2, event_state[0:3], event_state[3:6], event_state[6:9], event_state[9:12]
        ) < 0.0
        reached_status = "reached_0p01pc" if np.isclose(stop, 0.01) else "reached_target"
        status = reached_status if bound else "unbound"
    else:
        final_state = solution.y[:, -1]
        final_separation = float(np.linalg.norm(final_state[0:3] - final_state[3:6]))
        unbound_and_separated = (
            pair_specific_energy(
                m1,
                m2,
                final_state[0:3],
                final_state[3:6],
                final_state[6:9],
                final_state[9:12],
            )
            >= 0.0
            and final_separation > 10.0 * initial_separation
        )
        status = "unbound" if unbound_and_separated else "timeout"

    sample_times = np.linspace(0.0, final_time, config.integration.output_samples)
    sampled = solution.sol(sample_times) if solution.sol is not None else np.repeat(solution.y[:, -1:], config.integration.output_samples, axis=1)

    columns: dict[str, list[float]] = {
        name: []
        for name in (
            "D_pc",
            "r1_x_pc", "r1_y_pc", "r1_z_pc",
            "r2_x_pc", "r2_y_pc", "r2_z_pc",
            "v1_x_pc_myr", "v1_y_pc_myr", "v1_z_pc_myr",
            "v2_x_pc_myr", "v2_y_pc_myr", "v2_z_pc_myr",
            "E_kin", "E_bh_bh", "E_soliton", "E_mech",
            "E_to_fdm", "E_to_fdm_rest", "E_budget",
            "P_to_fdm_x", "P_to_fdm_y", "P_to_fdm_z",
            "power_to_fdm", "power_to_fdm_rest",
            "force_to_fdm_x", "force_to_fdm_y", "force_to_fdm_z",
            "L_x", "L_y", "L_z", "eccentricity_osculating",
            "rho1_msun_pc3", "rho2_msun_pc3", "q1", "q2", "eta_nl1", "eta_nl2",
            "menc_to_bh1", "menc_to_bh2", "velocity_floor_used",
        )
    }
    for index in range(sample_times.size):
        state = sampled[:, index]
        r1, r2 = state[0:3], state[3:6]
        v1, v2 = state[6:9], state[9:12]
        drag1, drag2 = _drag_pair(config, soliton, r1, r2, v1, v2)
        kinetic, binary_potential, soliton_potential, mechanical = _energies(
            m1, m2, soliton, r1, r2, v1, v2
        )
        angular_momentum = m1 * np.cross(r1, v1) + m2 * np.cross(r2, v2)
        separation = float(np.linalg.norm(r1 - r2))
        force_to_fdm = -(drag1.force_msun_pc_myr2 + drag2.force_msun_pc_myr2)
        if config.model.drag:
            power_to_fdm = -float(
                drag1.force_msun_pc_myr2 @ v1 + drag2.force_msun_pc_myr2 @ v2
            )
            power_to_fdm_rest = -float(
                drag1.force_msun_pc_myr2 @ drag1.relative_velocity_pc_myr
                + drag2.force_msun_pc_myr2 @ drag2.relative_velocity_pc_myr
            )
        else:
            force_to_fdm = np.zeros(3, dtype=float)
            power_to_fdm = 0.0
            power_to_fdm_rest = 0.0
        values = {
            "D_pc": separation,
            "r1_x_pc": r1[0], "r1_y_pc": r1[1], "r1_z_pc": r1[2],
            "r2_x_pc": r2[0], "r2_y_pc": r2[1], "r2_z_pc": r2[2],
            "v1_x_pc_myr": v1[0], "v1_y_pc_myr": v1[1], "v1_z_pc_myr": v1[2],
            "v2_x_pc_myr": v2[0], "v2_y_pc_myr": v2[1], "v2_z_pc_myr": v2[2],
            "E_kin": kinetic,
            "E_bh_bh": binary_potential,
            "E_soliton": soliton_potential,
            "E_mech": mechanical,
            "E_to_fdm": state[12],
            "E_to_fdm_rest": state[13],
            "E_budget": mechanical + state[12],
            "P_to_fdm_x": state[14], "P_to_fdm_y": state[15], "P_to_fdm_z": state[16],
            "power_to_fdm": power_to_fdm,
            "power_to_fdm_rest": power_to_fdm_rest,
            "force_to_fdm_x": force_to_fdm[0],
            "force_to_fdm_y": force_to_fdm[1],
            "force_to_fdm_z": force_to_fdm[2],
            "L_x": angular_momentum[0], "L_y": angular_momentum[1], "L_z": angular_momentum[2],
            "eccentricity_osculating": _osculating_eccentricity(m1, m2, r1, r2, v1, v2),
            "rho1_msun_pc3": drag1.density_msun_pc3,
            "rho2_msun_pc3": drag2.density_msun_pc3,
            "q1": drag1.q, "q2": drag2.q,
            "eta_nl1": drag1.eta_nl, "eta_nl2": drag2.eta_nl,
            "menc_to_bh1": drag1.enclosed_to_bh_mass,
            "menc_to_bh2": drag2.enclosed_to_bh_mass,
            "velocity_floor_used": float(drag1.used_velocity_floor or drag2.used_velocity_floor),
        }
        for name, value in values.items():
            columns[name].append(float(value))

    timeseries: dict[str, np.ndarray] = {"t_myr": sample_times}
    timeseries.update({name: np.asarray(values) for name, values in columns.items()})
    budget = timeseries["E_budget"]
    energy_scale = max(
        abs(timeseries["E_mech"][0]),
        abs(timeseries["E_kin"][0]),
        abs(timeseries["E_bh_bh"][0]),
        abs(timeseries["E_soliton"][0]),
        np.finfo(float).tiny,
    )
    max_budget_error = float(np.max(np.abs(budget - budget[0])) / energy_scale)
    max_eta = float(max(np.max(timeseries["eta_nl1"]), np.max(timeseries["eta_nl2"])))
    min_mass_ratio = float(min(np.min(timeseries["menc_to_bh1"]), np.min(timeseries["menc_to_bh2"])))
    floor_used = bool(np.any(timeseries["velocity_floor_used"] > 0.0))
    soliton_binding_energy = soliton.virial_binding_energy()
    max_injection_ratio = float(
        np.max(np.maximum(timeseries["E_to_fdm_rest"], 0.0)) / soliton_binding_energy
    )
    if max_budget_error > config.integration.energy_budget_relerr_limit:
        status = "invalid"
    flags = calibration_flags(config.fdm.particle_mass_ev, (m1, m2))
    metadata = config.raw.get("metadata", {})
    if isinstance(metadata, dict) and (
        metadata.get("calibrated") is False
        or metadata.get("validated_profile_definition") is False
    ):
        flags.append("UNVALIDATED_SOLITON_PROFILE")
    flags.extend(
        integration_flags(
            status=status,
            max_eta_nl=max_eta,
            min_enclosed_to_bh_mass=min_mass_ratio,
            velocity_floor_used=floor_used,
            max_energy_budget_relerr=max_budget_error,
            energy_budget_tolerance=config.integration.energy_budget_relerr_limit,
            max_injection_to_binding=max_injection_ratio,
        )
    )
    flags = list(dict.fromkeys(flags))
    summary: dict[str, Any] = {
        "model_version": "0.1.0",
        "status": status,
        "t_fdm_myr": final_time if status in {"reached_0p01pc", "reached_target"} else None,
        "integration_time_myr": final_time,
        "D_stop_pc": stop,
        "D_initial_pc": initial_separation,
        "D_final_pc": float(timeseries["D_pc"][-1]),
        "E_initial": float(timeseries["E_mech"][0]),
        "E_final": float(timeseries["E_mech"][-1]),
        "E_to_fdm": float(timeseries["E_to_fdm"][-1]),
        "E_to_fdm_rest": float(timeseries["E_to_fdm_rest"][-1]),
        "P_to_fdm": [
            float(timeseries["P_to_fdm_x"][-1]),
            float(timeseries["P_to_fdm_y"][-1]),
            float(timeseries["P_to_fdm_z"][-1]),
        ],
        "soliton_virial_binding_energy": soliton_binding_energy,
        "max_fdm_injection_to_binding": max_injection_ratio,
        "max_energy_budget_relerr": max_budget_error,
        "max_q": float(max(np.max(timeseries["q1"]), np.max(timeseries["q2"]))),
        "max_eta_nl": max_eta,
        "min_menc_to_bh": min_mass_ratio,
        "validity_flags": flags,
    }
    return IntegrationResult(summary, timeseries, solution.message)
