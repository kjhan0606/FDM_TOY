"""Restartable kpc-to-common-nucleus integration with explicit force budgets."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .binary_evolution import BoundBinaryState
from .constants import G_INTERNAL
from .delay_budget import DelaySegment
from .environmental_friction import (
    FDMFrictionEvaluation,
    FrictionEvaluation,
    fdm_dynamical_friction,
    gaseous_dynamical_friction,
    stellar_dynamical_friction,
)
from .galaxy_environment import (
    CompositePotential,
    FDMBackground,
    GasBackground,
    NuclearEnvelope,
    StellarBackground,
)
from .kpc_to_pc import (
    InspiralPhase,
    InspiralState,
    hard_binary_radius_pc,
    influence_radius_pc,
    transition_state,
)
from .orbital_exchange import KeplerianElements, keplerian_elements_from_relative_state


@dataclass(frozen=True)
class StellarFrictionConfig:
    minimum_impact_parameter_pc: float
    maximum_impact_parameter_pc: float | None = None
    core_stall_enclosed_mass_ratio: float = 0.0


@dataclass(frozen=True)
class GasFrictionConfig:
    coulomb_logarithm: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.coulomb_logarithm) or self.coulomb_logarithm < 0.0:
            raise ValueError("gas Coulomb logarithm must be finite and non-negative")


@dataclass(frozen=True)
class KpcInspiralModel:
    host_potential: CompositePotential
    secondary_bh_mass_msun: float
    nuclear_envelope: NuclearEnvelope | None = None
    stellar_background: StellarBackground | None = None
    stellar_friction: StellarFrictionConfig | None = None
    gas_background: GasBackground | None = None
    gas_friction: GasFrictionConfig | None = None
    fdm_background: FDMBackground | None = None

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.secondary_bh_mass_msun)
            or self.secondary_bh_mass_msun <= 0.0
        ):
            raise ValueError("secondary SMBH mass must be finite and positive")
        if (self.stellar_background is None) != (self.stellar_friction is None):
            raise ValueError("stellar background and friction controls must be paired")
        if (self.gas_background is None) != (self.gas_friction is None):
            raise ValueError("gas background and friction controls must be paired")


@dataclass(frozen=True)
class KpcIntegrationConfig:
    target_radius_pc: float
    maximum_time_myr: float
    maximum_step_myr: float
    timestep_fraction: float = 0.02
    maximum_steps: int = 1_000_000
    sample_interval_steps: int = 1

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.target_radius_pc,
                self.maximum_time_myr,
                self.maximum_step_myr,
                self.timestep_fraction,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("kpc integration controls must be finite and positive")
        if self.timestep_fraction > 0.25:
            raise ValueError("timestep_fraction must not exceed 0.25")
        if self.maximum_steps < 1 or self.sample_interval_steps < 1:
            raise ValueError("step counts must be positive integers")


@dataclass(frozen=True)
class DualNucleusState:
    elapsed_myr: float
    position_pc: np.ndarray
    velocity_pc_myr: np.ndarray
    envelope_truncation_radius_pc: float | None
    completed_steps: int = 0

    def __post_init__(self) -> None:
        position = np.asarray(self.position_pc, dtype=float)
        velocity = np.asarray(self.velocity_pc_myr, dtype=float)
        if (
            position.shape != (3,)
            or velocity.shape != (3,)
            or np.any(~np.isfinite(position))
            or np.any(~np.isfinite(velocity))
            or not np.isfinite(self.elapsed_myr)
            or self.elapsed_myr < 0.0
            or self.completed_steps < 0
        ):
            raise ValueError("dual-nucleus state is invalid")
        if self.envelope_truncation_radius_pc is not None and (
            not np.isfinite(self.envelope_truncation_radius_pc)
            or self.envelope_truncation_radius_pc <= 0.0
        ):
            raise ValueError("envelope truncation radius must be positive")
        object.__setattr__(self, "position_pc", position)
        object.__setattr__(self, "velocity_pc_myr", velocity)

    @property
    def radius_pc(self) -> float:
        return float(np.linalg.norm(self.position_pc))


@dataclass(frozen=True)
class ForceBudget:
    conservative_acceleration_pc_myr2: np.ndarray
    stellar_acceleration_pc_myr2: np.ndarray
    gas_acceleration_pc_myr2: np.ndarray
    fdm_acceleration_pc_myr2: np.ndarray
    total_acceleration_pc_myr2: np.ndarray
    effective_perturber_mass_msun: float
    bound_envelope_mass_msun: float
    stellar: FrictionEvaluation | None
    gas: FrictionEvaluation | None
    fdm: FDMFrictionEvaluation | None

    @property
    def dissipative_acceleration_pc_myr2(self) -> np.ndarray:
        return (
            self.stellar_acceleration_pc_myr2
            + self.gas_acceleration_pc_myr2
            + self.fdm_acceleration_pc_myr2
        )


@dataclass(frozen=True)
class KpcInspiralSample:
    elapsed_myr: float
    radius_pc: float
    speed_pc_myr: float
    specific_orbital_energy_pc2_myr2: float
    drag_work_rate_pc2_myr3: float
    bound_envelope_mass_msun: float
    envelope_truncation_radius_pc: float | None


@dataclass(frozen=True)
class KpcInspiralResult:
    status: str
    final_state: DualNucleusState
    samples: tuple[KpcInspiralSample, ...]
    reason: str

    @property
    def delay_segment(self) -> DelaySegment:
        if self.status == "reached_common_nucleus":
            return DelaySegment("kpc_to_pc", "complete", self.final_state.elapsed_myr)
        if self.status in {"timeout", "stalled"}:
            return DelaySegment(
                "kpc_to_pc",
                "timeout",
                None,
                elapsed_lower_bound_myr=self.final_state.elapsed_myr,
            )
        if self.status == "checkpoint":
            return DelaySegment(
                "kpc_to_pc",
                "missing",
                None,
                elapsed_lower_bound_myr=self.final_state.elapsed_myr,
            )
        return DelaySegment("kpc_to_pc", "invalid", None)


@dataclass(frozen=True)
class KpcToHardConfig:
    """Controls for the static-host kpc-to-hard-binary approximation.

    The state is the secondary's coordinate relative to the primary.  The
    primary must be the central point mass already present in
    ``KpcInspiralModel.host_potential``; the relative equation adds the
    primary's reflex acceleration so the mutual term contains ``M1 + M2``.
    The remaining host components are spherical and time independent.
    Galaxy-merger torques, centre motion through an asymmetric host,
    triaxiality, and an evolving host lie outside this model and require a
    resolved zoom calibration rather than silent extrapolation.
    """

    primary_bh_mass_msun: float
    common_nucleus_radius_pc: float
    sigma_pc_myr: float
    maximum_time_myr: float
    maximum_step_myr: float
    hard_binary_radius_pc: float | None = None
    timestep_fraction: float = 0.02
    maximum_steps: int = 1_000_000
    sample_interval_steps: int = 1

    def __post_init__(self) -> None:
        values = [
            self.primary_bh_mass_msun,
            self.common_nucleus_radius_pc,
            self.sigma_pc_myr,
            self.maximum_time_myr,
            self.maximum_step_myr,
            self.timestep_fraction,
        ]
        if self.hard_binary_radius_pc is not None:
            values.append(self.hard_binary_radius_pc)
        array = np.asarray(values, dtype=float)
        if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
            raise ValueError("phase-aware kpc integration controls must be positive")
        if self.timestep_fraction > 0.25:
            raise ValueError("timestep_fraction must not exceed 0.25")
        if self.maximum_steps < 1 or self.sample_interval_steps < 1:
            raise ValueError("step counts must be positive integers")


@dataclass(frozen=True)
class KpcToHardState:
    """Restartable dynamical state together with its complete phase history."""

    dynamical_state: DualNucleusState
    inspiral_state: InspiralState
    transition_history: tuple[InspiralState, ...]

    def __post_init__(self) -> None:
        if not self.transition_history:
            raise ValueError("phase-aware state requires a transition history")
        if self.transition_history[-1] != self.inspiral_state:
            raise ValueError("transition history must end at inspiral_state")
        event_uid = self.inspiral_state.event_uid
        if any(state.event_uid != event_uid for state in self.transition_history):
            raise ValueError("transition history event_uid values must agree")
        elapsed = [state.elapsed_myr for state in self.transition_history]
        if any(later < earlier for earlier, later in zip(elapsed, elapsed[1:])):
            raise ValueError("transition history elapsed times must be non-decreasing")
        if self.inspiral_state.elapsed_myr > self.dynamical_state.elapsed_myr:
            raise ValueError("phase transition cannot postdate the dynamical state")


@dataclass(frozen=True)
class KpcToHardResult:
    status: str
    final_state: KpcToHardState
    samples: tuple[KpcInspiralSample, ...]
    reason: str
    binary_initial_state: BoundBinaryState | None

    @property
    def delay_segment(self) -> DelaySegment:
        elapsed = self.final_state.dynamical_state.elapsed_myr
        if self.status == "reached_hard_binary":
            return DelaySegment("kpc_to_hard", "complete", elapsed)
        if self.status in {"timeout", "stalled", "outside"}:
            return DelaySegment(
                "kpc_to_hard", "timeout", None, elapsed_lower_bound_myr=elapsed
            )
        if self.status == "checkpoint":
            return DelaySegment(
                "kpc_to_hard", "missing", None, elapsed_lower_bound_myr=elapsed
            )
        return DelaySegment("kpc_to_hard", "invalid", None)


def initial_dual_nucleus_state(
    *,
    position_pc: np.ndarray,
    velocity_pc_myr: np.ndarray,
    model: KpcInspiralModel,
) -> DualNucleusState:
    truncation = (
        None
        if model.nuclear_envelope is None
        else model.nuclear_envelope.initial_truncation_radius_pc
    )
    return DualNucleusState(
        elapsed_myr=0.0,
        position_pc=position_pc,
        velocity_pc_myr=velocity_pc_myr,
        envelope_truncation_radius_pc=truncation,
    )


def _selected_hard_binary_radius_pc(
    model: KpcInspiralModel, config: KpcToHardConfig
) -> float:
    conventional = hard_binary_radius_pc(
        config.primary_bh_mass_msun,
        model.secondary_bh_mass_msun,
        config.sigma_pc_myr,
    )
    selected = (
        conventional
        if config.hard_binary_radius_pc is None
        else config.hard_binary_radius_pc
    )
    influence = influence_radius_pc(
        config.primary_bh_mass_msun + model.secondary_bh_mass_msun,
        config.sigma_pc_myr,
    )
    if selected > min(influence, config.common_nucleus_radius_pc):
        raise ValueError(
            "hard-binary radius must lie inside both the influence and common-nucleus radii"
        )
    return float(selected)


def _static_host_scope_error(
    model: KpcInspiralModel, config: KpcToHardConfig
) -> str | None:
    central_mass = model.host_potential.central_point_mass_msun
    if not np.isclose(
        central_mass,
        config.primary_bh_mass_msun,
        rtol=1.0e-12,
        atol=0.0,
    ):
        return (
            "the static-host approximation requires host_potential.central_point_mass_msun "
            "to equal primary_bh_mass_msun"
        )
    return None


def _osculating_elements(
    state: DualNucleusState,
    model: KpcInspiralModel,
    config: KpcToHardConfig,
) -> KeplerianElements:
    return keplerian_elements_from_relative_state(
        total_mass=config.primary_bh_mass_msun + model.secondary_bh_mass_msun,
        displacement=state.position_pc,
        relative_velocity=state.velocity_pc_myr,
    )


def _append_phase(
    state: KpcToHardState,
    target: InspiralPhase,
    *,
    elements: KeplerianElements,
    reason: str,
) -> KpcToHardState:
    updated = transition_state(
        state.inspiral_state,
        target,
        elapsed_myr=state.dynamical_state.elapsed_myr,
        reason=reason,
        separation_pc=elements.separation,
        semimajor_axis_pc=elements.semimajor_axis,
        eccentricity=elements.eccentricity,
    )
    return KpcToHardState(
        dynamical_state=state.dynamical_state,
        inspiral_state=updated,
        transition_history=state.transition_history + (updated,),
    )


def _replace_dynamics(
    state: KpcToHardState, dynamics: DualNucleusState
) -> KpcToHardState:
    return KpcToHardState(
        dynamics, state.inspiral_state, state.transition_history
    )


def _advance_physical_phase(
    state: KpcToHardState,
    model: KpcInspiralModel,
    config: KpcToHardConfig,
) -> tuple[KpcToHardState, str | None]:
    """Update monotonic physical phases after one accepted dynamical state."""

    elements = _osculating_elements(state.dynamical_state, model, config)
    hard_radius = _selected_hard_binary_radius_pc(model, config)
    phase = state.inspiral_state.phase

    if phase is InspiralPhase.DUAL_NUCLEUS:
        if elements.separation > config.common_nucleus_radius_pc:
            return state, None
        if elements.specific_energy >= 0.0:
            state = _append_phase(
                state,
                InspiralPhase.COMMON_NUCLEUS_UNBOUND,
                elements=elements,
                reason=(
                    "secondary entered the common nucleus but its osculating "
                    "two-body energy remains non-negative"
                ),
            )
            return state, None
        state = _append_phase(
            state,
            InspiralPhase.BOUND_BINARY,
            elements=elements,
            reason="negative osculating two-body energy inside the common nucleus",
        )
        phase = state.inspiral_state.phase

    if phase is InspiralPhase.COMMON_NUCLEUS_UNBOUND:
        if elements.specific_energy >= 0.0:
            if elements.separation > config.common_nucleus_radius_pc:
                state = _append_phase(
                    state,
                    InspiralPhase.CENSORED,
                    elements=elements,
                    reason=(
                        "unbound common-nucleus passage exited the static-host "
                        "calibration domain"
                    ),
                )
                return state, "outside"
            return state, None
        state = _append_phase(
            state,
            InspiralPhase.BOUND_BINARY,
            elements=elements,
            reason="osculating two-body energy became negative inside the common nucleus",
        )
        phase = state.inspiral_state.phase

    if phase is InspiralPhase.BOUND_BINARY:
        if elements.specific_energy >= 0.0 or elements.semimajor_axis is None:
            state = _append_phase(
                state,
                InspiralPhase.CENSORED,
                elements=elements,
                reason=(
                    "the osculating bound orbit left the static-host approximation domain"
                ),
            )
            return state, "outside"
        if elements.semimajor_axis <= hard_radius:
            state = _append_phase(
                state,
                InspiralPhase.HARD_BINARY,
                elements=elements,
                reason="semimajor axis reached the event-specific hard-binary radius",
            )
    return state, None


def initial_kpc_to_hard_state(
    *,
    event_uid: str,
    dynamical_state: DualNucleusState,
    model: KpcInspiralModel,
    config: KpcToHardConfig,
) -> KpcToHardState:
    """Classify a relative state without equating common-nucleus entry with capture."""

    elements = _osculating_elements(dynamical_state, model, config)
    initial = InspiralState(
        event_uid=event_uid,
        phase=InspiralPhase.NUMERICAL_CAPTURE,
        elapsed_myr=dynamical_state.elapsed_myr,
        separation_pc=elements.separation,
        semimajor_axis_pc=elements.semimajor_axis,
        eccentricity=elements.eccentricity,
        reason="phase-aware static-host integration boundary",
    )
    state = KpcToHardState(dynamical_state, initial, (initial,))
    try:
        hard_radius = _selected_hard_binary_radius_pc(model, config)
    except ValueError as error:
        return _append_phase(
            state,
            InspiralPhase.INVALID,
            elements=elements,
            reason=str(error),
        )
    if elements.separation > config.common_nucleus_radius_pc:
        return _append_phase(
            state,
            InspiralPhase.DUAL_NUCLEUS,
            elements=elements,
            reason="SMBHs remain outside the common-nucleus radius",
        )
    if elements.specific_energy >= 0.0:
        return _append_phase(
            state,
            InspiralPhase.COMMON_NUCLEUS_UNBOUND,
            elements=elements,
            reason="SMBHs share a nucleus but remain osculating-two-body unbound",
        )
    state = _append_phase(
        state,
        InspiralPhase.BOUND_BINARY,
        elements=elements,
        reason="negative osculating two-body energy inside the common nucleus",
    )
    assert elements.semimajor_axis is not None
    if elements.semimajor_axis <= hard_radius:
        state = _append_phase(
            state,
            InspiralPhase.HARD_BINARY,
            elements=elements,
            reason="semimajor axis lies inside the event-specific hard-binary radius",
        )
    return state


def _binary_handoff(state: KpcToHardState) -> BoundBinaryState:
    axis = state.inspiral_state.semimajor_axis_pc
    eccentricity = state.inspiral_state.eccentricity
    if (
        state.inspiral_state.phase is not InspiralPhase.HARD_BINARY
        or axis is None
        or eccentricity is None
        or not 0.0 <= eccentricity < 1.0
    ):
        raise ValueError("only a valid hard bound orbit can enter binary evolution")
    # Binary evolution measures its own segment time; kpc elapsed time remains
    # in the transition history and kpc delay segment.
    return BoundBinaryState(
        elapsed_myr=0.0,
        semimajor_axis_pc=axis,
        eccentricity_squared=eccentricity**2,
    )


def _bound_envelope_mass(state: DualNucleusState, model: KpcInspiralModel) -> float:
    if model.nuclear_envelope is None:
        return 0.0
    assert state.envelope_truncation_radius_pc is not None
    return model.nuclear_envelope.mass_within(state.envelope_truncation_radius_pc)


def force_budget(state: DualNucleusState, model: KpcInspiralModel) -> ForceBudget:
    envelope_mass = _bound_envelope_mass(state, model)
    effective_mass = model.secondary_bh_mass_msun + envelope_mass
    conservative = model.host_potential.acceleration(state.position_pc)
    stellar = None
    gas = None
    fdm = None
    stellar_acceleration = np.zeros(3)
    gas_acceleration = np.zeros(3)
    fdm_acceleration = np.zeros(3)
    if model.stellar_background is not None:
        assert model.stellar_friction is not None
        stellar = stellar_dynamical_friction(
            perturber_mass_msun=effective_mass,
            position_pc=state.position_pc,
            velocity_pc_myr=state.velocity_pc_myr,
            background=model.stellar_background,
            minimum_impact_parameter_pc=(
                model.stellar_friction.minimum_impact_parameter_pc
            ),
            maximum_impact_parameter_pc=(
                model.stellar_friction.maximum_impact_parameter_pc
            ),
            core_stall_enclosed_mass_ratio=(
                model.stellar_friction.core_stall_enclosed_mass_ratio
            ),
        )
        stellar_acceleration = stellar.acceleration_pc_myr2
    if model.gas_background is not None:
        assert model.gas_friction is not None
        gas = gaseous_dynamical_friction(
            perturber_mass_msun=effective_mass,
            position_pc=state.position_pc,
            velocity_pc_myr=state.velocity_pc_myr,
            total_enclosed_mass_msun=model.host_potential.enclosed_mass(
                state.radius_pc
            ),
            background=model.gas_background,
            coulomb_logarithm=model.gas_friction.coulomb_logarithm,
        )
        gas_acceleration = gas.acceleration_pc_myr2
    if model.fdm_background is not None:
        fdm = fdm_dynamical_friction(
            perturber_mass_msun=effective_mass,
            position_pc=state.position_pc,
            velocity_pc_myr=state.velocity_pc_myr,
            background=model.fdm_background,
        )
        fdm_acceleration = fdm.acceleration_pc_myr2
    dissipative = stellar_acceleration + gas_acceleration + fdm_acceleration
    return ForceBudget(
        conservative_acceleration_pc_myr2=conservative,
        stellar_acceleration_pc_myr2=stellar_acceleration,
        gas_acceleration_pc_myr2=gas_acceleration,
        fdm_acceleration_pc_myr2=fdm_acceleration,
        total_acceleration_pc_myr2=conservative + dissipative,
        effective_perturber_mass_msun=effective_mass,
        bound_envelope_mass_msun=envelope_mass,
        stellar=stellar,
        gas=gas,
        fdm=fdm,
    )


def tidal_truncation_radius_pc(
    state: DualNucleusState, model: KpcInspiralModel
) -> float | None:
    """Return a monotonic circular-orbit Jacobi truncation bracket."""

    envelope = model.nuclear_envelope
    if envelope is None:
        return None
    assert state.envelope_truncation_radius_pc is not None
    current = state.envelope_truncation_radius_pc
    radius = state.radius_pc
    if radius == 0.0:
        return current
    enclosed = float(model.host_potential.enclosed_mass(radius))
    slope = model.host_potential.logarithmic_enclosed_mass_slope(radius)
    denominator = (3.0 - slope) * enclosed
    if denominator <= 0.0:
        return current
    candidate = current
    for _ in range(64):
        satellite_mass = model.secondary_bh_mass_msun + envelope.mass_within(candidate)
        updated = min(current, radius * (satellite_mass / denominator) ** (1.0 / 3.0))
        if abs(updated - candidate) <= 1.0e-12 * max(candidate, 1.0):
            candidate = updated
            break
        candidate = updated
    return float(max(candidate, np.finfo(float).tiny))


def _sample(state: DualNucleusState, model: KpcInspiralModel) -> KpcInspiralSample:
    budget = force_budget(state, model)
    energy = 0.5 * float(np.dot(state.velocity_pc_myr, state.velocity_pc_myr))
    energy += float(model.host_potential.potential(state.radius_pc))
    drag_work = float(
        np.dot(state.velocity_pc_myr, budget.dissipative_acceleration_pc_myr2)
    )
    return KpcInspiralSample(
        elapsed_myr=state.elapsed_myr,
        radius_pc=state.radius_pc,
        speed_pc_myr=float(np.linalg.norm(state.velocity_pc_myr)),
        specific_orbital_energy_pc2_myr2=energy,
        drag_work_rate_pc2_myr3=drag_work,
        bound_envelope_mass_msun=budget.bound_envelope_mass_msun,
        envelope_truncation_radius_pc=state.envelope_truncation_radius_pc,
    )


def _samples_with_final(
    samples: list[KpcInspiralSample],
    state: DualNucleusState,
    model: KpcInspiralModel,
) -> tuple[KpcInspiralSample, ...]:
    if not samples or samples[-1].elapsed_myr != state.elapsed_myr:
        samples.append(_sample(state, model))
    return tuple(samples)


def _derivative(
    position: np.ndarray,
    velocity: np.ndarray,
    template: DualNucleusState,
    model: KpcInspiralModel,
) -> tuple[np.ndarray, np.ndarray]:
    stage = DualNucleusState(
        elapsed_myr=template.elapsed_myr,
        position_pc=position,
        velocity_pc_myr=velocity,
        envelope_truncation_radius_pc=template.envelope_truncation_radius_pc,
        completed_steps=template.completed_steps,
    )
    return velocity, force_budget(stage, model).total_acceleration_pc_myr2


def advance_dual_nucleus_rk4(
    state: DualNucleusState, model: KpcInspiralModel, time_step_myr: float
) -> DualNucleusState:
    if not np.isfinite(time_step_myr) or time_step_myr <= 0.0:
        raise ValueError("time step must be finite and positive")
    x0 = state.position_pc
    v0 = state.velocity_pc_myr
    k1x, k1v = _derivative(x0, v0, state, model)
    k2x, k2v = _derivative(
        x0 + 0.5 * time_step_myr * k1x,
        v0 + 0.5 * time_step_myr * k1v,
        state,
        model,
    )
    k3x, k3v = _derivative(
        x0 + 0.5 * time_step_myr * k2x,
        v0 + 0.5 * time_step_myr * k2v,
        state,
        model,
    )
    k4x, k4v = _derivative(
        x0 + time_step_myr * k3x,
        v0 + time_step_myr * k3v,
        state,
        model,
    )
    provisional = DualNucleusState(
        elapsed_myr=state.elapsed_myr + time_step_myr,
        position_pc=x0
        + time_step_myr * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0,
        velocity_pc_myr=v0
        + time_step_myr * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
        envelope_truncation_radius_pc=state.envelope_truncation_radius_pc,
        completed_steps=state.completed_steps + 1,
    )
    truncation = tidal_truncation_radius_pc(provisional, model)
    return DualNucleusState(
        elapsed_myr=provisional.elapsed_myr,
        position_pc=provisional.position_pc,
        velocity_pc_myr=provisional.velocity_pc_myr,
        envelope_truncation_radius_pc=truncation,
        completed_steps=provisional.completed_steps,
    )


def _relative_reflex_acceleration(
    position_pc: np.ndarray, secondary_bh_mass_msun: float
) -> np.ndarray:
    radius = float(np.linalg.norm(position_pc))
    if radius == 0.0:
        raise ValueError("relative SMBH separation reached zero before hard handoff")
    return -G_INTERNAL * secondary_bh_mass_msun * position_pc / radius**3


def _phase_aware_derivative(
    position: np.ndarray,
    velocity: np.ndarray,
    template: DualNucleusState,
    model: KpcInspiralModel,
) -> tuple[np.ndarray, np.ndarray]:
    stage = DualNucleusState(
        elapsed_myr=template.elapsed_myr,
        position_pc=position,
        velocity_pc_myr=velocity,
        envelope_truncation_radius_pc=template.envelope_truncation_radius_pc,
        completed_steps=template.completed_steps,
    )
    acceleration = force_budget(stage, model).total_acceleration_pc_myr2
    acceleration = acceleration + _relative_reflex_acceleration(
        position, model.secondary_bh_mass_msun
    )
    return velocity, acceleration


def _advance_phase_aware_rk4(
    state: DualNucleusState, model: KpcInspiralModel, time_step_myr: float
) -> DualNucleusState:
    """Advance the secondary-primary relative coordinate by one RK4 step."""

    if not np.isfinite(time_step_myr) or time_step_myr <= 0.0:
        raise ValueError("time step must be finite and positive")
    x0 = state.position_pc
    v0 = state.velocity_pc_myr
    k1x, k1v = _phase_aware_derivative(x0, v0, state, model)
    k2x, k2v = _phase_aware_derivative(
        x0 + 0.5 * time_step_myr * k1x,
        v0 + 0.5 * time_step_myr * k1v,
        state,
        model,
    )
    k3x, k3v = _phase_aware_derivative(
        x0 + 0.5 * time_step_myr * k2x,
        v0 + 0.5 * time_step_myr * k2v,
        state,
        model,
    )
    k4x, k4v = _phase_aware_derivative(
        x0 + time_step_myr * k3x,
        v0 + time_step_myr * k3v,
        state,
        model,
    )
    provisional = DualNucleusState(
        elapsed_myr=state.elapsed_myr + time_step_myr,
        position_pc=x0
        + time_step_myr * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0,
        velocity_pc_myr=v0
        + time_step_myr * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
        envelope_truncation_radius_pc=state.envelope_truncation_radius_pc,
        completed_steps=state.completed_steps + 1,
    )
    truncation = tidal_truncation_radius_pc(provisional, model)
    return DualNucleusState(
        elapsed_myr=provisional.elapsed_myr,
        position_pc=provisional.position_pc,
        velocity_pc_myr=provisional.velocity_pc_myr,
        envelope_truncation_radius_pc=truncation,
        completed_steps=provisional.completed_steps,
    )


def _adaptive_step(
    state: DualNucleusState, model: KpcInspiralModel, config: KpcIntegrationConfig
) -> float:
    radius = state.radius_pc
    speed = float(np.linalg.norm(state.velocity_pc_myr))
    enclosed = float(model.host_potential.enclosed_mass(radius))
    dynamical_time = np.sqrt(radius**3 / (G_INTERNAL * enclosed))
    crossing_time = radius / max(speed, np.finfo(float).tiny)
    remaining = config.maximum_time_myr - state.elapsed_myr
    return float(
        min(
            config.maximum_step_myr,
            config.timestep_fraction * dynamical_time,
            config.timestep_fraction * crossing_time,
            remaining,
        )
    )


def _phase_aware_adaptive_step(
    state: DualNucleusState,
    model: KpcInspiralModel,
    config: KpcToHardConfig,
) -> float:
    """Select a relative-orbit step using the full mutual gravitating mass."""

    radius = state.radius_pc
    speed = float(np.linalg.norm(state.velocity_pc_myr))
    enclosed = float(model.host_potential.enclosed_mass(radius))
    mutual_enclosed = enclosed + model.secondary_bh_mass_msun
    dynamical_time = np.sqrt(radius**3 / (G_INTERNAL * mutual_enclosed))
    crossing_time = np.inf if speed == 0.0 else radius / speed
    remaining = config.maximum_time_myr - state.elapsed_myr
    return float(
        min(
            config.maximum_step_myr,
            config.timestep_fraction * dynamical_time,
            config.timestep_fraction * crossing_time,
            remaining,
        )
    )


def integrate_dual_nucleus(
    *,
    initial_state: DualNucleusState,
    model: KpcInspiralModel,
    config: KpcIntegrationConfig,
    step_budget: int | None = None,
) -> KpcInspiralResult:
    """Integrate to the geometric common-nucleus boundary.

    This compatibility path does not establish a physical bound or hard binary.
    Physical kpc-to-hard delays must use ``integrate_dual_nucleus_to_hard``.
    """

    if step_budget is not None and step_budget < 1:
        raise ValueError("step_budget must be positive when supplied")
    if initial_state.elapsed_myr > config.maximum_time_myr:
        raise ValueError("initial state lies beyond maximum_time_myr")
    state = initial_state
    samples = [_sample(state, model)]
    steps_this_call = 0
    while state.completed_steps < config.maximum_steps:
        if state.radius_pc <= config.target_radius_pc:
            return KpcInspiralResult(
                "reached_common_nucleus",
                state,
                _samples_with_final(samples, state, model),
                "secondary reached the event-specific common-nucleus boundary",
            )
        if state.elapsed_myr >= config.maximum_time_myr:
            return KpcInspiralResult(
                "timeout",
                state,
                _samples_with_final(samples, state, model),
                "available integration time was exhausted",
            )
        if step_budget is not None and steps_this_call >= step_budget:
            return KpcInspiralResult(
                "checkpoint",
                state,
                _samples_with_final(samples, state, model),
                "bounded step budget reached; resume from final_state",
            )
        budget = force_budget(state, model)
        if (
            budget.stellar is not None
            and budget.stellar.stalled
            and model.gas_background is None
            and (
                budget.fdm is None
                or np.linalg.norm(budget.fdm_acceleration_pc_myr2) == 0.0
            )
        ):
            return KpcInspiralResult(
                "stalled",
                state,
                _samples_with_final(samples, state, model),
                budget.stellar.reason,
            )
        time_step = _adaptive_step(state, model, config)
        if not np.isfinite(time_step) or time_step <= 0.0:
            return KpcInspiralResult(
                "invalid",
                state,
                _samples_with_final(samples, state, model),
                "adaptive time step became non-positive or non-finite",
            )
        state = advance_dual_nucleus_rk4(state, model, time_step)
        steps_this_call += 1
        if state.completed_steps % config.sample_interval_steps == 0:
            samples.append(_sample(state, model))
    return KpcInspiralResult(
        "invalid",
        state,
        _samples_with_final(samples, state, model),
        "maximum_steps was exhausted",
    )


def _terminal_phase_state(
    state: KpcToHardState,
    target: InspiralPhase,
    reason: str,
) -> KpcToHardState:
    if state.inspiral_state.terminal:
        return state
    updated = transition_state(
        state.inspiral_state,
        target,
        elapsed_myr=state.dynamical_state.elapsed_myr,
        reason=reason,
        separation_pc=state.dynamical_state.radius_pc,
    )
    return KpcToHardState(
        state.dynamical_state,
        updated,
        state.transition_history + (updated,),
    )


def _phase_aware_samples_with_final(
    samples: list[KpcInspiralSample],
    state: KpcToHardState,
    model: KpcInspiralModel,
) -> tuple[KpcInspiralSample, ...]:
    dynamics = state.dynamical_state
    if not samples or samples[-1].elapsed_myr != dynamics.elapsed_myr:
        samples.append(_phase_aware_sample(dynamics, model))
    return tuple(samples)


def _phase_aware_sample(
    state: DualNucleusState, model: KpcInspiralModel
) -> KpcInspiralSample:
    sample = _sample(state, model)
    reflex_potential = -G_INTERNAL * model.secondary_bh_mass_msun / state.radius_pc
    return replace(
        sample,
        specific_orbital_energy_pc2_myr2=(
            sample.specific_orbital_energy_pc2_myr2 + reflex_potential
        ),
    )


def integrate_dual_nucleus_to_hard(
    *,
    initial_state: KpcToHardState,
    model: KpcInspiralModel,
    config: KpcToHardConfig,
    step_budget: int | None = None,
) -> KpcToHardResult:
    """Evolve a fixed-primary, static spherical host until a hard binary forms.

    Entering the common-nucleus radius is only a geometric transition.  An
    unbound state continues to evolve until it becomes osculating-two-body
    bound, exits the approximation domain, stalls, or exhausts the available
    time.  Checkpoints retain the complete transition history.
    """

    if step_budget is not None and step_budget < 1:
        raise ValueError("step_budget must be positive when supplied")
    if initial_state.dynamical_state.elapsed_myr > config.maximum_time_myr:
        raise ValueError("initial state lies beyond maximum_time_myr")

    state = initial_state
    samples = [_phase_aware_sample(state.dynamical_state, model)]
    scope_error = _static_host_scope_error(model, config)
    try:
        _selected_hard_binary_radius_pc(model, config)
    except ValueError as error:
        scope_error = str(error)
    if scope_error is not None:
        state = _terminal_phase_state(state, InspiralPhase.INVALID, scope_error)
        return KpcToHardResult(
            "invalid", state, tuple(samples), scope_error, None
        )

    steps_this_call = 0
    while state.dynamical_state.completed_steps < config.maximum_steps:
        if state.inspiral_state.phase is InspiralPhase.HARD_BINARY:
            try:
                binary_state = _binary_handoff(state)
            except ValueError as error:
                state = _terminal_phase_state(
                    state, InspiralPhase.INVALID, str(error)
                )
                return KpcToHardResult(
                    "invalid",
                    state,
                    _phase_aware_samples_with_final(samples, state, model),
                    str(error),
                    None,
                )
            return KpcToHardResult(
                "reached_hard_binary",
                state,
                _phase_aware_samples_with_final(samples, state, model),
                "binary reached the event-specific hard-binary boundary",
                binary_state,
            )
        if state.inspiral_state.terminal:
            status = (
                "outside"
                if state.inspiral_state.phase is InspiralPhase.CENSORED
                else "invalid"
            )
            return KpcToHardResult(
                status,
                state,
                _phase_aware_samples_with_final(samples, state, model),
                state.inspiral_state.reason,
                None,
            )
        if state.dynamical_state.elapsed_myr >= config.maximum_time_myr:
            reason = "available static-host integration time was exhausted"
            state = _terminal_phase_state(state, InspiralPhase.CENSORED, reason)
            return KpcToHardResult(
                "timeout",
                state,
                _phase_aware_samples_with_final(samples, state, model),
                reason,
                None,
            )
        if step_budget is not None and steps_this_call >= step_budget:
            return KpcToHardResult(
                "checkpoint",
                state,
                _phase_aware_samples_with_final(samples, state, model),
                "bounded step budget reached; resume from final_state",
                None,
            )

        budget = force_budget(state.dynamical_state, model)
        if (
            budget.stellar is not None
            and budget.stellar.stalled
            and model.gas_background is None
            and (
                budget.fdm is None
                or np.linalg.norm(budget.fdm_acceleration_pc_myr2) == 0.0
            )
        ):
            state = _terminal_phase_state(
                state, InspiralPhase.STALLED, budget.stellar.reason
            )
            return KpcToHardResult(
                "stalled",
                state,
                _phase_aware_samples_with_final(samples, state, model),
                budget.stellar.reason,
                None,
            )

        time_step = _phase_aware_adaptive_step(
            state.dynamical_state, model, config
        )
        if not np.isfinite(time_step) or time_step <= 0.0:
            reason = "adaptive phase-aware time step became invalid"
            state = _terminal_phase_state(state, InspiralPhase.INVALID, reason)
            return KpcToHardResult(
                "invalid",
                state,
                _phase_aware_samples_with_final(samples, state, model),
                reason,
                None,
            )
        try:
            dynamics = _advance_phase_aware_rk4(
                state.dynamical_state, model, time_step
            )
            state = _replace_dynamics(state, dynamics)
            state, terminal_status = _advance_physical_phase(state, model, config)
        except ValueError as error:
            reason = f"phase-aware state left its valid domain: {error}"
            state = _terminal_phase_state(state, InspiralPhase.INVALID, reason)
            return KpcToHardResult(
                "invalid",
                state,
                _phase_aware_samples_with_final(samples, state, model),
                reason,
                None,
            )
        steps_this_call += 1
        if state.dynamical_state.completed_steps % config.sample_interval_steps == 0:
            samples.append(_phase_aware_sample(state.dynamical_state, model))
        if terminal_status == "outside":
            return KpcToHardResult(
                "outside",
                state,
                _phase_aware_samples_with_final(samples, state, model),
                state.inspiral_state.reason,
                None,
            )

    reason = "maximum phase-aware step count was exhausted"
    state = _terminal_phase_state(state, InspiralPhase.INVALID, reason)
    return KpcToHardResult(
        "invalid",
        state,
        _phase_aware_samples_with_final(samples, state, model),
        reason,
        None,
    )
