"""Physical state boundaries between numerical capture and a hard SMBH binary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from .capture_ledger import CaptureEvent
from .constants import G_INTERNAL


class InspiralPhase(str, Enum):
    NUMERICAL_CAPTURE = "numerical_capture"
    DUAL_NUCLEUS = "dual_nucleus"
    COMMON_NUCLEUS_UNBOUND = "common_nucleus_unbound"
    BOUND_BINARY = "bound_binary"
    HARD_BINARY = "hard_binary"
    ENVIRONMENT_FDM_DRIVEN = "environment_fdm_driven"
    GW_DRIVEN = "gw_driven"
    COALESCED = "coalesced"
    STALLED = "stalled"
    CENSORED = "censored"
    MULTIPLE = "multiple"
    INVALID = "invalid"


TERMINAL_PHASES = {
    InspiralPhase.COALESCED,
    InspiralPhase.STALLED,
    InspiralPhase.CENSORED,
    InspiralPhase.MULTIPLE,
    InspiralPhase.INVALID,
}


_ALLOWED_TRANSITIONS = {
    InspiralPhase.NUMERICAL_CAPTURE: {
        InspiralPhase.DUAL_NUCLEUS,
        InspiralPhase.COMMON_NUCLEUS_UNBOUND,
        InspiralPhase.BOUND_BINARY,
        InspiralPhase.HARD_BINARY,
        InspiralPhase.GW_DRIVEN,
        InspiralPhase.MULTIPLE,
        InspiralPhase.INVALID,
        InspiralPhase.CENSORED,
    },
    InspiralPhase.DUAL_NUCLEUS: {
        InspiralPhase.COMMON_NUCLEUS_UNBOUND,
        InspiralPhase.BOUND_BINARY,
        InspiralPhase.STALLED,
        InspiralPhase.CENSORED,
        InspiralPhase.MULTIPLE,
        InspiralPhase.INVALID,
    },
    InspiralPhase.COMMON_NUCLEUS_UNBOUND: {
        InspiralPhase.BOUND_BINARY,
        InspiralPhase.STALLED,
        InspiralPhase.CENSORED,
        InspiralPhase.MULTIPLE,
        InspiralPhase.INVALID,
    },
    InspiralPhase.BOUND_BINARY: {
        InspiralPhase.HARD_BINARY,
        InspiralPhase.STALLED,
        InspiralPhase.CENSORED,
        InspiralPhase.MULTIPLE,
        InspiralPhase.INVALID,
    },
    InspiralPhase.HARD_BINARY: {
        InspiralPhase.ENVIRONMENT_FDM_DRIVEN,
        InspiralPhase.GW_DRIVEN,
        InspiralPhase.STALLED,
        InspiralPhase.CENSORED,
        InspiralPhase.MULTIPLE,
        InspiralPhase.INVALID,
    },
    InspiralPhase.ENVIRONMENT_FDM_DRIVEN: {
        InspiralPhase.GW_DRIVEN,
        InspiralPhase.STALLED,
        InspiralPhase.CENSORED,
        InspiralPhase.MULTIPLE,
        InspiralPhase.INVALID,
    },
    InspiralPhase.GW_DRIVEN: {
        InspiralPhase.COALESCED,
        InspiralPhase.CENSORED,
        InspiralPhase.INVALID,
    },
}


@dataclass(frozen=True)
class TransitionRadii:
    influence_radius_pc: float
    hard_binary_radius_pc: float
    gw_transition_pc: float | None = None

    def __post_init__(self) -> None:
        values = [self.influence_radius_pc, self.hard_binary_radius_pc]
        if self.gw_transition_pc is not None:
            values.append(self.gw_transition_pc)
        if np.any(~np.isfinite(values)) or np.any(np.asarray(values) <= 0.0):
            raise ValueError("transition radii must be finite and positive")
        if self.hard_binary_radius_pc > self.influence_radius_pc:
            raise ValueError("the hard-binary radius cannot exceed the influence radius")


@dataclass(frozen=True)
class InspiralState:
    event_uid: str
    phase: InspiralPhase
    elapsed_myr: float
    separation_pc: float | None
    semimajor_axis_pc: float | None
    eccentricity: float | None
    reason: str

    def __post_init__(self) -> None:
        if not self.event_uid:
            raise ValueError("event_uid is required")
        if not np.isfinite(self.elapsed_myr) or self.elapsed_myr < 0.0:
            raise ValueError("elapsed time must be finite and non-negative")
        for value, name in (
            (self.separation_pc, "separation"),
            (self.semimajor_axis_pc, "semimajor axis"),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if self.eccentricity is not None and not (
            np.isfinite(self.eccentricity) and self.eccentricity >= 0.0
        ):
            raise ValueError("osculating eccentricity must be finite and non-negative")

    @property
    def terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES


def influence_radius_pc(total_binary_mass_msun: float, sigma_pc_myr: float) -> float:
    """Return ``G M_binary / sigma**2`` in the package's internal units."""

    if (
        not np.isfinite(total_binary_mass_msun)
        or not np.isfinite(sigma_pc_myr)
        or total_binary_mass_msun <= 0.0
        or sigma_pc_myr <= 0.0
    ):
        raise ValueError("binary mass and velocity dispersion must be positive")
    return float(G_INTERNAL * total_binary_mass_msun / sigma_pc_myr**2)


def hard_binary_radius_pc(
    mass1_msun: float, mass2_msun: float, sigma_pc_myr: float
) -> float:
    """Return the conventional hard-binary scale ``G mu / (4 sigma**2)``."""

    values = np.asarray([mass1_msun, mass2_msun, sigma_pc_myr], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("SMBH masses and velocity dispersion must be positive")
    reduced_mass = mass1_msun * mass2_msun / (mass1_msun + mass2_msun)
    return float(G_INTERNAL * reduced_mass / (4.0 * sigma_pc_myr**2))


def transition_radii(
    *,
    mass1_msun: float,
    mass2_msun: float,
    sigma_pc_myr: float,
    gw_transition_pc: float | None = None,
) -> TransitionRadii:
    return TransitionRadii(
        influence_radius_pc= influence_radius_pc(
            mass1_msun + mass2_msun, sigma_pc_myr
        ),
        hard_binary_radius_pc=hard_binary_radius_pc(
            mass1_msun, mass2_msun, sigma_pc_myr
        ),
        gw_transition_pc=gw_transition_pc,
    )


def initial_capture_state(event: CaptureEvent) -> InspiralState:
    state = event.binary_orbital_state
    return InspiralState(
        event_uid=event.event_uid,
        phase=InspiralPhase.NUMERICAL_CAPTURE,
        elapsed_myr=0.0,
        separation_pc=(None if state is None else state.separation_pc),
        semimajor_axis_pc=(None if state is None else state.semi_major_axis_pc),
        eccentricity=(None if state is None else state.eccentricity),
        reason="lagRamses numerical FOF compaction boundary",
    )


def classify_capture_state(
    event: CaptureEvent,
    *,
    common_nucleus_radius_pc: float,
    sigma_pc_myr: float,
    gw_transition_pc: float | None = None,
) -> InspiralState:
    """Classify a ledger event without equating capture with coalescence.

    The common-nucleus boundary remains an explicit host-model input.  A
    two-body binding diagnostic is consulted only after both SMBHs lie inside
    that boundary; it cannot by itself establish a bound binary at kpc scales.
    """

    if (
        not np.isfinite(common_nucleus_radius_pc)
        or common_nucleus_radius_pc <= 0.0
    ):
        raise ValueError("common_nucleus_radius_pc must be finite and positive")
    current = initial_capture_state(event)
    if event.classification == "MULTIPLE":
        return transition_state(
            current,
            InspiralPhase.MULTIPLE,
            elapsed_myr=0.0,
            reason="three or more members require an explicit multiple solver",
        )
    orbit = event.binary_orbital_state
    if orbit is None:
        return transition_state(
            current,
            InspiralPhase.INVALID,
            elapsed_myr=0.0,
            reason="binary relative state is undefined",
        )
    if orbit.separation_pc > common_nucleus_radius_pc:
        target = InspiralPhase.DUAL_NUCLEUS
        reason = "SMBHs remain outside the event-specific common-nucleus boundary"
    elif not orbit.bound:
        target = InspiralPhase.COMMON_NUCLEUS_UNBOUND
        reason = "SMBHs share a nucleus but the osculating two-body orbit is unbound"
    else:
        radii = transition_radii(
            mass1_msun=orbit.masses_msun[0],
            mass2_msun=orbit.masses_msun[1],
            sigma_pc_myr=sigma_pc_myr,
            gw_transition_pc=gw_transition_pc,
        )
        assert orbit.semi_major_axis_pc is not None
        if (
            radii.gw_transition_pc is not None
            and orbit.semi_major_axis_pc <= radii.gw_transition_pc
        ):
            target = InspiralPhase.GW_DRIVEN
            reason = "semimajor axis lies inside the supplied GW transition"
        elif orbit.semi_major_axis_pc <= radii.hard_binary_radius_pc:
            target = InspiralPhase.HARD_BINARY
            reason = "bound binary lies inside the event-specific hard-binary radius"
        else:
            target = InspiralPhase.BOUND_BINARY
            reason = "negative osculating energy inside the common nucleus"
    return transition_state(
        current, target, elapsed_myr=0.0, reason=reason
    )


def transition_state(
    state: InspiralState,
    target: InspiralPhase,
    *,
    elapsed_myr: float,
    reason: str,
    separation_pc: float | None = None,
    semimajor_axis_pc: float | None = None,
    eccentricity: float | None = None,
) -> InspiralState:
    """Apply one forward state transition and reject silent phase skipping."""

    if state.terminal:
        raise ValueError(f"terminal phase {state.phase.value} cannot transition")
    if target not in _ALLOWED_TRANSITIONS.get(state.phase, set()):
        raise ValueError(
            f"transition {state.phase.value} -> {target.value} is not allowed"
        )
    if not np.isfinite(elapsed_myr) or elapsed_myr < state.elapsed_myr:
        raise ValueError("elapsed time must be finite and monotonic")
    if not reason:
        raise ValueError("every state transition requires a reason")
    return replace(
        state,
        phase=target,
        elapsed_myr=float(elapsed_myr),
        separation_pc=(
            state.separation_pc if separation_pc is None else separation_pc
        ),
        semimajor_axis_pc=(
            state.semimajor_axis_pc
            if semimajor_axis_pc is None
            else semimajor_axis_pc
        ),
        eccentricity=(state.eccentricity if eccentricity is None else eccentricity),
        reason=reason,
    )
