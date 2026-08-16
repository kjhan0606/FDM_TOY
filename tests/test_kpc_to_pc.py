from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fdm_smbh_delay.capture_ledger import (
    CaptureEvent,
    CaptureMember,
    CapturePair,
)
from fdm_smbh_delay.constants import G_INTERNAL, KM_S_TO_PC_MYR
from fdm_smbh_delay.kpc_to_pc import (
    InspiralPhase,
    classify_capture_state,
    hard_binary_radius_pc,
    influence_radius_pc,
    initial_capture_state,
    transition_radii,
    transition_state,
)
from fdm_smbh_delay.lagramses import pair_orbital_state


def _event(
    *,
    separation_pc: float = 1.0,
    relative_speed_pc_myr: float | None = None,
    multiple: bool = False,
) -> CaptureEvent:
    mass = 1.0e8
    if relative_speed_pc_myr is None:
        relative_speed_pc_myr = np.sqrt(G_INTERNAL * 2.0 * mass / separation_pc)
    members = (
        CaptureMember(
            1,
            mass,
            np.array([0.5 * separation_pc, 0.0, 0.0]),
            np.array([0.0, 0.5 * relative_speed_pc_myr, 0.0]),
            0.0,
            0.0,
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            0.0,
        ),
        CaptureMember(
            2,
            mass,
            np.array([-0.5 * separation_pc, 0.0, 0.0]),
            np.array([0.0, -0.5 * relative_speed_pc_myr, 0.0]),
            0.0,
            0.0,
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            0.0,
        ),
    )
    orbit = pair_orbital_state(
        member_ids=(1, 2),
        masses_msun=(mass, mass),
        positions_pc=np.asarray([member.position_pc for member in members]),
        velocities_pc_myr=np.asarray([member.velocity_pc_myr for member in members]),
    )
    pairs = (CapturePair((1, 2), orbit, True, orbit.bound, True),)
    if multiple:
        classification = "MULTIPLE"
        members = members + (members[0],)
        pairs = ()
    else:
        classification = "BINARY"
    return CaptureEvent(
        event_uid="event-1",
        classification=classification,
        nstep_coarse=1,
        level=1,
        scale_factor=0.5,
        redshift=1.0,
        code_time=0.0,
        proper_time_code=0.0,
        numerical_merge_radius_pc=10.0,
        members=members,
        pairs=pairs,
        event_sha256="0" * 64,
        source_path=Path("ledger.jsonl"),
        first_line=1,
        last_line=5,
    )


def test_event_specific_influence_and_hard_binary_radii() -> None:
    sigma = 200.0 * KM_S_TO_PC_MYR
    influence = influence_radius_pc(2.0e8, sigma)
    hard = hard_binary_radius_pc(1.0e8, 1.0e8, sigma)
    radii = transition_radii(
        mass1_msun=1.0e8, mass2_msun=1.0e8, sigma_pc_myr=sigma
    )
    assert radii.influence_radius_pc == pytest.approx(influence)
    assert radii.hard_binary_radius_pc == pytest.approx(hard)
    assert hard < influence


def test_capture_outside_common_nucleus_is_not_called_a_binary() -> None:
    event = _event(separation_pc=10.0)
    state = classify_capture_state(
        event,
        common_nucleus_radius_pc=5.0,
        sigma_pc_myr=200.0 * KM_S_TO_PC_MYR,
    )
    assert state.phase is InspiralPhase.DUAL_NUCLEUS


def test_unbound_pair_inside_common_nucleus_remains_unbound() -> None:
    event = _event(separation_pc=1.0, relative_speed_pc_myr=5000.0)
    state = classify_capture_state(
        event,
        common_nucleus_radius_pc=5.0,
        sigma_pc_myr=200.0 * KM_S_TO_PC_MYR,
    )
    assert state.phase is InspiralPhase.COMMON_NUCLEUS_UNBOUND


def test_bound_and_hard_classification_uses_semimajor_axis() -> None:
    sigma = 200.0 * KM_S_TO_PC_MYR
    assert classify_capture_state(
        _event(separation_pc=3.0),
        common_nucleus_radius_pc=5.0,
        sigma_pc_myr=sigma,
    ).phase is InspiralPhase.BOUND_BINARY
    assert classify_capture_state(
        _event(separation_pc=0.01),
        common_nucleus_radius_pc=5.0,
        sigma_pc_myr=sigma,
    ).phase is InspiralPhase.HARD_BINARY


def test_multiple_is_terminal_without_arbitrary_pairing() -> None:
    state = classify_capture_state(
        _event(multiple=True),
        common_nucleus_radius_pc=5.0,
        sigma_pc_myr=200.0 * KM_S_TO_PC_MYR,
    )
    assert state.phase is InspiralPhase.MULTIPLE
    assert state.terminal


def test_state_transitions_are_monotonic_and_explicit() -> None:
    initial = initial_capture_state(_event(separation_pc=10.0))
    dual = transition_state(
        initial,
        InspiralPhase.DUAL_NUCLEUS,
        elapsed_myr=2.0,
        reason="host nuclei remain distinct",
    )
    with pytest.raises(ValueError, match="not allowed"):
        transition_state(
            dual,
            InspiralPhase.GW_DRIVEN,
            elapsed_myr=3.0,
            reason="illegal phase skip",
        )
    with pytest.raises(ValueError, match="monotonic"):
        transition_state(
            dual,
            InspiralPhase.BOUND_BINARY,
            elapsed_myr=1.0,
            reason="time reversal",
        )
