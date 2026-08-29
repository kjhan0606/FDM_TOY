from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fdm_smbh_delay.capture_ledger import CaptureEvent, CapturePair
from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.lagramses import PairOrbitalState, pair_orbital_state
from fdm_smbh_delay.nuclear_bridge import (
    BRIDGE_SCHEMA_VERSION,
    BridgeStatus,
    EnvironmentChannel,
    EnvironmentSnapshot,
    NuclearBridgeInput,
)


SOURCE_HASH = "a" * 64


def _pair(*, bound: bool = True) -> PairOrbitalState:
    separation = 2.0
    speed = np.sqrt(G_INTERNAL * 2.0e8 / separation)
    if not bound:
        speed *= 2.0
    return pair_orbital_state(
        member_ids=(11, 22),
        masses_msun=(1.0e8, 1.0e8),
        positions_pc=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        velocities_pc_myr=np.array(
            [[0.0, speed / 2.0, 0.0], [0.0, -speed / 2.0, 0.0]]
        ),
    )


def _stellar(status: str = "available", *, reason: str | None = None):
    return EnvironmentChannel(
        "stellar",
        status,
        density_msun_pc3=None if status != "available" else 1.0e4,
        enclosed_mass_msun=None if status != "available" else 1.0e8,
        bulk_velocity_pc_myr=None if status != "available" else np.zeros(3),
        velocity_dispersion_pc_myr=100.0 if status == "available" else None,
        reason=reason,
    )


def _gas(status: str = "absent", *, reason: str | None = "no gas channel in snapshot"):
    return EnvironmentChannel(
        "gas",
        status,
        density_msun_pc3=0.0 if status == "absent" else None,
        enclosed_mass_msun=0.0 if status == "absent" else None,
        bulk_velocity_pc_myr=np.zeros(3) if status == "absent" else None,
        reason=reason,
    )


def _fdm(status: str = "available", *, reason: str | None = None):
    return EnvironmentChannel(
        "fdm",
        status,
        density_msun_pc3=None if status != "available" else 5.0e6,
        enclosed_mass_msun=None if status != "available" else 1.0e9,
        bulk_velocity_pc_myr=None if status != "available" else np.zeros(3),
        core_radius_pc=2.3 if status == "available" else None,
        fdm_mode="analytic_unresolved" if status == "available" else None,
        resolved_wake=False if status == "available" else None,
        reason=reason,
    )


def _environment(
    *, event_uid: str = "evt-1", stellar_status: str = "available"
) -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        event_uid=event_uid,
        time_myr=1200.0,
        redshift=0.4,
        radius_pc=200.0,
        channels=(_stellar(stellar_status, reason="stellar data unavailable" if stellar_status != "available" else None), _gas(), _fdm()),
        source_case_id="hr5-run-1",
        source_sha256=SOURCE_HASH,
        source_path="/gpfs/example/capture.jsonl",
    )


def _bridge(*, environment: EnvironmentSnapshot | None = None, pair=None):
    return NuclearBridgeInput(
        event_uid="evt-1",
        run_id="hr5-run-1",
        capture_time_myr=1200.0,
        redshift=0.4,
        pair=_pair() if pair is None else pair,
        environment=_environment() if environment is None else environment,
        target_semimajor_axis_pc=1.0,
        source_path="/gpfs/example/capture.jsonl",
        source_sha256=SOURCE_HASH,
    )


def test_ready_bridge_round_trips_json(tmp_path: Path) -> None:
    bridge = _bridge()
    assert bridge.status == BridgeStatus.READY.value
    assert bridge.ready_for_integration
    assert bridge.physical_binding_status == "bound"
    path = bridge.write_json(tmp_path / "bridge.json")
    decoded = NuclearBridgeInput.read_json(path)
    assert decoded.as_dict() == bridge.as_dict()
    assert decoded.pair.semi_major_axis_pc == pytest.approx(bridge.pair.semi_major_axis_pc)


def test_missing_environment_is_not_interpreted_as_zero() -> None:
    bridge = _bridge(environment=_environment(stellar_status="missing"))
    assert bridge.status == BridgeStatus.MISSING_ENVIRONMENT.value
    assert not bridge.ready_for_integration
    assert "stellar data unavailable" in bridge.reasons


def test_censored_environment_propagates_status() -> None:
    environment = EnvironmentSnapshot(
        event_uid="evt-1",
        time_myr=1200.0,
        redshift=0.4,
        radius_pc=200.0,
        channels=(_stellar(), _gas(), _fdm("censored", reason="outside zoom support")),
        source_case_id="hr5-run-1",
        source_sha256=SOURCE_HASH,
        source_path="capture.jsonl",
    )
    bridge = _bridge(environment=environment)
    assert bridge.status == BridgeStatus.CENSORED.value
    assert not bridge.ready_for_integration


def test_live_fdm_wake_rejects_analytic_drag_contract() -> None:
    with pytest.raises(ValueError, match="resolved_wake"):
        EnvironmentChannel(
            "fdm",
            "available",
            density_msun_pc3=1.0,
            enclosed_mass_msun=1.0,
            bulk_velocity_pc_myr=np.zeros(3),
            core_radius_pc=1.0,
            fdm_mode="live_resolved",
            resolved_wake=False,
        )


def test_absent_channel_cannot_hide_positive_mass() -> None:
    with pytest.raises(ValueError, match="absent channel"):
        EnvironmentChannel(
            "gas",
            "absent",
            density_msun_pc3=1.0,
            enclosed_mass_msun=0.0,
            reason="declared absent",
        )


def test_environment_requires_all_three_channels() -> None:
    with pytest.raises(ValueError, match="stellar, gas, and fdm"):
        EnvironmentSnapshot(
            event_uid="evt-1",
            time_myr=1.0,
            redshift=0.0,
            radius_pc=10.0,
            channels=(_stellar(), _gas()),
            source_case_id="case",
            source_sha256=SOURCE_HASH,
            source_path="capture.jsonl",
        )


def test_capture_event_constructor_preserves_ledger_provenance() -> None:
    pair = _pair()
    event = CaptureEvent(
        event_uid="evt-1",
        classification="BINARY",
        nstep_coarse=4,
        level=15,
        scale_factor=0.7,
        redshift=0.4,
        code_time=2.0,
        proper_time_code=3.0,
        numerical_merge_radius_pc=5.0,
        members=(),
        pairs=(CapturePair((11, 22), pair, True, True, True),),
        event_sha256=SOURCE_HASH,
        source_path=Path("capture.jsonl"),
        first_line=1,
        last_line=3,
    )
    bridge = NuclearBridgeInput.from_capture_event(
        event,
        run_id="hr5-run-1",
        capture_time_myr=1200.0,
        environment=_environment(),
    )
    assert bridge.source_sha256 == SOURCE_HASH
    assert bridge.redshift == pytest.approx(event.redshift)


def test_unbound_capture_is_preserved_for_rebinding_model() -> None:
    bridge = _bridge(pair=_pair(bound=False))
    assert bridge.status == BridgeStatus.READY.value
    assert bridge.ready_for_integration
    assert bridge.physical_binding_status == "unbound_or_undefined"
    assert bridge.reasons


def test_schema_version_is_required_on_decode() -> None:
    record = _bridge().as_dict()
    record["schema_version"] = BRIDGE_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="unsupported nuclear bridge schema"):
        NuclearBridgeInput.from_dict(record)
