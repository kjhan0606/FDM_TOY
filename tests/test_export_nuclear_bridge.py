from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.nuclear_bridge import (
    EnvironmentChannel,
    EnvironmentSnapshot,
    NuclearBridgeInput,
)
from scripts.export_nuclear_bridge import export_bridge_input


def _ledger_rows(uid: str = "event-1") -> list[dict]:
    unit_length = (1.0 * u.pc).to_value(u.cm)
    unit_velocity = (1.0 * u.pc / u.Myr).to_value(u.cm / u.s)
    unit_mass = (1.0 * u.Msun).to_value(u.g)
    relative_speed = np.sqrt(G_INTERNAL * 2.0e8 / 2.0)
    begin = {
        "schema_version": 1,
        "record_type": "event_begin",
        "event_uid": uid,
        "classification": "BINARY",
        "nstep_coarse": 10,
        "ilevel": 1,
        "nmember": 2,
        "expected_pairs": 1,
        "aexp": 0.5,
        "redshift": 1.0,
        "t_code": -0.2,
        "texp": 0.4,
        "merge_radius_code": 2.0,
        "unit_length_cgs": unit_length,
        "unit_velocity_cgs": unit_velocity,
        "unit_mass_cgs": unit_mass,
        "boxlen": 100.0,
        "complete": False,
    }
    members = []
    for index, (sink_id, x, vy) in enumerate(
        ((7, 0.5, relative_speed / 2.0), (9, -0.5, -relative_speed / 2.0)),
        start=1,
    ):
        members.append(
            {
                "schema_version": 1,
                "record_type": "member",
                "event_uid": uid,
                "member_index": index,
                "sink_id": sink_id,
                "mass_code": 1.0e8,
                "position_code": [x, 0.0, 0.0],
                "velocity_code": [0.0, vy, 0.0],
                "formation_time_code": -0.4,
                "accreted_mass_code": 0.0,
                "spin_magnitude": 0.5,
                "spin_direction": [0.0, 0.0, 1.0],
                "gas_angular_momentum_code": [0.0, 0.0, 0.0],
            }
        )
    pair = {
        "schema_version": 1,
        "record_type": "pair",
        "event_uid": uid,
        "pair_index": 1,
        "sink_id_1": 7,
        "sink_id_2": 9,
        "within_rmerge": True,
        "two_body_bound": True,
        "legacy_pair_bound": True,
    }
    end = {
        "schema_version": 1,
        "record_type": "event_end",
        "event_uid": uid,
        "nmember": 2,
        "npair": 1,
        "complete": True,
    }
    return [begin, *members, pair, end]


def _write_ledger(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_environment(path: Path, event_uid: str = "event-1") -> None:
    snapshot = EnvironmentSnapshot(
        event_uid=event_uid,
        time_myr=1000.0,
        redshift=1.0,
        radius_pc=200.0,
        channels=(
            EnvironmentChannel(
                "stellar",
                "available",
                density_msun_pc3=1.0e4,
                enclosed_mass_msun=1.0e8,
                bulk_velocity_pc_myr=np.zeros(3),
                velocity_dispersion_pc_myr=100.0,
            ),
            EnvironmentChannel(
                "gas",
                "absent",
                density_msun_pc3=0.0,
                enclosed_mass_msun=0.0,
                bulk_velocity_pc_myr=np.zeros(3),
                reason="no gas component in this snapshot",
            ),
            EnvironmentChannel(
                "fdm",
                "available",
                density_msun_pc3=5.0e6,
                enclosed_mass_msun=1.0e9,
                bulk_velocity_pc_myr=np.zeros(3),
                core_radius_pc=2.3,
                fdm_mode="analytic_unresolved",
                resolved_wake=False,
            ),
        ),
        source_case_id="case-1",
        source_sha256="b" * 64,
        source_path="environment.json",
    )
    path.write_text(json.dumps(snapshot.as_dict()), encoding="utf-8")


def test_exporter_builds_bridge_from_complete_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "capture.jsonl"
    environment = tmp_path / "environment.json"
    output = tmp_path / "bridge.json"
    _write_ledger(ledger, _ledger_rows())
    _write_environment(environment)

    result = export_bridge_input(
        ledger_path=ledger,
        environment_path=environment,
        event_uid="event-1",
        run_id="run-1",
        capture_time_myr=1000.0,
        output_path=output,
    )
    bridge = NuclearBridgeInput.read_json(result)
    assert bridge.ready_for_integration
    assert bridge.source_sha256 != bridge.environment.source_sha256
    assert bridge.target_semimajor_axis_pc == pytest.approx(1.0)


def test_exporter_rejects_environment_uid_mismatch(tmp_path: Path) -> None:
    ledger = tmp_path / "capture.jsonl"
    environment = tmp_path / "environment.json"
    _write_ledger(ledger, _ledger_rows())
    _write_environment(environment, event_uid="other-event")
    with pytest.raises(ValueError, match="event_uid"):
        export_bridge_input(
            ledger_path=ledger,
            environment_path=environment,
            event_uid="event-1",
            run_id="run-1",
            capture_time_myr=1000.0,
            output_path=tmp_path / "bridge.json",
        )


def test_exporter_never_promotes_incomplete_ledger_tail(tmp_path: Path) -> None:
    ledger = tmp_path / "capture.jsonl"
    environment = tmp_path / "environment.json"
    _write_ledger(ledger, _ledger_rows()[:-1])
    _write_environment(environment)
    with pytest.raises(ValueError, match="incomplete event"):
        export_bridge_input(
            ledger_path=ledger,
            environment_path=environment,
            event_uid="event-1",
            run_id="run-1",
            capture_time_myr=1000.0,
            output_path=tmp_path / "bridge.json",
        )
