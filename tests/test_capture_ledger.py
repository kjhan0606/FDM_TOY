from __future__ import annotations

import copy
import json

from astropy import units as u
import numpy as np
import pytest

from fdm_smbh_delay.capture_ledger import (
    CaptureLedgerError,
    read_capture_ledger,
)
from fdm_smbh_delay.constants import G_INTERNAL


def _binary_rows(uid: str = "10-1-7-9-2") -> list[dict]:
    unit_length = (1.0 * u.pc).to_value(u.cm)
    unit_velocity = (1.0 * u.pc / u.Myr).to_value(u.cm / u.s)
    unit_mass = (1.0 * u.Msun).to_value(u.g)
    relative_speed = np.sqrt(G_INTERNAL * 2.0e8)
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
        ((7, 0.5, 0.5 * relative_speed), (9, -0.5, -0.5 * relative_speed)),
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


def _write_rows(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_capture_ledger_converts_code_units_and_recovers_binary(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, _binary_rows())
    ledger = read_capture_ledger(path)
    assert len(ledger.events) == 1
    event = ledger.events[0]
    assert event.numerical_merge_radius_pc == pytest.approx(2.0)
    assert event.members[0].mass_msun == pytest.approx(1.0e8)
    assert event.binary_orbital_state is not None
    assert event.binary_orbital_state.separation_pc == pytest.approx(1.0)
    assert event.binary_orbital_state.eccentricity == pytest.approx(0.0, abs=2.0e-14)


def test_exact_restart_event_is_deduplicated(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    rows = _binary_rows()
    _write_rows(path, rows + rows)
    ledger = read_capture_ledger(path)
    assert len(ledger.events) == 1
    assert ledger.duplicate_events == 1


def test_conflicting_restart_event_is_rejected(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    rows = _binary_rows()
    conflict = copy.deepcopy(rows)
    conflict[1]["mass_code"] *= 2.0
    _write_rows(path, rows + conflict)
    with pytest.raises(CaptureLedgerError, match="conflicting deterministic"):
        read_capture_ledger(path)


def test_incomplete_tail_is_never_promoted(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    rows = _binary_rows()[:-1]
    _write_rows(path, rows)
    with pytest.raises(CaptureLedgerError, match="incomplete event"):
        read_capture_ledger(path)
    ledger = read_capture_ledger(path, allow_incomplete_tail=True)
    assert ledger.events == ()
    assert ledger.incomplete_event_uids == ("10-1-7-9-2",)
