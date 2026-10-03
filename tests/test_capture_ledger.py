from __future__ import annotations

import copy
import json

from astropy import units as u
import numpy as np
import pytest

from fdm_smbh_delay.capture_ledger import (
    CaptureLedgerError,
    _close_code,
    read_capture_ledger as _read_capture_ledger,
)
from fdm_smbh_delay.constants import G_INTERNAL


def read_capture_ledger(path, **kwargs):
    # This module also tests historical bare-ledger diagnostics explicitly.
    return _read_capture_ledger(path, allow_legacy_events=True, **kwargs)


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


def _native_binary_rows() -> list[dict]:
    rows = _binary_rows()
    begin, first, second, pair, _ = rows
    mass1 = mass2 = 1.0e8
    position1 = np.asarray(first["position_code"])
    position2 = np.asarray(second["position_code"])
    velocity1 = np.asarray(first["velocity_code"])
    velocity2 = np.asarray(second["velocity_code"])
    dr = position2 - position1
    dv = velocity2 - velocity1
    radius = float(np.linalg.norm(dr))
    speed = float(np.linalg.norm(dv))
    reduced_mass = mass1 * mass2 / (mass1 + mass2)
    specific_h = np.cross(dr, dv)
    begin.update(
        factG_code=G_INTERNAL,
        total_mass_code=mass1 + mass2,
        com_position_code=[0.0, 0.0, 0.0],
        com_velocity_code=[0.0, 0.0, 0.0],
        max_pair_separation_code=radius,
    )
    pair.update(
        delta_position_code=dr.tolist(),
        separation_code=radius,
        delta_velocity_code=dv.tolist(),
        relative_speed_code=speed,
        reduced_mass_code=reduced_mass,
        relative_kinetic_code=0.5 * reduced_mass * speed**2,
        newtonian_potential_1overr_code=-G_INTERNAL * mass1 * mass2 / radius,
        two_body_specific_energy_code=0.5 * speed**2 - G_INTERNAL * (mass1 + mass2) / radius,
        specific_angular_momentum_code=specific_h.tolist(),
        relative_angular_momentum_code=(reduced_mass * specific_h).tolist(),
        legacy_binding_proxy_1overr2_code=G_INTERNAL * mass1 * mass2 / radius**2,
    )
    return rows


def _committed_batch_rows(
    *, step: int = 10, sink_ids: tuple[int, int] = (7, 9),
) -> list[dict]:
    rows = _native_binary_rows()
    uid = f"{step}-1-{sink_ids[0]}-{sink_ids[1]}-2"
    for row in rows:
        row["event_uid"] = uid
    rows[0].update(
        nstep_coarse=step, primary_sink_id=sink_ids[0],
        periodic_box_size_code=[100.0, 100.0, 100.0],
    )
    for row, sink_id in zip(rows[1:3], sink_ids):
        row.update(
            sink_id=sink_id, primary_sink_id=sink_ids[0],
            is_primary=sink_id == sink_ids[0],
        )
    rows[-2].update(sink_id_1=sink_ids[0], sink_id_2=sink_ids[1])
    batch_uid = f"{step}-1-2-1"
    return [
        {"schema_version": 1, "record_type": "batch_begin", "batch_uid": batch_uid,
         "nstep_coarse": step, "ilevel": 1, "nsink_before": 2,
         "nsink_after": 1, "expected_events": 1},
        *rows,
        {"schema_version": 1, "record_type": "batch_commit", "batch_uid": batch_uid,
         "nsink_after": 1},
    ]


def _attempt(restart_output: int = 0, resume_step: int = 0) -> dict:
    return {"schema_version": 1, "record_type": "attempt_begin",
            "restart_output": restart_output, "resume_step": resume_step}


def test_batched_capture_requires_post_compaction_commit(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    rows = [_attempt(), *_committed_batch_rows()]
    _write_rows(path, rows)
    assert [event.event_uid for event in read_capture_ledger(path).events] == ["10-1-7-9-2"]
    _write_rows(path, rows[:-1])
    with pytest.raises(CaptureLedgerError, match="incomplete batch"):
        read_capture_ledger(path)
    censored = read_capture_ledger(path, allow_incomplete_tail=True)
    assert censored.events == ()
    assert censored.censored_batch_uids == ("10-1-2-1",)


def test_native_tolerance_does_not_mask_tiny_code_unit_errors() -> None:
    assert not _close_code(1.0e-20, -1.0e-20)
    assert not _close_code(1.0e-12, 1.001e-12)
    assert _close_code(1.0e-20, 1.0e-20 * (1.0 + 1.0e-13))


def test_close_pair_far_from_origin_uses_coordinate_roundoff_scale(tmp_path) -> None:
    rows = _native_binary_rows()
    begin, first, second, pair, _ = rows
    first["position_code"] = [50.0, 0.0, 0.0]
    second["position_code"] = [50.000001, 0.0, 0.0]
    dr = second["position_code"][0] - first["position_code"][0]
    speed = abs(second["velocity_code"][1] - first["velocity_code"][1])
    begin["com_position_code"] = [50.0000005, 0.0, 0.0]
    begin["max_pair_separation_code"] = dr + 1.0e-16
    pair.update(
        delta_position_code=[dr, 0.0, 0.0], separation_code=dr + 1.0e-16,
        specific_angular_momentum_code=[0.0, 0.0, -dr * speed],
        relative_angular_momentum_code=[0.0, 0.0, -5.0e7 * dr * speed],
        newtonian_potential_1overr_code=-G_INTERNAL * 1.0e16 / dr,
        two_body_specific_energy_code=0.5 * speed**2 - G_INTERNAL * 2.0e8 / dr,
        legacy_binding_proxy_1overr2_code=G_INTERNAL * 1.0e16 / dr**2,
    )
    pair["two_body_bound"] = bool(pair["two_body_specific_energy_code"] < 0.0)
    pair["legacy_pair_bound"] = bool(
        pair["relative_kinetic_code"] < pair["legacy_binding_proxy_1overr2_code"]
    )
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, rows)
    assert read_capture_ledger(path).events[0].native_conservation_verified


def test_restarted_capture_censors_uncommitted_batch(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = _committed_batch_rows()
    second = _committed_batch_rows(step=11)
    rows = [_attempt(), *first[:-1],
            {"schema_version": 1, "record_type": "checkpoint",
             "output_number": 1, "nstep_coarse": 10}]
    _write_rows(path, rows)
    with pytest.raises(CaptureLedgerError, match="checkpoint outside"):
        read_capture_ledger(path, allow_incomplete_batches=True)
    rows = [_attempt(), *first[:-1], _attempt(1, 10), *second]
    # A restart cannot name a checkpoint that was never written.
    _write_rows(path, rows)
    with pytest.raises(CaptureLedgerError, match="no matching ledger checkpoint"):
        read_capture_ledger(path, allow_incomplete_batches=True)
    rows = [_attempt(), {"schema_version": 1, "record_type": "checkpoint",
                        "output_number": 1, "nstep_coarse": 10},
            *first[:-1], _attempt(1, 10), *second]
    _write_rows(path, rows)
    with pytest.raises(CaptureLedgerError, match="missing batch_commit"):
        read_capture_ledger(path, allow_incomplete_batches=False)
    ledger = read_capture_ledger(path)
    assert [event.event_uid for event in ledger.events] == ["11-1-7-9-2"]
    assert ledger.censored_batch_uids == ("10-1-2-1",)


def test_restart_censors_one_torn_jsonl_row_after_checkpoint(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    checkpoint = {"schema_version": 1, "record_type": "checkpoint",
                  "output_number": 1, "nstep_coarse": 10}
    rows = [_attempt(), checkpoint, *_committed_batch_rows()[:-1]]
    _write_rows(path, rows)
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"schema_version":1,"record_type":"batch_commit"\n\n')
        for row in [_attempt(1, 10), *_committed_batch_rows(step=11)]:
            stream.write(json.dumps(row) + "\n")
    ledger = read_capture_ledger(path)
    assert [event.event_uid for event in ledger.events] == ["11-1-7-9-2"]
    assert ledger.censored_batch_uids == ("10-1-2-1",)

    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"bad":\n')
    with pytest.raises(CaptureLedgerError, match="invalid JSON"):
        read_capture_ledger(path)


def test_missing_batch_commit_without_restart_is_not_censored(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    batch = _committed_batch_rows()
    _write_rows(path, [_attempt(), *batch[:-1], *batch])
    with pytest.raises(CaptureLedgerError, match="same attempt"):
        read_capture_ledger(path)
    _write_rows(path, [_attempt(), *batch[:-1]])
    with pytest.raises(CaptureLedgerError, match="incomplete batch"):
        read_capture_ledger(path, allow_incomplete_batches=True)


def test_restart_cutline_uses_checkpoint_record_order(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    rows = [_attempt(), *_committed_batch_rows(),
            {"schema_version": 1, "record_type": "checkpoint",
             "output_number": 1, "nstep_coarse": 10},
            _attempt(1, 10), *_committed_batch_rows(step=11)]
    _write_rows(path, rows)
    ledger = read_capture_ledger(path)
    assert [event.event_uid for event in ledger.events] == [
        "10-1-7-9-2", "11-1-7-9-2"
    ]
    # The same-step capture after the snapshot marker was not in that output.
    rows = [_attempt(),
            {"schema_version": 1, "record_type": "checkpoint",
             "output_number": 1, "nstep_coarse": 10},
            *_committed_batch_rows(), _attempt(1, 10),
            *_committed_batch_rows(step=11)]
    _write_rows(path, rows)
    assert [event.event_uid for event in read_capture_ledger(path).events] == [
        "11-1-7-9-2"
    ]


def test_committed_multiple_preserves_all_three_members_and_pairs(tmp_path) -> None:
    rows = _native_binary_rows()
    begin, first, second, _, end = rows
    uid = "10-1-7-11-3"
    begin.update(
        event_uid=uid, classification="MULTIPLE", nmember=3,
        expected_pairs=3, primary_sink_id=7,
        periodic_box_size_code=[100.0, 100.0, 100.0],
        total_mass_code=3.0, com_position_code=[1.0, 0.0, 0.0],
        com_velocity_code=[0.0, 0.0, 0.0], max_pair_separation_code=2.0,
        merge_radius_code=2.5, factG_code=1.0,
    )
    third = copy.deepcopy(second)
    members = [first, second, third]
    for index, (member, sink_id) in enumerate(zip(members, (7, 9, 11))):
        member.update(
            event_uid=uid, member_index=index + 1, sink_id=sink_id,
            primary_sink_id=7, is_primary=index == 0, mass_code=1.0,
            position_code=[float(index), 0.0, 0.0], velocity_code=[0.0, 0.0, 0.0],
        )
    pairs = []
    for pair_index, (id1, id2, separation) in enumerate(
        ((7, 9, 1.0), (7, 11, 2.0), (9, 11, 1.0)), start=1
    ):
        pair = copy.deepcopy(rows[-2])
        pair.update(
            event_uid=uid, pair_index=pair_index, sink_id_1=id1, sink_id_2=id2,
            delta_position_code=[separation, 0.0, 0.0], separation_code=separation,
            delta_velocity_code=[0.0, 0.0, 0.0], relative_speed_code=0.0,
            reduced_mass_code=0.5, relative_kinetic_code=0.0,
            newtonian_potential_1overr_code=-1.0 / separation,
            two_body_specific_energy_code=-2.0 / separation,
            specific_angular_momentum_code=[0.0, 0.0, 0.0],
            relative_angular_momentum_code=[0.0, 0.0, 0.0],
            legacy_binding_proxy_1overr2_code=1.0 / separation**2,
            within_rmerge=True, two_body_bound=True, legacy_pair_bound=True,
        )
        pairs.append(pair)
    end.update(event_uid=uid, nmember=3, npair=3)
    batch_uid = "10-1-3-1"
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, [
        _attempt(),
        {"schema_version": 1, "record_type": "batch_begin", "batch_uid": batch_uid,
         "nstep_coarse": 10, "ilevel": 1, "nsink_before": 3,
         "nsink_after": 1, "expected_events": 1},
        begin, *members, *pairs, end,
        {"schema_version": 1, "record_type": "batch_commit", "batch_uid": batch_uid,
         "nsink_after": 1},
    ])
    ledger = read_capture_ledger(path)
    assert len(ledger.events) == 1
    event = ledger.events[0]
    assert event.classification == "MULTIPLE"
    assert [member.sink_id for member in event.members] == [7, 9, 11]
    assert len(event.pairs) == 3
    assert event.binary_orbital_state is None
    assert event.native_conservation_verified


def test_bad_batch_conservation_and_bare_event_are_rejected(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    rows = [_attempt(), *_committed_batch_rows()]
    bad = copy.deepcopy(rows)
    bad[-1]["nsink_after"] = 2
    _write_rows(path, bad)
    with pytest.raises(CaptureLedgerError, match="invalid batch commit"):
        read_capture_ledger(path)
    _write_rows(path, rows + _binary_rows())
    with pytest.raises(CaptureLedgerError, match="bare event"):
        read_capture_ledger(path)


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


def test_legacy_event_is_not_admitted_by_default(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, _native_binary_rows())
    with pytest.raises(CaptureLedgerError, match="cannot prove post-compaction capture"):
        _read_capture_ledger(path)
    assert read_capture_ledger(path).events[0].native_conservation_verified


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


def test_preserved_multiple_is_ingested_without_selecting_a_binary(tmp_path) -> None:
    from fdm_smbh_delay.kpc_to_pc import InspiralPhase, classify_capture_state

    rows = _binary_rows("10-1-7-11-3-3FF0000000000000")
    begin, first, second, pair, end = rows
    begin.update(classification="MULTIPLE", nmember=3, expected_pairs=3,
                 multiple_members_preserved=True)
    third = copy.deepcopy(second)
    third.update(member_index=3, sink_id=11, position_code=[0.0, 0.5, 0.0])
    pairs = []
    for index, ids in enumerate(((7, 9), (7, 11), (9, 11)), start=1):
        item = copy.deepcopy(pair)
        item.update(pair_index=index, sink_id_1=ids[0], sink_id_2=ids[1])
        pairs.append(item)
    end.update(nmember=3, npair=3)
    rows = [begin, first, second, third, *pairs, end]
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, rows + rows)
    ledger = read_capture_ledger(path)
    event = ledger.events[0]
    assert ledger.duplicate_events == 1
    assert event.multiple_members_preserved is True
    assert [member.sink_id for member in event.members] == [7, 9, 11]
    assert len(event.pairs) == 3
    assert event.binary_orbital_state is None
    assert classify_capture_state(
        event, common_nucleus_radius_pc=10.0, sigma_pc_myr=100.0
    ).phase == InspiralPhase.MULTIPLE


@pytest.mark.parametrize("flag", [True, "true", 1, None])
def test_invalid_multiple_preservation_claim_is_rejected(tmp_path, flag) -> None:
    rows = _binary_rows()
    rows[0]["multiple_members_preserved"] = flag
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, rows)
    with pytest.raises(CaptureLedgerError, match="multiple_members"):
        read_capture_ledger(path)


def test_legacy_event_has_unknown_multiple_preservation_policy(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, _binary_rows())
    assert read_capture_ledger(path).events[0].multiple_members_preserved is None


@pytest.mark.parametrize("field", ["within_rmerge", "two_body_bound", "legacy_pair_bound"])
def test_capture_ledger_rejects_non_boolean_pair_flags(tmp_path, field: str) -> None:
    rows = _binary_rows()
    rows[-2][field] = "false"
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, rows)
    with pytest.raises(CaptureLedgerError, match=f"{field} must be a boolean"):
        read_capture_ledger(path)


def test_capture_ledger_recomputes_numerical_merge_radius_flag(tmp_path) -> None:
    rows = _binary_rows()
    rows[-2]["within_rmerge"] = False
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, rows)
    with pytest.raises(CaptureLedgerError, match="within_rmerge contradicts"):
        read_capture_ledger(path)


def test_capture_ledger_rejects_source_binding_flag_energy_disagreement(tmp_path) -> None:
    rows = _binary_rows()
    rows[-2]["two_body_specific_energy_code"] = 1.0
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, rows)
    with pytest.raises(CaptureLedgerError, match="two_body_bound contradicts"):
        read_capture_ledger(path)


def test_capture_ledger_rejects_noncontiguous_pair_indices(tmp_path) -> None:
    rows = _binary_rows()
    rows[-2]["pair_index"] = 2
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, rows)
    with pytest.raises(CaptureLedgerError, match="pair indices are not contiguous"):
        read_capture_ledger(path)


def test_capture_ledger_merge_flag_uses_periodic_minimum_image(tmp_path) -> None:
    rows = _binary_rows()
    rows[1]["position_code"] = [99.5, 0.0, 0.0]
    rows[2]["position_code"] = [0.5, 0.0, 0.0]
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, rows)
    event = read_capture_ledger(path).events[0]
    assert event.pairs[0].within_numerical_merge_radius
    assert event.pairs[0].orbital_state.separation_pc == pytest.approx(1.0)


def test_complete_native_conservation_diagnostics_are_checked(tmp_path) -> None:
    rows = _native_binary_rows()
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, rows)
    assert read_capture_ledger(path).events[0].binary_orbital_state is not None


def test_native_capture_uses_axis_specific_periodic_extents(tmp_path) -> None:
    rows = _native_binary_rows()
    begin, first, second, pair, _ = rows
    begin["periodic_box_size_code"] = [100.0, 20.0, 30.0]
    begin["com_position_code"] = [0.0, 0.0, 0.0]
    first["position_code"] = [0.0, 19.5, 0.0]
    second["position_code"] = [0.0, 0.5, 0.0]
    pair["delta_position_code"] = [0.0, 1.0, 0.0]
    pair["specific_angular_momentum_code"] = [0.0, 0.0, 0.0]
    pair["relative_angular_momentum_code"] = [0.0, 0.0, 0.0]
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, rows)
    event = read_capture_ledger(path).events[0]
    assert event.native_conservation_verified
    assert event.pairs[0].orbital_state is not None
    assert event.pairs[0].orbital_state.separation_pc == pytest.approx(1.0)
    com_y = event.pairs[0].orbital_state.centre_of_mass_position_pc[1]
    assert min(abs(com_y), abs(com_y - 20.0)) < 1.0e-12

    begin["periodic_box_size_code"] = [100.0, 10.0, 30.0]
    _write_rows(path, rows)
    with pytest.raises(CaptureLedgerError, match="source geometry|native"):
        read_capture_ledger(path)


def test_capture_ledger_rejects_child_schema_mismatch(tmp_path) -> None:
    rows = _binary_rows()
    rows[1]["schema_version"] = 2
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, rows)
    with pytest.raises(CaptureLedgerError, match="unsupported ledger schema"):
        read_capture_ledger(path)


@pytest.mark.parametrize(("row_index", "field", "replacement", "expected"), [
    (0, "total_mass_code", 3.0e8, "total-mass"),
    (0, "com_position_code", [1.0, 0.0, 0.0], "centre-of-mass position"),
    (0, "com_velocity_code", [1.0, 0.0, 0.0], "centre-of-mass velocity"),
    (0, "max_pair_separation_code", 2.0, "maximum pair separation"),
    (-2, "delta_position_code", [2.0, 0.0, 0.0], "delta_position_code"),
    (-2, "relative_kinetic_code", 0.0, "relative_kinetic_code"),
    (-2, "specific_angular_momentum_code", [0.0, 0.0, 0.0], "specific_angular_momentum_code"),
    (-2, "newtonian_potential_1overr_code", -1.0, "newtonian_potential_1overr_code"),
    (-2, "legacy_binding_proxy_1overr2_code", 0.0, "legacy_binding_proxy_1overr2_code"),
])
def test_capture_ledger_rejects_corrupt_native_conservation(
    tmp_path, row_index: int, field: str, replacement: object, expected: str
) -> None:
    rows = _native_binary_rows()
    rows[row_index][field] = replacement
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, rows)
    with pytest.raises(CaptureLedgerError, match=expected):
        read_capture_ledger(path)


def test_capture_ledger_rejects_partial_native_diagnostics(tmp_path) -> None:
    rows = _native_binary_rows()
    del rows[-2]["relative_kinetic_code"]
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, rows)
    with pytest.raises(CaptureLedgerError, match="incomplete native pair"):
        read_capture_ledger(path)


def test_capture_ledger_rejects_periodic_extents_without_native_diagnostics(tmp_path) -> None:
    rows = _binary_rows()
    rows[0]["periodic_box_size_code"] = [100.0, 100.0, 100.0]
    path = tmp_path / "ledger.jsonl"
    _write_rows(path, rows)
    with pytest.raises(CaptureLedgerError, match="incomplete native event"):
        read_capture_ledger(path)
