"""Producer-to-consumer regression using a native lagRamses v2 ledger."""

from pathlib import Path

from fdm_smbh_delay.capture_ledger import read_capture_ledger


FIXTURE = Path(__file__).parent / "fixtures" / "lagramses_native_v2_capture"


def test_native_lagramses_v2_capture_and_restart_lineage() -> None:
    ledger = read_capture_ledger(
        FIXTURE / "smbh_capture_ledger_v2.jsonl",
        output_root=FIXTURE,
    )

    assert ledger.duplicate_events == 0
    assert ledger.incomplete_event_uids == ()
    assert ledger.censored_batch_uids == ()
    assert len(ledger.events) == 2

    events = {event.classification: event for event in ledger.events}
    assert set(events) == {"BINARY", "MULTIPLE"}

    binary = events["BINARY"]
    assert binary.event_uid == "0-1-1-2-2"
    assert [member.sink_id for member in binary.members] == [1, 2]
    assert len(binary.members) == 2
    assert len(binary.pairs) == 1
    assert binary.binary_orbital_state is not None

    multiple = events["MULTIPLE"]
    assert multiple.event_uid == "0-1-3-5-3"
    assert [member.sink_id for member in multiple.members] == [3, 4, 5]
    assert len(multiple.members) == 3
    assert len(multiple.pairs) == 3
    assert multiple.binary_orbital_state is None

    for event in ledger.events:
        assert event.native_conservation_verified
        assert event.post_compaction_verified
        assert event.lineage_verified
