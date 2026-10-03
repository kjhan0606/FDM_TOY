from __future__ import annotations

import hashlib
import json

import pytest

from fdm_smbh_delay.backreaction import (
    BackreactionEvidence,
    BackreactionTrackPoint,
    assess_live_frozen_backreaction,
    read_verified_backreaction_delay_record,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decision():
    separations = (1.0, 2.0, 4.0, 8.0)
    live = tuple(
        BackreactionTrackPoint(radius, -2.0 / radius, -1.0 / radius, 0.2)
        for radius in separations
    )
    frozen = tuple(
        BackreactionTrackPoint(radius, -2.0 / radius, -1.0 / radius, 0.2)
        for radius in separations
    )
    evidence = BackreactionEvidence(
        model="cdm",
        live_checkpoint_id="checkpoint-1",
        frozen_checkpoint_id="checkpoint-1",
        live_source_path="live.json",
        live_source_sha256=_sha(b"live"),
        frozen_source_path="frozen.json",
        frozen_source_sha256=_sha(b"frozen"),
        live_force_accounting="live_resolved",
        frozen_force_accounting="frozen_background",
        maximum_live_relative_energy_error=1.0e-4,
        maximum_frozen_relative_energy_error=1.0e-4,
        minimum_live_orbital_resolution_cells=8.0,
        minimum_frozen_orbital_resolution_cells=8.0,
    )
    return assess_live_frozen_backreaction(
        model="cdm", live_points=live, frozen_points=frozen, evidence=evidence
    )


def _record(path, source: str, decision) -> None:
    source_path = path.parent / "integrated.json"
    source_path.write_text(source, encoding="utf-8")
    record = {
        "schema_version": 1,
        "status": "integrated_delay",
        "name": "cdm_kpc_to_hard",
        "model": "cdm",
        "decision_sha256": _sha(
            json.dumps(decision.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        ),
        "delay_myr": 12.5,
        "start_separation_pc": 8.0,
        "end_separation_pc": 1.0,
        "source_case_id": "case-1",
        "source": {"path": source_path.name, "sha256": _sha(source.encode())},
    }
    path.write_text(json.dumps(record), encoding="utf-8")


def test_verified_delay_record_binds_decision_interval_and_source(tmp_path) -> None:
    decision = _decision()
    record = tmp_path / "delay.json"
    _record(record, "solver-integrated-delay-v1\n", decision)
    segment = read_verified_backreaction_delay_record(record, decision=decision)
    assert segment.status == "complete"
    assert segment.delay_myr == pytest.approx(12.5)
    assert segment.source_case_id == "case-1"
    assert segment.source_sha256 == _sha(b"solver-integrated-delay-v1\n")


def test_verified_delay_record_rejects_changed_source(tmp_path) -> None:
    decision = _decision()
    record = tmp_path / "delay.json"
    _record(record, "solver-integrated-delay-v1\n", decision)
    (tmp_path / "integrated.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA-256 differs"):
        read_verified_backreaction_delay_record(record, decision=decision)


def test_verified_delay_record_rejects_outside_overlap(tmp_path) -> None:
    decision = _decision()
    record = tmp_path / "delay.json"
    _record(record, "solver-integrated-delay-v1\n", decision)
    raw = json.loads(record.read_text(encoding="utf-8"))
    raw["start_separation_pc"] = 16.0
    record.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the measured overlap"):
        read_verified_backreaction_delay_record(record, decision=decision)
