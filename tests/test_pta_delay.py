from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from fdm_smbh_delay.backreaction import (
    BackreactionTrackPoint,
    read_verified_backreaction_decision,
)
from fdm_smbh_delay.capture_ledger import read_capture_ledger
from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.nuclear_bridge import (
    EnvironmentChannel,
    EnvironmentSnapshot,
    NuclearBridgeInput,
)
from fdm_smbh_delay.pta_delay_cli import main


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _track(path: Path, scale: float = 1.0) -> None:
    points = [
        BackreactionTrackPoint(radius, -2.0 * scale / radius, -1.0 / radius, 0.2).as_dict()
        for radius in (1.0, 2.0, 4.0, 8.0)
    ]
    path.write_text(
        json.dumps({"schema_version": 1, "status": "measured_track", "track": points}),
        encoding="utf-8",
    )


def _capture_bridge(tmp_path: Path) -> Path:
    speed = np.sqrt(G_INTERNAL * 2.0e8 / 8.0)
    uid = "capture-1"
    begin = {
        "schema_version": 1, "record_type": "event_begin", "event_uid": uid,
        "classification": "BINARY", "nstep_coarse": 10, "ilevel": 1,
        "nmember": 2, "expected_pairs": 1, "aexp": 0.5, "redshift": 1.0,
        "t_code": -0.2, "texp": 0.4, "merge_radius_code": 10.0,
        "unit_length_cgs": (1.0 * u.pc).to_value(u.cm),
        "unit_velocity_cgs": (1.0 * u.pc / u.Myr).to_value(u.cm / u.s),
        "unit_mass_cgs": (1.0 * u.Msun).to_value(u.g),
        "boxlen": 100.0, "complete": False,
    }
    members = [
        {
            "schema_version": 1, "record_type": "member", "event_uid": uid,
            "member_index": index, "sink_id": sink_id, "mass_code": 1.0e8,
            "position_code": [x, 0.0, 0.0], "velocity_code": [0.0, vy, 0.0],
            "formation_time_code": -0.4, "accreted_mass_code": 0.0,
            "spin_magnitude": 0.5, "spin_direction": [0.0, 0.0, 1.0],
            "gas_angular_momentum_code": [0.0, 0.0, 0.0],
        }
        for index, (sink_id, x, vy) in enumerate(
            ((7, 4.0, speed / 2.0), (9, -4.0, -speed / 2.0)), start=1
        )
    ]
    pair = {
        "schema_version": 1, "record_type": "pair", "event_uid": uid,
        "pair_index": 1, "sink_id_1": 7, "sink_id_2": 9,
        "within_rmerge": True, "two_body_bound": True, "legacy_pair_bound": True,
    }
    end = {
        "schema_version": 1, "record_type": "event_end", "event_uid": uid,
        "nmember": 2, "npair": 1, "complete": True,
    }
    ledger_path = tmp_path / "capture.jsonl"
    ledger_path.write_text(
        "".join(json.dumps(row) + "\n" for row in (begin, *members, pair, end)),
        encoding="utf-8",
    )
    event = read_capture_ledger(ledger_path).events[0]
    absent_stars = EnvironmentChannel(
        "stellar", "absent", density_msun_pc3=0.0, enclosed_mass_msun=0.0,
        bulk_velocity_pc_myr=np.zeros(3), reason="synthetic empty stellar channel",
    )
    absent_gas = EnvironmentChannel(
        "gas", "absent", density_msun_pc3=0.0, enclosed_mass_msun=0.0,
        bulk_velocity_pc_myr=np.zeros(3), reason="synthetic empty gas channel",
    )
    fdm = EnvironmentChannel(
        "fdm", "available", density_msun_pc3=1.0e6,
        enclosed_mass_msun=1.0e9, bulk_velocity_pc_myr=np.zeros(3),
        core_radius_pc=2.0, fdm_mode="analytic_unresolved", resolved_wake=False,
    )
    environment = EnvironmentSnapshot(
        event_uid=uid, time_myr=1000.0, redshift=1.0, radius_pc=8.0,
        channels=(absent_stars, absent_gas, fdm), source_case_id=uid,
        source_sha256="c" * 64, source_path=str(ledger_path),
    )
    bridge = NuclearBridgeInput.from_capture_event(
        event, run_id="synthetic-run", capture_time_myr=1000.0,
        environment=environment,
    )
    return bridge.write_json(tmp_path / "bridge.json")


def _make_inputs(
    tmp_path: Path, *, model: str = "fdm"
) -> tuple[Path, Path, Path, Path, Path]:
    bridge = _capture_bridge(tmp_path)
    live = tmp_path / "live.json"
    frozen = tmp_path / "frozen.json"
    _track(live)
    _track(frozen, scale=0.99)
    manifest = tmp_path / "manifest.json"
    common = {
        "checkpoint_id": "checkpoint-1",
        "force_accounting": "live_wave_only" if model == "fdm" else "live_resolved",
        "maximum_relative_energy_error": 1.0e-4,
        "minimum_orbital_resolution_cells": 8.0,
    }
    live_record = dict(common)
    live_record["source"] = {"path": live.name, "sha256": _sha(live)}
    frozen_record = dict(common)
    frozen_record["force_accounting"] = "frozen_background"
    frozen_record["source"] = {"path": frozen.name, "sha256": _sha(frozen)}
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "model": model, "live": live_record, "frozen": frozen_record}
        ),
        encoding="utf-8",
    )
    decision = tmp_path / "decision.json"
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/assess_live_frozen_backreaction.py", str(manifest), str(decision)],
        check=True,
        capture_output=True,
        text=True,
    )
    verified = read_verified_backreaction_decision(decision)
    integrated = tmp_path / "integrated.json"
    integrated.write_text("integrated by accepted estimator\n", encoding="utf-8")
    delay_record = tmp_path / "delay.json"
    decision_digest = hashlib.sha256(
        json.dumps(verified.as_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    delay_record.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "integrated_delay",
                "name": "kpc_to_pc",
                "model": model,
                "decision_sha256": decision_digest,
                "delay_myr": 12.5,
                "start_separation_pc": 8.0,
                "end_separation_pc": 1.0,
                "source_case_id": "capture-1",
                "source": {"path": integrated.name, "sha256": _sha(integrated)},
            }
        ),
        encoding="utf-8",
    )
    fdm = tmp_path / "fdm.json"
    fdm.write_text(
        json.dumps(
            {
                "status": "reached_0p01pc",
                "t_fdm_myr": 30.0,
                "D_initial_pc": 1.0,
                "D_stop_pc": 0.01,
                "validity_flags": [],
                "source_case_id": "fdm-case-1",
                "source_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    gw = tmp_path / "gw.json"
    gw.write_text(
        json.dumps(
            {
                "name": "gravitational_wave",
                "status": "complete",
                "delay_myr": 4.0,
                "elapsed_lower_bound_myr": 0.0,
                "reason": None,
                "source_case_id": "gw-case-1",
                "source_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    return bridge, decision, delay_record, fdm, gw


def test_verified_pta_driver_composes_all_three_intervals(tmp_path, capsys) -> None:
    bridge, decision, delay, fdm, gw = _make_inputs(tmp_path)
    assert main(
        [
            "--sink-time", "1 Gyr",
            "--capture-bridge", str(bridge),
            "--backreaction-decision", str(decision),
            "--backreaction-delay-record", str(delay),
            "--fdm-summary", str(fdm),
            "--gw-record", str(gw),
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["estimate"]["status"] == "complete"
    assert output["estimate"]["total_delay_myr"] == 46.5


def test_pta_driver_rejects_kpc_record_that_stops_before_one_pc(tmp_path) -> None:
    bridge, decision, delay, fdm, gw = _make_inputs(tmp_path)
    record = json.loads(delay.read_text(encoding="utf-8"))
    record["end_separation_pc"] = 2.0
    delay.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="1 pc FDM handoff"):
        main(
            [
                "--sink-time", "1 Gyr",
                "--capture-bridge", str(bridge),
                "--backreaction-decision", str(decision),
                "--backreaction-delay-record", str(delay),
                "--fdm-summary", str(fdm),
                "--gw-record", str(gw),
            ]
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("start_separation_pc", 7.0, "start separation differs"),
        ("source_case_id", "different-capture", "event UID differs"),
    ],
)
def test_pta_driver_rejects_kpc_record_not_bound_to_capture(
    tmp_path, field, value, error
) -> None:
    bridge, decision, delay, fdm, gw = _make_inputs(tmp_path)
    record = json.loads(delay.read_text(encoding="utf-8"))
    record[field] = value
    delay.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        main([
            "--sink-time", "1 Gyr", "--capture-bridge", str(bridge),
            "--backreaction-decision", str(decision),
            "--backreaction-delay-record", str(delay),
            "--fdm-summary", str(fdm), "--gw-record", str(gw),
        ])


def test_pta_driver_rejects_changed_capture_ledger_and_sink_time(tmp_path) -> None:
    bridge, decision, delay, fdm, gw = _make_inputs(tmp_path)
    arguments = [
        "--capture-bridge", str(bridge),
        "--backreaction-decision", str(decision),
        "--backreaction-delay-record", str(delay),
        "--fdm-summary", str(fdm), "--gw-record", str(gw),
    ]
    with pytest.raises(ValueError, match="sink time differs"):
        main(["--sink-time", "1001 Myr", *arguments])
    ledger = tmp_path / "capture.jsonl"
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    records[0]["t_code"] = -0.21
    ledger.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        main(["--sink-time", "1 Gyr", *arguments])


def test_pta_driver_rejects_cdm_kpc_stage_in_pure_fdm_chain(tmp_path) -> None:
    bridge, decision, delay, fdm, gw = _make_inputs(tmp_path, model="cdm")
    with pytest.raises(ValueError, match="requires an FDM kpc"):
        main([
            "--sink-time", "1 Gyr", "--capture-bridge", str(bridge),
            "--backreaction-decision", str(decision),
            "--backreaction-delay-record", str(delay),
            "--fdm-summary", str(fdm), "--gw-record", str(gw),
        ])
