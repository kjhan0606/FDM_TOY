from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fdm_smbh_delay.backreaction import (
    BackreactionTrackPoint,
    read_verified_backreaction_decision,
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


def _make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    live = tmp_path / "live.json"
    frozen = tmp_path / "frozen.json"
    _track(live)
    _track(frozen, scale=0.99)
    manifest = tmp_path / "manifest.json"
    common = {
        "checkpoint_id": "checkpoint-1",
        "force_accounting": "live_resolved",
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
            {"schema_version": 1, "model": "cdm", "live": live_record, "frozen": frozen_record}
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
                "model": "cdm",
                "decision_sha256": decision_digest,
                "delay_myr": 12.5,
                "start_separation_pc": 8.0,
                "end_separation_pc": 1.0,
                "source_case_id": "case-1",
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
    return decision, delay_record, fdm, gw


def test_verified_pta_driver_composes_all_three_intervals(tmp_path, capsys) -> None:
    decision, delay, fdm, gw = _make_inputs(tmp_path)
    assert main(
        [
            "--sink-time", "1 Gyr",
            "--backreaction-decision", str(decision),
            "--backreaction-delay-record", str(delay),
            "--fdm-summary", str(fdm),
            "--gw-record", str(gw),
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["estimate"]["status"] == "complete"
    assert output["estimate"]["total_delay_myr"] == 46.5
