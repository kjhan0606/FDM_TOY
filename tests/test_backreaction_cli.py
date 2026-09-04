from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from fdm_smbh_delay.backreaction import BackreactionTrackPoint
from fdm_smbh_delay.backreaction import read_verified_backreaction_decision


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _track(path: Path, *, scale: float = 1.0) -> None:
    points = [
        BackreactionTrackPoint(
            separation_pc=radius,
            orbital_power_pc2_myr3=-2.0 * scale / radius,
            orbital_torque_msun_pc2_myr=-1.0 * scale / radius,
            eccentricity=0.2,
        ).as_dict()
        for radius in (1.0, 2.0, 4.0, 8.0)
    ]
    path.write_text(
        json.dumps(
            {"schema_version": 1, "status": "measured_track", "track": points},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _manifest(tmp_path: Path, *, frozen_scale: float = 0.99) -> Path:
    live = tmp_path / "live.json"
    frozen = tmp_path / "frozen.json"
    _track(live)
    _track(frozen, scale=frozen_scale)
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
            {
                "schema_version": 1,
                "model": "cdm",
                "live": live_record,
                "frozen": frozen_record,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_cli_verifies_source_hashes_and_writes_decision(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "decision.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/assess_live_frozen_backreaction.py",
            str(manifest),
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "offline_acceptable"
    assert record["input_manifest"]["sha256"] == _sha(manifest)
    assert record["gates"]["maximum_rate_fractional_difference"] == 0.2
    verified = read_verified_backreaction_decision(output)
    assert verified.offline_acceptable


def test_saved_decision_status_edit_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "decision.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/assess_live_frozen_backreaction.py",
            str(manifest),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    record["status"] = "offline_acceptable"
    record["reasons"] = ["manually edited"]
    output.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from current evidence"):
        read_verified_backreaction_decision(output)


def test_cli_reports_runtime_required_for_a_large_change(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, frozen_scale=1.5)
    output = tmp_path / "decision.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/assess_live_frozen_backreaction.py",
            str(manifest),
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "runtime_required"


def test_cli_rejects_changed_track_bytes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    live = tmp_path / "live.json"
    live.write_text(live.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    output = tmp_path / "decision.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/assess_live_frozen_backreaction.py",
            str(manifest),
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert not output.exists()


def test_cli_rejects_looser_custom_gate(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["gates"] = {"maximum_rate_fractional_difference": 0.5}
    manifest.write_text(json.dumps(record), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/assess_live_frozen_backreaction.py",
            str(manifest),
            str(tmp_path / "decision.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
