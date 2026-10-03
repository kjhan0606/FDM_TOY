from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from scripts import assess_qe_box_control as cli


def test_candidate_cli_publishes_once_without_replacing_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "status": "qe_no_box_controlled_bins_censored",
        "candidate_rows": [],
        "production_calibration_row_admitted": False,
    }
    monkeypatch.setattr(
        cli, "prepare_qe_calibration_candidate", lambda *args: payload
    )
    output = tmp_path / "candidate.json"
    monkeypatch.setattr(sys, "argv", [
        "assess_qe_box_control.py", "--profile-id", "test",
        "--resolution-pair", "pair.json", "--doubled-box", "box.json",
        "--candidate-output", str(output),
    ])
    cli.main()
    assert json.loads(output.read_text()) == payload
    with pytest.raises(FileExistsError):
        cli.main()
    assert json.loads(output.read_text()) == payload
    assert list(tmp_path.glob(".candidate.json.partial-*")) == []
