from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from launch_torch_wave_case import _output_argument  # noqa: E402
import snapshot_torch_provenance  # noqa: E402
from snapshot_torch_provenance import _snapshot_run  # noqa: E402


def test_output_argument_accepts_both_cli_forms(tmp_path: Path) -> None:
    expected = (tmp_path / "run").resolve()
    assert _output_argument(["reference", "--output", str(expected)]) == expected
    assert _output_argument([f"--output={expected}", "reference"]) == expected
    with pytest.raises(ValueError, match="--output"):
        _output_argument(["reference"])


def test_source_snapshot_is_repeatable_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.uldm").write_text("{}")
    (run / "fdm_adapter_metadata.json").write_text(
        json.dumps(
            {
                "backend": "pytorch_cpu",
                "adapter_revision": revision,
                "adapter_worktree_clean": False,
                "reference_initial_state": "/reference",
                "pyul_revision": "upstream-revision",
            }
        )
    )

    first = _snapshot_run(PROJECT, run)
    monkeypatch.setattr(
        snapshot_torch_provenance.platform,
        "platform",
        lambda: "different-restart-host",
    )
    second = _snapshot_run(PROJECT, run)

    assert first == second
    assert len(first["source_files"]) == 3
    manifest = json.loads(
        (run / "torch_solver_provenance" / "manifest.json").read_text()
    )
    assert manifest["source_files"][0]["origin"] == "launch_worktree"
    frozen = (
        run
        / "torch_solver_provenance"
        / "source"
        / "scripts"
        / "run_torch_wave_case.py"
    )
    frozen.write_text("changed")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        _snapshot_run(PROJECT, run)
