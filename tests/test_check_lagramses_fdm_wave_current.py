from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from fdm_smbh_delay.fdm_wave_stencil import FDMSameLevelStencil, FDMShardStencilResult
from scripts import check_lagramses_fdm_wave_current as cli


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ledger_path = tmp_path / "sample-ledger.json"
    ledger_path.write_text("verified synthetic ledger")
    wave = tmp_path / "fdm_00001.out00001"
    amr = tmp_path / "amr_00001.out00001"
    provenance_path = tmp_path / "fdm_outer_wave_provenance_00001.txt"
    sample = SimpleNamespace(
        raw_provenance_path=provenance_path,
        raw_provenance_sha256="a" * 64,
        wave_snapshot_files=(SimpleNamespace(path=wave),),
        amr_topology_files=(SimpleNamespace(path=amr),),
    )
    ledger = SimpleNamespace(
        source_path=ledger_path, source_sha256="b" * 64,
        samples=(sample,),
    )
    provenance = SimpleNamespace(
        source_path=provenance_path, mpi_ncpu=1,
        fdm_use_hjm=False, fdm_first_wave_level=1,
        hbar_code=1.0, psi_snapshot_prefix="fdm_00001.out",
        leaf_cell_count=8.0, complete_current_stencil_cell_count=8.0,
        leaf_mass_code=5.0, integrated_current_code=(0.0, 0.0, 0.0),
    )
    measurement = FDMSameLevelStencil(
        status="same_level_current_complete_pending_writer_identity",
        level=1, owner_leaf_cells=8, complete_leaf_stencil_cells=8,
        incomplete_leaf_stencil_cells=0, refined_neighbour_stencil_cells=0,
        leaf_mass_code=5.0, integrated_current_code=(0.0, 0.0, 0.0),
        central_gradient_square_proxy_code=0.0,
        interpretation="synthetic constant wave",
    )
    result = FDMShardStencilResult(
        wave_path=wave, amr_path=amr, owner_rank=1, ncpu=1,
        nlevelmax=1, boxlen_code=1.0, measurement=measurement,
    )
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "lageunha")
    monkeypatch.setattr(
        cli, "read_verified_dual_soliton_relaxation_sample_ledger",
        lambda path: ledger,
    )
    monkeypatch.setattr(
        cli, "read_lagramses_fdm_outer_wave_provenance", lambda path: provenance
    )
    monkeypatch.setattr(cli, "inspect_fdm_shard", lambda *args, **kwargs: SimpleNamespace(nlevelmax=1))
    monkeypatch.setattr(cli, "measure_fdm_shard_same_level_stencil", lambda *args, **kwargs: result)
    output = tmp_path / "current.json"
    monkeypatch.setattr(sys, "argv", [
        "check_lagramses_fdm_wave_current.py",
        "--sample-ledger", str(ledger_path),
        "--sample-index", "0", "--simple-boundary", "false",
        "--output", str(output),
    ])
    return ledger, output


def test_wave_current_cli_publishes_source_bound_identity_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, output = _inputs(tmp_path, monkeypatch)
    assert cli.main() == 0
    record = json.loads(output.read_text())
    assert record["status"] == "saved_wave_writer_current_identity_matches"
    assert record["sample_ledger"]["sha256"] == ledger.source_sha256
    assert record["identity"]["reconstructed_mass_code"] == 5.0
    with pytest.raises(SystemExit):
        cli.main()


def test_wave_current_cli_rejects_changed_ledger_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, output = _inputs(tmp_path, monkeypatch)
    calls = 0

    def changed(_path: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            return SimpleNamespace(**{**vars(ledger), "source_sha256": "c" * 64})
        return ledger

    monkeypatch.setattr(cli, "read_verified_dual_soliton_relaxation_sample_ledger", changed)
    with pytest.raises(ValueError, match="changed during"):
        cli.main()
    assert not output.exists()
