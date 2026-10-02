from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from fdm_smbh_delay.fdm_leaf_mass import FDMLeafMassIdentity
from scripts import check_lagramses_fdm_mass_series as cli


def test_atomic_mass_series_publication_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "mass-series.json"
    cli._publish_json_without_overwrite(target, {"status": "first"})
    with pytest.raises(FileExistsError):
        cli._publish_json_without_overwrite(target, {"status": "second"})
    assert json.loads(target.read_text()) == {"status": "first"}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["mass-series.json"]


def _mock_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, changed_after: bool):
    samples = tuple(
        SimpleNamespace(
            time_code=float(index),
            raw_provenance_path=tmp_path / f"raw_{index}.txt",
            wave_snapshot_files=(SimpleNamespace(path=tmp_path / "fdm_00001.out00001"),),
            amr_topology_files=(SimpleNamespace(path=tmp_path / "amr_00001.out00001"),),
        )
        for index in range(3)
    )
    first = SimpleNamespace(
        samples=samples,
        source_path=tmp_path / "ledger.json",
        source_sha256="a" * 64,
        as_dict=lambda: {"samples": 3},
    )
    after = SimpleNamespace(
        samples=samples,
        source_path=first.source_path,
        source_sha256=("b" if changed_after else "a") * 64,
        as_dict=lambda: {"samples": 3},
    )
    calls = iter((first, after))
    monkeypatch.setattr(cli, "read_verified_dual_soliton_relaxation_sample_ledger", lambda path: next(calls))
    monkeypatch.setattr(
        cli,
        "read_lagramses_fdm_outer_wave_provenance",
        lambda path: SimpleNamespace(
            source_path=path,
            mpi_ncpu=1,
            fdm_use_hjm=False,
            fdm_first_wave_level=1,
        ),
    )
    monkeypatch.setattr(cli, "summarize_owned_leaf_amplitudes", lambda *args, **kwargs: object())

    def identity(_summaries, provenance, **_kwargs):
        return FDMLeafMassIdentity(
            status="raw_leaf_mass_reconstruction_matches_writer",
            provenance_path=provenance.source_path,
            coarse_cells_per_box=1,
            owner_ranks=(1,),
            leaf_cells_by_level=(8,),
            mass_code_by_level=(1.0,),
            reconstructed_mass_code=1.0,
            writer_mass_code=1.0,
            mass_relative_difference=0.0,
            reconstructed_leaf_cells=8,
            writer_leaf_cells=8.0,
        )

    monkeypatch.setattr(cli, "check_fdm_leaf_mass_identity", identity)
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "LagEunha")
    target = tmp_path / "mass-series.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_lagramses_fdm_mass_series.py",
            "--sample-ledger", str(first.source_path),
            "--coarse-cells-per-box", "1",
            "--simple-boundary", "false",
            "--output", str(target),
        ],
    )
    return target


def test_mass_series_rechecks_sources_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _mock_inputs(tmp_path, monkeypatch, changed_after=False)
    assert cli.main() == 0
    record = json.loads(target.read_text())
    assert record["status"] == "mass_series_within_limit_pending_other_conservation"
    assert len(record["snapshot_identity_checks"]) == 3
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2


def test_changed_ledger_prevents_mass_series_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _mock_inputs(tmp_path, monkeypatch, changed_after=True)
    with pytest.raises(ValueError, match="changed during"):
        cli.main()
    assert not target.exists()
