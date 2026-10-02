from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from fdm_smbh_delay.fdm_leaf_mass import FDMLeafMassIdentity
from fdm_smbh_delay.fdm_potential_coupling import FDMPotentialCoupling
from scripts import check_lagramses_fdm_potential_coupling as cli


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "gravity-binding.json"
    source.write_text("bound gravity sources")
    samples = tuple(
        SimpleNamespace(
            raw_provenance_path=tmp_path / f"raw_{index}.txt",
            wave_snapshot_files=(SimpleNamespace(path=tmp_path / "fdm_00001.out00001"),),
            amr_topology_files=(SimpleNamespace(path=tmp_path / "amr_00001.out00001"),),
        )
        for index in range(3)
    )
    binding = {
        "sample_ledger": {"path": str(tmp_path / "sample-ledger.json")},
        "particle_density_included": False,
        "samples": [
            {
                "raw_fdm_provenance": {"path": str(sample.raw_provenance_path)},
                "gravity_snapshot_files": [{"path": str(tmp_path / "grav_00001.out00001")}],
            }
            for sample in samples
        ],
    }
    monkeypatch.setattr(cli, "read_verified_fdm_gravity_source_binding", lambda path: binding)
    monkeypatch.setattr(
        cli, "read_verified_dual_soliton_relaxation_sample_ledger",
        lambda path: SimpleNamespace(samples=samples),
    )
    monkeypatch.setattr(
        cli, "read_lagramses_fdm_outer_wave_provenance",
        lambda path: SimpleNamespace(
            source_path=path, mpi_ncpu=1,
            fdm_use_hjm=False, fdm_first_wave_level=1,
        ),
    )
    monkeypatch.setattr(cli, "summarize_owned_leaf_potential_coupling", lambda *args, **kwargs: object())

    def measured(_summaries, provenance, **_kwargs):
        identity = FDMLeafMassIdentity(
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
        return FDMPotentialCoupling(
            status="potential_coupling_measured_pending_full_hamiltonian",
            provenance_path=provenance.source_path,
            coarse_cells_per_box=1,
            leaf_mass_identity=identity,
            density_times_potential_code_by_level=(2.0,),
            integrated_density_times_potential_code=2.0,
            interpretation="not Hamiltonian",
        )

    monkeypatch.setattr(cli, "reconstruct_fdm_potential_coupling", measured)
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "LagEunha")
    output = tmp_path / "potential.json"
    monkeypatch.setattr(
        sys, "argv", [
            "check_lagramses_fdm_potential_coupling.py",
            "--gravity-binding", str(source),
            "--coarse-cells-per-box", "1",
            "--simple-boundary", "false",
            "--output", str(output),
        ],
    )
    return source, output


def test_potential_series_keeps_mass_identity_and_source_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = _inputs(tmp_path, monkeypatch)
    assert cli.main() == 0
    record = json.loads(output.read_text())
    assert record["status"] == "potential_coupling_series_pending_full_hamiltonian"
    assert record["gravity_binding"]["sha256"] == cli._sha256(source)
    assert len(record["samples"]) == 3
    with pytest.raises(SystemExit):
        cli.main()


def test_changed_binding_prevents_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = _inputs(tmp_path, monkeypatch)
    original = cli.reconstruct_fdm_potential_coupling

    def mutate(summaries, provenance, **kwargs):
        source.write_text("changed gravity binding")
        return original(summaries, provenance, **kwargs)

    monkeypatch.setattr(cli, "reconstruct_fdm_potential_coupling", mutate)
    with pytest.raises(ValueError, match="changed during"):
        cli.main()
    assert not output.exists()
