from __future__ import annotations

from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from fdm_smbh_delay import fdm_gravity_binding as binding
from test_fdm_gravity_shard import _gravity
from test_fdm_shard_format import _shard


def _sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ledger_path = tmp_path / "sample-ledger.json"
    ledger_path.write_text("verified sample identity")
    samples = []
    for number in range(1, 4):
        directory = tmp_path / f"output_{number:05d}"
        directory.mkdir()
        raw = directory / f"fdm_outer_wave_provenance_{number:05d}.txt"
        raw.write_text("raw provenance")
        wave_paths = []
        amr_paths = []
        gravity_paths = []
        for rank in (1, 2):
            wave = directory / f"fdm_{number:05d}.out{rank:05d}"
            amr = directory / f"amr_{number:05d}.out{rank:05d}"
            gravity = directory / f"grav_{number:05d}.out{rank:05d}"
            wave.write_bytes(_shard())
            amr.write_bytes(b"AMR distinct source")
            gravity.write_bytes(_gravity())
            wave_paths.append(SimpleNamespace(path=wave))
            amr_paths.append(SimpleNamespace(path=amr))
            gravity_paths.append(gravity)
        (directory / "POISSON_PHI_VALID").write_text(
            f"LAGRAMSES_POISSON_PHI_VALID_V1\n {number} 2 {float(number)} 1.0\n"
        )
        samples.append(SimpleNamespace(
            raw_provenance_path=raw,
            raw_provenance_sha256="a" * 64,
            wave_snapshot_files=tuple(wave_paths),
            amr_topology_files=tuple(amr_paths),
            nstep_coarse=number,
            time_code=float(number),
            aexp=1.0,
        ))
    ledger = SimpleNamespace(
        source_path=ledger_path,
        source_sha256=binding._sha256(ledger_path),
        samples=tuple(samples),
        as_dict=lambda: {"sample_count": 3},
    )
    monkeypatch.setattr(
        binding, "read_verified_dual_soliton_relaxation_sample_ledger",
        lambda path: ledger,
    )
    monkeypatch.setattr(
        binding, "read_lagramses_fdm_outer_wave_provenance",
        lambda path: SimpleNamespace(
            mpi_ncpu=2,
            psi_snapshot_prefix=f"fdm_{Path(path).parent.name.removeprefix('output_')}.out",
            source_path=Path(path),
        ),
    )
    return ledger, gravity_paths


def test_complete_gravity_and_phi_marker_binding_is_recheckable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sources(tmp_path, monkeypatch)
    target = tmp_path / "gravity-binding.json"
    record = binding.materialize_fdm_gravity_source_binding(
        tmp_path / "sample-ledger.json", target
    )
    assert record["status"] == "fdm_gravity_sources_bound_pending_hamiltonian"
    assert len(record["samples"]) == 3
    assert len(record["samples"][0]["gravity_snapshot_files"]) == 2
    assert binding.read_verified_fdm_gravity_source_binding(target) == record
    with pytest.raises(ValueError, match="already exists"):
        binding.materialize_fdm_gravity_source_binding(
            tmp_path / "sample-ledger.json", target
        )


def test_missing_marker_rank_or_changed_gravity_source_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, gravity_paths = _sources(tmp_path, monkeypatch)
    target = tmp_path / "gravity-binding.json"
    binding.materialize_fdm_gravity_source_binding(ledger.source_path, target)
    gravity_paths[0].write_bytes(
        _gravity().replace(struct.pack("<d", 1.0), struct.pack("<d", 2.0), 1)
    )
    with pytest.raises(ValueError, match="no longer matches"):
        binding.read_verified_fdm_gravity_source_binding(target)
    gravity_paths[0].write_bytes(_gravity() + b"x")
    with pytest.raises(ValueError, match="trailing bytes"):
        binding.read_verified_fdm_gravity_source_binding(target)
    gravity_paths[0].write_bytes(_gravity())
    marker = ledger.samples[0].raw_provenance_path.parent / "POISSON_PHI_VALID"
    marker.write_text("LAGRAMSES_POISSON_PHI_VALID_V1\n 99 2 1.0 1.0\n")
    with pytest.raises(ValueError, match="differs"):
        binding.read_verified_fdm_gravity_source_binding(target)
    marker.write_text("LAGRAMSES_POISSON_PHI_VALID_V1\n 1 2 1.0 1.0\n")
    gravity_paths[0].unlink()
    with pytest.raises(ValueError, match="censored_missing"):
        binding.read_verified_fdm_gravity_source_binding(target)


def test_grouped_gravity_shards_keep_rank_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, gravity_paths = _sources(tmp_path, monkeypatch)
    group = ledger.samples[-1].raw_provenance_path.parent / "group_00001"
    group.mkdir()
    for source in gravity_paths:
        source.rename(group / source.name)
    target = tmp_path / "grouped-gravity-binding.json"
    record = binding.materialize_fdm_gravity_source_binding(ledger.source_path, target)
    assert all(
        "group_00001" in item["path"]
        for item in record["samples"][-1]["gravity_snapshot_files"]
    )
    assert binding.read_verified_fdm_gravity_source_binding(target) == record
