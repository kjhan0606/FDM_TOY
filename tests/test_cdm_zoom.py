from __future__ import annotations

from pathlib import Path

from fdm_smbh_delay.cdm_zoom import assess_noncompacting_cdm_zoom_run
from fdm_smbh_delay.dm_run_provenance import read_dark_matter_run_provenance


def _write_provenance(tmp_path: Path, *, radius: str | None, mode: str | None) -> Path:
    values = {
        "dark_matter_model": "cdm",
        "pic_enabled": ".true.",
        "sidm_enabled": ".false.",
        "fdm_enabled": ".false.",
        "nstep_coarse": "42",
        "time_code": "1.0d0",
        "aexp": "5.0d-1",
        "build_git_hash": "a" * 40,
        "namelist_copy": "namelist.txt",
        "compilation_copy": "compilation.txt",
        "smbh_capture_ledger_enabled": ".true.",
        "smbh_capture_ledger_file": "smbh_capture_ledger_v1.jsonl",
        "dm_transport": "collisionless_nbody",
        "force_accounting": "resolved_collisionless_only",
    }
    if radius is not None:
        values["smbh_merge_radius_cells"] = radius
    if mode is not None:
        values["smbh_compaction_mode"] = mode
    path = tmp_path / "dm_run_provenance_00042.txt"
    path.write_text(
        "# dm_run_provenance_v1\n"
        + "".join(f"{key} = {value}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    return path


def test_noncompacting_cdm_zoom_requires_native_zero_merge_radius(tmp_path: Path) -> None:
    ready = assess_noncompacting_cdm_zoom_run(
        read_dark_matter_run_provenance(
            _write_provenance(tmp_path, radius="0.0d0", mode="no_finite_radius_rmerge_zero")
        )
    )
    assert ready.ready
    assert ready.as_dict()["status"] == "cdm_noncompacting_zoom_provenance_ready"

    compacting = assess_noncompacting_cdm_zoom_run(
        read_dark_matter_run_provenance(
            _write_provenance(tmp_path, radius="1.0d0", mode="enabled")
        )
    )
    assert not compacting.ready
    assert "smbh_merge_radius_cells=0" in compacting.reasons[0]


def test_incomplete_compaction_provenance_is_rejected(tmp_path: Path) -> None:
    try:
        read_dark_matter_run_provenance(_write_provenance(tmp_path, radius="0.0d0", mode=None))
    except ValueError as error:
        assert "must appear together" in str(error)
    else:
        raise AssertionError("partial compaction provenance must be rejected")
