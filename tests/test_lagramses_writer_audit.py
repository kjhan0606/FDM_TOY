from __future__ import annotations

import json
from pathlib import Path

from fdm_smbh_delay.lagramses_writer_audit import (
    audit_lagramses_writer_force_accounting,
    write_writer_force_accounting_audit,
)


def _source(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "output_amr.kjhan.f90"
    path.write_text(text, encoding="utf-8")
    return path


def test_writer_audit_reports_current_missing_tokens_without_runtime_claim(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        """
        ! force_accounting = resolved_collisionless_only
        write(unit,*) 'dm_transport = collisionless_nbody'
        write(unit,*) 'sidm_max_scatter_probability = ', pmax_value
        """,
    )
    audit = audit_lagramses_writer_force_accounting(source)
    assert audit.status == "source_token_missing"
    assert not audit.tokens_present
    assert audit.source_sha256 is not None
    assert any("CDM".lower() in reason.lower() for reason in audit.reasons)


def test_writer_audit_requires_both_model_tokens_and_source_evidence(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        """
        write(unit,*) 'force_accounting = resolved_collisionless_only'
        write(unit,*) 'dm_transport = collisionless_nbody'
        write(unit,*) 'force_accounting = resolved_collisionless_plus_scattering'
        write(unit,*) 'sidm_max_scatter_probability = ', pmax_value
        ! sidm_Pmax is intentionally absent as executable source
        """,
    )
    audit = audit_lagramses_writer_force_accounting(source)
    assert audit.status == "source_token_missing"
    assert audit.models["cdm"]["token_present"]
    assert audit.models["sidm"]["token_present"]
    assert not audit.models["sidm"]["evidence"]["sidm_Pmax"]


def test_writer_audit_accepts_tokens_only_as_a_runtime_test_prerequisite(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        """
        if (dark_matter_model == 'cdm') then
          dm_transport = collisionless_nbody
          collisionless_force = true
          write(unit,*) 'force_accounting = resolved_collisionless_only'
        endif
        if (dark_matter_model == 'sidm') then
          sidm_Pmax = sidm_scatter_probability_max
          sidm_max_scatter_probability = sidm_Pmax
          write(unit,*) 'force_accounting = resolved_collisionless_plus_scattering'
        endif
        """,
    )
    audit = audit_lagramses_writer_force_accounting(source)
    assert audit.status == "source_tokens_present_requires_runtime_test"
    output = tmp_path / "audit.json"
    write_writer_force_accounting_audit(output, audit)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == audit.status
    assert "runtime-derived values" in saved["interpretation"]


def test_writer_audit_missing_source_is_fail_closed(tmp_path: Path) -> None:
    audit = audit_lagramses_writer_force_accounting(tmp_path / "missing.f90")
    assert audit.status == "source_unavailable"
    assert audit.source_sha256 is None
