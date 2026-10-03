from __future__ import annotations

from pathlib import Path

import pytest

from scripts import audit_qe_adjacent_pilot as audit


def test_adjacent_pilot_audit_caches_runs_and_rechecks_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,run_id,effective_grid_cells\n"
        "case_a,a128,128\ncase_a,a256,256\ncase_a,a512,512\n"
    )
    checks: list[str] = []
    loads: list[str] = []

    def require(run: Path, *, case_id: str, resolution: int) -> dict[str, str]:
        checks.append(run.name)
        return {"source": f"{case_id}-{resolution}"}

    def load(label: str, run: Path) -> dict:
        loads.append(label)
        return {"label": label}

    monkeypatch.setattr(audit, "_require_complete", require)
    monkeypatch.setattr(audit, "load_convergence_run", load)
    monkeypatch.setattr(
        audit, "pair_occupancy",
        lambda fine, coarse, **kwargs: {
            "status": "common_resolved_separation_binned",
            "resolved_complete_orbits": [12, 12],
            "orbit_count_eligible_bins": 0,
        },
    )
    result = audit.audit_adjacent_pilot(manifest, tmp_path / "torch")
    assert result["pair_count"] == 2
    assert result["production_calibration_row_admitted"] is False
    assert len(loads) == 3
    assert checks.count("a256") == 3
    assert all(not pair["production_calibration_row_admitted"] for pair in result["pairs"])


def test_adjacent_pilot_audit_rejects_changed_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,run_id,effective_grid_cells\n"
        "case_a,a256,256\ncase_a,a512,512\n"
    )
    seen: dict[str, int] = {}

    def require(run: Path, *, case_id: str, resolution: int) -> dict[str, str]:
        seen[run.name] = seen.get(run.name, 0) + 1
        return {"source": f"{run.name}-{seen[run.name]}"}

    monkeypatch.setattr(audit, "_require_complete", require)
    monkeypatch.setattr(audit, "load_convergence_run", lambda label, run: {})
    monkeypatch.setattr(audit, "pair_occupancy", lambda fine, coarse, **kwargs: {})
    with pytest.raises(ValueError, match="input changed"):
        audit.audit_adjacent_pilot(manifest, tmp_path / "torch")


def test_adjacent_pilot_report_refuses_to_replace_existing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.json"
    audit._write_new_report(path, {"status": "first"})
    with pytest.raises(FileExistsError):
        audit._write_new_report(path, {"status": "second"})
    assert path.read_text().count("first") == 1
    assert "second" not in path.read_text()
