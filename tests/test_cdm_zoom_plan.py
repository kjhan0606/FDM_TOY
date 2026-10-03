from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fdm_smbh_delay.cdm_zoom_plan import load_cdm_noncompacting_zoom_plan


def _specification() -> dict[str, object]:
    return yaml.safe_load(Path("configs/cdm_noncompacting_zoom_grid.yaml").read_text())


def _write_specification(tmp_path: Path, record: dict[str, object]) -> Path:
    path = tmp_path / "cdm_zoom.yaml"
    path.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    return path


def test_cdm_noncompacting_zoom_plan_requires_resolution_pairs_phase_and_cadence() -> None:
    plan = load_cdm_noncompacting_zoom_plan("configs/cdm_noncompacting_zoom_grid.yaml")
    assert len(plan.grid.cases) == 4
    assert plan.minimum_complete_outputs == 15
    assert plan.as_dict()["status"] == "cdm_noncompacting_zoom_preflight"


def test_cdm_noncompacting_zoom_plan_rejects_finite_radius_compaction_and_coarse_output(
    tmp_path: Path,
) -> None:
    record = _specification()
    record["noncompacting"]["smbh_merge_radius_cells"] = 1.0
    with pytest.raises(ValueError, match="requires smbh_merge_radius_cells=0"):
        load_cdm_noncompacting_zoom_plan(_write_specification(tmp_path, record))

    record = _specification()
    record["noncompacting"]["maximum_output_cadence_myr"] = 1.0
    with pytest.raises(ValueError, match="output cadence is too coarse"):
        load_cdm_noncompacting_zoom_plan(_write_specification(tmp_path, record))


def test_cdm_noncompacting_zoom_plan_rejects_target_underresolution(tmp_path: Path) -> None:
    record = _specification()
    record["numerics"][0]["finest_cell_size_pc"] = 1.0
    with pytest.raises(ValueError, match="four-cell resolution gate"):
        load_cdm_noncompacting_zoom_plan(_write_specification(tmp_path, record))
