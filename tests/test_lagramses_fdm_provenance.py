from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from fdm_smbh_delay.lagramses_fdm_provenance import (
    read_lagramses_fdm_outer_wave_provenance,
)


def _write_record(path: Path, *, source_schema_version: int = 2, **replacements: str) -> None:
    values = {
        "time_code": "1.0000000000000000D+00",
        "aexp": "5.0000000000000000D-01",
        "nstep_coarse": "42",
        "m_axion_ev": "1.0000000000000000D-21",
        "hbar_code": "2.0000000000000000D-03",
        "fdm_use_hjm": "F",
        "fdm_first_wave_level": "12",
        "analytic_fdm_drag_enabled": ".false.",
        "force_accounting": "resolved_wave_only",
        "leaf_mass_code": "3.0000000000000000D+00",
        "integrated_current_code": "1.0D-02 -2.0D-02 3.0D-02",
        "leaf_cell_count": "100.0",
        "complete_current_stencil_cell_count": "98.0",
        "complete_current_stencil_fraction": "9.8D-01",
        "psi_snapshot_prefix": "fdm_00042.out",
    }
    if source_schema_version == 2:
        values.update(
            {
                "fdm_dual_soliton_ic": "T",
                "fdm_dual_soliton_profile_c": "9.1D-02",
                "fdm_dual_soliton_rho0": "4.0D+00 3.0D+00",
                "fdm_dual_soliton_rc_box": "1.0D-01 1.0D-01",
                "fdm_dual_soliton_center_box_1": "3.5D-01 5.0D-01 5.0D-01",
                "fdm_dual_soliton_center_box_2": "6.5D-01 5.0D-01 5.0D-01",
                "fdm_dual_soliton_velocity_1": "0.0D+00 2.0D-02 0.0D+00",
                "fdm_dual_soliton_velocity_2": "0.0D+00 -2.0D-02 0.0D+00",
                "fdm_dual_soliton_phase": "0.0D+00 1.0D+00",
            }
        )
    elif source_schema_version != 1:
        raise ValueError("test writer only supports raw provenance V1/V2")
    values.update(replacements)
    lines = [f"# fdm_outer_wave_provenance_v{source_schema_version}"]
    lines.extend(f"{key} = {value}" for key, value in values.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_reader_accepts_lagramses_writer_format_and_keeps_it_raw(tmp_path: Path) -> None:
    path = tmp_path / "provenance.txt"
    _write_record(path)
    record = read_lagramses_fdm_outer_wave_provenance(path)
    assert record.integrated_current_code == pytest.approx((0.01, -0.02, 0.03))
    assert record.source_schema_version == 2
    assert record.fdm_dual_soliton_ic
    assert record.fdm_dual_soliton_centres_box is not None
    assert record.fdm_dual_soliton_centres_box[0] == pytest.approx((0.35, 0.5, 0.5))
    assert record.fdm_dual_soliton_centres_box[1] == pytest.approx((0.65, 0.5, 0.5))
    assert record.decision()["status"] == "available_raw_provenance"
    assert "postprocessing remains required" in record.decision()["reason"]


@pytest.mark.parametrize(
    ("replacements", "message"),
    [
        ({"analytic_fdm_drag_enabled": "T"}, "analytic FDM drag"),
        ({"force_accounting": "analytic_drag"}, "not resolved_wave_only"),
        (
            {
                "complete_current_stencil_cell_count": "90.0",
                "complete_current_stencil_fraction": "0.90",
            },
            "coverage lies below",
        ),
    ],
)
def test_raw_record_decision_censors_force_or_stencil_failures(
    tmp_path: Path, replacements: dict[str, str], message: str
) -> None:
    path = tmp_path / "provenance.txt"
    _write_record(path, **replacements)
    decision = read_lagramses_fdm_outer_wave_provenance(path).decision()
    assert decision["status"] == "censored"
    assert message in decision["reason"]


@pytest.mark.parametrize(
    ("replacements", "message"),
    [
        ({"complete_current_stencil_fraction": "0.97"}, "does not match"),
        ({"psi_snapshot_prefix": "output_00042/fdm.out"}, "must be a basename"),
    ],
)
def test_reader_rejects_inconsistent_or_nonportable_provenance(
    tmp_path: Path, replacements: dict[str, str], message: str
) -> None:
    path = tmp_path / "provenance.txt"
    _write_record(path, **replacements)
    with pytest.raises(ValueError, match=message):
        read_lagramses_fdm_outer_wave_provenance(path)


def test_reader_accepts_legacy_v1_but_marks_dual_seed_runtime_state_absent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provenance-v1.txt"
    _write_record(path, source_schema_version=1)
    record = read_lagramses_fdm_outer_wave_provenance(path)
    assert record.source_schema_version == 1
    assert record.fdm_dual_soliton_ic is None
    assert record.as_dict()["fdm_dual_soliton"] is None


def test_reader_rejects_enabled_v2_dual_soliton_with_invalid_core_radius(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provenance-v2.txt"
    _write_record(path, fdm_dual_soliton_rc_box="1.0D-01 5.0D-01")
    with pytest.raises(ValueError, match="invalid component parameters"):
        read_lagramses_fdm_outer_wave_provenance(path)


def test_cli_writes_atomic_raw_provenance_decision(tmp_path: Path) -> None:
    source = tmp_path / "provenance.txt"
    output = tmp_path / "validated" / "provenance.json"
    _write_record(source)
    subprocess.run(
        [
            sys.executable,
            "scripts/validate_lagramses_fdm_outer_wave_provenance.py",
            str(source),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "lagramses_fdm_outer_wave_provenance"
    assert record["decision"]["status"] == "available_raw_provenance"
