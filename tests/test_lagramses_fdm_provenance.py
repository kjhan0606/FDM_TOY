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
    if source_schema_version in {2, 3, 4, 5}:
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
        raise ValueError("test writer only supports raw provenance V1/V2/V3/V4/V5")
    if source_schema_version >= 3:
        values.update({"mpi_ncpu": "2", "restart_parent_output": "0"})
    if source_schema_version >= 4:
        values.update({"execution_instance_id": "instance-a"})
    if source_schema_version == 5:
        values.update({"restart_parent_execution_instance_id": "none"})
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


def test_reader_accepts_v3_expected_mpi_shard_count_without_claiming_lineage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provenance-v3.txt"
    _write_record(path, source_schema_version=3, restart_parent_output="41")
    record = read_lagramses_fdm_outer_wave_provenance(path)
    assert record.source_schema_version == 3
    assert record.mpi_ncpu == 2
    assert record.restart_parent_output == 41
    assert record.as_dict()["mpi_ncpu"] == 2
    legacy_path = tmp_path / "provenance-v2.txt"
    _write_record(legacy_path)
    assert "mpi_ncpu" not in read_lagramses_fdm_outer_wave_provenance(legacy_path).as_dict()


@pytest.mark.parametrize(
    ("replacements", "message"),
    [
        ({"mpi_ncpu": "0"}, "mpi_ncpu must lie in"),
        ({"mpi_ncpu": "100000"}, "mpi_ncpu must lie in"),
        ({"restart_parent_output": "-1"}, "restart_parent_output must be non-negative"),
    ],
)
def test_reader_rejects_invalid_v3_output_set_fields(
    tmp_path: Path, replacements: dict[str, str], message: str
) -> None:
    path = tmp_path / "provenance-v3.txt"
    _write_record(path, source_schema_version=3, **replacements)
    with pytest.raises(ValueError, match=message):
        read_lagramses_fdm_outer_wave_provenance(path)


def test_reader_accepts_v4_execution_instance_without_upgrading_v3_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provenance-v4.txt"
    _write_record(path, source_schema_version=4, execution_instance_id="run-42.alpha")
    record = read_lagramses_fdm_outer_wave_provenance(path)
    assert record.source_schema_version == 4
    assert record.execution_instance_id == "run-42.alpha"
    assert record.as_dict()["execution_instance_id"] == "run-42.alpha"


@pytest.mark.parametrize("identifier", ("", "contains space", "slash/not-allowed"))
def test_reader_rejects_invalid_v4_execution_instance_identifier(
    tmp_path: Path, identifier: str
) -> None:
    path = tmp_path / "provenance-v4.txt"
    _write_record(path, source_schema_version=4, execution_instance_id=identifier)
    with pytest.raises(ValueError, match="execution_instance_id is invalid"):
        read_lagramses_fdm_outer_wave_provenance(path)


def test_reader_requires_an_exact_v5_parent_execution_token_for_restarts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provenance-v5.txt"
    _write_record(
        path,
        source_schema_version=5,
        restart_parent_output="41",
        restart_parent_execution_instance_id="instance-parent",
    )
    record = read_lagramses_fdm_outer_wave_provenance(path)
    assert record.restart_parent_execution_instance_id == "instance-parent"
    assert record.as_dict()["restart_parent_execution_instance_id"] == "instance-parent"


@pytest.mark.parametrize(
    ("replacements", "message"),
    [
        (
            {"restart_parent_output": "0", "restart_parent_execution_instance_id": "instance-a"},
            "initial V5 provenance",
        ),
        (
            {"restart_parent_output": "1", "restart_parent_execution_instance_id": "none"},
            "restart_parent_execution_instance_id is invalid",
        ),
    ],
)
def test_reader_rejects_unbound_v5_parent_execution_identity(
    tmp_path: Path, replacements: dict[str, str], message: str
) -> None:
    path = tmp_path / "provenance-v5.txt"
    _write_record(path, source_schema_version=5, **replacements)
    with pytest.raises(ValueError, match=message):
        read_lagramses_fdm_outer_wave_provenance(path)


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
