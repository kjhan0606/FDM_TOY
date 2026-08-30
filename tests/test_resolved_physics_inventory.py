from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from fdm_smbh_delay.resolved_physics_inventory import (
    assess_lagramses_resolved_physics_inventory,
    read_lagramses_resolved_physics_inventory,
)


def _write_inventory(
    root: Path,
    *,
    model: str = "fdm",
    updates: dict[str, str] | None = None,
    grouped: bool = False,
) -> Path:
    label = "00042"
    output_directory = root / f"output_{label}"
    directory = output_directory / "group_00001" if grouped else output_directory
    directory.mkdir(parents=True)
    (output_directory / "COMPLETE").write_text(label + "\n", encoding="utf-8")
    values = {
        "output_number": label,
        "nstep_coarse": "42",
        "time_code": "1.25d0",
        "aexp": "5.0d-1",
        "dark_matter_model": model,
        "raw_snapshot_directory": (
            f"output_{label}/group_00001/" if grouped else f"output_{label}/"
        ),
        "completion_marker": "COMPLETE",
        "star_formation_enabled": ".false.",
        "stars_channel_status": "absent",
        "stars_particle_snapshot_prefix": "none",
        "gas_channel_status": "absent",
        "gas_snapshot_prefix": "none",
        "dark_matter_channel_status": "available" if model != "none" else "absent",
        "particle_snapshot_prefix": "part_00042.out" if model in {"cdm", "sidm"} else "none",
        "potential_snapshot_prefix": "grav_00042.out",
        "potential_checkpoint_status": "validated",
        "sink_info_file": "sink_00042.info",
        "force_source_ledger_status": "unavailable",
        "force_source_ledger_reason": "no_source_decomposition_in_normal_output",
        "conservation_ledger_status": "unavailable",
        "conservation_ledger_reason": "no_time_series_in_normal_output",
    }
    if model == "sidm":
        values.update(
            {
                "sidm_scattering_ledger_status": "unavailable",
                "sidm_scattering_ledger_reason": "no_cumulative_scatter_counter_in_normal_output",
            }
        )
    if model == "fdm":
        values.update(
            {
                "fdm_field_snapshot_status": "available",
                "fdm_field_snapshot_prefix": "fdm_00042.out",
                "fdm_wave_provenance_status": "unavailable",
                "fdm_wave_provenance_path": "none",
                "fdm_force_accounting": "resolved_wave_only",
            }
        )
    if updates:
        values.update(updates)
    for key in (
        "stars_particle_snapshot_prefix",
        "gas_snapshot_prefix",
        "particle_snapshot_prefix",
        "potential_snapshot_prefix",
        "sink_info_file",
        "fdm_field_snapshot_prefix",
    ):
        value = values.get(key)
        if value and value != "none":
            (directory / value).touch()
    if values["potential_checkpoint_status"] == "validated":
        (output_directory / "POISSON_PHI_VALID").write_text(
            "LAGRAMSES_POISSON_PHI_VALID_V1\n42 21 1.25 0.5\n",
            encoding="utf-8",
        )
    if values.get("fdm_wave_provenance_status") == "available":
        (output_directory / f"fdm_outer_wave_provenance_{label}.txt").write_text(
            "# fdm_outer_wave_provenance_v2\n"
            "time_code = 1.25d0\n"
            "aexp = 5.0d-1\n"
            "nstep_coarse = 42\n"
            "m_axion_ev = 1.0d-21\n"
            "hbar_code = 2.0d-3\n"
            "fdm_use_hjm = F\n"
            "fdm_first_wave_level = 12\n"
            "analytic_fdm_drag_enabled = .false.\n"
            "force_accounting = resolved_wave_only\n"
            "leaf_mass_code = 3.0d0\n"
            "integrated_current_code = 1.0d-2 -2.0d-2 3.0d-2\n"
            "leaf_cell_count = 100.0\n"
            "complete_current_stencil_cell_count = 98.0\n"
            "complete_current_stencil_fraction = 0.98\n"
            "psi_snapshot_prefix = fdm_00042.out\n"
            "fdm_dual_soliton_ic = F\n"
            "fdm_dual_soliton_profile_c = 0.0\n"
            "fdm_dual_soliton_rho0 = 0.0 0.0\n"
            "fdm_dual_soliton_rc_box = 0.0 0.0\n"
            "fdm_dual_soliton_center_box_1 = 0.0 0.0 0.0\n"
            "fdm_dual_soliton_center_box_2 = 0.0 0.0 0.0\n"
            "fdm_dual_soliton_velocity_1 = 0.0 0.0 0.0\n"
            "fdm_dual_soliton_velocity_2 = 0.0 0.0 0.0\n"
            "fdm_dual_soliton_phase = 0.0 0.0\n",
            encoding="utf-8",
        )
    inventory = directory / f"resolved_physics_inventory_{label}.txt"
    inventory.write_text(
        "# lagramses_resolved_physics_inventory_v1\n"
        "# Raw file availability only; this record is not a force or delay measurement.\n"
        + "".join(f"{key} = {value}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    return inventory


def test_current_normal_fdm_inventory_is_censored_not_zero_force(tmp_path: Path) -> None:
    inventory = read_lagramses_resolved_physics_inventory(_write_inventory(tmp_path))
    decision = assess_lagramses_resolved_physics_inventory(
        inventory,
        stars_required=False,
        gas_required=False,
    )
    assert inventory.dark_matter_model == "fdm"
    assert inventory.fdm_field_snapshot_status == "available"
    assert decision.status == "censored"
    assert "force-source ledger is unavailable: no_source_decomposition_in_normal_output" in decision.reasons
    assert "conservation ledger is unavailable: no_time_series_in_normal_output" in decision.reasons
    assert "FDM wave provenance is unavailable" in decision.reasons
    assert decision.as_dict()["inventory"]["fdm"]["force_accounting"] == "resolved_wave_only"


def test_v1_rejects_unattested_available_force_or_conservation_ledgers(tmp_path: Path) -> None:
    source = _write_inventory(
        tmp_path,
        updates={
            "force_source_ledger_status": "available",
            "force_source_ledger_reason": "resolved_source_work_available",
        },
    )
    with pytest.raises(ValueError, match="force_source_ledger_status is unsupported"):
        read_lagramses_resolved_physics_inventory(source)


def test_rejects_missing_complete_marker_and_cross_output_wave_path(tmp_path: Path) -> None:
    source = _write_inventory(tmp_path)
    (source.parent / "COMPLETE").write_text("00041\n", encoding="utf-8")
    with pytest.raises(ValueError, match="COMPLETE marker differs"):
        read_lagramses_resolved_physics_inventory(source)

    other = _write_inventory(
        tmp_path / "other",
        updates={
            "fdm_wave_provenance_status": "available",
            "fdm_wave_provenance_path": "output_00041/fdm_outer_wave_provenance_00041.txt",
        },
    )
    with pytest.raises(ValueError, match="FDM wave provenance path differs"):
        read_lagramses_resolved_physics_inventory(other)


def test_accepts_grouped_normal_output_and_rejects_missing_snapshot_or_wave_file(
    tmp_path: Path,
) -> None:
    grouped = _write_inventory(tmp_path, grouped=True)
    inventory = read_lagramses_resolved_physics_inventory(grouped)
    assert inventory.raw_snapshot_directory == "output_00042/group_00001/"
    assert assess_lagramses_resolved_physics_inventory(
        inventory, stars_required=False, gas_required=False
    ).status == "censored"

    missing_snapshot = _write_inventory(tmp_path / "missing-snapshot")
    (missing_snapshot.parent / "fdm_00042.out").unlink()
    with pytest.raises(ValueError, match="FDM field snapshot prefix does not name a file"):
        read_lagramses_resolved_physics_inventory(missing_snapshot)

    missing_wave = _write_inventory(
        tmp_path / "missing-wave",
        updates={
            "fdm_wave_provenance_status": "available",
            "fdm_wave_provenance_path": "output_00042/fdm_outer_wave_provenance_00042.txt",
        },
    )
    (missing_wave.parent / "fdm_outer_wave_provenance_00042.txt").unlink()
    with pytest.raises(ValueError, match="cannot validate FDM wave provenance"):
        read_lagramses_resolved_physics_inventory(missing_wave)


def test_rejects_particle_star_ambiguity_even_when_stars_are_not_required(tmp_path: Path) -> None:
    source = _write_inventory(
        tmp_path,
        model="cdm",
        updates={
            "stars_channel_status": "requires_particle_classification",
            "stars_particle_snapshot_prefix": "part_00042.out",
        },
    )
    decision = assess_lagramses_resolved_physics_inventory(
        read_lagramses_resolved_physics_inventory(source),
        stars_required=False,
        gas_required=False,
    )
    assert decision.status == "censored"
    assert decision.reasons[0] == "stars channel requires particle classification"
    assert any("force-source ledger is unavailable" in reason for reason in decision.reasons)


def test_cli_writes_atomic_censored_record_without_submitting_work(tmp_path: Path) -> None:
    source = _write_inventory(tmp_path)
    output = tmp_path / "assessment.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/assess_lagramses_resolved_physics_inventory.py",
            str(source),
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "censored"
