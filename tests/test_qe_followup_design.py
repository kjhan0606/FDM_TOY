from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from fdm_smbh_delay.qe_followup_design import (
    build_qe_followup_design,
    read_verified_qe_followup_design,
    verify_qe_design_comparison_runs,
    verify_qe_design_run_request,
)
from scripts import summarize_pyul_convergence as comparison_cli
from scripts.plan_wave_calibration_runs import build_plan


def test_registered_q100e000_followup_is_bound_but_has_no_results(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cases = root / "results/wave_calibration_qe_extension/physical_cases.csv"
    manifest = root / "results/wave_calibration_qe_followup_q100e000/run_manifest.csv"
    path = root / "results/wave_calibration_qe_followup_q100e000/design.json"
    design, digest = read_verified_qe_followup_design(
        path, physical_cases=cases, run_manifest=manifest,
    )
    assert len(digest) == 64
    assert design["production_calibration_row_admitted"] is False
    assert design["separation_bin_edges_pc"] == [0.43, 0.438]
    plan = build_plan(
        manifest, cases, tmp_path / "initial", tmp_path / "torch",
        tmp_path / "pyul", qe_design_path=path,
    )
    assert [row.qe_design_role for row in plan] == [
        "coarse", "fine", "doubled_box_control",
    ]
    assert all(row.case_duration_myr == 0.1 for row in plan)
    assert all(row.status_detail == "seed_missing" for row in plan)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    cases = tmp_path / "cases.csv"
    cases.write_text(
        "case_id,mass_ratio_q,eccentricity,initial_separation_pc,core_radius_pc,"
        "kepler_period_myr\n"
        "qe_test,0.3,0.3,0.572,2.2,0.002\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,run_id,effective_grid_cells,box_size_pc,finest_cell_size_pc\n"
        "qe_test,qe_test_n256,256,26.4,0.103125\n"
        "qe_test,qe_test_n512,512,26.4,0.0515625\n",
        encoding="utf-8",
    )
    return cases, manifest


def _design(cases: Path, manifest: Path, *, memory_gib: float = 80.0) -> dict:
    return build_qe_followup_design(
        case_id="qe_test", physical_cases=cases, run_manifest=manifest,
        coarse_resolution=256, fine_resolution=512,
        separation_bin_edges_pc=(0.43, 0.45), duration_myr=0.02,
        reference_gpu_memory_gib=memory_gib,
    )


def test_design_binds_fixed_bins_run_triplet_and_resource_warning(tmp_path: Path) -> None:
    cases, manifest = _inputs(tmp_path)
    design = _design(cases, manifest)
    assert design["production_calibration_row_admitted"] is False
    assert design["separation_bin_edges_pc"] == [0.43, 0.45]
    assert design["necessary_nominal_orbits"] == 8
    assert design["doubled_box_control"]["resolution"] == 1024
    assert design["doubled_box_control"]["box_size_pc"] == pytest.approx(52.8)
    assert design["doubled_box_control"]["cell_size_pc"] == pytest.approx(0.0515625)
    assert design["doubled_box_control"]["uniform_grid_memory_estimate_gib"] == 128.0
    assert design["doubled_box_control"]["exceeds_reference_gpu_memory"] is True
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design), encoding="utf-8")
    saved, digest = read_verified_qe_followup_design(
        path, physical_cases=cases, run_manifest=manifest,
    )
    assert saved == design
    assert len(digest) == 64


def test_design_rejects_short_duration_and_invalid_edges(tmp_path: Path) -> None:
    cases, manifest = _inputs(tmp_path)
    with pytest.raises(ValueError, match="necessary orbit budget"):
        build_qe_followup_design(
            case_id="qe_test", physical_cases=cases, run_manifest=manifest,
            coarse_resolution=256, fine_resolution=512,
            separation_bin_edges_pc=(0.43, 0.45), duration_myr=0.01,
        )
    with pytest.raises(ValueError, match="fixed physical separation edges"):
        build_qe_followup_design(
            case_id="qe_test", physical_cases=cases, run_manifest=manifest,
            coarse_resolution=256, fine_resolution=512,
            separation_bin_edges_pc=(0.45, 0.43), duration_myr=0.02,
        )


def test_design_rejects_tampering_or_changed_source(tmp_path: Path) -> None:
    cases, manifest = _inputs(tmp_path)
    design = _design(cases, manifest)
    path = tmp_path / "design.json"
    design["separation_bin_edges_pc"][0] = 0.44
    path.write_text(json.dumps(design), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its bound"):
        read_verified_qe_followup_design(
            path, physical_cases=cases, run_manifest=manifest,
        )
    path.write_text(json.dumps(_design(cases, manifest)), encoding="utf-8")
    cases.write_text(cases.read_text().replace("0.3,0.3", "0.4,0.3"), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its bound"):
        read_verified_qe_followup_design(
            path, physical_cases=cases, run_manifest=manifest,
        )


def test_design_cli_refuses_to_overwrite(tmp_path: Path) -> None:
    cases, manifest = _inputs(tmp_path)
    output = tmp_path / "registered.json"
    command = [
        sys.executable, "scripts/register_qe_followup_design.py",
        "--case-id", "qe_test", "--cases", str(cases),
        "--manifest", str(manifest), "--coarse-resolution", "256",
        "--fine-resolution", "512", "--separation-bin-edges-pc", "0.43,0.45",
        "--duration-myr", "0.02", "--output", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    first = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert output.read_bytes() == first


def test_design_run_binding_rejects_changed_role_geometry_or_duration(tmp_path: Path) -> None:
    cases, manifest = _inputs(tmp_path)
    design = _design(cases, manifest)
    verify_qe_design_run_request(
        design, case_id="qe_test", role="fine", resolution=512,
        box_size_pc=26.4, duration_myr=0.02,
    )
    verify_qe_design_run_request(
        design, case_id="qe_test", role="fine", resolution=512,
        box_size_pc=26.4, duration_myr=1e-6, initial_state_only=True,
    )
    with pytest.raises(ValueError, match="run request disagrees"):
        verify_qe_design_run_request(
            design, case_id="qe_test", role="fine", resolution=512,
            box_size_pc=26.4, duration_myr=1e-6,
        )
    with pytest.raises(ValueError, match="run request disagrees"):
        verify_qe_design_run_request(
            design, case_id="qe_test", role="fine", resolution=512,
            box_size_pc=26.4, duration_myr=0.0, initial_state_only=True,
        )
    with pytest.raises(ValueError, match="exceeds the declared GPU memory"):
        verify_qe_design_run_request(
            design, case_id="qe_test", role="doubled_box_control", resolution=1024,
            box_size_pc=52.8, duration_myr=0.03,
        )
    reviewed_capacity = _design(cases, manifest, memory_gib=160.0)
    verify_qe_design_run_request(
        reviewed_capacity, case_id="qe_test", role="doubled_box_control",
        resolution=1024, box_size_pc=52.8, duration_myr=0.03,
    )
    for role, resolution, box, duration in (
        ("fine", 768, 26.4, 0.02),
        ("fine", 512, 52.8, 0.02),
        ("fine", 512, 26.4, 0.01),
        ("doubled_box_control", 1024, 26.4, 0.02),
    ):
        with pytest.raises(ValueError, match="disagrees"):
            verify_qe_design_run_request(
                reviewed_capacity, case_id="qe_test", role=role, resolution=resolution,
                box_size_pc=box, duration_myr=duration,
            )


def test_design_comparison_requires_both_bound_roles(tmp_path: Path) -> None:
    cases, manifest = _inputs(tmp_path)
    design = _design(cases, manifest, memory_gib=160.0)
    source = tmp_path / "design.json"
    source.write_text(json.dumps(design), encoding="utf-8")
    _, digest = read_verified_qe_followup_design(
        source, physical_cases=cases, run_manifest=manifest,
    )
    for role, resolution, box in (
        ("coarse", 256, 26.4),
        ("fine", 512, 26.4),
        ("doubled_box_control", 1024, 52.8),
    ):
        run = tmp_path / role
        run.mkdir()
        (run / "fdm_adapter_metadata.json").write_text(json.dumps({
            "case_id": "qe_test", "resolution": resolution,
            "box_size_pc": box, "duration_myr": 0.02,
            "qe_design_binding": {
                "status": "qe_prospective_design_bound_not_a_calibration_release",
                "design_sha256": design["design_sha256"],
                "file_sha256": digest, "role": role,
            },
        }), encoding="utf-8")
    assert verify_qe_design_comparison_runs(
        design, digest, (tmp_path / "fine", tmp_path / "coarse")
    ) == "resolution_pair"
    assert verify_qe_design_comparison_runs(
        design, digest, (tmp_path / "doubled_box_control", tmp_path / "fine")
    ) == "doubled_box"
    with pytest.raises(ValueError, match="registered reference and comparison roles"):
        verify_qe_design_comparison_runs(
            design, digest, (tmp_path / "coarse", tmp_path / "doubled_box_control")
        )
    with pytest.raises(ValueError, match="registered reference and comparison roles"):
        verify_qe_design_comparison_runs(
            design, digest, (tmp_path / "coarse", tmp_path / "fine")
        )
    metadata_path = tmp_path / "fine" / "fdm_adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["qe_design_binding"]["file_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks the registered design binding"):
        verify_qe_design_comparison_runs(
            design, digest, (tmp_path / "fine", tmp_path / "coarse")
        )


def test_registered_comparison_uses_only_bound_edges_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases, manifest = _inputs(tmp_path)
    design = _design(cases, manifest)
    source = tmp_path / "design.json"
    source.write_text(json.dumps(design), encoding="utf-8")
    _, digest = read_verified_qe_followup_design(
        source, physical_cases=cases, run_manifest=manifest,
    )
    for role, resolution in (("coarse", 256), ("fine", 512)):
        run = tmp_path / role
        run.mkdir()
        (run / "fdm_adapter_metadata.json").write_text(json.dumps({
            "case_id": "qe_test", "resolution": resolution,
            "box_size_pc": 26.4, "duration_myr": 0.02,
            "qe_design_binding": {
                "status": "qe_prospective_design_bound_not_a_calibration_release",
                "design_sha256": design["design_sha256"],
                "file_sha256": digest, "role": role,
            },
        }), encoding="utf-8")
    seen = []
    monkeypatch.setattr(comparison_cli, "load_convergence_run", lambda label, path: label)
    def summarize(loaded, *, separation_bins, minimum_orbits_per_separation_bin,
                  separation_bin_edges_pc):
        seen.append((separation_bins, minimum_orbits_per_separation_bin,
                     separation_bin_edges_pc))
        return {"status": "synthetic_comparison"}
    monkeypatch.setattr(comparison_cli, "summarize_convergence", summarize)
    output = tmp_path / "comparison.json"
    arguments = [
        "summarize_pyul_convergence.py",
        f"n512={tmp_path / 'fine'}", f"n256={tmp_path / 'coarse'}",
        "--qe-design", str(source), "--qe-design-cases", str(cases),
        "--qe-design-manifest", str(manifest), "--output", str(output),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    assert comparison_cli.main() == 0
    assert seen == [(1, 8, (0.43, 0.45))]
    saved = json.loads(output.read_text())
    assert saved["qe_design_binding"]["comparison_kind"] == "resolution_pair"
    with pytest.raises(FileExistsError):
        comparison_cli.main()
    monkeypatch.setattr(sys, "argv", arguments + [
        "--separation-bin-edges-pc", "0.1,0.2"
    ])
    with pytest.raises(SystemExit):
        comparison_cli.main()
