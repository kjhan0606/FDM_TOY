from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fdm_smbh_delay.pure_fdm_outer_submission import (
    assess_pure_fdm_outer_submission,
    build_fdm_writer_runtime_attestation,
    read_verified_pure_fdm_outer_submission,
)
from fdm_smbh_delay.pure_fdm_zoom import preflight_pure_fdm_outer_zoom
from fdm_smbh_delay.zoom_calibration import load_zoom_grid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outer_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    specification = Path("configs/pure_fdm_outer_zoom_grid.yaml").resolve()
    grid = load_zoom_grid(specification)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(grid.as_dict(), indent=2, sort_keys=True) + "\n")
    preflight = tmp_path / "preflight.json"
    preflight_record = preflight_pure_fdm_outer_zoom(grid).as_dict()
    preflight.write_text(json.dumps(preflight_record, indent=2, sort_keys=True) + "\n")
    source = tmp_path / "output_amr.kjhan.f90"
    source.write_text(
        "if (dark_matter_model == 'fdm') then\n"
        "  fdm_outer_ledger_enabled = .true.\n"
        "  fdm_first_wave_level = 17\n"
        "  fdm_use_hjm = .false.\n"
        "  write(unit,*) 'fdm_force_accounting = resolved_wave_only'\n"
        "endif\n",
        encoding="utf-8",
    )
    return specification, manifest, preflight, source


def _fdm_sidecar(path: Path) -> None:
    path.write_text(
        "# dm_run_provenance_v1\n"
        "dark_matter_model = fdm\n"
        "pic_enabled = .false.\n"
        "sidm_enabled = .false.\n"
        "fdm_enabled = .true.\n"
        "nstep_coarse = 12\n"
        "time_code = 1.0d0\n"
        "aexp = 5.0d-1\n"
        "build_git_hash = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "namelist_copy = namelist.txt\n"
        "compilation_copy = compilation.txt\n"
        "smbh_capture_ledger_enabled = .true.\n"
        "smbh_capture_ledger_file = smbh_capture_ledger_v1.jsonl\n"
        "m_axion_ev = 1.0d-22\n"
        "fdm_use_hjm = .false.\n"
        "fdm_first_wave_level = 17\n"
        "fdm_outer_ledger_enabled = .true.\n"
        "fdm_force_accounting = resolved_wave_only\n",
        encoding="utf-8",
    )


def _runtime_supporting_files(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "COMPLETE").write_text("", encoding="utf-8")
    (output_dir / "namelist.txt").write_text(
        "&RUN_PARAMS\n"
        "use_fdm=.true.\n"
        "/\n"
        "&FDM_PARAMS\n"
        "fdm_use_hjm=.false.\n"
        "fdm_outer_ledger=.true.\n"
        "fdm_first_wave_level=17\n"
        "/\n",
        encoding="utf-8",
    )
    (output_dir / "namelist.txt").write_text(
        (output_dir / "namelist.txt").read_text(encoding="utf-8")
        + "&AMR_PARAMS\nlevelmax=17\n/\n",
        encoding="utf-8",
    )
    (output_dir / "compilation.txt").write_text(
        "last commit = " + "a" * 40 + "\n", encoding="utf-8"
    )
    (output_dir.parent / "run.log").write_text(
        "Working with nproc =    2 for ndim = 3\n"
        "Working with nproc =    2 for ndim = 3\n"
        "Run completed\n"
        "Run completed\n",
        encoding="utf-8",
    )


def _raw_fdm_outer_provenance(output_dir: Path) -> None:
    label = output_dir.name.removeprefix("output_")
    (output_dir / f"fdm_outer_wave_provenance_{label}.txt").write_text(
        f"# fdm_outer_wave_provenance_v3\n"
        "time_code = 1.0d0\n"
        "aexp = 5.0d-1\n"
        "nstep_coarse = 12\n"
        "mpi_ncpu = 2\n"
        "restart_parent_output = 0\n"
        "m_axion_ev = 1.0d-22\n"
        "hbar_code = 2.0d-3\n"
        "fdm_use_hjm = F\n"
        "fdm_first_wave_level = 17\n"
        "fdm_dual_soliton_ic = F\n"
        "fdm_dual_soliton_profile_c = 0.0d0\n"
        "fdm_dual_soliton_rho0 = 0.0d0 0.0d0\n"
        "fdm_dual_soliton_rc_box = 0.0d0 0.0d0\n"
        "fdm_dual_soliton_center_box_1 = 0.1d0 0.1d0 0.1d0\n"
        "fdm_dual_soliton_center_box_2 = 0.9d0 0.9d0 0.9d0\n"
        "fdm_dual_soliton_velocity_1 = 0.0d0 0.0d0 0.0d0\n"
        "fdm_dual_soliton_velocity_2 = 0.0d0 0.0d0 0.0d0\n"
        "fdm_dual_soliton_phase = 0.0d0 0.0d0\n"
        "analytic_fdm_drag_enabled = .false.\n"
        "force_accounting = resolved_wave_only\n"
        "leaf_mass_code = 3.0d0\n"
        "integrated_current_code = 1.0d-2 -2.0d-2 3.0d-2\n"
        "leaf_cell_count = 100.0\n"
        "complete_current_stencil_cell_count = 100.0\n"
        "complete_current_stencil_fraction = 1.0d0\n"
        f"psi_snapshot_prefix = fdm_{label}.out\n",
        encoding="utf-8",
    )


def _runtime_attestation(
    tmp_path: Path, specification: Path, source: Path
) -> Path:
    del specification
    executable = tmp_path / "ramses"
    executable.write_bytes(b"compiled writer integration fixture\n")
    executable.chmod(0o755)
    output_dir = tmp_path / "output_00042"
    _runtime_supporting_files(output_dir)
    _raw_fdm_outer_provenance(output_dir)
    sidecar = output_dir / "dm_run_provenance_00042.txt"
    _fdm_sidecar(sidecar)
    record = build_fdm_writer_runtime_attestation(
        source, executable, sidecar, operator_confirmed=True
    )
    path = tmp_path / "runtime_attestation.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def test_runtime_attestation_requires_explicit_operator_confirmation(
    tmp_path: Path,
) -> None:
    specification, _, _, source = _outer_inputs(tmp_path)
    del specification
    executable = tmp_path / "ramses"
    executable.write_bytes(b"compiled writer integration fixture\n")
    executable.chmod(0o755)
    output_dir = tmp_path / "output_00042"
    _runtime_supporting_files(output_dir)
    _raw_fdm_outer_provenance(output_dir)
    sidecar = output_dir / "dm_run_provenance_00042.txt"
    _fdm_sidecar(sidecar)
    with pytest.raises(ValueError, match="operator_confirmed"):
        build_fdm_writer_runtime_attestation(source, executable, sidecar)


def test_outer_submission_requires_compiled_writer_attestation(tmp_path: Path) -> None:
    specification, manifest, preflight, source = _outer_inputs(tmp_path)
    decision = assess_pure_fdm_outer_submission(
        specification, manifest, preflight, source
    )
    assert decision.status == "not_ready_writer_runtime_attestation"
    assert not decision.ready
    assert decision.outer_case_count == 20
    assert decision.nested_request_count == 10
    assert any("runtime attestation" in reason for reason in decision.reasons)


def test_outer_submission_accepts_exact_inputs_after_runtime_test(tmp_path: Path) -> None:
    specification, manifest, preflight, source = _outer_inputs(tmp_path)
    attestation = _runtime_attestation(tmp_path, specification, source)
    decision = assess_pure_fdm_outer_submission(
        specification, manifest, preflight, source, attestation
    )
    assert decision.status == "ready_for_operator_submission"
    assert decision.ready
    assert decision.runtime_attestation is not None
    assert decision.as_dict()["execution"]["scheduler"] == "Slurm only"


def test_saved_submission_decision_is_rebuilt_before_verification(tmp_path: Path) -> None:
    specification, manifest, preflight, source = _outer_inputs(tmp_path)
    attestation = _runtime_attestation(tmp_path, specification, source)
    decision = assess_pure_fdm_outer_submission(
        specification, manifest, preflight, source, attestation
    )
    record_path = tmp_path / "submission.json"
    record_path.write_text(
        json.dumps(decision.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verified = read_verified_pure_fdm_outer_submission(
        record_path, specification, manifest, preflight, source, attestation
    )
    assert verified.ready

    saved = json.loads(record_path.read_text(encoding="utf-8"))
    saved["interpretation"] = "edited by operator"
    record_path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match current inputs"):
        read_verified_pure_fdm_outer_submission(
            record_path, specification, manifest, preflight, source, attestation
        )


def test_outer_submission_rejects_tampered_manifest_before_runtime_attestation(
    tmp_path: Path,
) -> None:
    specification, manifest, preflight, source = _outer_inputs(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["cases"][0]["replicate"] = 99
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    decision = assess_pure_fdm_outer_submission(
        specification, manifest, preflight, source
    )
    assert decision.status == "not_ready_outer_manifest_or_preflight"
    assert any("manifest" in reason for reason in decision.reasons)


def test_outer_submission_rejects_missing_fdm_writer_token(tmp_path: Path) -> None:
    specification, manifest, preflight, source = _outer_inputs(tmp_path)
    source.write_text("! no executable FDM accounting token\n", encoding="utf-8")
    decision = assess_pure_fdm_outer_submission(
        specification, manifest, preflight, source
    )
    assert decision.status == "not_ready_writer_source"
    assert any("fdm" in reason.lower() for reason in decision.reasons)


@pytest.mark.parametrize("field", ("fdm_force_accounting", "fdm_outer_ledger_enabled"))
def test_outer_submission_rejects_invalid_runtime_sidecar(
    tmp_path: Path, field: str
) -> None:
    specification, manifest, preflight, source = _outer_inputs(tmp_path)
    attestation = _runtime_attestation(tmp_path, specification, source)
    sidecar = tmp_path / "output_00042" / "dm_run_provenance_00042.txt"
    text = sidecar.read_text(encoding="utf-8")
    if field == "fdm_force_accounting":
        text = text.replace("fdm_force_accounting = resolved_wave_only", "fdm_force_accounting = analytic_drag")
    else:
        text = text.replace("fdm_outer_ledger_enabled = .true.", "fdm_outer_ledger_enabled = .false.")
    sidecar.write_text(text, encoding="utf-8")
    # Keep the attestation's digest honest; the reader must then reject the
    # semantic sidecar value rather than merely reporting a stale hash.
    record = json.loads(attestation.read_text(encoding="utf-8"))
    record["fdm_sidecar"]["sha256"] = _sha256(sidecar)
    attestation.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    decision = assess_pure_fdm_outer_submission(
        specification, manifest, preflight, source, attestation
    )
    assert decision.status == "not_ready_writer_runtime_attestation"
    assert any("sidecar" in reason or "accounting" in reason for reason in decision.reasons)


@pytest.mark.parametrize(
    ("artifact", "expected"),
    (
        ("COMPLETE", "COMPLETE marker"),
        ("compilation.txt", "copied compilation"),
    ),
)
def test_runtime_attestation_requires_completed_supporting_files(
    tmp_path: Path, artifact: str, expected: str
) -> None:
    specification, _, _, source = _outer_inputs(tmp_path)
    executable = tmp_path / "ramses"
    executable.write_bytes(b"compiled writer integration fixture\n")
    executable.chmod(0o755)
    output_dir = tmp_path / "output_00042"
    _runtime_supporting_files(output_dir)
    _raw_fdm_outer_provenance(output_dir)
    sidecar = output_dir / "dm_run_provenance_00042.txt"
    _fdm_sidecar(sidecar)
    (output_dir / artifact).unlink()
    with pytest.raises(ValueError, match=expected):
        build_fdm_writer_runtime_attestation(
            source, executable, sidecar, operator_confirmed=True
        )


def test_runtime_attestation_rejects_compilation_build_mismatch(tmp_path: Path) -> None:
    specification, _, _, source = _outer_inputs(tmp_path)
    del specification
    executable = tmp_path / "ramses"
    executable.write_bytes(b"compiled writer integration fixture\n")
    executable.chmod(0o755)
    output_dir = tmp_path / "output_00042"
    _runtime_supporting_files(output_dir)
    _raw_fdm_outer_provenance(output_dir)
    sidecar = output_dir / "dm_run_provenance_00042.txt"
    _fdm_sidecar(sidecar)
    (output_dir / "compilation.txt").write_text(
        "last commit = " + "b" * 40 + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="last commit"):
        build_fdm_writer_runtime_attestation(
            source, executable, sidecar, operator_confirmed=True
        )


def test_runtime_attestation_rejects_missing_success_log(tmp_path: Path) -> None:
    specification, _, _, source = _outer_inputs(tmp_path)
    del specification
    executable = tmp_path / "ramses"
    executable.write_bytes(b"compiled writer integration fixture\n")
    executable.chmod(0o755)
    output_dir = tmp_path / "output_00042"
    _runtime_supporting_files(output_dir)
    _raw_fdm_outer_provenance(output_dir)
    sidecar = output_dir / "dm_run_provenance_00042.txt"
    _fdm_sidecar(sidecar)
    (tmp_path / "run.log").write_text(
        "Working with nproc =    2 for ndim = 3\n"
        "Working with nproc =    2 for ndim = 3\n"
        "Run stopped\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="completion marker"):
        build_fdm_writer_runtime_attestation(
            source, executable, sidecar, operator_confirmed=True
        )


def test_runtime_attestation_rejects_singleton_mpi_launch(tmp_path: Path) -> None:
    specification, _, _, source = _outer_inputs(tmp_path)
    del specification
    executable = tmp_path / "ramses"
    executable.write_bytes(b"compiled writer integration fixture\n")
    executable.chmod(0o755)
    output_dir = tmp_path / "output_00042"
    _runtime_supporting_files(output_dir)
    _raw_fdm_outer_provenance(output_dir)
    sidecar = output_dir / "dm_run_provenance_00042.txt"
    _fdm_sidecar(sidecar)
    (tmp_path / "run.log").write_text(
        "Working with nproc =    1 for ndim = 3\n"
        "Working with nproc =    1 for ndim = 3\n"
        "Run completed\n"
        "Run completed\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multi-rank MPI"):
        build_fdm_writer_runtime_attestation(
            source, executable, sidecar, operator_confirmed=True
        )
