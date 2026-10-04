"""Fail-closed comparison checks for reciprocal TSC time-step diagnostics."""

import json

import pytest

from scripts import analyze_periodic_tsc_dt_followup as analysis


def _comparison_fixture(tmp_path, monkeypatch):
    reference = tmp_path / "first" / "n256_f100"
    followup = tmp_path / "more"
    paths = (reference, followup / "f050", followup / "f025")
    for path in paths:
        path.mkdir(parents=True)
        (path / "conservation_summary.json").write_text("{}")
        (path / "fdm_adapter_metadata.json").write_text(json.dumps({
            "wave_smbh_coupling": "periodic_tsc_reciprocal",
            "experimental_coupling_not_a_calibration_release": True,
        }))
    source = (
        ("scripts/run_torch_wave_case.py", "runner-hash"),
        (analysis._TSC_SOURCE, "tsc-hash"),
    )
    records = {}
    for path, factor, steps in zip(paths, (1.0, 0.5, 0.25), (1, 2, 4), strict=True):
        records[path.name] = {
            "label": path.name,
            "time_step_factor": factor,
            "wave_steps_per_saved_interval": steps,
            "launch_hashes": ("wave-hash", "particle-hash"),
            "source_fingerprints": source,
            "physical_metadata": {"box_size_pc": 26.4},
            "matter_particles": {"Plummer Radius": 0.05},
            "wave_time_step_code": 4e-8 * factor,
            "time_myr": [0.0, 1e-6, 2e-6],
            "change_from_initial": {"combined_energy": [0.0, -10 * factor, -5 * factor]},
            "maximum_sampled_prefix_error_over_transfer": 0.02 * factor,
        }
    monkeypatch.setattr(
        analysis, "_read_run",
        lambda root, label, factor, steps: records[label].copy(),
    )
    monkeypatch.setattr(analysis, "_sha256", lambda path: "summary-hash")
    monkeypatch.setattr(
        analysis, "_launch_owner",
        lambda path, project, sources: {
            "slurm_job_id": "1", "source_commit": "a" * 40,
            "launch_owner_sha256": "owner-hash",
        },
    )
    return reference, followup, records, paths


def test_tsc_comparison_requires_common_source_and_marks_noncalibrating(
    tmp_path, monkeypatch,
) -> None:
    reference, followup, records, _ = _comparison_fixture(tmp_path, monkeypatch)
    result = analysis.summarize(reference, followup)
    assert result["status"].endswith("not_a_calibration_release")
    assert [row["first_save_hamiltonian_change_msun_pc2_myr2"] for row in result["runs"]] == [
        -10.0, -5.0, -2.5,
    ]
    assert all(row["calibration_eligible"] is False for row in result["runs"])
    records["f025"]["source_fingerprints"] = ((analysis._TSC_SOURCE, "changed"),)
    with pytest.raises(ValueError, match="common source"):
        analysis.summarize(reference, followup)


def test_tsc_comparison_rejects_mode_or_time_mismatch(tmp_path, monkeypatch) -> None:
    reference, followup, records, paths = _comparison_fixture(tmp_path, monkeypatch)
    (paths[1] / "fdm_adapter_metadata.json").write_text(json.dumps({
        "wave_smbh_coupling": "legacy_plummer",
        "experimental_coupling_not_a_calibration_release": False,
    }))
    with pytest.raises(ValueError, match="scope is unverified"):
        analysis.summarize(reference, followup)
    (paths[1] / "fdm_adapter_metadata.json").write_text(json.dumps({
        "wave_smbh_coupling": "periodic_tsc_reciprocal",
        "experimental_coupling_not_a_calibration_release": True,
    }))
    records["f050"]["time_myr"] = [0.0, 1.1e-6, 2.2e-6]
    with pytest.raises(ValueError, match="matched physical save times"):
        analysis.summarize(reference, followup)
