"""Common-time and non-calibrating guards for the per-step TSC comparison."""

import json

import numpy as np
import pytest

from scripts import analyze_periodic_tsc_step_trace as analysis


def test_observed_order_is_defined_only_without_sign_change() -> None:
    assert analysis._observed_orders([16.0, 4.0, 1.0, 0.25]) == pytest.approx(
        [2.0, 2.0, 2.0]
    )
    assert analysis._observed_orders([-4.0, 1.0, 0.0]) == [None, None]


def test_step_trace_rejects_changed_source_and_common_time(tmp_path, monkeypatch) -> None:
    reference = tmp_path / "reference" / "n256_f100"
    trace_root = tmp_path / "traces"
    common_path = tmp_path / "common.json"
    labels = ("n256_f100", "f050", "f025", "f0125")
    paths = (reference, *(trace_root / label for label in labels[1:]))
    sources = (("src/fdm_smbh_delay/periodic_mesh_coupling.py", "same"),)
    records = {}
    common_runs = []
    for index, path in enumerate(paths):
        stride = 2**index
        times = np.arange(10 * stride + 1, dtype=float) / stride
        item = {
            "source_fingerprints": sources,
            "launch_hashes": ("wave", "particle"),
            "physical_configuration": {"Plummer Radius": 0.1},
            "physical_metadata": {"box_size_pc": 26.4},
            "wave_time_step_code": 8e-9 / stride,
            "time_step_factor": 1.0 / stride,
            "wave_steps_per_common_save": stride,
            "first_common_save_hamiltonian_change_msun_pc2_myr2": -16.0 / stride**2,
            "last_common_save_hamiltonian_change_msun_pc2_myr2": -8.0 / stride**2,
            "last_common_save_separation_pc": 0.57 + 1e-6 / stride,
        }
        series = np.zeros(times.size, dtype=[("time_myr", float)])
        series["time_myr"] = times
        records[str(path)] = (item, series)
        if index < 3:
            common_runs.append({
                "source_fingerprints": [list(row) for row in sources],
                "launch_hashes": ["wave", "particle"],
                "time_step_factor": 1.0 / stride,
                "first_save_hamiltonian_change_msun_pc2_myr2": -16.0 / stride**2,
                "final_save_hamiltonian_change_msun_pc2_myr2": -8.0 / stride**2,
                "time_myr": np.arange(11, dtype=float).tolist(),
            })
    common_record = {
        "status": "periodic_tsc_temporal_diagnostic_not_a_calibration_release",
        "reference_run": str(reference),
        "analysis_source_sha256": {
            "scripts/analyze_periodic_tsc_dt_followup.py": "hash",
            "scripts/analyze_qe_startup_component_dt.py": "hash",
            "scripts/analyze_pyul_wave_run.py": "hash",
        },
        "runs": common_runs,
    }
    common_path.write_text(json.dumps(common_record))
    monkeypatch.setattr(
        analysis, "_read_trace",
        lambda path, factor, saves, stop: records[str(path)],
    )
    monkeypatch.setattr(analysis, "_sha256", lambda path: "hash")
    result = analysis.summarize(reference, trace_root, common_path)
    assert result["status"].endswith("not_a_calibration_release")
    assert result["first_common_save_drift_observed_pair_orders"] == pytest.approx(
        [2.0, 2.0, 2.0]
    )
    records[str(paths[3])][0]["source_fingerprints"] = (("other", "changed"),)
    with pytest.raises(ValueError, match="differ in source"):
        analysis.summarize(reference, trace_root, common_path)
    records[str(paths[3])][0]["source_fingerprints"] = sources
    common_runs[1]["time_myr"][1] = 1.5
    common_path.write_text(json.dumps(common_record))
    with pytest.raises(ValueError, match="differs from matched common-time run"):
        analysis.summarize(reference, trace_root, common_path)
