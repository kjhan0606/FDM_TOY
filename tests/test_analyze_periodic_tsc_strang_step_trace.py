"""Source and common-time guards for the experimental Strang trace audit."""

import json

import numpy as np
import pytest

from scripts import analyze_periodic_tsc_strang_step_trace as analysis


def test_strang_trace_requires_common_source_and_phase_scaling(
    tmp_path, monkeypatch,
) -> None:
    root = tmp_path / "traces"
    records = {}
    for index, (label, factor, saves, stop) in enumerate(analysis._RUNS):
        run = root / label
        run.mkdir(parents=True)
        (run / "initial_compact_phase_jump.json").write_text(json.dumps({
            "max_neighbour_phase_difference_over_pi": 0.08 * factor,
        }))
        records[str(run)] = ({
            "source_fingerprints": (("source.py", "same"),),
            "launch_hashes": ("wave", "particle"),
            "physical_configuration": {"Plummer Radius": 0.1},
            "physical_metadata": {"box_size_pc": 26.4},
            "wave_time_step_code": 8e-9 * factor,
            "time_step_factor": factor,
            "first_common_save_hamiltonian_change_msun_pc2_myr2": -16 * factor**2,
            "last_common_save_hamiltonian_change_msun_pc2_myr2": -8 * factor**2,
            "last_common_save_separation_pc": 0.57 + 1e-6 * factor**2,
        }, np.zeros(stop + 1))
    monkeypatch.setattr(
        analysis, "_read_trace",
        lambda run, factor, saves, stop, coupling: records[str(run)],
    )
    monkeypatch.setattr(analysis, "_sha256", lambda path: "hash")
    result = analysis.summarize(root)
    assert result["status"].endswith("not_a_calibration_release")
    assert result["first_common_save_drift_observed_pair_orders"] == pytest.approx(
        [2.0, 2.0, 2.0]
    )
    assert result["last_common_save_separation_observed_pair_orders"] == pytest.approx(
        [2.0, 2.0]
    )
    records[str(root / "f0125")][0]["source_fingerprints"] = (("source.py", "changed"),)
    with pytest.raises(ValueError, match="differ in source"):
        analysis.summarize(root)
    records[str(root / "f0125")][0]["source_fingerprints"] = (("source.py", "same"),)
    (root / "f0125" / "initial_compact_phase_jump.json").write_text(json.dumps({
        "max_neighbour_phase_difference_over_pi": 0.02,
    }))
    with pytest.raises(ValueError, match="phase jump does not scale"):
        analysis.summarize(root)
