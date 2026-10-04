"""Small-grid and provenance checks for fixed-source wave diagnostics."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.audit_wave_only_startup import (
    _evolve_fixed_sources, _loaded_numerical_source_paths, _reference_interval,
)


def test_fixed_source_wave_matches_initial_budget_and_preserves_mass() -> None:
    generator = torch.Generator().manual_seed(2718)
    real = torch.randn((8, 8, 8), generator=generator, dtype=torch.float64)
    imaginary = torch.randn((8, 8, 8), generator=generator, dtype=torch.float64)
    wave = torch.complex(real, imaginary)
    arguments = {
        "box_length_code": 4.0,
        "masses_code": np.array([0.1, 0.03]),
        "positions_code": np.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        "plummer_radius_code": 0.2,
        "physical_interval_code": 0.001,
    }
    runs = [_evolve_fixed_sources(wave, steps=steps, **arguments)
            for steps in (1, 2, 4)]
    assert [run["wave_steps"] for run in runs] == [1, 2, 4]
    assert [run["wave_time_step_code"] for run in runs] == pytest.approx(
        [0.001, 0.0005, 0.00025]
    )
    for run in runs:
        assert all(np.isfinite(value) for value in run["final_components_code"].values())
        assert abs(run["wave_mass_relative_change"]) < 1.0e-12
        assert run["initial_hamiltonian_code"] == pytest.approx(
            sum(run["initial_components_code"].values())
        )
    for run in runs[1:]:
        assert run["initial_components_code"] == pytest.approx(
            runs[0]["initial_components_code"]
        )


def test_fixed_source_wave_rejects_invalid_step_count() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _evolve_fixed_sources(
            torch.ones((8, 8, 8), dtype=torch.complex128),
            box_length_code=4.0,
            masses_code=np.array([0.1, 0.03]),
            positions_code=np.zeros((2, 3)),
            plummer_radius_code=0.2,
            physical_interval_code=0.001,
            steps=0,
        )


def test_rejects_stale_imported_numerical_operator(monkeypatch, tmp_path: Path) -> None:
    from scripts import audit_wave_only_startup as audit_module

    project = Path(__file__).resolve().parents[1]
    assert set(_loaded_numerical_source_paths(project)) == {
        "src/fdm_smbh_delay/torch_wave.py", "src/fdm_smbh_delay/pyul.py",
    }
    monkeypatch.setattr(audit_module._loaded_torch_wave, "__file__", str(tmp_path / "stale.py"))
    with pytest.raises(ValueError, match="not from this project"):
        _loaded_numerical_source_paths(project)


def test_reference_interval_requires_launch_hashes_and_same_physics(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    reference = tmp_path / "f100"
    seed.mkdir()
    reference.mkdir()
    unit_fields = {
        "box_size_pc": 26.4, "particle_mass_ev": 1e-21,
        "pyul_length_unit_m": 1.0, "pyul_time_unit_s": 1.0,
        "pyul_mass_unit_kg": 1.0, "pyul_energy_unit_j": 1.0,
    }
    (seed / "fdm_adapter_metadata.json").write_text(json.dumps(unit_fields))
    config = {"Matter Particles": {
        "Mass Units": "solar_masses", "Plummer Radius": 0.05,
        "Condition": [[1.0], [0.3]],
    }}
    (seed / "config.uldm").write_text(json.dumps(config))
    seed_hashes = {"wave": "a" * 64, "particle": "b" * 64}
    metadata = {
        **unit_fields,
        "reference_initial_state": str(seed),
        "reference_initial_wave_sha256": seed_hashes["wave"],
        "reference_initial_particle_sha256": seed_hashes["particle"],
        "case_id": "qe_q030_e030_a020", "resolution": 256,
        "backend": "pytorch_cuda", "analytic_fdm_drag": False,
        "time_step_factor": 1.0, "save_number": 58500,
        "actual_wave_steps": 58500,
        "diagnostic_stop_after_save": 10,
        "nbody_rk4_substeps_per_wave_step": 9,
        "duration_myr": 0.1,
        "wave_time_step_code": 0.1 / 58500,
        "torch_version": torch.__version__,
    }
    metadata_path = reference / "fdm_adapter_metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    config_path = reference / "config.uldm"
    config_path.write_text(json.dumps(config))
    (reference / "torch_run_summary.json").write_text(json.dumps({
        "status": "diagnostic_partial", "saved_intervals": 10,
        "planned_wave_steps": 58500,
    }))
    source_dir = reference / "torch_solver_provenance/source"
    numerical_hashes = {}
    for relative in (
        "scripts/run_torch_wave_case.py", "src/fdm_smbh_delay/torch_wave.py",
        "src/fdm_smbh_delay/pyul.py",
    ):
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative)
        numerical_hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest_path = reference / "torch_solver_provenance/manifest.json"
    manifest_path.write_text(json.dumps({
        "status": "source_snapshot", "run": str(reference),
        "input_records": {
            "fdm_adapter_metadata_sha256": hashlib.sha256(
                metadata_path.read_bytes()
            ).hexdigest(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
        "source_files": [
            {"path": relative, "sha256": digest}
            for relative, digest in numerical_hashes.items()
        ],
    }))
    interval, wave_step, ladder, _record = _reference_interval(
        seed=seed, reference=reference, seed_hashes=seed_hashes,
        numerical_hashes=numerical_hashes,
    )
    assert interval == pytest.approx(0.1 / 58500)
    assert wave_step == pytest.approx(0.1 / 58500)
    assert ladder == (1, 2, 4)

    metadata["reference_initial_wave_sha256"] = "c" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="matching launch"):
        _reference_interval(seed=seed, reference=reference,
                            seed_hashes=seed_hashes,
                            numerical_hashes=numerical_hashes)

    metadata["reference_initial_wave_sha256"] = seed_hashes["wave"]
    metadata_path.write_text(json.dumps(metadata))
    manifest = json.loads(manifest_path.read_text())
    manifest["input_records"]["fdm_adapter_metadata_sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    config["Matter Particles"]["Plummer Radius"] = 0.06
    config_path.write_text(json.dumps(config))
    manifest["input_records"]["config_sha256"] = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="matching launch"):
        _reference_interval(seed=seed, reference=reference,
                            seed_hashes=seed_hashes,
                            numerical_hashes=numerical_hashes)
