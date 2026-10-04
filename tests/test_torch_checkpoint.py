from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import run_torch_wave_case  # noqa: E402
from fdm_smbh_delay.pyul import (  # noqa: E402
    PYUL_MYR_S,
    PYUL_PARSEC_M,
    PYUL_SOLAR_MASS_KG,
)
from fdm_smbh_delay.run_metadata import (  # noqa: E402
    validate_torch_calibration_completion,
)
from fdm_smbh_delay.torch_wave import (  # noqa: E402
    apply_potential_half_kick_in_place,
    apply_kinetic_phase_in_place,
    periodic_poisson_torch,
    spectral_grid,
    wave_density,
)


def _state(offset: float = 0.0) -> np.ndarray:
    return np.arange(12, dtype=float).reshape(2, 6) + offset


def _wave(value: float):
    return torch.full(
        (4, 4, 4), complex(value, -value), dtype=torch.complex128
    )


def test_diagnostic_stop_writes_restartable_partial_not_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference"
    (reference / "Outputs/3Wfn").mkdir(parents=True)
    (reference / "Outputs/NBody").mkdir()
    np.save(reference / "Outputs/3Wfn/P3D_#000.npy", np.ones((16, 16, 16), dtype=complex))
    np.save(
        reference / "Outputs/NBody/NTM_#000.npy",
        np.array([[-0.5, 0, 0, 0, 0, 0], [0.5, 0, 0, 0, 0, 0]], dtype=float),
    )
    energy_unit = PYUL_SOLAR_MASS_KG * (PYUL_PARSEC_M / PYUL_MYR_S) ** 2
    (reference / "fdm_adapter_metadata.json").write_text(json.dumps({
        "adapter_revision": "test",
        "box_size_pc": 4.0,
        "resolution": 16,
        "case_id": "diagnostic",
        "pyul_length_unit_m": PYUL_PARSEC_M,
        "pyul_time_unit_s": PYUL_MYR_S,
        "pyul_mass_unit_kg": PYUL_SOLAR_MASS_KG,
        "pyul_energy_unit_j": energy_unit,
    }))
    (reference / "config.uldm").write_text(json.dumps({
        "Matter Particles": {
            "Plummer Radius": 0.1,
            "Condition": [[1.0, [-0.5, 0, 0], [0, 0, 0]],
                          [1.0, [0.5, 0, 0], [0, 0, 0]]],
        },
        "Duration": {"Time Duration": 0.001},
        "Save Options": {"Number": 4},
    }))
    (reference / "reproducibility.uldm").write_text("test\n")
    output = tmp_path / "partial"
    monkeypatch.setattr(sys, "argv", [
        "run_torch_wave_case.py", str(reference), "--output", str(output),
        "--duration-myr", "0.001", "--save-number", "4",
        "--save-3d-number", "0", "--movie-frame-number", "0",
        "--checkpoint-every-saves", "1", "--device", "cpu",
        "--diagnostic-stop-after-save", "1",
    ])
    assert run_torch_wave_case.main() == 0
    summary = json.loads((output / "torch_run_summary.json").read_text())
    metadata = json.loads((output / "fdm_adapter_metadata.json").read_text())
    assert metadata["reference_initial_wave_sha256"] == hashlib.sha256(
        (reference / "Outputs/3Wfn/P3D_#000.npy").read_bytes()
    ).hexdigest()
    assert metadata["reference_initial_particle_sha256"] == hashlib.sha256(
        (reference / "Outputs/NBody/NTM_#000.npy").read_bytes()
    ).hexdigest()
    assert summary["status"] == "diagnostic_partial"
    assert summary["saved_intervals"] == 1
    assert summary["actual_wave_steps"] < summary["planned_wave_steps"]
    assert len(list((output / "Outputs/NBody").glob("NTM_#*.npy"))) == 2
    assert (output / "Checkpoints/latest.json").is_file()
    with pytest.raises(ValueError, match="not complete"):
        validate_torch_calibration_completion(
            output,
            expected_case_id="diagnostic",
            expected_resolution=16,
            expected_duration_myr=0.001,
            expected_saved_intervals=4,
            expected_saved_3d_states=1,
            expected_rk4_substeps=9,
            expected_checkpoint_interval=1,
            expected_run_id="partial",
        )
    monkeypatch.setattr(sys, "argv", [
        "run_torch_wave_case.py", str(reference), "--output", str(output),
        "--duration-myr", "0.001", "--save-number", "4",
        "--save-3d-number", "0", "--movie-frame-number", "0",
        "--checkpoint-every-saves", "1", "--device", "cpu", "--resume",
    ])
    assert run_torch_wave_case.main() == 0
    resumed_summary = json.loads((output / "torch_run_summary.json").read_text())
    assert resumed_summary["status"] == "diagnostic_complete"
    assert resumed_summary["saved_intervals"] == 4
    assert len(list((output / "Outputs/NBody").glob("NTM_#*.npy"))) == 5
    with pytest.raises(ValueError, match="not complete"):
        validate_torch_calibration_completion(
            output,
            expected_case_id="diagnostic",
            expected_resolution=16,
            expected_duration_myr=0.001,
            expected_saved_intervals=4,
            expected_saved_3d_states=1,
            expected_rk4_substeps=9,
            expected_checkpoint_interval=1,
            expected_run_id="partial",
        )
    resumed_summary["status"] = "complete"
    (output / "torch_run_summary.json").write_text(json.dumps(resumed_summary))
    with pytest.raises(ValueError, match="diagnostic evolution cannot be a calibration run"):
        validate_torch_calibration_completion(
            output,
            expected_case_id="diagnostic",
            expected_resolution=16,
            expected_duration_myr=0.001,
            expected_saved_intervals=4,
            expected_saved_3d_states=1,
            expected_rk4_substeps=9,
            expected_checkpoint_interval=1,
            expected_run_id="partial",
        )


@pytest.mark.parametrize(
    "extra", (
        ["--checkpoint-every-saves", "0", "--diagnostic-stop-after-save", "1"],
        ["--resume", "--diagnostic-stop-after-save", "1"],
        ["--diagnostic-stop-after-save", "4"],
    ),
)
def test_diagnostic_stop_rejects_unsafe_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", [
        "run_torch_wave_case.py", str(tmp_path / "missing"),
        "--output", str(tmp_path / "output"),
        "--duration-myr", "0.001", "--save-number", "4",
        *extra,
    ])
    with pytest.raises(ValueError, match="checkpointed run"):
        run_torch_wave_case.main()


def test_remaining_time_uses_only_steps_completed_since_resume() -> None:
    assert run_torch_wave_case._estimated_remaining_seconds(
        elapsed_seconds=20.0,
        start_step=600,
        step=800,
        total_steps=1200,
    ) == pytest.approx(40.0)
    assert run_torch_wave_case._estimated_remaining_seconds(
        elapsed_seconds=0.0,
        start_step=600,
        step=600,
        total_steps=1200,
    ) is None


def test_restart_rejects_changed_launch_hashed_seed() -> None:
    requested = {key: 1 for key in run_torch_wave_case._RESTART_METADATA_KEYS}
    requested.update({
        "reference_initial_wave_sha256": "same-wave",
        "reference_initial_particle_sha256": "same-particles",
    })
    saved = dict(requested)
    run_torch_wave_case._require_resume_metadata(saved, requested)
    legacy = dict(saved)
    del legacy["reference_initial_wave_sha256"]
    del legacy["reference_initial_particle_sha256"]
    run_torch_wave_case._require_resume_metadata(legacy, requested)
    saved["reference_initial_wave_sha256"] = "changed-wave"
    with pytest.raises(ValueError, match="reference initial content changed"):
        run_torch_wave_case._require_resume_metadata(saved, requested)


@pytest.mark.parametrize(
    "field", (
        "time_step_factor", "nbody_rk4_substeps_per_wave_step",
        "wave_density_layout", "compact_potential_layout", "potential_phase_layout",
        "total_potential_lifetime",
        "saved_energy_density_lifetime",
    )
)
def test_restart_rejects_missing_or_changed_numerical_layout(field: str) -> None:
    requested = {key: 1 for key in run_torch_wave_case._RESTART_METADATA_KEYS}
    requested["wave_density_layout"] = "real_imag_addcmul_v1"
    requested["compact_potential_layout"] = "x_slab32_inplace_rsqrt_v1"
    requested["potential_phase_layout"] = "complex_real_imag_inplace_trig_v1"
    requested["total_potential_lifetime"] = (
        "recompute_before_first_kick_release_before_save_v1"
    )
    requested["saved_energy_density_lifetime"] = (
        "release_before_kinetic_fft_rebuild_for_output_v1"
    )
    run_torch_wave_case._require_resume_metadata(dict(requested), requested)
    saved = dict(requested)
    del saved[field]
    with pytest.raises(ValueError, match=field):
        run_torch_wave_case._require_resume_metadata(saved, requested)
    saved[field] = "other_layout"
    with pytest.raises(ValueError, match=field):
        run_torch_wave_case._require_resume_metadata(saved, requested)


def test_torch_checkpoint_round_trip_and_replaces_the_previous_pair(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    run_torch_wave_case._save_checkpoint(
        run=run,
        wavefunction=_wave(1.0),
        state=_state(),
        step=320,
        save_index=32,
    )
    run_torch_wave_case._save_checkpoint(
        run=run,
        wavefunction=_wave(2.0),
        state=_state(1.0),
        step=640,
        save_index=64,
    )
    measured_wave, measured_state, step, save_index = (
        run_torch_wave_case._load_checkpoint(
            run=run, device=torch.device("cpu")
        )
    )
    torch.testing.assert_close(measured_wave, _wave(2.0))
    np.testing.assert_allclose(measured_state, _state(1.0))
    assert (step, save_index) == (640, 64)
    assert not (run / "Checkpoints/wave_000032.npy").exists()
    assert not (run / "Checkpoints/state_000032.npz").exists()


def test_interrupted_checkpoint_publication_keeps_the_previous_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    run_torch_wave_case._save_checkpoint(
        run=run,
        wavefunction=_wave(1.0),
        state=_state(),
        step=320,
        save_index=32,
    )
    original_replace = run_torch_wave_case.os.replace

    def interrupt_before_state_publish(source, destination) -> None:
        if Path(destination).name == "state_000064.npz":
            raise OSError("simulated interruption")
        original_replace(source, destination)

    monkeypatch.setattr(
        run_torch_wave_case.os,
        "replace",
        interrupt_before_state_publish,
    )
    with pytest.raises(OSError, match="simulated interruption"):
        run_torch_wave_case._save_checkpoint(
            run=run,
            wavefunction=_wave(2.0),
            state=_state(1.0),
            step=640,
            save_index=64,
        )
    measured_wave, measured_state, step, save_index = (
        run_torch_wave_case._load_checkpoint(
            run=run, device=torch.device("cpu")
        )
    )
    torch.testing.assert_close(measured_wave, _wave(1.0))
    np.testing.assert_allclose(measured_state, _state())
    assert (step, save_index) == (320, 32)


def test_torch_checkpoint_rejects_marker_state_disagreement(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    run_torch_wave_case._save_checkpoint(
        run=run,
        wavefunction=_wave(1.0),
        state=_state(),
        step=320,
        save_index=32,
    )
    marker_path = run / "Checkpoints/latest.json"
    marker = json.loads(marker_path.read_text())
    marker["step"] = 321
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="checkpoint state disagree"):
        run_torch_wave_case._load_checkpoint(
            run=run, device=torch.device("cpu")
        )


def test_self_gravitating_wave_restart_matches_uninterrupted_split_steps(
    tmp_path: Path,
) -> None:
    """The separable drift preserves the actual split-step restart trajectory."""

    grid = spectral_grid(
        resolution=8, box_length=4.0, time_step=0.001,
        device=torch.device("cpu"),
    )
    rng = np.random.default_rng(2394)
    initial = torch.as_tensor(
        1.0 + 0.01 * rng.normal(size=(8, 8, 8))
        + 0.01j * rng.normal(size=(8, 8, 8)),
        dtype=torch.complex128,
    )

    def advance(wavefunction: torch.Tensor, count: int) -> torch.Tensor:
        for _ in range(count):
            density = wave_density(wavefunction)
            potential = periodic_poisson_torch(
                density, grid.poisson_inverse_wavenumber_squared
            )
            apply_potential_half_kick_in_place(wavefunction, potential, 0.001)
            spectrum = torch.fft.fftn(wavefunction)
            apply_kinetic_phase_in_place(spectrum, grid.kinetic_axis_phase)
            wavefunction = torch.fft.ifftn(spectrum)
            density = wave_density(wavefunction)
            potential = periodic_poisson_torch(
                density, grid.poisson_inverse_wavenumber_squared
            )
            apply_potential_half_kick_in_place(wavefunction, potential, 0.001)
        return wavefunction

    uninterrupted = advance(initial.clone(), 4)
    run = tmp_path / "run"
    run.mkdir()
    partial = advance(initial.clone(), 2)
    run_torch_wave_case._save_checkpoint(
        run=run, wavefunction=partial, state=_state(), step=2, save_index=2,
    )
    resumed, state, step, save_index = run_torch_wave_case._load_checkpoint(
        run=run, device=torch.device("cpu"),
    )
    assert (step, save_index) == (2, 2)
    np.testing.assert_array_equal(state, _state())
    torch.testing.assert_close(resumed, partial, rtol=0, atol=0)
    torch.testing.assert_close(advance(resumed, 2), uninterrupted, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.sum(uninterrupted.abs().square()),
        torch.sum(initial.abs().square()),
        rtol=1e-12, atol=1e-12,
    )
