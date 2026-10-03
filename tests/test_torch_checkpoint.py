from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import run_torch_wave_case  # noqa: E402
from fdm_smbh_delay.torch_wave import (  # noqa: E402
    apply_kinetic_phase_in_place,
    periodic_poisson_torch,
    spectral_grid,
)


def _state(offset: float = 0.0) -> np.ndarray:
    return np.arange(12, dtype=float).reshape(2, 6) + offset


def _wave(value: float):
    return torch.full(
        (4, 4, 4), complex(value, -value), dtype=torch.complex128
    )


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
            density = wavefunction.abs().square()
            potential = periodic_poisson_torch(
                density, grid.poisson_inverse_wavenumber_squared
            )
            wavefunction.mul_(torch.exp(-0.5j * 0.001 * potential))
            spectrum = torch.fft.fftn(wavefunction)
            apply_kinetic_phase_in_place(spectrum, grid.kinetic_axis_phase)
            wavefunction = torch.fft.ifftn(spectrum)
            density = wavefunction.abs().square()
            potential = periodic_poisson_torch(
                density, grid.poisson_inverse_wavenumber_squared
            )
            wavefunction.mul_(torch.exp(-0.5j * 0.001 * potential))
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
