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


def _state(offset: float = 0.0) -> np.ndarray:
    return np.arange(12, dtype=float).reshape(2, 6) + offset


def _wave(value: float):
    return torch.full(
        (4, 4, 4), complex(value, -value), dtype=torch.complex128
    )


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
