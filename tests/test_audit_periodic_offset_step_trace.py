"""Slabwise wave and SMBH translation checks for the offset audit."""

import numpy as np
import pytest

from scripts.audit_periodic_offset_step_trace import (
    compare_bodies, max_rolled_wave_difference,
)


def test_slabwise_whole_cell_wave_translation(tmp_path) -> None:
    rng = np.random.default_rng(618)
    wave = rng.normal(size=(16, 16, 16)) + 1j * rng.normal(size=(16, 16, 16))
    first = tmp_path / "base.npy"
    second = tmp_path / "whole.npy"
    np.save(first, wave)
    np.save(second, np.roll(wave, 1, axis=0))
    absolute, relative = max_rolled_wave_difference(first, second)
    assert absolute == 0.0
    assert relative == 0.0
    shifted = np.roll(wave, 1, axis=0)
    shifted[2, 3, 4] += 1e-4
    np.save(second, shifted)
    absolute, relative = max_rolled_wave_difference(first, second)
    assert absolute == pytest.approx(1e-4)
    assert relative > 0.0


def test_whole_and_half_cell_body_offsets(tmp_path) -> None:
    roots = {label: tmp_path / label for label in ("base", "whole", "half")}
    for root in roots.values():
        (root / "Outputs/NBody").mkdir(parents=True)
    base = np.array([
        -0.2, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.2, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=float).reshape(2, 6)
    for index in range(2):
        states = {name: base.copy() for name in roots}
        states["whole"][:, 0] += 0.1
        states["half"][:, 0] += 0.05
        states["half"][1, 0] += 1e-4 * index
        for label, root in roots.items():
            np.save(root / f"Outputs/NBody/NTM_#{index:03d}.npy",
                    states[label].reshape(12))
    result = compare_bodies(
        roots["base"], roots["whole"], roots["half"], stop=1,
        whole_shift_code=0.1, length_pc=2.0,
    )
    assert result["maximum_whole_cell_body_difference_code"] < 1e-15
    assert result["endpoint_half_minus_base_separation_pc"] == pytest.approx(
        2e-4
    )
