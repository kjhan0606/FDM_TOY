"""Fourier and compact-state translation gates for a periodic offset seed."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.derive_periodic_offset_seed import (
    shifted_config, shifted_particle_state, shifted_wave,
)


def test_half_cell_fourier_shift_matches_plane_wave_and_mass() -> None:
    n = 8
    coordinates = np.arange(n, dtype=float)
    plane = np.exp(2j * np.pi * coordinates / n)
    wave = torch.as_tensor(
        np.broadcast_to(plane[:, None, None], (n, n, n)).copy(),
        dtype=torch.complex128,
    )
    translated = shifted_wave(wave, axis=0, shift_cells=0.5)
    expected = np.broadcast_to(
        np.exp(2j * np.pi * (coordinates - 0.5)[:, None, None] / n),
        (n, n, n),
    )
    np.testing.assert_allclose(translated.numpy(), expected, rtol=0, atol=1e-15)
    assert abs(float(torch.sum(translated.abs().square()))
               - float(torch.sum(wave.abs().square()))) < 1e-12


def test_whole_cell_shift_is_exact_roll() -> None:
    rng = np.random.default_rng(481)
    wave = torch.as_tensor(
        rng.normal(size=(8, 8, 8)) + 1j * rng.normal(size=(8, 8, 8)),
        dtype=torch.complex128,
    )
    translated = shifted_wave(wave, axis=1, shift_cells=1.0)
    torch.testing.assert_close(translated, torch.roll(wave, 1, dims=1),
                               rtol=0, atol=0)


def test_particle_and_config_translation_keep_physical_units() -> None:
    body = np.array([
        0.1, 0.0, 0.0, 0.0, 2.0, 0.0,
        -0.3, 0.0, 0.0, 0.0, -2.0, 0.0,
    ], dtype=float)
    config = {
        "Matter Particles": {"Condition": [
            [1.0, [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [1.0, [-3.0, 0.0, 0.0], [0.0, -2.0, 0.0]],
        ]},
        "ULDM Solitons": {
            "Condition": [[10.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0]],
            "Embedded": [],
        },
    }
    units = SimpleNamespace(length_pc=10.0)
    shifted = shifted_particle_state(
        body, axis=0, shift_code=0.05, box_code=4.0,
    )
    shifted_input = shifted_config(
        config, axis=0, shift_pc=0.5, original_state=body, units=units,
    )
    np.testing.assert_allclose(
        shifted.reshape(2, 6)[:, :3] * units.length_pc,
        [item[1] for item in shifted_input["Matter Particles"]["Condition"]],
        rtol=0, atol=1e-14,
    )
    assert config["Matter Particles"]["Condition"][0][1][0] == 1.0
    assert shifted_input["ULDM Solitons"]["Condition"][0][1] == [
        0.5, 0.0, 0.0,
    ]


def test_particle_shift_fails_closed_outside_direct_binary_domain() -> None:
    body = np.zeros(12)
    body[0] = 1.9
    with pytest.raises(ValueError, match="leave nonperiodic binary domain"):
        shifted_particle_state(
            body, axis=0, shift_code=0.2, box_code=4.0,
        )
