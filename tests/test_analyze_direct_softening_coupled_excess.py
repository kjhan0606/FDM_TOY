import numpy as np
import pytest
import torch

from scripts.analyze_direct_softening_coupled_excess import (
    isolated_plummer_kdk, plummer_acceleration,
)
from fdm_smbh_delay.periodic_mesh_coupling import (
    drift_binary_tsc, kick_binary_tsc,
)


def test_pair_acceleration_conserves_linear_momentum():
    state = np.array([
        [-0.4, 0.1, 0.0, 0.0, 0.2, 0.0],
        [0.6, -0.2, 0.0, 0.0, -0.3, 0.0],
    ])
    masses = np.array([3.0, 1.0])
    acceleration = plummer_acceleration(state, masses, 0.1)
    np.testing.assert_allclose(
        np.sum(masses[:, None] * acceleration, axis=0),
        np.zeros(3), atol=1e-15,
    )


def test_isolated_kdk_reverses_and_preserves_centre_of_mass():
    state = np.array([
        [-0.3, 0.0, 0.0, 0.0, 0.4, 0.0],
        [0.9, 0.0, 0.0, 0.0, -1.2, 0.0],
    ])
    masses = np.array([3.0, 1.0])
    history = isolated_plummer_kdk(state, masses, 0.15, 1e-3, 40)
    momentum = np.einsum("i,tij->tj", masses, history[:, :, 3:])
    np.testing.assert_allclose(
        momentum, np.broadcast_to(momentum[0], momentum.shape), atol=2e-14,
    )
    reversed_state = history[-1].copy()
    reversed_state[:, 3:] *= -1
    reverse = isolated_plummer_kdk(reversed_state, masses, 0.15, 1e-3, 40)
    np.testing.assert_allclose(reverse[-1, :, :3], state[:, :3], atol=2e-14)
    np.testing.assert_allclose(reverse[-1, :, 3:], -state[:, 3:], atol=2e-14)


def test_kdk_refinement_and_invalid_inputs():
    state = np.array([
        [-0.25, 0.0, 0.0, 0.0, 0.5, 0.0],
        [0.75, 0.0, 0.0, 0.0, -0.5, 0.0],
    ])
    masses = np.array([1.0, 1.0])
    endpoints = [isolated_plummer_kdk(
        state, masses, 0.1, 0.01 / n, n,
    )[-1] for n in (4, 8, 16)]
    coarse_error = np.linalg.norm(endpoints[0] - endpoints[1])
    fine_error = np.linalg.norm(endpoints[1] - endpoints[2])
    assert 3.5 < coarse_error / fine_error < 4.5
    with pytest.raises(ValueError, match="finite positive"):
        plummer_acceleration(state, masses, 0.0)
    with pytest.raises(ValueError, match="positive"):
        isolated_plummer_kdk(state, masses, 0.1, 0.0, 4)


def test_isolated_kdk_matches_zero_wave_tsc_subflow():
    state = np.array([
        [-0.3, 0.02, 0.0, 0.0, 0.4, 0.0],
        [0.8, -0.01, 0.0, 0.0, -1.2, 0.0],
    ])
    masses = np.array([3.0, 1.0])
    step = 1e-3
    zero_wave = torch.zeros((8, 8, 8), dtype=torch.float64)
    expected = kick_binary_tsc(
        state=state, masses=masses, wave_potential=zero_wave,
        box_length=4.0, plummer_radius=0.15, time_step=step / 2,
        force_scheme="spectral_momentum",
    )
    expected = drift_binary_tsc(
        state=expected, box_length=4.0, time_step=step,
    )
    expected = kick_binary_tsc(
        state=expected, masses=masses, wave_potential=zero_wave,
        box_length=4.0, plummer_radius=0.15, time_step=step / 2,
        force_scheme="spectral_momentum",
    )
    actual = isolated_plummer_kdk(state, masses, 0.15, step, 1)[-1]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-18)
