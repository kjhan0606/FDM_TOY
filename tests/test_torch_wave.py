from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fdm_smbh_delay.torch_wave import (
    advance_binary_rk4,
    periodic_poisson_torch,
    plummer_potential_torch,
    sample_potential_and_acceleration,
    spectral_grid,
)
from fdm_smbh_delay.wave_response import periodic_poisson_code


def test_torch_poisson_matches_numpy_spectral_solver() -> None:
    rng = np.random.default_rng(1341)
    density = rng.random((16, 16, 16))
    grid = spectral_grid(
        resolution=16,
        box_length=8.0,
        time_step=1.0e-3,
        device=torch.device("cpu"),
    )
    measured = periodic_poisson_torch(
        torch.as_tensor(density), grid.poisson_inverse_wavenumber_squared
    ).numpy()
    expected = periodic_poisson_code(density, 8.0)
    np.testing.assert_allclose(measured, expected, rtol=2.0e-14, atol=2.0e-14)


def test_torch_plummer_potential_has_expected_central_value() -> None:
    grid = spectral_grid(
        resolution=16,
        box_length=8.0,
        time_step=1.0e-3,
        device=torch.device("cpu"),
    )
    potential = plummer_potential_torch(
        coordinate=grid.coordinate,
        masses=np.array([2.0]),
        positions=np.zeros((1, 3)),
        plummer_radius=0.1,
    )
    assert float(potential[8, 8, 8]) == pytest.approx(-20.0)


def test_field_sampler_recovers_a_linear_acceleration() -> None:
    grid = spectral_grid(
        resolution=16,
        box_length=8.0,
        time_step=1.0e-3,
        device=torch.device("cpu"),
    )
    x = grid.coordinate[:, None, None]
    y = grid.coordinate[None, :, None]
    z = grid.coordinate[None, None, :]
    potential = x + 2.0 * y + 3.0 * z
    values, acceleration = sample_potential_and_acceleration(
        potential=potential,
        positions=np.array([[0.25, -0.25, 0.5]]),
        box_length=8.0,
    )
    assert values[0] == pytest.approx(1.25)
    np.testing.assert_allclose(acceleration[0], [-1.0, -2.0, -3.0])


def test_binary_rk4_preserves_centre_of_mass_symmetry() -> None:
    state = np.array(
        [[-0.5, 0.0, 0.0, 0.0, -0.5, 0.0],
         [0.5, 0.0, 0.0, 0.0, 0.5, 0.0]]
    )
    advanced = advance_binary_rk4(
        state=state,
        masses=np.array([1.0, 1.0]),
        external_acceleration=np.zeros((2, 3)),
        plummer_radius=0.05,
        time_step=1.0e-3,
        substeps=2,
    ).reshape(2, 6)
    np.testing.assert_allclose(advanced[0], -advanced[1], atol=2.0e-15)
