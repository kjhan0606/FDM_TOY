from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fdm_smbh_delay.torch_wave import (
    advance_binary_rk4,
    apply_kinetic_phase_in_place,
    periodic_poisson_torch,
    plummer_potential_torch,
    sample_potential_and_acceleration,
    spectral_grid,
    wave_energy_components,
)
from fdm_smbh_delay.wave_response import periodic_poisson_code


def test_separable_kinetic_phase_matches_cubic_reference_and_preserves_norm() -> None:
    grid = spectral_grid(
        resolution=8, box_length=4.0, time_step=0.003, device=torch.device("cpu")
    )
    rng = np.random.default_rng(2391)
    initial = torch.as_tensor(
        rng.normal(size=(8, 8, 8)) + 1j * rng.normal(size=(8, 8, 8)),
        dtype=torch.complex128,
    )
    wave_number_squared = grid.kinetic_axis_wavenumber_squared
    full_k_squared = (
        wave_number_squared[:, None, None]
        + wave_number_squared[None, :, None]
        + wave_number_squared[None, None, :]
    )
    original_k = torch.fft.fftn(initial)
    expected = original_k * torch.exp(-0.5j * 0.003 * full_k_squared)
    measured = original_k.clone()
    apply_kinetic_phase_in_place(measured, grid.kinetic_axis_phase)
    assert grid.kinetic_axis_phase.shape == (8,)
    torch.testing.assert_close(measured, expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(
        torch.sum(torch.fft.ifftn(measured).abs().square()),
        torch.sum(initial.abs().square()),
        rtol=1e-13,
        atol=1e-13,
    )


def test_axis_marginal_kinetic_energy_matches_cubic_reference() -> None:
    grid = spectral_grid(
        resolution=8, box_length=4.0, time_step=0.003, device=torch.device("cpu")
    )
    rng = np.random.default_rng(783)
    wavefunction = torch.as_tensor(
        rng.normal(size=(8, 8, 8)) + 1j * rng.normal(size=(8, 8, 8)),
        dtype=torch.complex128,
    )
    density = wavefunction.abs().square()
    potential = torch.zeros_like(density)
    kinetic, self_gravity, interaction, mass = wave_energy_components(
        wavefunction=wavefunction,
        density=density,
        wave_potential=potential,
        compact_potential=potential,
        kinetic_axis_wavenumber_squared=grid.kinetic_axis_wavenumber_squared,
        cell_volume=grid.cell_volume,
    )
    axis = grid.kinetic_axis_wavenumber_squared
    full_k_squared = axis[:, None, None] + axis[None, :, None] + axis[None, None, :]
    expected = (
        0.5 * grid.cell_volume / wavefunction.numel()
        * torch.sum(full_k_squared * torch.fft.fftn(wavefunction).abs().square())
    )
    assert kinetic == pytest.approx(float(expected), rel=1e-13)
    assert self_gravity == 0.0
    assert interaction == 0.0
    assert mass == pytest.approx(float(grid.cell_volume * density.sum()))


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
