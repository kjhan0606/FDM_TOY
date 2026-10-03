from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fdm_smbh_delay.torch_wave import (
    apply_potential_half_kick_in_place,
    advance_binary_rk4,
    apply_kinetic_phase_in_place,
    periodic_poisson_torch,
    plummer_potential_torch,
    sample_potential_and_acceleration,
    spectral_grid,
    wave_density,
    wave_energy_components,
)
from fdm_smbh_delay.wave_response import periodic_poisson_code


def test_wave_density_matches_complex_magnitude_without_mutating_input() -> None:
    rng = np.random.default_rng(2826)
    original = torch.as_tensor(
        rng.normal(size=(8, 8, 8)) + 1j * rng.normal(size=(8, 8, 8)),
        dtype=torch.complex128,
    )
    before = original.clone()
    measured = wave_density(original)
    expected = original.abs().square()
    torch.testing.assert_close(measured, expected, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(original, before, rtol=0, atol=0)
    assert measured.dtype == torch.float64
    assert measured.data_ptr() != original.data_ptr()
    with pytest.raises(ValueError, match="complex dtype"):
        wave_density(torch.ones(3, dtype=torch.float64))


def test_potential_half_kick_matches_complex_exponential_and_preserves_norm() -> None:
    rng = np.random.default_rng(2827)
    initial = torch.as_tensor(
        rng.normal(size=(8, 8, 8)) + 1j * rng.normal(size=(8, 8, 8)),
        dtype=torch.complex128,
    )
    potential = torch.as_tensor(rng.normal(size=(8, 8, 8)))
    potential_before = potential.clone()
    expected = initial * torch.exp(-0.5j * 0.003 * potential)
    measured = initial.clone()
    apply_potential_half_kick_in_place(measured, potential, 0.003)
    torch.testing.assert_close(measured, expected, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(potential, potential_before, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.sum(measured.abs().square()), torch.sum(initial.abs().square()),
        rtol=2e-15, atol=2e-15,
    )
    with pytest.raises(ValueError, match="incompatible"):
        apply_potential_half_kick_in_place(measured, potential, -0.003)


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


def test_torch_plummer_slabs_match_full_two_mass_reference() -> None:
    grid = spectral_grid(
        resolution=40, box_length=8.0, time_step=1.0e-3,
        device=torch.device("cpu"),
    )
    masses = np.array([2.0, 0.7])
    positions = np.array([[0.13, -0.21, 0.42], [-0.35, 0.12, -0.25]])
    measured = plummer_potential_torch(
        coordinate=grid.coordinate, masses=masses, positions=positions,
        plummer_radius=0.07,
    )
    x, y, z = np.meshgrid(
        grid.coordinate.numpy(), grid.coordinate.numpy(),
        grid.coordinate.numpy(), indexing="ij",
    )
    expected = np.zeros((40, 40, 40))
    for mass, position in zip(masses, positions, strict=True):
        radius_squared = (
            (x - position[0]) ** 2 + (y - position[1]) ** 2
            + (z - position[2]) ** 2 + 0.07**2
        )
        expected -= mass / np.sqrt(radius_squared)
    np.testing.assert_allclose(measured.numpy(), expected, rtol=2e-14, atol=2e-14)


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
