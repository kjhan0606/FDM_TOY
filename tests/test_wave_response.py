import numpy as np
import pytest

from fdm_smbh_delay.wave_response import (
    centred_grid,
    multipole_amplitudes,
    periodic_centre_of_mass,
    periodic_poisson_code,
    plummer_potential_code,
    spectral_wave_fields,
)


def test_periodic_centre_and_spherical_modes() -> None:
    n = 64
    box = 12.0
    coordinate = np.linspace(-box / 2.0, box / 2.0, n, endpoint=False)
    target = np.array([1.0, -0.5, 0.5])
    dx = coordinate[:, None, None] - target[0]
    dy = coordinate[None, :, None] - target[1]
    dz = coordinate[None, None, :] - target[2]
    density = np.exp(-(dx * dx + dy * dy + dz * dz))
    centre = periodic_centre_of_mass(density, box)
    np.testing.assert_allclose(centre, target, atol=1.0e-5)
    x, y, z, radius = centred_grid(n, box, centre)
    modes = multipole_amplitudes(
        density,
        x,
        y,
        z,
        radius,
        radius < 3.0,
        (box / n) ** 3,
    )
    assert modes.l1_fraction < 1.0e-3
    assert modes.l2_fraction < 1.0e-3


def test_spectral_plane_wave_currents() -> None:
    n = 16
    box = 2.0 * np.pi
    coordinate = np.linspace(-box / 2.0, box / 2.0, n, endpoint=False)
    wavefunction = np.exp(2j * coordinate[:, None, None]) * np.ones((1, n, n))
    fields = spectral_wave_fields(wavefunction, np.zeros((n, n, n)), box)
    np.testing.assert_allclose(fields.density, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(fields.mass_current[0], 2.0, atol=1.0e-11)
    np.testing.assert_allclose(fields.mass_current[1], 0.0, atol=1.0e-11)
    np.testing.assert_allclose(fields.kinetic_energy_density, 2.0, atol=1.0e-11)
    np.testing.assert_allclose(
        fields.schrodinger_energy_current[0], 4.0, atol=1.0e-10
    )


def test_code_potentials_have_expected_conventions() -> None:
    density = np.zeros((16, 16, 16))
    density[8, 8, 8] = 1.0
    wave_potential = periodic_poisson_code(density, 8.0)
    assert np.mean(wave_potential) == pytest.approx(0.0, abs=1.0e-15)
    bh_potential = plummer_potential_code(
        shape=density.shape,
        box_size=8.0,
        masses=np.array([2.0]),
        positions=np.zeros((1, 3)),
        plummer_radius=0.1,
    )
    assert np.all(bh_potential < 0.0)
    assert bh_potential[8, 8, 8] == pytest.approx(-20.0)
