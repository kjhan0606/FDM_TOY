import numpy as np
import pytest

from fdm_smbh_delay.wave_response import (
    MultipoleAmplitudes,
    centred_grid,
    multipole_amplitudes,
    periodic_centre_of_mass,
    periodic_point_centre,
    periodic_poisson_code,
    plummer_potential_code,
    rotate_multipoles_to_frame,
    spectral_wave_fields,
    windowed_dominant_frequency,
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
    assert modes.l1_fraction**2 == pytest.approx(
        abs(modes.l1_m0) ** 2 + 2.0 * abs(modes.l1_m1) ** 2
    )
    assert modes.l2_fraction**2 == pytest.approx(
        abs(modes.l2_m0) ** 2
        + 2.0 * abs(modes.l2_m1) ** 2
        + 2.0 * abs(modes.l2_m2) ** 2
    )


def test_density_multipoles_retain_dipole_phase() -> None:
    n = 48
    box = 10.0
    centre = np.zeros(3)
    x, y, z, radius = centred_grid(n, box, centre)
    safe_radius = np.where(radius > 0.0, radius, 1.0)
    spherical = np.exp(-radius**2)
    selection = radius < 3.0
    x_dipole = multipole_amplitudes(
        spherical * (1.0 + 0.2 * x / safe_radius),
        x,
        y,
        z,
        radius,
        selection,
        (box / n) ** 3,
    )
    y_dipole = multipole_amplitudes(
        spherical * (1.0 + 0.2 * y / safe_radius),
        x,
        y,
        z,
        radius,
        selection,
        (box / n) ** 3,
    )
    assert abs(x_dipole.l1_m1.real) > 100.0 * abs(x_dipole.l1_m1.imag)
    assert abs(y_dipole.l1_m1.imag) > 100.0 * abs(y_dipole.l1_m1.real)
    assert abs(x_dipole.l1_m1) == pytest.approx(abs(y_dipole.l1_m1), rel=1.0e-3)


def test_multipole_rotation_preserves_each_invariant_amplitude() -> None:
    multipoles = MultipoleAmplitudes(
        mass=2.0,
        l1_fraction=np.nan,
        l2_fraction=np.nan,
        l1_m0=complex(0.17),
        l1_m1=complex(-0.21, 0.08),
        l2_m0=complex(0.11),
        l2_m1=complex(-0.04, 0.09),
        l2_m2=complex(0.06, -0.12),
    )
    angle = 0.63
    radial = np.array([np.cos(angle), np.sin(angle), 0.0])
    tangential = np.array([-np.sin(angle), np.cos(angle), 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    rotated = rotate_multipoles_to_frame(
        multipoles, radial, tangential, normal
    )
    original_l1 = np.sqrt(
        abs(multipoles.l1_m0) ** 2 + 2.0 * abs(multipoles.l1_m1) ** 2
    )
    original_l2 = np.sqrt(
        abs(multipoles.l2_m0) ** 2
        + 2.0 * abs(multipoles.l2_m1) ** 2
        + 2.0 * abs(multipoles.l2_m2) ** 2
    )
    assert rotated.l1_fraction == pytest.approx(original_l1)
    assert rotated.l2_fraction == pytest.approx(original_l2)


def test_radial_dipole_is_real_negative_m1_in_orbital_frame() -> None:
    radial = np.array([0.0, 1.0, 0.0])
    tangential = np.array([-1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    global_multipoles = MultipoleAmplitudes(
        mass=1.0,
        l1_fraction=1.0,
        l2_fraction=0.0,
        l1_m0=0.0j,
        l1_m1=complex(0.0, np.sqrt(3.0 / 2.0)),
        l2_m0=0.0j,
        l2_m1=0.0j,
        l2_m2=0.0j,
    )
    rotated = rotate_multipoles_to_frame(
        global_multipoles, radial, tangential, normal
    )
    assert rotated.l1_m0 == pytest.approx(0.0j)
    assert rotated.l1_m1 == pytest.approx(complex(-np.sqrt(3.0 / 2.0)))


def test_periodic_point_centre_crosses_box_boundary() -> None:
    centre = periodic_point_centre(
        np.array([[4.8, 1.0, 0.0], [-4.8, 1.0, 0.0]]),
        np.array([1.0, 1.0]),
        10.0,
    )
    assert abs(abs(centre[0]) - 5.0) < 1.0e-12
    assert centre[1] == pytest.approx(1.0)
    assert centre[2] == pytest.approx(0.0)


def test_windowed_frequency_recovers_a_detrended_sinusoid() -> None:
    time = np.linspace(0.0, 2.0, 2049)
    signal = 3.0 + 0.2 * time + np.sin(2.0 * np.pi * 7.0 * time)
    peak = windowed_dominant_frequency(time, signal)
    assert peak.frequency_inverse_time == pytest.approx(7.0, rel=1.0e-3)
    assert peak.period_time == pytest.approx(1.0 / 7.0, rel=1.0e-3)
    assert peak.peak_power_fraction > 0.6


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
