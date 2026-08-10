import numpy as np
import pytest

from fdm_smbh_delay.interaction import (
    coupled_hamiltonian,
    gauge_align_point_estimator,
    periodic_poisson_potential,
    single_bh_interaction_snapshot,
    wave_bh_interaction_energy,
)


def test_cross_energy_is_counted_once() -> None:
    density = np.full((2, 2, 2), 3.0)
    potential = np.full((2, 2, 2), -4.0)
    assert wave_bh_interaction_energy(density, potential, 0.5) == -48.0
    total = coupled_hamiltonian(
        wave_kinetic=10.0,
        wave_self_gravity=-3.0,
        wave_bh_interaction=-48.0,
        bh_kinetic=7.0,
        bh_mutual_gravity=-2.0,
    )
    assert float(total) == -36.0


def test_point_estimator_gauge_alignment_preserves_changes() -> None:
    grid = np.array([-10.0, -8.0, -3.0])
    point = np.array([-7.0, -5.0, 0.0])
    aligned, offset = gauge_align_point_estimator(point, grid)
    assert offset == -3.0
    np.testing.assert_allclose(aligned, grid)


def test_grid_force_is_negative_energy_gradient() -> None:
    n = 32
    box_size = 16.0
    cell_size = box_size / n
    coordinates = np.linspace(-box_size / 2.0, box_size / 2.0, n, endpoint=False)
    radius_squared = (
        coordinates[:, None, None] ** 2
        + coordinates[None, :, None] ** 2
        + coordinates[None, None, :] ** 2
    )
    density = np.exp(-radius_squared / 2.0)
    wave_potential = periodic_poisson_potential(density, cell_size)

    def snapshot(x_position: float):
        return single_bh_interaction_snapshot(
            density_msun_pc3=density,
            wave_potential_pc2_myr2=wave_potential,
            box_size_pc=box_size,
            bh_mass_msun=1.0e6,
            bh_position_pc=np.array([x_position, 0.0, 0.0]),
            plummer_radius_pc=0.25,
        )

    position = 1.0
    increment = 1.0e-3
    minus = snapshot(position - increment)
    centre = snapshot(position)
    plus = snapshot(position + increment)
    finite_difference_force = -(
        plus.grid_interaction_energy - minus.grid_interaction_energy
    ) / (2.0 * increment)
    assert centre.force_on_bh_from_grid_energy[0] == pytest.approx(
        finite_difference_force, rel=1.0e-5
    )
    assert centre.force_on_bh_from_grid_energy[0] < 0.0
