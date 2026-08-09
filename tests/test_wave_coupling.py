from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.wave_coupling import build_wave_source_fields, normalized_gaussian_kernel


def _grid() -> np.ndarray:
    coordinates = np.arange(-5.0, 6.0)
    return np.stack(np.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), axis=-1)


def test_kernel_has_exact_discrete_normalization() -> None:
    kernel = normalized_gaussian_kernel(_grid(), np.zeros(3), 1.2, 1.0)
    assert np.sum(kernel) == pytest.approx(1.0, abs=2.0e-16)


def test_deposited_energy_and_momentum_close_exchange() -> None:
    positions = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 3.0, 0.0], [0.0, -2.0, 0.0]])
    forces = np.array([[0.0, -4.0, 0.0], [0.0, 5.0, 0.0]])
    bulk = np.array([0.0, 1.0, 0.0])
    fields = build_wave_source_fields(
        grid_positions_pc=_grid(),
        cell_volume_pc3=1.0,
        bh_positions_pc=positions,
        bh_velocities_pc_myr=velocities,
        forces_on_bh_msun_pc_myr2=forces,
        fdm_bulk_velocities_pc_myr=bulk,
        sigma_pc=1.0,
    )
    assert np.allclose(np.sum(fields.momentum_density_rate, axis=(0, 1, 2)), -np.sum(forces, axis=0))
    assert np.sum(fields.energy_lab_density_rate) == -np.sum(forces * velocities)
    assert np.sum(fields.excitation_energy_density_rate) == -np.sum(forces * (velocities - bulk))
