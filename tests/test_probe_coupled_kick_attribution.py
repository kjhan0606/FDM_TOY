"""Factorized joint-kick replay against the unsplit Strang solver order."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fdm_smbh_delay.periodic_mesh_coupling import (
    drift_binary_tsc, kick_binary_tsc,
)
from fdm_smbh_delay.torch_wave import (
    apply_kinetic_phase_in_place,
    apply_potential_half_kick_in_place,
    periodic_poisson_torch,
    spectral_grid,
    wave_density,
)
from scripts.probe_coupled_kick_attribution import (
    attributed_step, body_momentum, compact_field,
)
from scripts.probe_wave_only_momentum_control import spectral_wave_momentum_torch


def test_factorized_step_matches_unsplit_and_ledger_telescopes() -> None:
    rng = np.random.default_rng(3902)
    wave = torch.as_tensor(
        1.0 + 0.01 * rng.normal(size=(8, 8, 8))
        + 0.01j * rng.normal(size=(8, 8, 8)),
        dtype=torch.complex128,
    )
    body = np.array([
        [-0.45, 0.01, 0.02, 0.01, -0.08, 0.0],
        [0.45, -0.01, -0.02, -0.01, 0.06, 0.0],
    ])
    masses = np.array([0.7, 1.3])
    dt = 0.001
    box = 4.0
    grid = spectral_grid(
        resolution=8, box_length=box, time_step=dt,
        device=torch.device("cpu"),
    )
    wave_potential = periodic_poisson_torch(
        wave_density(wave), grid.poisson_inverse_wavenumber_squared
    )
    compact_potential = compact_field(
        body, masses, grid=grid, box_length=box, device=torch.device("cpu")
    )
    reference_wave = wave.clone()
    reference_body = body.copy()
    apply_potential_half_kick_in_place(
        reference_wave, wave_potential + compact_potential, dt
    )
    reference_body = kick_binary_tsc(
        state=reference_body, masses=masses,
        wave_potential=wave_potential, box_length=box,
        plummer_radius=0.1, time_step=0.5 * dt,
        force_scheme="spectral_momentum",
    )
    spectrum = torch.fft.fftn(reference_wave)
    apply_kinetic_phase_in_place(spectrum, grid.kinetic_axis_phase)
    reference_wave = torch.fft.ifftn(spectrum)
    reference_body = drift_binary_tsc(
        state=reference_body, box_length=box, time_step=dt
    )
    next_wave_potential = periodic_poisson_torch(
        wave_density(reference_wave), grid.poisson_inverse_wavenumber_squared
    )
    next_compact_potential = compact_field(
        reference_body, masses, grid=grid, box_length=box,
        device=torch.device("cpu"),
    )
    apply_potential_half_kick_in_place(
        reference_wave, next_wave_potential + next_compact_potential, dt
    )
    reference_body = kick_binary_tsc(
        state=reference_body, masses=masses,
        wave_potential=next_wave_potential, box_length=box,
        plummer_radius=0.1, time_step=0.5 * dt,
        force_scheme="spectral_momentum",
    )
    before_total = spectral_wave_momentum_torch(wave, box) + body_momentum(
        body, masses
    )
    (actual_wave, actual_body, _, _, changes) = attributed_step(
        wave.clone(), body.copy(), masses, wave_potential, compact_potential,
        grid=grid, box_length=box, plummer_radius=0.1, time_step=dt,
    )
    torch.testing.assert_close(actual_wave, reference_wave, rtol=0, atol=2e-15)
    np.testing.assert_allclose(actual_body, reference_body, rtol=0, atol=1e-15)
    after_total = spectral_wave_momentum_torch(
        actual_wave, box
    ) + body_momentum(actual_body, masses)
    np.testing.assert_allclose(
        sum((changes[key] for key in (
            "wave_self_first", "wave_compact_first", "body_first",
            "wave_drift", "wave_self_second", "wave_compact_second",
            "body_second",
        )), np.zeros(3)),
        after_total - before_total, rtol=0, atol=1e-12,
    )


def test_large_phase_factorization_over_two_kicks() -> None:
    rng = np.random.default_rng(5217)
    wave = torch.as_tensor(
        rng.normal(size=(8, 8, 8)) + 1j * rng.normal(size=(8, 8, 8)),
        dtype=torch.complex128,
    )
    self_potential = torch.as_tensor(
        2.0e4 * rng.normal(size=(8, 8, 8)), dtype=torch.float64,
    )
    compact_potential = torch.as_tensor(
        2.0e4 * rng.normal(size=(8, 8, 8)), dtype=torch.float64,
    )
    dt = 0.001
    unsplit = wave.clone()
    split = wave.clone()
    alternate = wave.clone()
    for _ in range(2):
        apply_potential_half_kick_in_place(
            unsplit, self_potential + compact_potential, dt,
        )
        apply_potential_half_kick_in_place(split, self_potential, dt)
        apply_potential_half_kick_in_place(split, compact_potential, dt)
        apply_potential_half_kick_in_place(
            alternate, compact_potential, dt,
        )
        apply_potential_half_kick_in_place(alternate, self_potential, dt)
    scale = float(torch.max(torch.abs(unsplit)))
    assert 10.0 < 0.5 * dt * float(torch.max(torch.abs(
        self_potential + compact_potential
    ))) < 100.0
    assert float(torch.max(torch.abs(split - unsplit))) / scale < 1e-12
    assert float(torch.max(torch.abs(alternate - unsplit))) / scale < 1e-12
