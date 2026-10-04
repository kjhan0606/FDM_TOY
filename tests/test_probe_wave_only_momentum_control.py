"""Wave-only null momentum operator, mass and split-run regression."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fdm_smbh_delay.torch_wave import spectral_grid, wave_density
from scripts.audit_periodic_tsc_momentum import (
    spectral_wave_diagnostics, spectral_wave_momentum,
)
from scripts.probe_wave_only_momentum_control import (
    evolve_wave_only, spectral_wave_diagnostics_torch,
    spectral_wave_momentum_torch,
)


def test_torch_null_momentum_matches_numpy_and_zeros_real_nyquist() -> None:
    rng = np.random.default_rng(2204)
    wave = np.asarray(
        1.0 + 0.03 * rng.normal(size=(8, 8, 8))
        + 0.05j * rng.normal(size=(8, 8, 8)),
        dtype=np.complex128,
    )
    np.testing.assert_allclose(
        spectral_wave_momentum_torch(torch.as_tensor(wave), 4.0),
        spectral_wave_momentum(wave, 4.0), rtol=0, atol=1e-14,
    )
    torch_diagnostic = spectral_wave_diagnostics_torch(
        torch.as_tensor(wave), 4.0, high_shell=True,
    )
    numpy_diagnostic = spectral_wave_diagnostics(wave, 4.0)
    assert torch_diagnostic["summation_rounding_floor_code"] == pytest.approx(
        numpy_diagnostic["summation_rounding_floor_code"], rel=1e-14
    )
    assert torch_diagnostic["high_frequency_power_fraction"] == pytest.approx(
        numpy_diagnostic["high_frequency_power_fraction"], rel=1e-14
    )
    axis = np.arange(8)
    nyquist = np.broadcast_to(
        1.0 + 0.2 * (-1.0) ** axis[:, None, None], wave.shape
    ).astype(np.complex128)
    np.testing.assert_allclose(
        spectral_wave_momentum_torch(torch.as_tensor(nyquist), 4.0),
        0.0, rtol=0, atol=1e-13,
    )


def test_wave_only_split_run_matches_full_and_conserves_mass() -> None:
    rng = np.random.default_rng(1208)
    initial = torch.as_tensor(
        1.0 + 0.03 * rng.normal(size=(8, 8, 8))
        + 0.03j * rng.normal(size=(8, 8, 8)),
        dtype=torch.complex128,
    )
    grid = spectral_grid(
        resolution=8, box_length=4.0, time_step=0.001,
        device=torch.device("cpu"),
    )
    full, full_momentum = evolve_wave_only(
        initial.clone(), grid=grid, time_step=0.001, steps=4,
    )
    partial, first_momentum = evolve_wave_only(
        initial.clone(), grid=grid, time_step=0.001, steps=2,
    )
    resumed, second_momentum = evolve_wave_only(
        partial, grid=grid, time_step=0.001, steps=2,
    )
    torch.testing.assert_close(full, resumed, rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        full_momentum,
        np.concatenate((first_momentum, second_momentum[1:])),
        rtol=0, atol=1e-13,
    )
    assert float(torch.sum(wave_density(full))) == pytest.approx(
        float(torch.sum(wave_density(initial))), rel=1e-12
    )
