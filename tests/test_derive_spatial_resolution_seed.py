import numpy as np
import pytest

from scripts.derive_spatial_resolution_seed import (
    derive_metadata_and_config, relative_wave_mass_change,
    spectral_resample_wave,
)


@pytest.mark.parametrize("resolution", [192, 384])
def test_spectral_resample_preserves_low_periodic_modes(resolution):
    x = np.arange(256, dtype=float) / 256
    wave = np.exp(2j * np.pi * 7 * x)[:, None, None]
    wave = wave * np.exp(-2j * np.pi * 9 * x)[None, :, None]
    wave = np.broadcast_to(wave, (256, 256, 256))
    changed = spectral_resample_wave(wave, resolution)
    new = np.arange(resolution, dtype=float) / resolution
    expected = (
        np.exp(2j * np.pi * 7 * new)[:, None, None]
        * np.exp(-2j * np.pi * 9 * new)[None, :, None]
    )
    for index in (0, resolution // 3, resolution - 1):
        np.testing.assert_allclose(
            changed[index], np.broadcast_to(expected[index], (resolution, resolution)),
            rtol=0, atol=1e-12,
        )
    assert abs(relative_wave_mass_change(wave, changed)) < 1e-12


def test_invalid_parent_rejected():
    config = {
        "Spatial Resolution": 256,
        "Simulation Box": {"Box Length": 26.4, "Length Units": "pc"},
        "Matter Particles": {"Plummer Radius": 0.0515625, "Position Units": "pc"},
    }
    metadata = {
        "case_id": "qe_q030_e030_a020", "resolution": 256,
        "analytic_fdm_drag": False, "box_size_pc": 26.4,
        "cell_size_pc": 26.4 / 256, "plummer_radius_pc": 0.0515625,
        "qe_design_binding": {"status": "old_design"},
    }
    changed, new_config = derive_metadata_and_config(
        metadata, config, resolution=192, output_name="test",
        parent="/some/parent", parent_sha256={},
        source_commit="0" * 40, source_sha256="1" * 64,
    )
    assert changed["resolution"] == new_config["Spatial Resolution"] == 192
    assert changed["plummer_radius_pc"] == 0.0515625
    assert "qe_design_binding" not in changed
    assert changed["spatial_resolution_binding"]["calibration_eligible"] is False
    metadata["cell_size_pc"] *= 2
    with pytest.raises(ValueError, match="inconsistent"):
        derive_metadata_and_config(
            metadata, config, resolution=192, output_name="test",
            parent="/some/parent", parent_sha256={},
            source_commit="0" * 40, source_sha256="1" * 64,
        )
