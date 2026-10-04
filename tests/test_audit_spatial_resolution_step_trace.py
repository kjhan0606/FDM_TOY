import copy

import pytest

from scripts.audit_spatial_resolution_step_trace import validate_fixed_physics


def test_fixed_physics_accepts_only_grid_change():
    config = {
        "Spatial Resolution": 256,
        "Matter Particles": {"Plummer Radius": 0.0515625},
    }
    metadata = {
        "box_size_pc": 26.4,
        "cell_size_pc": 26.4 / 256,
        "plummer_radius_pc": 0.0515625,
        "wave_time_step_code": 2e-12,
    }
    changed_config = copy.deepcopy(config)
    changed_config["Spatial Resolution"] = 384
    changed_metadata = copy.deepcopy(metadata)
    changed_metadata["cell_size_pc"] = 26.4 / 384
    validate_fixed_physics(
        metadata, config, changed_metadata, changed_config, resolution=384,
    )
    changed_config["Matter Particles"]["Plummer Radius"] *= 0.75
    with pytest.raises(ValueError, match="changes physics"):
        validate_fixed_physics(
            metadata, config, changed_metadata, changed_config, resolution=384,
        )
    changed_config["Matter Particles"]["Plummer Radius"] = 0.0515625
    changed_metadata["wave_time_step_code"] *= 2
    with pytest.raises(ValueError, match="changes physics"):
        validate_fixed_physics(
            metadata, config, changed_metadata, changed_config, resolution=384,
        )
