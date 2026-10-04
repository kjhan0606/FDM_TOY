import copy

import pytest

from scripts.derive_orbit_prefix_seed import (
    PLANNED_DURATION_MYR, derive_metadata,
)


def test_orbit_seed_retains_physical_inputs_and_is_not_calibration():
    metadata = {
        "case_id": "qe_q030_e030_a020", "resolution": 256,
        "analytic_fdm_drag": False, "semi_major_axis_pc": 0.44,
        "plummer_radius_pc": 0.0515625,
        "pyul_length_unit_m": 3.7439093785040434e20,
        "pyul_time_unit_s": 2.369427842304837e18,
        "pyul_mass_unit_kg": 1.4014034281125392e35,
        "pyul_energy_unit_j": 3.498862348523442e39,
        "qe_design_binding": {
            "status": "qe_prospective_design_bound_not_a_calibration_release"
        },
    }
    config = {
        "Spatial Resolution": 256,
        "Matter Particles": {
            "Position Units": "pc", "Plummer Radius": 0.0515625,
            "Condition": [[76973140.82894911], [23091942.24868473]],
        },
    }
    original = copy.deepcopy(metadata)
    changed = derive_metadata(
        metadata, config, parent="/parent", parent_sha256={},
        output_name="diagnostic", source_commit="0" * 40,
        source_sha256="1" * 64,
    )
    assert metadata == original
    assert changed["run_id"] == "diagnostic"
    assert changed["plummer_radius_pc"] == metadata["plummer_radius_pc"]
    assert "qe_design_binding" not in changed
    binding = changed["orbit_prefix_binding"]
    assert binding["calibration_eligible"] is False
    assert 1 < PLANNED_DURATION_MYR / binding["initial_kepler_period_myr"] < 1.5
    changed_parent = copy.deepcopy(metadata)
    changed_parent["analytic_fdm_drag"] = True
    with pytest.raises(ValueError, match="registered"):
        derive_metadata(
            changed_parent, config, parent="/parent", parent_sha256={},
            output_name="diagnostic", source_commit="0" * 40,
            source_sha256="1" * 64,
        )
