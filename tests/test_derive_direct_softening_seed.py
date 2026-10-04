"""Units and single-operator scope for direct SMBH–SMBH softening seeds."""

from pathlib import Path

import pytest

from scripts.derive_direct_softening_seed import derive_config_and_metadata


def _parent() -> tuple[dict, dict]:
    cell_pc = 26.4 / 256
    config = {
        "Matter Particles": {
            "Plummer Radius": 0.5 * cell_pc,
            "Condition": [
                [7.0, [0.132, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [3.0, [-0.44, 0.0, 0.0], [0.0, -1.0, 0.0]],
            ],
        },
        "ULDM Solitons": {"Condition": [[1.0, [0.0, 0.0, 0.0]]]},
    }
    metadata = {
        "case_id": "qe_q030_e030_a020",
        "resolution": 256,
        "box_size_pc": 26.4,
        "plummer_radius_pc": 0.5 * cell_pc,
        "analytic_fdm_drag": False,
        "run_id": "original",
    }
    return config, metadata


@pytest.mark.parametrize("fraction", (0.25, 1.0))
def test_only_direct_plummer_radius_changes(fraction: float) -> None:
    config, metadata = _parent()
    changed, changed_metadata, radius = derive_config_and_metadata(
        config, metadata, cell_fraction=fraction, output_name="derived",
        parent=Path("/scratch/parent"), parent_hashes={"wave": "hash"},
        source_commit="a" * 40, source_hash="source-hash",
    )
    assert radius == pytest.approx(fraction * 26.4 / 256)
    assert changed["Matter Particles"]["Plummer Radius"] == radius
    assert changed["Matter Particles"]["Condition"] == config[
        "Matter Particles"
    ]["Condition"]
    assert changed["ULDM Solitons"] == config["ULDM Solitons"]
    assert changed_metadata["plummer_radius_pc"] == radius
    assert changed_metadata["direct_softening_binding"]["changed_operator"] == (
        "direct_smbh_smbh_plummer_only"
    )
    assert metadata["run_id"] == "original"
    assert config["Matter Particles"]["Plummer Radius"] == 0.5 * 26.4 / 256


def test_unregistered_radius_fails_closed() -> None:
    config, metadata = _parent()
    with pytest.raises(ValueError, match="quarter- and one-cell"):
        derive_config_and_metadata(
            config, metadata, cell_fraction=0.75, output_name="derived",
            parent=Path("/scratch/parent"), parent_hashes={},
            source_commit="a" * 40, source_hash="source-hash",
        )
