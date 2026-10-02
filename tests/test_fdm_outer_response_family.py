from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.fdm_orbital_response import project_fdm_response_family_to_orbit
from fdm_smbh_delay.fdm_outer_response import FDMOuterResponseTable
from fdm_smbh_delay.fdm_outer_response_family import FDMOuterResponseFamily


def _table(value: float, *, status: str = "calibrated", low: float = 10.0) -> FDMOuterResponseTable:
    vector = np.array([value, -value, 0.0])
    # Rank-one endpoints rotate their null spaces; their convex mixture must
    # remain PSD even though element-wise interpolation need not.
    direction = np.array([value, 1.0, 0.0])
    tensor = np.outer(direction, direction)
    return FDMOuterResponseTable(
        radii_pc=np.array([low, 20.0]),
        drift_acceleration_pc_myr2=np.array([vector, vector]),
        diffusion_tensor_pc2_myr3=np.array([tensor, tensor]),
        response_status=status,
        component_frame="orbital_rtn",
    )


def _family(**replacements: object) -> FDMOuterResponseFamily:
    fields = dict(
        mass_ratios_q=(0.1, 0.5),
        eccentricities=(0.0, 0.8),
        tables=((_table(1.0), _table(2.0)), (_table(3.0), _table(4.0))),
        source_sha256=(("a" * 64, "b" * 64), ("c" * 64, "d" * 64)),
    )
    fields.update(replacements)
    return FDMOuterResponseFamily(**fields)


def test_qe_family_interpolates_drift_and_psd_diffusion_convexly() -> None:
    family = _family()
    response = family.decision(0.3, 0.4, 15.0)
    assert response["status"] == "interpolated_candidate_pending_physical_validation"
    assert response["drift_acceleration_pc_myr2"] == pytest.approx([2.5, -2.5, 0.0])
    expected = sum(
        0.25 * table.evaluate(15.0)["diffusion_tensor_pc2_myr3"]
        for row in family.tables for table in row
    )
    assert response["diffusion_tensor_pc2_myr3"] == pytest.approx(expected)
    assert np.min(np.linalg.eigvalsh(response["diffusion_tensor_pc2_myr3"])) >= -1.0e-12
    assert response["source_sha256"] == ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def test_qe_family_censors_extrapolation_missing_corner_and_radial_gap() -> None:
    family = _family()
    assert family.decision(0.05, 0.4, 15.0)["status"] == "censored"
    assert family.decision(0.3, 0.9, 15.0)["status"] == "censored"
    assert family.decision(0.3, 0.4, 9.0)["status"] == "censored"
    bad = _family(tables=((_table(1.0), _table(2.0)),
                          (_table(3.0), _table(4.0, status="uncalibrated"))))
    assert bad.decision(0.3, 0.4, 15.0)["status"] == "censored"
    assert bad.decision(0.1, 0.0, 15.0)["status"] != "censored"
    gap = _family(tables=((_table(1.0), _table(2.0)),
                          (_table(3.0), _table(4.0, low=16.0))))
    assert gap.decision(0.3, 0.4, 15.0)["status"] == "censored"


def test_qe_family_orbital_projection_and_radial_censor() -> None:
    family = _family()
    first = project_fdm_response_family_to_orbit(
        family, position_pc=np.array([15.0, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, 1.0, 0.0]),
        mass_ratio_q=0.3, eccentricity=0.4,
    )
    assert first.status == "projected_candidate_pending_physical_validation"
    assert first.drift_acceleration_pc_myr2 == pytest.approx([2.5, -2.5, 0.0])
    assert first.source_sha256 == ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    turned = project_fdm_response_family_to_orbit(
        family, position_pc=rotation @ np.array([15.0, 0.0, 0.0]),
        velocity_pc_myr=rotation @ np.array([0.0, 1.0, 0.0]),
        mass_ratio_q=0.3, eccentricity=0.4,
    )
    assert turned.drift_acceleration_pc_myr2 == pytest.approx(
        rotation @ first.drift_acceleration_pc_myr2
    )
    assert turned.diffusion_tensor_pc2_myr3 == pytest.approx(
        rotation @ first.diffusion_tensor_pc2_myr3 @ rotation.T
    )
    radial = project_fdm_response_family_to_orbit(
        family, position_pc=np.array([15.0, 0.0, 0.0]),
        velocity_pc_myr=np.array([1.0, 0.0, 0.0]),
        mass_ratio_q=0.3, eccentricity=0.4,
    )
    assert radial.status == "censored"
    assert radial.drift_acceleration_pc_myr2 is None


def test_qe_family_snapshots_mutable_grid_containers() -> None:
    tables = [[_table(1.0), _table(2.0)], [_table(3.0), _table(4.0)]]
    source_hashes = [["a" * 64, "b" * 64], ["c" * 64, "d" * 64]]
    family = _family(tables=tables, source_sha256=source_hashes)
    before = family.decision(0.3, 0.4, 15.0)
    tables[0][0] = _table(100.0)
    source_hashes[0][0] = "f" * 64
    after = family.decision(0.3, 0.4, 15.0)
    assert after["drift_acceleration_pc_myr2"] == pytest.approx(
        before["drift_acceleration_pc_myr2"]
    )
    assert after["source_sha256"] == before["source_sha256"]


@pytest.mark.parametrize("replacement", [
    {"mass_ratios_q": (0.5, 0.1)},
    {"eccentricities": (0.0, 1.0)},
    {"source_sha256": (("not-a-hash", "b" * 64), ("c" * 64, "d" * 64))},
    {"tables": ((_table(1.0),), (_table(3.0),))},
])
def test_qe_family_rejects_invalid_grid_or_source(replacement: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="q/e|SHA-256"):
        _family(**replacement)
