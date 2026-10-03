from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.galaxy_environment import CompositePotential, DehnenProfile
from fdm_smbh_delay.host_orbit import (
    project_fdm_response_family_in_host,
    spherical_host_turning_points,
)
from fdm_smbh_delay.fdm_outer_response import FDMOuterResponseTable
from fdm_smbh_delay.fdm_outer_response_family import FDMOuterResponseFamily
from fdm_smbh_delay.profile_table import TabulatedSphericalProfile


def _family() -> FDMOuterResponseFamily:
    table = FDMOuterResponseTable(
        radii_pc=np.array([10.0, 200.0]),
        drift_acceleration_pc_myr2=np.array([[0.0, -1.0, 0.0]] * 2),
        diffusion_tensor_pc2_myr3=np.array([np.eye(3)] * 2),
        response_status="calibrated", component_frame="orbital_rtn",
        diffusion_convention="velocity_covariance_rate",
    )
    return FDMOuterResponseFamily(
        mass_ratios_q=(0.2,), eccentricities=(0.5,),
        tables=((table,),), source_sha256=(("a" * 64,),),
    )


def test_kepler_host_turning_points_include_primary_reflex_mass() -> None:
    primary = 1.0e8
    secondary = 2.0e7
    axis = 100.0
    eccentricity = 0.5
    peri = axis * (1.0 - eccentricity)
    speed = np.sqrt(G_INTERNAL * (primary + secondary) * (1.0 + eccentricity) / peri)
    result = spherical_host_turning_points(
        host_potential=CompositePotential((), central_point_mass_msun=primary),
        position_pc=np.array([peri, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, speed, 0.0]),
        secondary_bh_mass_msun=secondary,
        radial_support_pc=(10.0, 400.0),
    )
    assert result.status == "bound_candidate_static_host"
    assert result.pericentre_pc == pytest.approx(50.0)
    assert result.apocentre_pc == pytest.approx(150.0)
    assert result.eccentricity == pytest.approx(eccentricity)
    mu = G_INTERNAL * (primary + secondary)
    for turning_radius in (result.pericentre_pc, result.apocentre_pc):
        turning_energy = (
            -mu / turning_radius
            + 0.5 * result.specific_angular_momentum_pc2_myr**2 / turning_radius**2
        )
        assert turning_energy == pytest.approx(result.specific_energy_pc2_myr2)


def test_extended_host_circular_orbit_is_not_smbh_only_kepler_eccentricity() -> None:
    host = CompositePotential(
        (DehnenProfile(1.0e10, 500.0, 1.0),),
        central_point_mass_msun=1.0e7,
    )
    secondary = 1.0e7
    radius = 100.0
    speed = np.sqrt(G_INTERNAL * (host.enclosed_mass(radius) + secondary) / radius)
    result = spherical_host_turning_points(
        host_potential=host,
        position_pc=np.array([radius, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, speed, 0.0]),
        secondary_bh_mass_msun=secondary,
        radial_support_pc=(10.0, 1000.0),
    )
    assert result.status == "bound_candidate_static_host"
    assert result.eccentricity == pytest.approx(0.0, abs=1.0e-10)


def test_host_orbit_censors_unbracketed_and_radial_states() -> None:
    host = CompositePotential((), central_point_mass_msun=1.0e8)
    state = dict(
        host_potential=host,
        position_pc=np.array([50.0, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, 20.0, 0.0]),
        secondary_bh_mass_msun=2.0e7,
    )
    assert spherical_host_turning_points(
        **state, radial_support_pc=(40.0, 60.0),
    ).status == "censored"
    assert spherical_host_turning_points(
        **{**state, "velocity_pc_myr": np.array([20.0, 0.0, 0.0])},
        radial_support_pc=(10.0, 1000.0),
    ).status == "censored"


def test_host_orbit_rejects_invalid_support_without_extrapolation() -> None:
    with pytest.raises(ValueError, match="support"):
        spherical_host_turning_points(
            host_potential=CompositePotential((), central_point_mass_msun=1.0e8),
            position_pc=np.array([50.0, 0.0, 0.0]),
            velocity_pc_myr=np.array([0.0, 1.0, 0.0]),
            secondary_bh_mass_msun=2.0e7,
            radial_support_pc=(50.0, 100.0),
        )


def test_host_eccentricity_drives_source_bound_qe_projection() -> None:
    primary = 1.0e8
    secondary = 2.0e7
    peri = 50.0
    speed = np.sqrt(G_INTERNAL * (primary + secondary) * 1.5 / peri)
    result = project_fdm_response_family_in_host(
        family=_family(),
        host_potential=CompositePotential((), central_point_mass_msun=primary),
        position_pc=np.array([peri, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, speed, 0.0]),
        secondary_bh_mass_msun=secondary,
        radial_support_pc=(10.0, 400.0),
    )
    assert result.status == "candidate_pending_zoom_validation"
    assert result.host_orbit.eccentricity == pytest.approx(0.5)
    assert result.projected_response.mass_ratio_q == pytest.approx(0.2)
    assert result.projected_response.drift_acceleration_pc_myr2 == pytest.approx(
        [0.0, -1.0, 0.0]
    )


def test_host_projection_censors_qe_or_turning_point_outside_support() -> None:
    host = CompositePotential((), central_point_mass_msun=1.0e8)
    values = dict(
        family=_family(), host_potential=host,
        position_pc=np.array([50.0, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, 20.0, 0.0]),
        secondary_bh_mass_msun=2.0e7,
    )
    assert project_fdm_response_family_in_host(
        **values, radial_support_pc=(40.0, 60.0),
    ).status == "censored"
    outside_q = project_fdm_response_family_in_host(
        **{**values, "secondary_bh_mass_msun": 3.0e7},
        radial_support_pc=(10.0, 400.0),
    )
    assert outside_q.status == "censored"
    assert outside_q.projected_response is None


def test_host_turning_points_censor_tabulated_profile_boundary() -> None:
    profile = TabulatedSphericalProfile(
        radii_pc=np.array([20.0, 100.0]),
        density_msun_pc3=np.zeros(2),
        enclosed_mass_msun=np.zeros(2),
        potential_pc2_myr2=np.zeros(2),
    )
    host = CompositePotential((profile,), central_point_mass_msun=1.0e8)
    result = spherical_host_turning_points(
        host_potential=host,
        position_pc=np.array([50.0, 0.0, 0.0]),
        velocity_pc_myr=np.array([0.0, 20.0, 0.0]),
        secondary_bh_mass_msun=2.0e7,
        radial_support_pc=(10.0, 200.0),
    )
    assert result.status == "censored"
    assert "outside tabulated support" in result.reason
