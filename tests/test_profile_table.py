from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.fdm_outer_halo import FDMOuterHaloClosure
from fdm_smbh_delay.profile_table import (
    EnvironmentProfileBundle,
    TabulatedSphericalProfile,
)


SOURCE_HASH = "c" * 64


def _profile(*, core_radius_pc: float | None = None) -> TabulatedSphericalProfile:
    return TabulatedSphericalProfile(
        radii_pc=np.array([1.0, 2.0, 4.0, 8.0]),
        density_msun_pc3=np.array([100.0, 60.0, 25.0, 8.0]),
        enclosed_mass_msun=np.array([1.0e4, 2.0e4, 4.0e4, 7.0e4]),
        potential_pc2_myr2=np.array([-10.0, -8.0, -5.0, -3.0]),
        core_radius_pc=core_radius_pc,
    )


def test_profile_interpolates_and_acceleration_has_internal_units() -> None:
    profile = _profile()
    assert profile.density(2.0) == pytest.approx(60.0)
    assert profile.enclosed_mass(4.0) == pytest.approx(4.0e4)
    position = np.array([4.0, 0.0, 0.0])
    acceleration = profile.acceleration(position)
    assert acceleration[0] == pytest.approx(-G_INTERNAL * 4.0e4 / 4.0**2)
    assert np.allclose(acceleration[1:], 0.0)


def test_profile_rejects_outside_support_instead_of_extrapolating() -> None:
    profile = _profile()
    with pytest.raises(ValueError, match="outside tabulated support"):
        profile.density(0.5)
    with pytest.raises(ValueError, match="outside tabulated support"):
        profile.enclosed_mass(8.1)
    with pytest.raises(ValueError, match="outside tabulated support"):
        profile.acceleration(np.array([0.5, 0.0, 0.0]))


def test_profile_bundle_round_trip_preserves_provenance() -> None:
    bundle = EnvironmentProfileBundle(
        event_uid="event-1",
        stellar=_profile(),
        gas=None,
        fdm=_profile(core_radius_pc=2.0),
        source_case_id="zoom-1",
        source_sha256=SOURCE_HASH,
        source_path="profiles.json",
        outer_fdm_closure=FDMOuterHaloClosure(
            radii_pc=np.array([10.0, 30.0]),
            mass_current_msun_pc2_myr=np.zeros((2, 3)),
            coherence_time_myr=np.array([1.0, 2.0]),
            de_broglie_wavelength_pc=np.array([20.0, 30.0]),
            velocity_diffusion_pc2_myr3=np.array([0.0, 0.01]),
            density_gradient_scale_pc=np.array([15.0, 25.0]),
        ),
    )
    decoded = EnvironmentProfileBundle.from_dict(bundle.as_dict())
    assert decoded.event_uid == bundle.event_uid
    assert decoded.source_sha256 == SOURCE_HASH
    assert decoded.profile("stellar").enclosed_mass(4.0) == pytest.approx(4.0e4)
    assert decoded.profile("gas") is None
    assert decoded.outer_fdm_closure is not None


def test_profile_bundle_rejects_invalid_provenance() -> None:
    with pytest.raises(ValueError, match="64 characters"):
        EnvironmentProfileBundle(
            event_uid="event-1",
            stellar=None,
            gas=None,
            fdm=None,
            source_case_id="zoom-1",
            source_sha256="bad",
            source_path="profiles.json",
        )
