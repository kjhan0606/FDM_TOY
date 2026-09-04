from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.bridge_adapter import (
    BridgeMemberAssignment,
    build_kpc_model_from_profile_bundle,
    initial_kpc_to_hard_state_from_bridge,
)
from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.galaxy_environment import (
    CompositePotential,
    FDMBackground,
    StellarBackground,
)
from fdm_smbh_delay.kpc_inspiral import (
    KpcInspiralModel,
    KpcToHardConfig,
    StellarFrictionConfig,
)
from fdm_smbh_delay.nuclear_bridge import (
    BridgeStatus,
    EnvironmentChannel,
    EnvironmentSnapshot,
    NuclearBridgeInput,
)
from fdm_smbh_delay.profile_table import (
    EnvironmentProfileBundle,
    TabulatedSphericalProfile,
)
from fdm_smbh_delay.soliton import SchiveSoliton
from fdm_smbh_delay.lagramses import pair_orbital_state


def _bridge(*, stellar_status: str = "available", fdm_mode: str = "analytic_unresolved"):
    separation = 2.0
    speed = np.sqrt(G_INTERNAL * 2.0e8 / separation)
    pair = pair_orbital_state(
        member_ids=(11, 22),
        masses_msun=(1.0e8, 1.0e8),
        positions_pc=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        velocities_pc_myr=np.array([[0.0, speed / 2.0, 0.0], [0.0, -speed / 2.0, 0.0]]),
    )
    stellar = EnvironmentChannel(
        "stellar",
        stellar_status,
        density_msun_pc3=1.0e4 if stellar_status == "available" else None,
        enclosed_mass_msun=1.0e8 if stellar_status == "available" else None,
        bulk_velocity_pc_myr=np.zeros(3) if stellar_status == "available" else None,
        velocity_dispersion_pc_myr=100.0 if stellar_status == "available" else None,
        reason=None if stellar_status == "available" else "stellar snapshot missing",
    )
    gas = EnvironmentChannel(
        "gas",
        "absent",
        density_msun_pc3=0.0,
        enclosed_mass_msun=0.0,
        bulk_velocity_pc_myr=np.zeros(3),
        reason="no gas component",
    )
    fdm = EnvironmentChannel(
        "fdm",
        "available",
        density_msun_pc3=5.0e6,
        enclosed_mass_msun=1.0e9,
        bulk_velocity_pc_myr=np.zeros(3),
        core_radius_pc=2.3,
        fdm_mode=fdm_mode,
        resolved_wake=fdm_mode == "live_resolved",
    )
    environment = EnvironmentSnapshot(
        event_uid="event-1",
        time_myr=1000.0,
        redshift=1.0,
        radius_pc=200.0,
        channels=(stellar, gas, fdm),
        source_case_id="case-1",
        source_sha256="b" * 64,
        source_path="environment.json",
    )
    return NuclearBridgeInput(
        event_uid="event-1",
        run_id="run-1",
        capture_time_myr=1000.0,
        redshift=1.0,
        pair=pair,
        environment=environment,
        target_semimajor_axis_pc=1.0,
        source_path="capture.jsonl",
        source_sha256="a" * 64,
    )


def _model(*, fdm_mode: str = "analytic_unresolved", with_stars: bool = True):
    soliton = SchiveSoliton.from_mass(1.0e9, 2.3, "total_profile")
    return KpcInspiralModel(
        host_potential=CompositePotential((), central_point_mass_msun=1.0e8),
        secondary_bh_mass_msun=1.0e8,
        stellar_background=(
            None
            if not with_stars
            else StellarBackground(soliton, 100.0, np.zeros(3))
        ),
        stellar_friction=(
            None if not with_stars else StellarFrictionConfig(0.1)
        ),
        fdm_background=FDMBackground(
            soliton, 1.0e-21, 1.0, np.zeros(3), mode=fdm_mode
        ),
    )


def _config() -> KpcToHardConfig:
    return KpcToHardConfig(
        primary_bh_mass_msun=1.0e8,
        common_nucleus_radius_pc=10.0,
        sigma_pc_myr=100.0,
        hard_binary_radius_pc=1.0,
        maximum_time_myr=1.0,
        maximum_step_myr=0.01,
    )


def _tabulated_profile(*, core_radius_pc: float | None = None) -> TabulatedSphericalProfile:
    return TabulatedSphericalProfile(
        radii_pc=np.array([1.0, 2.0, 4.0, 8.0]),
        density_msun_pc3=np.array([100.0, 60.0, 25.0, 8.0]),
        enclosed_mass_msun=np.array([1.0e4, 2.0e4, 4.0e4, 7.0e4]),
        potential_pc2_myr2=np.array([-10.0, -8.0, -5.0, -3.0]),
        core_radius_pc=core_radius_pc,
    )


def _profile_bundle(*, event_uid: str = "event-1", gas=None):
    return EnvironmentProfileBundle(
        event_uid=event_uid,
        stellar=_tabulated_profile(),
        gas=gas,
        fdm=_tabulated_profile(core_radius_pc=2.3),
        source_case_id="case-1",
        source_sha256="d" * 64,
        source_path="profiles.json",
    )


def test_adapter_maps_secondary_minus_primary_state() -> None:
    state = initial_kpc_to_hard_state_from_bridge(
        _bridge(),
        model=_model(),
        config=_config(),
        assignment=BridgeMemberAssignment(11, 22),
    )
    assert state.inspiral_state.phase.value == "bound_binary"
    assert state.dynamical_state.position_pc[0] == pytest.approx(-2.0)
    assert state.dynamical_state.velocity_pc_myr[1] < 0.0


def test_adapter_reverses_orientation_when_member_order_is_swapped() -> None:
    state = initial_kpc_to_hard_state_from_bridge(
        _bridge(),
        model=_model(),
        config=_config(),
        assignment=BridgeMemberAssignment(22, 11),
    )
    assert state.dynamical_state.position_pc[0] == pytest.approx(2.0)
    assert state.dynamical_state.velocity_pc_myr[1] > 0.0


def test_adapter_requires_ready_environment() -> None:
    bridge = _bridge(stellar_status="missing")
    assert bridge.status == BridgeStatus.MISSING_ENVIRONMENT.value
    with pytest.raises(ValueError, match="not ready"):
        initial_kpc_to_hard_state_from_bridge(
            bridge,
            model=_model(),
            config=_config(),
            assignment=BridgeMemberAssignment(11, 22),
        )


def test_adapter_rejects_mass_mismatch() -> None:
    model = KpcInspiralModel(
        host_potential=CompositePotential((), central_point_mass_msun=1.0e8),
        secondary_bh_mass_msun=5.0e7,
        stellar_background=StellarBackground(
            SchiveSoliton.from_mass(1.0e9, 2.3, "total_profile"),
            100.0,
            np.zeros(3),
        ),
        stellar_friction=StellarFrictionConfig(0.1),
        fdm_background=_model().fdm_background,
    )
    with pytest.raises(ValueError, match="secondary mass"):
        initial_kpc_to_hard_state_from_bridge(
            _bridge(),
            model=model,
            config=_config(),
            assignment=BridgeMemberAssignment(11, 22),
        )


def test_adapter_rejects_fdm_mode_mismatch() -> None:
    with pytest.raises(ValueError, match="FDM bridge mode"):
        initial_kpc_to_hard_state_from_bridge(
            _bridge(fdm_mode="live_resolved"),
            model=_model(fdm_mode="analytic_unresolved"),
            config=_config(),
            assignment=BridgeMemberAssignment(11, 22),
        )


def test_adapter_rejects_available_stars_without_profile() -> None:
    with pytest.raises(ValueError, match="stellar environment"):
        initial_kpc_to_hard_state_from_bridge(
            _bridge(),
            model=_model(with_stars=False),
            config=_config(),
            assignment=BridgeMemberAssignment(11, 22),
        )


def test_builder_constructs_explicit_stellar_and_fdm_backgrounds() -> None:
    bundle = _profile_bundle()
    model = build_kpc_model_from_profile_bundle(
        _bridge(),
        profiles=bundle,
        host_potential=CompositePotential(
            (bundle.stellar, bundle.fdm), central_point_mass_msun=1.0e8
        ),
        secondary_bh_mass_msun=1.0e8,
        stellar_friction=StellarFrictionConfig(0.1),
        fdm_particle_mass_ev=1.0e-21,
        fdm_alpha_df=1.0,
    )
    assert model.stellar_background is not None
    assert model.gas_background is None
    assert model.fdm_background is not None
    assert model.fdm_background.mode == "analytic_unresolved"


def test_builder_rejects_profile_event_mismatch() -> None:
    with pytest.raises(ValueError, match="event_uid"):
        build_kpc_model_from_profile_bundle(
            _bridge(),
            profiles=_profile_bundle(event_uid="other-event"),
            host_potential=CompositePotential((), central_point_mass_msun=1.0e8),
            secondary_bh_mass_msun=1.0e8,
            stellar_friction=StellarFrictionConfig(0.1),
            fdm_particle_mass_ev=1.0e-21,
            fdm_alpha_df=1.0,
        )


def test_builder_rejects_absent_channel_profile() -> None:
    with pytest.raises(ValueError, match="absent gas channel"):
        build_kpc_model_from_profile_bundle(
            _bridge(),
            profiles=_profile_bundle(gas=_tabulated_profile()),
            host_potential=CompositePotential((), central_point_mass_msun=1.0e8),
            secondary_bh_mass_msun=1.0e8,
            stellar_friction=StellarFrictionConfig(0.1),
            fdm_particle_mass_ev=1.0e-21,
            fdm_alpha_df=1.0,
        )


def test_builder_requires_fdm_controls() -> None:
    with pytest.raises(ValueError, match="particle mass and alpha_df"):
        build_kpc_model_from_profile_bundle(
            _bridge(),
            profiles=_profile_bundle(),
            host_potential=CompositePotential((), central_point_mass_msun=1.0e8),
            secondary_bh_mass_msun=1.0e8,
            stellar_friction=StellarFrictionConfig(0.1),
        )
