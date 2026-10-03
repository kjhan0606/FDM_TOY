"""Adapters from the versioned bridge contract to the kpc inspiral solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kpc_inspiral import (
    GasFrictionConfig,
    KpcInspiralModel,
    KpcToHardConfig,
    KpcToHardState,
    StellarFrictionConfig,
    initial_dual_nucleus_state,
    initial_kpc_to_hard_state,
)
from .galaxy_environment import (
    CompositePotential,
    FDMBackground,
    GasBackground,
    NuclearEnvelope,
    StellarBackground,
)
from .nuclear_bridge import BridgeStatus, NuclearBridgeInput
from .profile_table import EnvironmentProfileBundle, TabulatedSphericalProfile


@dataclass(frozen=True)
class BridgeMemberAssignment:
    """Explicit mapping from ledger member IDs to solver primary/secondary."""

    primary_member_id: int
    secondary_member_id: int

    def __post_init__(self) -> None:
        if self.primary_member_id == self.secondary_member_id:
            raise ValueError("primary and secondary member IDs must differ")


def _profile_for_available_channel(
    bridge: NuclearBridgeInput,
    profiles: EnvironmentProfileBundle,
    name: str,
) -> TabulatedSphericalProfile | None:
    channel = bridge.environment.channel(name)
    profile = profiles.profile(name)
    if channel.status == "available" and profile is None:
        raise ValueError(f"available {name} channel has no radial profile")
    if channel.status == "absent" and profile is not None:
        raise ValueError(f"absent {name} channel cannot have a radial profile")
    if channel.status == "absent":
        return None
    if channel.status != "available":
        raise ValueError(
            f"cannot build a kpc model from {name} channel status={channel.status}"
        )
    assert profile is not None
    return profile


def build_kpc_model_from_profile_bundle(
    bridge: NuclearBridgeInput,
    *,
    profiles: EnvironmentProfileBundle,
    host_potential: CompositePotential,
    secondary_bh_mass_msun: float,
    stellar_friction: StellarFrictionConfig | None = None,
    gas_friction: GasFrictionConfig | None = None,
    fdm_particle_mass_ev: float | None = None,
    fdm_alpha_df: float | None = None,
    gas_rotation_axis: np.ndarray | None = None,
    nuclear_envelope: NuclearEnvelope | None = None,
) -> KpcInspiralModel:
    """Build a kpc model from explicit radial closure tables.

    The bridge supplies local closure values and the profile bundle supplies
    radial dependence.  No profile is inferred from a single snapshot and no
    table is extrapolated outside its tabulated radius range.  For a pure-FDM
    study, the outer FDM halo profile must be supplied explicitly through
    ``host_potential``; the parsec-scale soliton is not silently extrapolated
    to kpc radii.  Any baryonic components are likewise explicit, and no CDM
    component is manufactured or silently reused as stellar friction.
    """

    if bridge.status != BridgeStatus.READY.value:
        raise ValueError(
            f"bridge is not ready for model construction: {bridge.status}; reasons={bridge.reasons}"
        )
    if profiles.event_uid != bridge.event_uid:
        raise ValueError("profile bundle and bridge event_uid values must agree")

    stellar_profile = _profile_for_available_channel(bridge, profiles, "stellar")
    gas_profile = _profile_for_available_channel(bridge, profiles, "gas")
    fdm_profile = _profile_for_available_channel(bridge, profiles, "fdm")

    stellar_background = None
    if stellar_profile is not None:
        channel = bridge.environment.channel("stellar")
        if channel.velocity_dispersion_pc_myr is None or channel.bulk_velocity_pc_myr is None:
            raise ValueError("available stellar channel lacks velocity closure values")
        if stellar_friction is None:
            raise ValueError("available stellar channel requires stellar_friction controls")
        stellar_background = StellarBackground(
            stellar_profile,
            channel.velocity_dispersion_pc_myr,
            channel.bulk_velocity_pc_myr,
        )

    gas_background = None
    if gas_profile is not None:
        channel = bridge.environment.channel("gas")
        if channel.sound_speed_pc_myr is None or channel.rotation_fraction is None:
            raise ValueError("available gas channel lacks sound-speed/rotation closure values")
        if gas_friction is None:
            raise ValueError("available gas channel requires gas_friction controls")
        if gas_rotation_axis is None:
            raise ValueError("available gas channel requires an explicit gas_rotation_axis")
        gas_background = GasBackground(
            gas_profile,
            channel.sound_speed_pc_myr,
            channel.rotation_fraction,
            gas_rotation_axis,
        )

    fdm_background = None
    if fdm_profile is not None:
        channel = bridge.environment.channel("fdm")
        if channel.bulk_velocity_pc_myr is None or channel.fdm_mode is None:
            raise ValueError("available FDM channel lacks velocity/mode closure values")
        if fdm_profile.core_radius_pc is None:
            raise ValueError("available FDM profile requires core_radius_pc")
        if fdm_particle_mass_ev is None or fdm_alpha_df is None:
            raise ValueError("available FDM channel requires particle mass and alpha_df")
        fdm_background = FDMBackground(
            fdm_profile,
            fdm_particle_mass_ev,
            fdm_alpha_df,
            channel.bulk_velocity_pc_myr,
            mode=channel.fdm_mode,
        )

    return KpcInspiralModel(
        host_potential=host_potential,
        secondary_bh_mass_msun=secondary_bh_mass_msun,
        nuclear_envelope=nuclear_envelope,
        stellar_background=stellar_background,
        stellar_friction=stellar_friction if stellar_background is not None else None,
        gas_background=gas_background,
        gas_friction=gas_friction if gas_background is not None else None,
        fdm_background=fdm_background,
    )


def _mass_for_member(bridge: NuclearBridgeInput, member_id: int) -> float:
    pair = bridge.pair
    for member, mass in zip(pair.member_ids, pair.masses_msun, strict=True):
        if member == member_id:
            return float(mass)
    raise ValueError(f"member ID {member_id} is absent from the bridge pair")


def _relative_secondary_state(
    bridge: NuclearBridgeInput, assignment: BridgeMemberAssignment
) -> tuple[np.ndarray, np.ndarray]:
    """Return secondary-minus-primary position and velocity."""

    pair = bridge.pair
    if {
        assignment.primary_member_id,
        assignment.secondary_member_id,
    } != set(pair.member_ids):
        raise ValueError("primary/secondary IDs must match the bridge pair")
    sign = (
        -1.0
        if (
            assignment.primary_member_id == pair.member_ids[0]
            and assignment.secondary_member_id == pair.member_ids[1]
        )
        else 1.0
    )
    return (
        sign * np.asarray(pair.separation_vector_pc, dtype=float),
        sign * np.asarray(pair.relative_velocity_pc_myr, dtype=float),
    )


def _validate_environment_model_contract(
    bridge: NuclearBridgeInput, model: KpcInspiralModel
) -> None:
    for name in ("stellar", "gas", "fdm"):
        channel = bridge.environment.channel(name)
        supplied = getattr(model, f"{name}_background")
        if channel.status == "available" and supplied is None:
            raise ValueError(
                f"{name} environment is available in the bridge but the kpc model has no background"
            )
        if channel.status == "absent" and supplied is not None:
            raise ValueError(
                f"{name} environment is explicitly absent but the kpc model supplies a background"
            )
    fdm_channel = bridge.environment.channel("fdm")
    if fdm_channel.status == "available":
        assert model.fdm_background is not None
        if model.fdm_background.mode != fdm_channel.fdm_mode:
            raise ValueError(
                "FDM bridge mode and kpc model mode disagree; refusing a double-counted force"
            )


def initial_kpc_to_hard_state_from_bridge(
    bridge: NuclearBridgeInput,
    *,
    model: KpcInspiralModel,
    config: KpcToHardConfig,
    assignment: BridgeMemberAssignment,
) -> KpcToHardState:
    """Convert a ready bridge record into a restartable kpc solver state.

    The host profiles and friction controls remain explicit model inputs.  A
    local environment snapshot cannot silently manufacture a radial profile.
    """

    if bridge.status != BridgeStatus.READY.value:
        raise ValueError(
            f"bridge is not ready for integration: {bridge.status}; reasons={bridge.reasons}"
        )
    _validate_environment_model_contract(bridge, model)
    primary_mass = _mass_for_member(bridge, assignment.primary_member_id)
    secondary_mass = _mass_for_member(bridge, assignment.secondary_member_id)
    if not np.isclose(
        config.primary_bh_mass_msun, primary_mass, rtol=1.0e-12, atol=0.0
    ):
        raise ValueError("KpcToHardConfig primary mass disagrees with bridge member")
    if not np.isclose(
        model.secondary_bh_mass_msun, secondary_mass, rtol=1.0e-12, atol=0.0
    ):
        raise ValueError("KpcInspiralModel secondary mass disagrees with bridge member")
    position, velocity = _relative_secondary_state(bridge, assignment)
    dynamics = initial_dual_nucleus_state(
        position_pc=position,
        velocity_pc_myr=velocity,
        model=model,
    )
    return initial_kpc_to_hard_state(
        event_uid=bridge.event_uid,
        dynamical_state=dynamics,
        model=model,
        config=config,
    )
