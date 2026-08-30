"""FDM SMBH delay toy model."""

from .bridge_adapter import (
    BridgeMemberAssignment,
    build_kpc_model_from_profile_bundle,
    initial_kpc_to_hard_state_from_bridge,
)
from .capture_fdm_seed import (
    CaptureDerivedDualSMBHSinkPair,
    CaptureFDMSeedAssignment,
    CaptureFDMSeedFrame,
    CaptureFDMSeedFrameSpecification,
    CaptureSMBHMassProjection,
    derive_dual_smbh_sink_pair_from_capture,
    verify_mass_projection_source,
    verify_pure_fdm_seed_matches_capture_sink_pair,
)
from .capture_seed_binding import (
    CaptureSeedMaterializationBinding,
    assess_capture_seed_materialization_binding,
)
from .capture_seed_assembly import (
    CaptureSolitonConfiguration,
    assemble_capture_derived_pure_fdm_seed,
    capture_derived_seed_mapping,
    capture_soliton_configuration_from_mapping,
)
from .dm_run_provenance import (
    CaptureDMRunBinding,
    DarkMatterRunProvenance,
    bind_capture_event_to_dark_matter_run,
    read_dark_matter_run_provenance,
)
from .cdm_zoom_materialization import (
    CDMNonCompactingZoomRunContract,
    assess_cdm_noncompacting_zoom_run_inputs,
    materialize_cdm_noncompacting_zoom_run_contract,
)
from .cdm_zoom_runtime_identity import (
    CDMNonCompactingZoomRuntimeIdentity,
    assess_cdm_noncompacting_zoom_runtime_identity,
)
from .dm_comparison import (
    DMComparisonCaptureEnsemble,
    DMComparisonCaptureRegistration,
    DMComparisonFamilyManifest,
    DMComparisonFamilyPreflight,
    DMComparisonPhysicsAssessment,
    DMComparisonPhysicsInput,
    DMComparisonSmokeAssessment,
    HashedArtifact,
    assess_dm_comparison_physics_inputs,
    assess_dm_comparison_smoke_outputs,
    preflight_dm_comparison_family,
    read_dm_comparison_capture_registration,
    read_dm_comparison_family_manifest,
    read_dm_comparison_physics_input,
    read_verified_dm_comparison_capture_ensemble,
    register_dm_comparison_capture_ensemble,
)
from .model_specific_physics import (
    MODEL_SPECIFIC_PHYSICS_RESULT_SCHEMA_VERSION,
    ModelSpecificEvidence,
    ModelSpecificPhaseEnsemble,
    ModelSpecificResolutionAssessment,
    ResolvedEnvironmentChannel,
    ResolvedModelPhysicsResult,
    assess_model_specific_phase_ensemble,
    compare_model_specific_resolution_pair,
    read_resolved_model_physics_result,
)
from .config import CaseConfig, load_config
from .fdm_outer_halo import FDMOuterHaloClosure
from .fdm_outer_wave_ledger import FDMOuterWaveLedger
from .lagramses_fdm_provenance import (
    LagRamsesFDMOuterWaveProvenance,
    read_lagramses_fdm_outer_wave_provenance,
)
from .dual_soliton_seed import (
    DualSMBHSinkSeed,
    DualSolitonComponent,
    PureFDMDualSolitonSeed,
    load_pure_fdm_dual_soliton_seed,
    materialize_pure_fdm_dual_soliton_seed,
    pure_fdm_dual_soliton_seed_from_mapping,
)
from .dual_soliton_preflight import (
    DualSolitonRunPreflight,
    DualSolitonRuntimeIdentity,
    preflight_pure_fdm_dual_soliton_run,
    validate_pure_fdm_dual_soliton_runtime_identity,
)
from .dual_soliton_relaxation import (
    DualSolitonRelaxationAssessment,
    DualSolitonRelaxationEvidence,
    RelaxationConservationThresholds,
    assess_dual_soliton_relaxation,
)
from .outer_inner_handoff import (
    HandoffDecision,
    HandoffGateConfig,
    HandoffRatePoint,
    HandoffSimilarityState,
    validate_outer_inner_handoff,
)
from .fdm_outer_response import FDMOuterResponseTable
from .pure_fdm_zoom import (
    DeferredNestedZoomRequest,
    NestedZoomCheckpointContract,
    PureFDMOuterZoomPreflight,
    bind_nested_zoom_checkpoint,
    preflight_pure_fdm_outer_zoom,
)
from .pure_fdm_outer_results import (
    PureFDMOuterConvergenceResult,
    PureFDMOuterPhaseEnsemble,
    PureFDMOuterRunResult,
    PureFDMOuterStage,
    compare_pure_fdm_outer_resolution_pair,
    assess_pure_fdm_outer_phase_ensemble,
    read_pure_fdm_outer_result,
)
from .pure_fdm_outer_evaluation import (
    PureFDMOuterEnsembleEvaluation,
    PureFDMOuterPhysicsAssessment,
    PureFDMOuterResolutionAssessment,
    PureFDMOuterResultIndex,
    evaluate_pure_fdm_outer_result_index,
    load_pure_fdm_outer_result_index,
)
from .pure_fdm_nested_registration import (
    PureFDMNestedPhysicsRegistration,
    PureFDMNestedRegistrationManifest,
    PureFDMNestedZoomRegistration,
    build_pure_fdm_nested_registration_manifest,
)
from .nuclear_bridge import (
    EnvironmentChannel,
    EnvironmentSnapshot,
    NuclearBridgeInput,
)
from .profile_table import EnvironmentProfileBundle, TabulatedSphericalProfile
from .orbit import IntegrationResult, integrate_case

__all__ = [
    "CaseConfig",
    "BridgeMemberAssignment",
    "CaptureDerivedDualSMBHSinkPair",
    "CaptureFDMSeedAssignment",
    "CaptureFDMSeedFrame",
    "CaptureFDMSeedFrameSpecification",
    "CaptureSMBHMassProjection",
    "CaptureSeedMaterializationBinding",
    "CaptureSolitonConfiguration",
    "CaptureDMRunBinding",
    "CDMNonCompactingZoomRunContract",
    "CDMNonCompactingZoomRuntimeIdentity",
    "DMComparisonCaptureEnsemble",
    "DMComparisonCaptureRegistration",
    "DMComparisonFamilyManifest",
    "DMComparisonFamilyPreflight",
    "DMComparisonPhysicsAssessment",
    "DMComparisonPhysicsInput",
    "DMComparisonSmokeAssessment",
    "MODEL_SPECIFIC_PHYSICS_RESULT_SCHEMA_VERSION",
    "ModelSpecificEvidence",
    "ModelSpecificPhaseEnsemble",
    "ModelSpecificResolutionAssessment",
    "EnvironmentProfileBundle",
    "DarkMatterRunProvenance",
    "FDMOuterHaloClosure",
    "FDMOuterWaveLedger",
    "LagRamsesFDMOuterWaveProvenance",
    "DualSMBHSinkSeed",
    "DualSolitonRunPreflight",
    "DualSolitonRuntimeIdentity",
    "DualSolitonRelaxationAssessment",
    "DualSolitonRelaxationEvidence",
    "DualSolitonComponent",
    "PureFDMDualSolitonSeed",
    "RelaxationConservationThresholds",
    "FDMOuterResponseTable",
    "DeferredNestedZoomRequest",
    "NestedZoomCheckpointContract",
    "HandoffDecision",
    "HandoffGateConfig",
    "HandoffRatePoint",
    "HandoffSimilarityState",
    "HashedArtifact",
    "EnvironmentChannel",
    "EnvironmentSnapshot",
    "IntegrationResult",
    "NuclearBridgeInput",
    "PureFDMOuterZoomPreflight",
    "PureFDMOuterConvergenceResult",
    "PureFDMOuterPhaseEnsemble",
    "PureFDMOuterEnsembleEvaluation",
    "PureFDMOuterPhysicsAssessment",
    "PureFDMOuterResolutionAssessment",
    "PureFDMOuterResultIndex",
    "PureFDMNestedPhysicsRegistration",
    "PureFDMNestedRegistrationManifest",
    "PureFDMNestedZoomRegistration",
    "PureFDMOuterRunResult",
    "PureFDMOuterStage",
    "ResolvedEnvironmentChannel",
    "ResolvedModelPhysicsResult",
    "TabulatedSphericalProfile",
    "assemble_capture_derived_pure_fdm_seed",
    "assess_cdm_noncompacting_zoom_run_inputs",
    "assess_cdm_noncompacting_zoom_runtime_identity",
    "build_kpc_model_from_profile_bundle",
    "capture_derived_seed_mapping",
    "capture_soliton_configuration_from_mapping",
    "derive_dual_smbh_sink_pair_from_capture",
    "bind_nested_zoom_checkpoint",
    "bind_capture_event_to_dark_matter_run",
    "initial_kpc_to_hard_state_from_bridge",
    "integrate_case",
    "load_config",
    "load_pure_fdm_dual_soliton_seed",
    "materialize_pure_fdm_dual_soliton_seed",
    "materialize_cdm_noncompacting_zoom_run_contract",
    "pure_fdm_dual_soliton_seed_from_mapping",
    "preflight_pure_fdm_dual_soliton_run",
    "validate_pure_fdm_dual_soliton_runtime_identity",
    "assess_dual_soliton_relaxation",
    "assess_capture_seed_materialization_binding",
    "assess_dm_comparison_physics_inputs",
    "assess_dm_comparison_smoke_outputs",
    "assess_model_specific_phase_ensemble",
    "read_lagramses_fdm_outer_wave_provenance",
    "read_dark_matter_run_provenance",
    "compare_pure_fdm_outer_resolution_pair",
    "compare_model_specific_resolution_pair",
    "assess_pure_fdm_outer_phase_ensemble",
    "evaluate_pure_fdm_outer_result_index",
    "load_pure_fdm_outer_result_index",
    "build_pure_fdm_nested_registration_manifest",
    "preflight_pure_fdm_outer_zoom",
    "preflight_dm_comparison_family",
    "read_dm_comparison_capture_registration",
    "read_dm_comparison_family_manifest",
    "read_dm_comparison_physics_input",
    "read_verified_dm_comparison_capture_ensemble",
    "read_resolved_model_physics_result",
    "read_pure_fdm_outer_result",
    "validate_outer_inner_handoff",
    "verify_mass_projection_source",
    "verify_pure_fdm_seed_matches_capture_sink_pair",
    "register_dm_comparison_capture_ensemble",
]
__version__ = "0.1.0"
