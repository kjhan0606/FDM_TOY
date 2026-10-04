"""Fail-closed scope checks for the spectral-momentum endpoint audit."""

import pytest

from scripts import audit_periodic_tsc_strang_momentum_candidate as audit


@pytest.mark.parametrize("label", tuple(audit._LEVELS))
def test_candidate_scope_requires_exact_diagnostic_prefix(label: str) -> None:
    factor, saves, stop = audit._LEVELS[label]
    metadata = {
        "case_id": "qe_q030_e030_a020",
        "resolution": 256,
        "backend": "pytorch_cuda",
        "wave_smbh_coupling": "periodic_tsc_strang_momentum",
        "binary_integrator": "joint_kick_drift_kick_spectral_momentum_v1",
        "smbh_force_is_interaction_energy_gradient": False,
        "coupled_hamiltonian_ledger": True,
        "experimental_coupling_not_a_calibration_release": True,
        "analytic_fdm_drag": False,
        "time_step_factor": factor,
        "save_number": saves,
        "actual_wave_steps": saves,
        "diagnostic_stop_after_save": stop,
        "nbody_rk4_substeps_per_wave_step": 0,
    }
    solver = {
        "status": "diagnostic_partial", "actual_wave_steps": stop,
        "saved_intervals": stop, "peak_device_memory_bytes": 12345,
    }
    conservation = {
        "status": "diagnostic_partial", "samples": stop + 1,
        "wave_smbh_coupling": "periodic_tsc_strang_momentum",
        "smbh_force_is_interaction_energy_gradient": False,
        "max_total_energy_drift_over_energy_transfer": 0.002,
    }
    assert audit._validate_scope(metadata, solver, conservation, label) == (
        factor, stop
    )
    changed = dict(metadata)
    changed["smbh_force_is_interaction_energy_gradient"] = True
    with pytest.raises(ValueError, match="metadata differs"):
        audit._validate_scope(changed, solver, conservation, label)
    changed_solver = dict(solver, status="complete")
    with pytest.raises(ValueError, match="requested prefix"):
        audit._validate_scope(metadata, changed_solver, conservation, label)
    changed_conservation = dict(
        conservation, max_total_energy_drift_over_energy_transfer=float("nan")
    )
    with pytest.raises(ValueError, match="requested prefix"):
        audit._validate_scope(metadata, solver, changed_conservation, label)


def test_candidate_scope_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="unknown"):
        audit._validate_scope({}, {}, {}, "f00625")
