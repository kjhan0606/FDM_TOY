"""Spectral-wave and binary-momentum unit checks for the TSC audit."""

import json
from pathlib import Path

import numpy as np
import pytest

from fdm_smbh_delay.pyul import (
    PYUL_MYR_S, PYUL_PARSEC_M, PYUL_SOLAR_MASS_KG,
)
from scripts import audit_periodic_tsc_momentum as audit
from scripts.audit_periodic_tsc_momentum import (
    _body_momentum,
    spectral_wave_momentum,
)


def test_periodic_plane_wave_momentum_and_integer_grid_shift() -> None:
    resolution = 16
    box_length = 4.0
    density = 1.7
    axis = np.arange(resolution) / resolution
    phase = 2.0 * np.pi * (
        axis[:, None, None] - 2.0 * axis[None, :, None]
        + axis[None, None, :]
    )
    wave = np.sqrt(density) * np.exp(1j * phase)
    expected = density * box_length**3 * 2.0 * np.pi / box_length * np.array(
        [1.0, -2.0, 1.0]
    )
    measured = spectral_wave_momentum(wave, box_length)
    np.testing.assert_allclose(measured, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        spectral_wave_momentum(np.roll(wave, 3, axis=0), box_length),
        expected, rtol=1e-13, atol=1e-13,
    )
    np.testing.assert_allclose(
        spectral_wave_momentum(np.conj(wave), box_length),
        -expected, rtol=1e-13, atol=1e-13,
    )


def test_momentum_rejects_invalid_wave_and_sums_binary() -> None:
    with pytest.raises(ValueError, match="finite cubic complex"):
        spectral_wave_momentum(np.ones((8, 8, 8)), 4.0)
    state = np.array([
        [0, 0, 0, 1, 2, 3],
        [0, 0, 0, -4, 5, -6],
    ])
    np.testing.assert_allclose(
        _body_momentum(state, np.array([2.0, 3.0])),
        [-10.0, 19.0, -12.0],
    )


def test_momentum_audit_binds_reference_and_final_checkpoint(
    tmp_path, monkeypatch,
) -> None:
    reference = tmp_path / "seed"
    (reference / "Outputs/3Wfn").mkdir(parents=True)
    (reference / "Outputs/NBody").mkdir()
    initial_wave = reference / "Outputs/3Wfn/P3D_#000.npy"
    initial_body = reference / "Outputs/NBody/NTM_#000.npy"
    np.save(initial_wave, np.ones((8, 8, 8), dtype=np.complex128))
    body = np.array([
        [-0.5, 0, 0, 0, 0, 0],
        [0.5, 0, 0, 0, 0, 0],
    ], dtype=float)
    np.save(initial_body, body)
    run = tmp_path / "f100"
    (run / "Checkpoints").mkdir(parents=True)
    (run / "Outputs/NBody").mkdir(parents=True)
    (run / "fdm_adapter_metadata.json").write_text(json.dumps({
        "reference_initial_state": str(reference),
        "reference_initial_wave_sha256": audit._sha256(initial_wave),
        "reference_initial_particle_sha256": audit._sha256(initial_body),
        "box_size_pc": 4.0,
        "pyul_length_unit_m": 2.0 * PYUL_PARSEC_M,
        "pyul_time_unit_s": 3.0 * PYUL_MYR_S,
        "pyul_mass_unit_kg": 4.0 * PYUL_SOLAR_MASS_KG,
        "pyul_energy_unit_j": (
            4.0 * PYUL_SOLAR_MASS_KG
            * (2.0 * PYUL_PARSEC_M / (3.0 * PYUL_MYR_S)) ** 2
        ),
    }))
    (run / "config.uldm").write_text(json.dumps({
        "Matter Particles": {"Condition": [[2.0], [3.0]]},
    }))
    final_body = body.copy()
    final_body[0, 3] = 0.1
    final_body[1, 3] = -0.1
    np.save(run / "Outputs/NBody/NTM_#010.npy", final_body)
    np.save(run / "Outputs/ULDMass.npy", np.full(11, 8.0))
    np.save(run / "Outputs/ekandqlist.npy", np.zeros(11))
    np.save(run / "Checkpoints/wave_000010.npy", np.ones((8, 8, 8), dtype=np.complex128))
    np.savez(
        run / "Checkpoints/state_000010.npz", state=final_body,
        step=np.int64(10), save_index=np.int64(10),
    )
    (run / "Checkpoints/latest.json").write_text(json.dumps({
        "wave": "wave_000010.npy", "state": "state_000010.npz",
        "step": 10, "save_index": 10,
    }))
    monkeypatch.setattr(
        audit, "_read_trace",
        lambda path, factor, saves, stop, coupling: (
            {
                "slurm_job_id": "123", "source_commit": "test",
                "source_fingerprints": (("source.py", "same"),),
                "launch_owner_sha256": "owner",
                "conservation_summary_sha256": "conservation",
                "conservation_timeseries_sha256": "series",
                "provenance_manifest_sha256": "manifest",
            }, None
        ),
    )
    result = audit.summarize(run, factor=1.0, saves=58500, stop=10)
    assert result["calibration_eligible"] is False
    assert result["analysis_source_sha256"] == audit._sha256(Path(audit.__file__))
    np.testing.assert_allclose(result["total_momentum_change_code"], [-0.025, 0, 0])
    np.testing.assert_allclose(
        result["total_momentum_change_msun_pc_myr"], [-1.0 / 15.0, 0, 0]
    )
    assert result["momentum_error_over_exchange"] == pytest.approx(1.0)
    mass_path = run / "Outputs/ULDMass.npy"
    np.save(mass_path, np.full(11, 7.0))
    with pytest.raises(ValueError, match="mass or kinetic energy disagrees"):
        audit.summarize(run, factor=1.0, saves=58500, stop=10)
    np.save(mass_path, np.full(11, 8.0))
    np.save(run / "Outputs/NBody/NTM_#010.npy", body)
    with pytest.raises(ValueError, match="checkpoint differs"):
        audit.summarize(run, factor=1.0, saves=58500, stop=10)
