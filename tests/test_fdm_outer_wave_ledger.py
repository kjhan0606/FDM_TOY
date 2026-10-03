from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.fdm_outer_wave_ledger import FDMOuterWaveLedger


def _ledger(**changes: object) -> FDMOuterWaveLedger:
    values: dict[str, object] = {
        "source_path": "/gpfs/example/outer-wave-ledger.json",
        "source_sha256": "a" * 64,
        "force_ledger_sha256": "b" * 64,
        "field_snapshot_index_sha256": "c" * 64,
        "profile_snapshot_index_sha256": "d" * 64,
        "sample_times_myr": np.array([0.0, 1.0]),
        "core_centres_pc": np.zeros((2, 3)),
        "fdm_current_msun_pc2_myr": np.ones((2, 3)),
        "coherence_time_myr": np.array([0.1, 0.2]),
        "de_broglie_wavelength_pc": np.array([1.0, 1.0]),
        "granule_power": np.array([0.01, 0.02]),
        "dipole_l1_real_imag": np.zeros((2, 2)),
        "quadrupole_l2_real_imag": np.zeros((2, 2)),
        "wake_extent_pc": np.array([2.0, 2.0]),
        "hjm_wave_seam_clearance_pc": np.array([3.0, 3.0]),
        "boundary_clearance_pc": np.array([3.0, 3.0]),
        "maximum_relative_hamiltonian_error": 1.0e-5,
        "maximum_relative_mass_error": 1.0e-5,
        "maximum_relative_angular_momentum_error": 2.0e-5,
    }
    values.update(changes)
    return FDMOuterWaveLedger(**values)


def test_wave_ledger_round_trips_all_required_outer_observables() -> None:
    ledger = _ledger(granule_power=np.array([0.0, 0.02]))
    restored = FDMOuterWaveLedger.from_dict(ledger.as_dict())
    assert restored.source_sha256 == ledger.source_sha256
    assert np.array_equal(restored.dipole_l1_real_imag, ledger.dipole_l1_real_imag)
    assert restored.force_accounting == "live_wave_only"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"force_accounting": "analytic_fdm_drag"}, "resolved-wave force"),
        ({"hjm_wave_seam_clearance_pc": np.array([2.0, 2.0])}, "inside the protected"),
        ({"sample_times_myr": np.array([1.0, 1.0])}, "strictly increasing"),
    ],
)
def test_wave_ledger_rejects_analytic_force_or_invalid_wave_provenance(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _ledger(**changes)
