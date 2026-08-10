from pathlib import Path

import pytest

from fdm_smbh_delay.pyul import ordered_output_paths, pyul_unit_system


def test_pyul_units_from_explicit_metadata() -> None:
    units = pyul_unit_system(
        {
            "pyul_length_unit_m": 3.0857e16,
            "pyul_time_unit_s": 365.0 * 86400.0 * 1.0e6,
            "pyul_mass_unit_kg": 1.989e30,
            "pyul_energy_unit_j": 1.989e30
            * (3.0857e16 / (365.0 * 86400.0 * 1.0e6)) ** 2,
        }
    )
    assert units.length_pc == pytest.approx(1.0)
    assert units.time_myr == pytest.approx(1.0)
    assert units.mass_msun == pytest.approx(1.0)
    assert units.energy_msun_pc2_myr2 == pytest.approx(1.0)
    assert units.angular_momentum_msun_pc2_myr == pytest.approx(1.0)


def test_pyul_fallback_matches_reference_run() -> None:
    units = pyul_unit_system({"particle_mass_ev": 1.0e-21})
    assert units.length_pc == pytest.approx(
        3.7439093785040434e20 / 3.0857e16, rel=2.0e-12
    )
    assert units.mass_msun == pytest.approx(
        1.4014034281125392e35 / 1.989e30, rel=2.0e-12
    )


def test_ordered_output_paths(tmp_path: Path) -> None:
    for index in (2, 0, 1):
        (tmp_path / f"P3D_#{index:03d}.npy").touch()
    paths = ordered_output_paths(tmp_path, "P3D_#*.npy")
    assert [path.name for path in paths] == [
        "P3D_#000.npy",
        "P3D_#001.npy",
        "P3D_#002.npy",
    ]
    with pytest.raises(FileNotFoundError):
        ordered_output_paths(tmp_path, "missing*.npy")
