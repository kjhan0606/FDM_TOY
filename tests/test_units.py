from __future__ import annotations

import textwrap

import numpy as np
import pytest
from astropy import units as u

from fdm_smbh_delay.config import load_config
from fdm_smbh_delay.units import UnitValidationError, parse_quantity, parse_vector


def test_quantity_and_vector_conversion() -> None:
    assert parse_quantity("1 kpc", u.pc, "x") == pytest.approx(1000.0)
    vector = parse_vector(["1 km/s", "0 km/s", "-1 km/s"], u.pc / u.Myr, "v")
    assert vector[0] == pytest.approx(1.022712165)
    assert np.allclose(vector[[0, 2]], -vector[[2, 0]])


def test_dimensional_values_reject_bare_numbers() -> None:
    with pytest.raises(UnitValidationError, match="explicit unit"):
        parse_quantity(1.0, u.pc, "length")


def test_load_config_requires_soliton_mass_definition(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        textwrap.dedent(
            """
            model:
              fdm_bulk_velocity: ["0 km/s", "0 km/s", "0 km/s"]
            binary:
              M1: "1e6 Msun"
              M2: "1e6 Msun"
              separation: "1 pc"
            fdm:
              particle_mass: "1e-21 eV"
              soliton_mass: "1e8 Msun"
              core_radius: "10 pc"
            integration:
              max_time: "1 Myr"
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mass_definition"):
        load_config(path)


def test_example_configs_are_valid() -> None:
    for path in (
        "configs/koo2024.yaml",
        "configs/boey2025_fiducial.yaml",
        "configs/lagramses_m22_example.yaml",
    ):
        assert load_config(path).fdm.build_soliton().total_mass_msun > 0.0
