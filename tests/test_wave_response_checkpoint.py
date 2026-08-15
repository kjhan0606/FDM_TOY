import importlib.util
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_pyul_wave_response.py"
)
_SPEC = importlib.util.spec_from_file_location("analyze_pyul_wave_response", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_resume_rows = _MODULE._resume_rows
_write_rows = _MODULE._write_rows


def test_wave_response_checkpoint_round_trip(tmp_path) -> None:
    response_path = tmp_path / "response.partial.csv"
    radial_path = tmp_path / "radial.partial.csv"
    response_rows = [
        {"sample": 0, "time_myr": 0.0, "diagnostic": np.nan},
        {"sample": 1, "time_myr": 0.5, "diagnostic": 3.0},
    ]
    radial_rows = [
        {"sample": sample, "time_myr": 0.5 * sample, "radius": radius}
        for sample in range(2)
        for radius in (1.0, 2.0)
    ]
    _write_rows(response_path, response_rows)
    _write_rows(radial_path, radial_rows)

    measured_response, measured_radial = _resume_rows(
        response_path=response_path,
        radial_path=radial_path,
        radial_bins=2,
        times_myr=np.asarray([0.0, 0.5, 1.0]),
    )

    assert len(measured_response) == 2
    assert np.isnan(measured_response[0]["diagnostic"])
    assert measured_response[1]["diagnostic"] == pytest.approx(3.0)
    assert measured_radial == radial_rows


def test_wave_response_checkpoint_requires_both_tables(tmp_path) -> None:
    response_path = tmp_path / "response.partial.csv"
    radial_path = tmp_path / "radial.partial.csv"
    _write_rows(response_path, [{"sample": 0, "time_myr": 0.0}])

    with pytest.raises(ValueError, match="checkpoint tables are incomplete"):
        _resume_rows(
            response_path=response_path,
            radial_path=radial_path,
            radial_bins=2,
            times_myr=np.asarray([0.0]),
        )


def test_wave_response_checkpoint_recovers_one_table_lead(tmp_path) -> None:
    response_path = tmp_path / "response.partial.csv"
    radial_path = tmp_path / "radial.partial.csv"
    response_rows = [
        {"sample": 0, "time_myr": 0.0},
        {"sample": 1, "time_myr": 0.5},
    ]
    radial_rows = [
        {"sample": 0, "time_myr": 0.0, "radius": radius}
        for radius in (1.0, 2.0)
    ]
    _write_rows(response_path, response_rows)
    _write_rows(radial_path, radial_rows)

    measured_response, measured_radial = _resume_rows(
        response_path=response_path,
        radial_path=radial_path,
        radial_bins=2,
        times_myr=np.asarray([0.0, 0.5]),
    )

    assert measured_response == response_rows[:1]
    assert measured_radial == radial_rows


@pytest.mark.parametrize(
    ("radial_rows", "message"),
    [
        (
            [{"sample": 0, "time_myr": 0.0}],
            "radial checkpoint has the wrong size",
        ),
        (
            [
                {"sample": 1, "time_myr": 0.0},
                {"sample": 1, "time_myr": 0.0},
            ],
            "radial checkpoint samples are inconsistent",
        ),
        (
            [
                {"sample": 0, "time_myr": 0.25},
                {"sample": 0, "time_myr": 0.25},
            ],
            "radial checkpoint times do not match the run",
        ),
    ],
)
def test_wave_response_checkpoint_rejects_invalid_radial_table(
    tmp_path, radial_rows, message
) -> None:
    response_path = tmp_path / "response.partial.csv"
    radial_path = tmp_path / "radial.partial.csv"
    _write_rows(response_path, [{"sample": 0, "time_myr": 0.0}])
    _write_rows(radial_path, radial_rows)

    with pytest.raises(ValueError, match=message):
        _resume_rows(
            response_path=response_path,
            radial_path=radial_path,
            radial_bins=2,
            times_myr=np.asarray([0.0]),
        )
