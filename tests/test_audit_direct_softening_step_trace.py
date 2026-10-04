"""Single-operator and trajectory-distance checks for softening audit."""

from copy import deepcopy

import numpy as np

from scripts.audit_direct_softening_step_trace import (
    matching_except_direct_plummer, separation_series,
)


def test_only_direct_plummer_config_difference_is_accepted() -> None:
    baseline = {
        "Matter Particles": {
            "Plummer Radius": 0.05,
            "Condition": [[7.0, [0.1, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        },
        "ULDM Solitons": {"Condition": [[1.0, [0.0, 0.0, 0.0]]]},
    }
    candidate = deepcopy(baseline)
    candidate["Matter Particles"]["Plummer Radius"] = 0.1
    assert matching_except_direct_plummer(candidate, baseline)
    candidate["Matter Particles"]["Condition"][0][1][0] += 0.01
    assert not matching_except_direct_plummer(candidate, baseline)


def test_separation_series_uses_code_to_pc_conversion(tmp_path) -> None:
    directory = tmp_path / "run/Outputs/NBody"
    directory.mkdir(parents=True)
    for index, distance in enumerate((0.25, 0.375)):
        state = np.zeros((2, 6))
        state[1, 0] = distance
        np.save(directory / f"NTM_#{index:03d}.npy", state.reshape(12))
    np.testing.assert_allclose(
        separation_series(tmp_path / "run", stop=1, length_pc=8.0),
        [2.0, 3.0], rtol=0, atol=0,
    )
