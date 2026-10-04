"""Regression check for a common physical-time directional trajectory."""

import numpy as np

from scripts.audit_transverse_offset_step_trace import separation_series


def test_directional_separation_series_uses_particle_distance(tmp_path) -> None:
    path = tmp_path / "run"
    (path / "Outputs/NBody").mkdir(parents=True)
    for index, distance in enumerate((0.25, 0.3)):
        state = np.zeros((2, 6), dtype=float)
        state[1, :3] = (distance, 0.0, 0.0)
        np.save(path / f"Outputs/NBody/NTM_#{index:03d}.npy",
                state.reshape(12))
    np.testing.assert_allclose(
        separation_series(path, stop=1, length_pc=4.0),
        [1.0, 1.2], rtol=0, atol=1e-15,
    )
