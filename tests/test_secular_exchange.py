import numpy as np
import pytest

from fdm_smbh_delay.secular_exchange import (
    phase_cycle_average,
    unwrapped_orbital_phase,
)


def test_unwrapped_phase_counts_multiple_circular_orbits() -> None:
    phase = np.linspace(0.0, 4.5 * np.pi, 100)
    displacement = np.column_stack(
        (np.cos(phase), np.sin(phase), np.zeros_like(phase))
    )
    velocity = np.column_stack(
        (-np.sin(phase), np.cos(phase), np.zeros_like(phase))
    )
    measured = unwrapped_orbital_phase(displacement, velocity)
    np.testing.assert_allclose(measured, phase, atol=1.0e-14)


def test_phase_cycle_average_recovers_linear_rate() -> None:
    time = np.linspace(0.0, 2.4, 49)
    phase = 2.0 * np.pi * time
    value = 2.0 + 3.0 * time
    result = phase_cycle_average(time=time, phase=phase, value=value)
    assert result.cycle_index.size == 2
    np.testing.assert_allclose(result.duration, 1.0)
    np.testing.assert_allclose(result.rate, 3.0)
    np.testing.assert_allclose(result.mean_value, [3.5, 6.5])


def test_cycle_average_requires_a_complete_orbit() -> None:
    time = np.linspace(0.0, 0.9, 10)
    with pytest.raises(ValueError, match="no complete orbit"):
        phase_cycle_average(
            time=time, phase=2.0 * np.pi * time, value=np.ones_like(time)
        )
