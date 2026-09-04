import numpy as np
import pytest

from fdm_smbh_delay.secular_exchange import (
    moving_block_bootstrap_rate,
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


def test_velocity_guided_phase_resolves_advances_larger_than_pi() -> None:
    phase = np.array([0.0, 0.8 * np.pi, 1.9 * np.pi, 2.95 * np.pi])
    displacement = np.column_stack(
        (np.cos(phase), np.sin(phase), np.zeros_like(phase))
    )
    velocity = np.column_stack(
        (-np.sin(phase), np.cos(phase), np.zeros_like(phase))
    )

    measured = unwrapped_orbital_phase(
        displacement,
        velocity,
        time=phase,
    )

    np.testing.assert_allclose(measured, phase, atol=1.0e-14)


def test_velocity_guided_phase_rejects_retrograde_sample() -> None:
    phase = np.array([0.0, 0.2, 0.1])
    displacement = np.column_stack(
        (np.cos(phase), np.sin(phase), np.zeros_like(phase))
    )
    velocity = np.column_stack(
        (
            [-np.sin(phase[0]), -np.sin(phase[1]), np.sin(phase[2])],
            [np.cos(phase[0]), np.cos(phase[1]), -np.cos(phase[2])],
            np.zeros_like(phase),
        )
    )

    with pytest.raises(ValueError, match="phase velocity must remain positive"):
        unwrapped_orbital_phase(displacement, velocity, time=np.arange(3.0))


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


def test_cycle_average_can_mark_optional_nonfinite_diagnostic_cycles() -> None:
    time = np.linspace(0.0, 3.0, 13)
    phase = 2.0 * np.pi * time
    value = 2.0 + 3.0 * time
    value[1] = np.nan

    with pytest.raises(ValueError, match="cycle inputs must be finite"):
        phase_cycle_average(time=time, phase=phase, value=value)

    result = phase_cycle_average(
        time=time,
        phase=phase,
        value=value,
        allow_nonfinite_value=True,
    )
    assert result.cycle_index.size == 3
    assert np.isnan(result.mean_value[0])
    assert np.isnan(result.rate[0])
    np.testing.assert_allclose(result.mean_value[1:], [6.5, 9.5])
    np.testing.assert_allclose(result.rate[1:], 3.0)


def test_block_bootstrap_preserves_constant_rate() -> None:
    interval = moving_block_bootstrap_rate(
        rate=np.full(12, -3.0),
        duration=np.linspace(0.8, 1.2, 12),
        block_length=4,
        samples=100,
        seed=11,
    )
    assert interval.estimate == pytest.approx(-3.0)
    assert interval.lower_95 == pytest.approx(-3.0)
    assert interval.upper_95 == pytest.approx(-3.0)


def test_block_bootstrap_is_reproducible() -> None:
    arguments = {
        "rate": np.arange(10.0),
        "duration": np.ones(10),
        "block_length": 3,
        "samples": 80,
        "seed": 7,
    }
    first = moving_block_bootstrap_rate(**arguments)
    second = moving_block_bootstrap_rate(**arguments)
    assert first == second
