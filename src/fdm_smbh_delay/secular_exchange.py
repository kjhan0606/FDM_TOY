"""Orbit-resolved averages for SMBH energy and angular-momentum exchange."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_trapezoid = getattr(np, "trapezoid", np.trapz)


@dataclass(frozen=True)
class CycleAveragedSeries:
    cycle_index: np.ndarray
    start_time: np.ndarray
    end_time: np.ndarray
    mean_time: np.ndarray
    duration: np.ndarray
    start_value: np.ndarray
    end_value: np.ndarray
    mean_value: np.ndarray
    rate: np.ndarray


@dataclass(frozen=True)
class BlockBootstrapInterval:
    estimate: float
    lower_95: float
    upper_95: float
    block_length: int
    samples: int


def moving_block_bootstrap_rate(
    *,
    rate: np.ndarray,
    duration: np.ndarray,
    block_length: int,
    samples: int = 2000,
    seed: int = 1729,
) -> BlockBootstrapInterval:
    """Estimate a time-weighted mean and its interval from circular blocks.

    The input cycles should cover a local interval over which a stationary
    mean is physically meaningful.  The interval measures correlated
    cycle-to-cycle variation and excludes numerical-resolution systematics.
    """

    rate_array = np.asarray(rate, dtype=float)
    duration_array = np.asarray(duration, dtype=float)
    if (
        rate_array.ndim != 1
        or duration_array.shape != rate_array.shape
        or rate_array.size < 2
    ):
        raise ValueError("rate and duration must be equal vectors with two samples")
    if np.any(~np.isfinite(rate_array)) or np.any(~np.isfinite(duration_array)):
        raise ValueError("bootstrap inputs must be finite")
    if np.any(duration_array <= 0.0):
        raise ValueError("cycle durations must be positive")
    if not 1 <= block_length <= rate_array.size:
        raise ValueError("block length must lie within the input series")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")

    estimate = float(np.sum(rate_array * duration_array) / np.sum(duration_array))
    generator = np.random.default_rng(seed)
    block_count = int(np.ceil(rate_array.size / block_length))
    offsets = np.arange(block_length)
    bootstrap_means = np.empty(samples)
    for draw in range(samples):
        starts = generator.integers(0, rate_array.size, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % rate_array.size).ravel()
        indices = indices[: rate_array.size]
        bootstrap_means[draw] = np.sum(
            rate_array[indices] * duration_array[indices]
        ) / np.sum(duration_array[indices])
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return BlockBootstrapInterval(
        estimate=estimate,
        lower_95=float(lower),
        upper_95=float(upper),
        block_length=block_length,
        samples=samples,
    )


def unwrapped_orbital_phase(
    displacement: np.ndarray, relative_velocity: np.ndarray
) -> np.ndarray:
    """Measure orbital phase in the plane normal to the initial angular momentum."""

    position = np.asarray(displacement, dtype=float)
    velocity = np.asarray(relative_velocity, dtype=float)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("displacement must have shape (samples, 3)")
    if velocity.shape != position.shape:
        raise ValueError("relative velocity must match the displacement shape")
    if position.shape[0] < 2:
        raise ValueError("at least two orbital states are required")
    if np.any(~np.isfinite(position)) or np.any(~np.isfinite(velocity)):
        raise ValueError("orbital states must be finite")

    angular_momentum = np.cross(position[0], velocity[0])
    angular_momentum_norm = float(np.linalg.norm(angular_momentum))
    position_norm = float(np.linalg.norm(position[0]))
    if angular_momentum_norm <= 0.0 or position_norm <= 0.0:
        raise ValueError("the initial state must define an orbital plane")
    normal = angular_momentum / angular_momentum_norm
    radial_basis = position[0] / position_norm
    transverse_basis = np.cross(normal, radial_basis)
    phase = np.unwrap(
        np.arctan2(position @ transverse_basis, position @ radial_basis)
    )
    if np.any(np.diff(phase) <= 0.0):
        raise ValueError("orbital phase must increase between saved states")
    return phase


def phase_cycle_average(
    *, time: np.ndarray, phase: np.ndarray, value: np.ndarray
) -> CycleAveragedSeries:
    """Average one quantity between successive equal-phase orbital boundaries."""

    time_array = np.asarray(time, dtype=float)
    phase_array = np.asarray(phase, dtype=float)
    value_array = np.asarray(value, dtype=float)
    if not (
        time_array.ndim == phase_array.ndim == value_array.ndim == 1
        and time_array.size == phase_array.size == value_array.size
    ):
        raise ValueError("time, phase, and value must be equal-length vectors")
    if time_array.size < 2:
        raise ValueError("at least two samples are required")
    if np.any(
        ~np.isfinite(time_array)
        | ~np.isfinite(phase_array)
        | ~np.isfinite(value_array)
    ):
        raise ValueError("cycle inputs must be finite")
    if np.any(np.diff(time_array) <= 0.0) or np.any(np.diff(phase_array) <= 0.0):
        raise ValueError("time and phase must increase strictly")

    cycle_count = int(
        np.floor((phase_array[-1] - phase_array[0]) / (2.0 * np.pi) + 1.0e-12)
    )
    if cycle_count < 1:
        raise ValueError("the saved interval contains no complete orbit")
    boundary_phase = phase_array[0] + 2.0 * np.pi * np.arange(cycle_count + 1)
    boundary_time = np.interp(boundary_phase, phase_array, time_array)
    boundary_value = np.interp(boundary_phase, phase_array, value_array)

    mean_value = np.empty(cycle_count)
    for index in range(cycle_count):
        inside = (phase_array > boundary_phase[index]) & (
            phase_array < boundary_phase[index + 1]
        )
        segment_time = np.concatenate(
            ([boundary_time[index]], time_array[inside], [boundary_time[index + 1]])
        )
        segment_value = np.concatenate(
            ([boundary_value[index]], value_array[inside], [boundary_value[index + 1]])
        )
        duration = boundary_time[index + 1] - boundary_time[index]
        mean_value[index] = _trapezoid(segment_value, segment_time) / duration

    start_time = boundary_time[:-1]
    end_time = boundary_time[1:]
    duration = end_time - start_time
    start_value = boundary_value[:-1]
    end_value = boundary_value[1:]
    return CycleAveragedSeries(
        cycle_index=np.arange(cycle_count, dtype=int),
        start_time=start_time,
        end_time=end_time,
        mean_time=0.5 * (start_time + end_time),
        duration=duration,
        start_value=start_value,
        end_value=end_value,
        mean_value=mean_value,
        rate=(end_value - start_value) / duration,
    )
