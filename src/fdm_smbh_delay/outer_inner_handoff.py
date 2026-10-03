"""Censor-first validation for an outer-FDM to inner-wave handoff."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HandoffRatePoint:
    """Orbit-averaged exchange diagnostics at one separation."""

    separation_pc: float
    orbital_power_pc2_myr3: float
    orbital_torque_msun_pc2_myr: float
    eccentricity: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.separation_pc,
                self.orbital_power_pc2_myr3,
                self.orbital_torque_msun_pc2_myr,
                self.eccentricity,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("handoff rate point must be finite")
        if self.separation_pc <= 0.0:
            raise ValueError("handoff separation must be positive")
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("handoff eccentricity must lie in [0, 1)")


@dataclass(frozen=True)
class HandoffSimilarityState:
    """Dimensionless state that must agree across the two calculations."""

    fdm_particle_mass_ev: float
    soliton_mass_msun: float
    core_radius_pc: float
    eta_sp: float
    binary_mass_ratio: float
    binary_to_soliton_mass_ratio: float
    eccentricity: float
    semimajor_axis_over_core: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.fdm_particle_mass_ev,
                self.soliton_mass_msun,
                self.core_radius_pc,
                self.eta_sp,
                self.binary_mass_ratio,
                self.binary_to_soliton_mass_ratio,
                self.eccentricity,
                self.semimajor_axis_over_core,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("finite similarity parameters are required")
        if np.any(values[[0, 1, 2, 3, 4, 5, 7]] <= 0.0):
            raise ValueError("positive similarity parameters are required")
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("similarity eccentricity must lie in [0, 1)")


@dataclass(frozen=True)
class HandoffGateConfig:
    minimum_overlap_factor: float = 2.0
    minimum_overlap_points: int = 3
    maximum_rate_fractional_difference: float = 0.20
    maximum_eccentricity_difference: float = 0.02
    maximum_similarity_fractional_difference: float = 0.02
    maximum_log_separation_match: float = 0.15
    rate_floor_fraction: float = 1.0e-8

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.minimum_overlap_factor,
                self.maximum_rate_fractional_difference,
                self.maximum_eccentricity_difference,
                self.maximum_similarity_fractional_difference,
                self.maximum_log_separation_match,
                self.rate_floor_fraction,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("handoff gate controls must be positive and finite")
        if self.minimum_overlap_factor <= 1.0:
            raise ValueError("minimum_overlap_factor must exceed one")
        if self.minimum_overlap_points < 2:
            raise ValueError("minimum_overlap_points must be at least two")


@dataclass(frozen=True)
class HandoffDecision:
    """A non-throwing physical handoff decision."""

    status: str
    overlap_low_pc: float | None
    overlap_high_pc: float | None
    matched_points: int
    maximum_power_fractional_difference: float | None
    maximum_torque_fractional_difference: float | None
    maximum_eccentricity_difference: float | None
    reasons: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def _validate_points(points: tuple[HandoffRatePoint, ...], name: str) -> None:
    if len(points) < 2:
        raise ValueError(f"{name} must contain at least two points")
    separations = np.array([point.separation_pc for point in points], dtype=float)
    if np.any(np.diff(separations) <= 0.0):
        raise ValueError(f"{name} separations must be strictly increasing")


def _fractional_difference(left: float, right: float, floor: float) -> float:
    denominator = max(abs(left), abs(right), floor)
    return abs(left - right) / denominator


def _similarity_reasons(
    outer: HandoffSimilarityState,
    inner: HandoffSimilarityState,
    config: HandoffGateConfig,
) -> list[str]:
    reasons: list[str] = []
    for name in (
        "fdm_particle_mass_ev",
        "soliton_mass_msun",
        "core_radius_pc",
        "eta_sp",
        "binary_mass_ratio",
        "binary_to_soliton_mass_ratio",
        "semimajor_axis_over_core",
    ):
        left = float(getattr(outer, name))
        right = float(getattr(inner, name))
        if _fractional_difference(left, right, 1.0e-300) > config.maximum_similarity_fractional_difference:
            reasons.append(f"similarity mismatch: {name}")
    if abs(outer.eccentricity - inner.eccentricity) > config.maximum_eccentricity_difference:
        reasons.append("similarity mismatch: eccentricity")
    return reasons


def validate_outer_inner_handoff(
    *,
    outer_points: tuple[HandoffRatePoint, ...],
    inner_points: tuple[HandoffRatePoint, ...],
    outer_similarity: HandoffSimilarityState,
    inner_similarity: HandoffSimilarityState,
    config: HandoffGateConfig = HandoffGateConfig(),
) -> HandoffDecision:
    """Validate a positive-width, single-counted handoff.

    The two point sets are compared in log separation.  This function never
    combines delays and never interpolates a rate beyond the supplied points;
    a failed physical gate is returned as ``censored``.
    """

    _validate_points(outer_points, "outer_points")
    _validate_points(inner_points, "inner_points")
    reasons = _similarity_reasons(outer_similarity, inner_similarity, config)

    outer_radii = np.array([point.separation_pc for point in outer_points])
    inner_radii = np.array([point.separation_pc for point in inner_points])
    overlap_low = max(outer_radii[0], inner_radii[0])
    overlap_high = min(outer_radii[-1], inner_radii[-1])
    if overlap_high <= overlap_low:
        reasons.append("outer and inner supports do not overlap")
        return HandoffDecision(
            "censored", None, None, 0, None, None, None, tuple(reasons)
        )
    if overlap_high / overlap_low < config.minimum_overlap_factor:
        reasons.append("overlap has insufficient separation width")

    outer_candidates = [
        point for point in outer_points if overlap_low <= point.separation_pc <= overlap_high
    ]
    inner_candidates = [
        point for point in inner_points if overlap_low <= point.separation_pc <= overlap_high
    ]
    pairs: list[tuple[HandoffRatePoint, HandoffRatePoint]] = []
    used_inner: set[int] = set()
    for outer in outer_candidates:
        distances = [
            (abs(math.log(inner.separation_pc / outer.separation_pc)), index)
            for index, inner in enumerate(inner_candidates)
            if index not in used_inner
        ]
        if not distances:
            continue
        distance, index = min(distances)
        if distance <= config.maximum_log_separation_match:
            pairs.append((outer, inner_candidates[index]))
            used_inner.add(index)

    if len(pairs) < config.minimum_overlap_points:
        reasons.append("overlap has too few matched rate points")

    all_power = [
        abs(point.orbital_power_pc2_myr3) for point in outer_points + inner_points
    ]
    all_torque = [
        abs(point.orbital_torque_msun_pc2_myr)
        for point in outer_points + inner_points
    ]
    power_floor = max(max(all_power, default=0.0), np.finfo(float).tiny) * config.rate_floor_fraction
    torque_floor = max(max(all_torque, default=0.0), np.finfo(float).tiny) * config.rate_floor_fraction
    power_differences: list[float] = []
    torque_differences: list[float] = []
    eccentricity_differences: list[float] = []
    unresolved = False
    for outer, inner in pairs:
        if (
            abs(outer.orbital_power_pc2_myr3) <= power_floor
            or abs(inner.orbital_power_pc2_myr3) <= power_floor
        ):
            unresolved = True
        if (
            abs(outer.orbital_torque_msun_pc2_myr) <= torque_floor
            or abs(inner.orbital_torque_msun_pc2_myr) <= torque_floor
        ):
            unresolved = True
        if np.sign(outer.orbital_power_pc2_myr3) != np.sign(inner.orbital_power_pc2_myr3):
            reasons.append("orbital power changes sign across handoff")
        if np.sign(outer.orbital_torque_msun_pc2_myr) != np.sign(inner.orbital_torque_msun_pc2_myr):
            reasons.append("orbital torque changes sign across handoff")
        power_differences.append(
            _fractional_difference(
                outer.orbital_power_pc2_myr3,
                inner.orbital_power_pc2_myr3,
                power_floor,
            )
        )
        torque_differences.append(
            _fractional_difference(
                outer.orbital_torque_msun_pc2_myr,
                inner.orbital_torque_msun_pc2_myr,
                torque_floor,
            )
        )
        eccentricity_differences.append(abs(outer.eccentricity - inner.eccentricity))

    if unresolved:
        reasons.append("one or more paired power/torque rates are unresolved")

    maximum_power = max(power_differences, default=None)
    maximum_torque = max(torque_differences, default=None)
    maximum_eccentricity = max(eccentricity_differences, default=None)
    if maximum_power is not None and maximum_power > config.maximum_rate_fractional_difference:
        reasons.append("orbital power rates disagree beyond the handoff gate")
    if maximum_torque is not None and maximum_torque > config.maximum_rate_fractional_difference:
        reasons.append("orbital torque rates disagree beyond the handoff gate")
    if maximum_eccentricity is not None and maximum_eccentricity > config.maximum_eccentricity_difference:
        reasons.append("eccentricity evolution disagrees beyond the handoff gate")

    return HandoffDecision(
        "accepted" if not reasons else "censored",
        overlap_low,
        overlap_high,
        len(pairs),
        maximum_power,
        maximum_torque,
        maximum_eccentricity,
        tuple(dict.fromkeys(reasons)),
    )
