"""Preflight a resolution-paired non-compacting CDM SMBH zoom manifest."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .constants import G_INTERNAL
from .zoom_calibration import ZoomGrid, load_zoom_grid


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and non-negative") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        numeric = float(value)
        result = int(numeric)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if result != numeric or result < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return result


@dataclass(frozen=True)
class CDMNonCompactingZoomPlan:
    """Static requirements before any CDM kpc-to-pc zoom is submitted."""

    grid: ZoomGrid
    target_separation_pc: float
    smbh_merge_radius_cells: float
    maximum_output_cadence_myr: float
    samples_per_secular_block: int
    minimum_secular_blocks: int
    position_coordinate: str
    time_coordinate: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_separation_pc",
            _positive(self.target_separation_pc, "target_separation_pc"),
        )
        object.__setattr__(
            self,
            "smbh_merge_radius_cells",
            _nonnegative(self.smbh_merge_radius_cells, "smbh_merge_radius_cells"),
        )
        object.__setattr__(
            self,
            "maximum_output_cadence_myr",
            _positive(self.maximum_output_cadence_myr, "maximum_output_cadence_myr"),
        )
        object.__setattr__(
            self,
            "samples_per_secular_block",
            _integer(self.samples_per_secular_block, "samples_per_secular_block", minimum=5),
        )
        object.__setattr__(
            self,
            "minimum_secular_blocks",
            _integer(self.minimum_secular_blocks, "minimum_secular_blocks", minimum=3),
        )
        if self.position_coordinate not in {"comoving", "physical"}:
            raise ValueError("position_coordinate must be comoving or physical")
        if self.time_coordinate != "proper":
            raise ValueError("time_coordinate must be proper")

    @property
    def minimum_complete_outputs(self) -> int:
        return self.samples_per_secular_block * self.minimum_secular_blocks

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "cdm_noncompacting_zoom_preflight",
            "interpretation": (
                "pre-submission CDM zoom requirements only; this does not submit "
                "a run, accept a rate, or estimate a physical coalescence time"
            ),
            "manifest_sha256": self.grid.manifest_sha256,
            "case_count": len(self.grid.cases),
            "target_separation_pc": self.target_separation_pc,
            "smbh_merge_radius_cells": self.smbh_merge_radius_cells,
            "maximum_output_cadence_myr": self.maximum_output_cadence_myr,
            "samples_per_secular_block": self.samples_per_secular_block,
            "minimum_secular_blocks": self.minimum_secular_blocks,
            "minimum_complete_outputs": self.minimum_complete_outputs,
            "position_coordinate": self.position_coordinate,
            "time_coordinate": self.time_coordinate,
            "execution": "operator Slurm submission only after this preflight",
        }


def _target_orbital_period_myr(total_mass_msun: float, separation_pc: float) -> float:
    return 2.0 * math.pi * math.sqrt(separation_pc**3 / (G_INTERNAL * total_mass_msun))


def preflight_cdm_noncompacting_zoom(plan: CDMNonCompactingZoomPlan) -> CDMNonCompactingZoomPlan:
    """Validate the no-finite-radius-compaction resolution and cadence gates."""

    if not plan.grid.cases:
        raise ValueError("CDM non-compacting zoom manifest is empty")
    if plan.smbh_merge_radius_cells != 0.0:
        raise ValueError("CDM non-compacting zoom requires smbh_merge_radius_cells=0")
    grouped: dict[tuple[str, int], list[Any]] = {}
    phase_replicates: dict[str, set[int]] = {}
    for case in plan.grid.cases:
        physics = case.physics
        if physics.dark_matter_model != "cdm":
            raise ValueError("CDM non-compacting zoom manifest contains a non-CDM case")
        if physics.initial_separation_pc <= plan.target_separation_pc:
            raise ValueError("CDM zoom initial separation must exceed the target separation")
        if plan.target_separation_pc / case.numerics.finest_cell_size_pc < 4.0:
            raise ValueError("CDM zoom target violates the four-cell resolution gate")
        if case.numerics.minimum_softening_pc > case.numerics.finest_cell_size_pc:
            raise ValueError("CDM zoom softening cannot exceed the finest cell size")
        target_period = _target_orbital_period_myr(
            physics.binary_total_mass_msun,
            plan.target_separation_pc,
        )
        if plan.maximum_output_cadence_myr > target_period / 16.0:
            raise ValueError("CDM zoom output cadence is too coarse at the target orbit")
        grouped.setdefault((physics.physics_id, case.replicate), []).append(case)
        phase_replicates.setdefault(physics.physics_id, set()).add(case.replicate)
    if any(len(group) < 2 for group in grouped.values()):
        raise ValueError("CDM non-compacting zoom requires a resolution pair per phase")
    if any(len(replicates) < 2 for replicates in phase_replicates.values()):
        raise ValueError("CDM non-compacting zoom requires at least two phase replicas")
    return plan


def load_cdm_noncompacting_zoom_plan(path: str | Path) -> CDMNonCompactingZoomPlan:
    """Load one CDM zoom grid plus its explicit non-compacting controls."""

    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read CDM non-compacting zoom manifest: {error}") from error
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("CDM non-compacting zoom manifest schema is unsupported")
    controls = raw.get("noncompacting")
    expected_controls = {
        "target_separation_pc",
        "smbh_merge_radius_cells",
        "maximum_output_cadence_myr",
        "samples_per_secular_block",
        "minimum_secular_blocks",
        "position_coordinate",
        "time_coordinate",
    }
    if not isinstance(controls, Mapping) or set(controls) != expected_controls:
        raise ValueError("CDM non-compacting zoom controls are invalid")
    plan = CDMNonCompactingZoomPlan(
        grid=load_zoom_grid(source),
        target_separation_pc=controls.get("target_separation_pc"),
        smbh_merge_radius_cells=controls.get("smbh_merge_radius_cells"),
        maximum_output_cadence_myr=controls.get("maximum_output_cadence_myr"),
        samples_per_secular_block=controls.get("samples_per_secular_block"),
        minimum_secular_blocks=controls.get("minimum_secular_blocks"),
        position_coordinate=controls.get("position_coordinate"),
        time_coordinate=controls.get("time_coordinate"),
    )
    return preflight_cdm_noncompacting_zoom(plan)
