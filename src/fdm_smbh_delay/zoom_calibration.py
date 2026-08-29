"""Controlled galaxy-merger zoom manifests and convergence-gated delay rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import yaml

from .delay_budget import DelaySegment


ZOOM_SCHEMA_VERSION = 1
ZOOM_STAGE_NAMES = (
    "numerical_capture",
    "common_nucleus",
    "bound_binary",
    "hard_binary",
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ZoomPhysicsPoint:
    host_stellar_mass_msun: float
    host_scale_radius_pc: float
    host_inner_slope: float
    binary_total_mass_msun: float
    mass_ratio_q: float
    gas_fraction: float
    gas_rotation_fraction: float
    initial_orbit_eccentricity: float
    initial_separation_pc: float
    nuclear_envelope_to_secondary_bh_mass: float
    dark_matter_model: str
    fdm_particle_mass_ev: float | None = None
    fdm_core_radius_pc: float | None = None
    fdm_soliton_mass_msun: float | None = None

    def __post_init__(self) -> None:
        stellar_mass = float(self.host_stellar_mass_msun)
        if not np.isfinite(stellar_mass) or stellar_mass < 0.0:
            raise ValueError("host_stellar_mass_msun must be finite and non-negative")
        positive = np.asarray(
            [
                self.host_scale_radius_pc,
                self.binary_total_mass_msun,
                self.initial_separation_pc,
            ],
            dtype=float,
        )
        bounded = np.asarray(
            [
                self.host_inner_slope,
                self.mass_ratio_q,
                self.gas_fraction,
                self.gas_rotation_fraction,
                self.initial_orbit_eccentricity,
                self.nuclear_envelope_to_secondary_bh_mass,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(positive)) or np.any(positive <= 0.0):
            raise ValueError("zoom masses, scales, and initial separation must be positive")
        if np.any(~np.isfinite(bounded)):
            raise ValueError("zoom dimensionless coordinates must be finite")
        if not 0.0 <= self.host_inner_slope < 3.0:
            raise ValueError("host_inner_slope must satisfy 0 <= gamma < 3")
        if not 0.0 < self.mass_ratio_q <= 1.0:
            raise ValueError("mass_ratio_q must satisfy 0 < q <= 1")
        if not 0.0 <= self.gas_fraction < 1.0:
            raise ValueError("gas_fraction must lie in [0, 1)")
        if not 0.0 <= self.gas_rotation_fraction <= 1.0:
            raise ValueError("gas_rotation_fraction must lie in [0, 1]")
        if not 0.0 <= self.initial_orbit_eccentricity < 1.0:
            raise ValueError("initial orbit eccentricity must lie in [0, 1)")
        if self.nuclear_envelope_to_secondary_bh_mass < 0.0:
            raise ValueError("nuclear envelope mass ratio cannot be negative")
        if self.dark_matter_model not in {"cdm", "fdm"}:
            raise ValueError("dark_matter_model must be cdm or fdm")
        fdm_values = (
            self.fdm_particle_mass_ev,
            self.fdm_core_radius_pc,
            self.fdm_soliton_mass_msun,
        )
        if self.dark_matter_model == "fdm":
            if any(value is None for value in fdm_values) or any(
                not np.isfinite(value) or value <= 0.0
                for value in fdm_values
                if value is not None
            ):
                raise ValueError("FDM zooms require positive particle/core/soliton values")
        elif any(value is not None for value in fdm_values):
            raise ValueError("CDM zooms cannot carry FDM-only coordinates")

    @property
    def physics_id(self) -> str:
        return "zoomphys-" + _canonical_sha256(asdict(self))[:16]


@dataclass(frozen=True)
class ZoomNumerics:
    levelmax: int
    finest_cell_size_pc: float
    collisionless_particle_mass_msun: float
    minimum_softening_pc: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.finest_cell_size_pc,
                self.collisionless_particle_mass_msun,
                self.minimum_softening_pc,
            ],
            dtype=float,
        )
        if self.levelmax < 1 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("zoom numerical resolution is invalid")


@dataclass(frozen=True)
class GalaxyMergerZoomCase:
    physics: ZoomPhysicsPoint
    numerics: ZoomNumerics
    replicate: int = 0

    def __post_init__(self) -> None:
        if self.replicate < 0:
            raise ValueError("zoom replicate must be non-negative")

    @property
    def case_id(self) -> str:
        return (
            f"{self.physics.physics_id}-l{self.numerics.levelmax}"
            f"-r{self.replicate:02d}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ZOOM_SCHEMA_VERSION,
            "case_id": self.case_id,
            "physics_id": self.physics.physics_id,
            "replicate": self.replicate,
            "physics": asdict(self.physics),
            "numerics": asdict(self.numerics),
        }


@dataclass(frozen=True)
class ZoomGrid:
    cases: tuple[GalaxyMergerZoomCase, ...]
    manifest_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ZOOM_SCHEMA_VERSION,
            "status": "galaxy_merger_zoom_manifest",
            "manifest_sha256": self.manifest_sha256,
            "cases": [case.as_dict() for case in self.cases],
        }


def _physics_from_mapping(mapping: dict[str, Any]) -> ZoomPhysicsPoint:
    converted = dict(mapping)
    for field in (
        "host_stellar_mass_msun",
        "host_scale_radius_pc",
        "host_inner_slope",
        "binary_total_mass_msun",
        "mass_ratio_q",
        "gas_fraction",
        "gas_rotation_fraction",
        "initial_orbit_eccentricity",
        "initial_separation_pc",
        "nuclear_envelope_to_secondary_bh_mass",
        "fdm_particle_mass_ev",
        "fdm_core_radius_pc",
        "fdm_soliton_mass_msun",
    ):
        if converted.get(field) is not None:
            try:
                converted[field] = float(converted[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"zoom field {field} must be numeric") from error
    return ZoomPhysicsPoint(**converted)


def _numerics_from_mapping(mapping: dict[str, Any]) -> ZoomNumerics:
    try:
        return ZoomNumerics(
            levelmax=int(mapping["levelmax"]),
            finest_cell_size_pc=float(mapping["finest_cell_size_pc"]),
            collisionless_particle_mass_msun=float(
                mapping["collisionless_particle_mass_msun"]
            ),
            minimum_softening_pc=float(mapping["minimum_softening_pc"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("zoom numerical record is incomplete or non-numeric") from error


def build_zoom_grid(specification: dict[str, Any], *, maximum_cases: int = 1000) -> ZoomGrid:
    """Build a baseline-plus-variations grid with a resolution pair per point."""

    if specification.get("schema_version") != ZOOM_SCHEMA_VERSION:
        raise ValueError("zoom-grid schema_version must be 1")
    baseline_mapping = specification.get("baseline")
    variations = specification.get("variations", [])
    numerics_records = specification.get("numerics")
    replicates = int(specification.get("replicates", 1))
    if (
        not isinstance(baseline_mapping, dict)
        or not isinstance(variations, list)
        or not isinstance(numerics_records, list)
        or len(numerics_records) < 2
        or replicates < 1
        or maximum_cases < 1
    ):
        raise ValueError("zoom-grid baseline, variations, numerics, or replicates are invalid")

    physics_points = [_physics_from_mapping(baseline_mapping)]
    for variation in variations:
        if not isinstance(variation, dict) or not variation:
            raise ValueError("every zoom variation must be a non-empty mapping")
        unknown = set(variation) - set(baseline_mapping)
        if unknown:
            raise ValueError(f"zoom variation has unknown fields: {sorted(unknown)}")
        physics_points.append(_physics_from_mapping(baseline_mapping | variation))
    physics_ids = [point.physics_id for point in physics_points]
    if len(physics_ids) != len(set(physics_ids)):
        raise ValueError("zoom physical points are duplicated")

    numerics = [_numerics_from_mapping(record) for record in numerics_records]
    levels = [record.levelmax for record in numerics]
    if len(levels) != len(set(levels)):
        raise ValueError("zoom numerical levels are duplicated")
    cell_sizes = [record.finest_cell_size_pc for record in numerics]
    if cell_sizes != sorted(cell_sizes, reverse=True):
        raise ValueError("zoom numerics must be ordered from coarse to fine")

    cases = tuple(
        GalaxyMergerZoomCase(point, numerical, replicate)
        for point in physics_points
        for replicate in range(replicates)
        for numerical in numerics
    )
    if len(cases) > maximum_cases:
        raise ValueError(
            f"zoom grid has {len(cases)} cases, exceeding maximum_cases={maximum_cases}"
        )
    manifest_payload = [case.as_dict() for case in cases]
    return ZoomGrid(cases, _canonical_sha256(manifest_payload))


def load_zoom_grid(path: str | Path, *, maximum_cases: int = 1000) -> ZoomGrid:
    resolved = Path(path).expanduser().resolve()
    specification = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(specification, dict):
        raise ValueError("zoom-grid file must contain a mapping")
    return build_zoom_grid(specification, maximum_cases=maximum_cases)


@dataclass(frozen=True)
class ZoomStageResult:
    status: str
    elapsed_since_capture_myr: float | None
    separation_pc: float | None

    def __post_init__(self) -> None:
        if self.status not in {"complete", "timeout", "not_reached", "invalid"}:
            raise ValueError("unsupported zoom stage status")
        values = [
            value
            for value in (self.elapsed_since_capture_myr, self.separation_pc)
            if value is not None
        ]
        if np.any(~np.isfinite(values)) or np.any(np.asarray(values) < 0.0):
            raise ValueError("zoom stage values must be finite and non-negative")
        if self.status == "complete" and (
            self.elapsed_since_capture_myr is None or self.separation_pc is None
        ):
            raise ValueError("complete zoom stages require time and separation")


@dataclass(frozen=True)
class ZoomRunResult:
    case: GalaxyMergerZoomCase
    source_path: Path
    source_sha256: str
    capture_event_uid: str
    stages: tuple[tuple[str, ZoomStageResult], ...]
    analytic_kpc_to_hard_delay_myr: float | None
    maximum_relative_energy_error: float
    maximum_relative_angular_momentum_error: float
    minimum_transition_radius_cells: float
    integration_time_myr: float

    def stage(self, name: str) -> ZoomStageResult:
        return dict(self.stages)[name]


def read_zoom_result(path: str | Path, case: GalaxyMergerZoomCase) -> ZoomRunResult:
    resolved = Path(path).expanduser().resolve()
    record = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        record.get("schema_version") != ZOOM_SCHEMA_VERSION
        or record.get("case_id") != case.case_id
        or record.get("case") != case.as_dict()
        or not record.get("capture_event_uid")
    ):
        raise ValueError("zoom result provenance does not match its manifest case")
    stage_records = record.get("stages")
    if not isinstance(stage_records, dict) or set(stage_records) != set(ZOOM_STAGE_NAMES):
        raise ValueError("zoom result must contain every named stage")
    stages = tuple(
        (
            name,
            ZoomStageResult(
                status=stage_records[name]["status"],
                elapsed_since_capture_myr=stage_records[name].get(
                    "elapsed_since_capture_myr"
                ),
                separation_pc=stage_records[name].get("separation_pc"),
            ),
        )
        for name in ZOOM_STAGE_NAMES
    )
    complete_times = [
        stage.elapsed_since_capture_myr
        for _, stage in stages
        if stage.status == "complete"
    ]
    if complete_times != sorted(complete_times):
        raise ValueError("zoom stage times are not physically ordered")
    first_incomplete = next(
        (index for index, (_, stage) in enumerate(stages) if stage.status != "complete"),
        len(stages),
    )
    if any(stage.status == "complete" for _, stage in stages[first_incomplete + 1 :]):
        raise ValueError("a later zoom stage completed after an incomplete stage")
    capture_stage = dict(stages)["numerical_capture"]
    if (
        capture_stage.status != "complete"
        or capture_stage.elapsed_since_capture_myr != 0.0
    ):
        raise ValueError("numerical_capture must be a complete zero-time boundary")
    complete_separations = [
        stage.separation_pc
        for _, stage in stages
        if stage.status == "complete"
    ]
    if complete_separations != sorted(complete_separations, reverse=True):
        raise ValueError("zoom stage separations are not monotonically decreasing")

    analytic = record.get("analytic_kpc_to_hard_delay_myr")
    if analytic is not None:
        analytic = float(analytic)
        if not np.isfinite(analytic) or analytic <= 0.0:
            raise ValueError("analytic zoom delay must be finite and positive")
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("zoom result diagnostics are absent")
    values = np.asarray(
        [
            diagnostics["maximum_relative_energy_error"],
            diagnostics["maximum_relative_angular_momentum_error"],
            diagnostics["minimum_transition_radius_cells"],
            record["integration_time_myr"],
        ],
        dtype=float,
    )
    if (
        np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or values[2] <= 0.0
        or values[3] <= 0.0
    ):
        raise ValueError("zoom result diagnostics are invalid")
    return ZoomRunResult(
        case=case,
        source_path=resolved,
        source_sha256=_file_sha256(resolved),
        capture_event_uid=str(record["capture_event_uid"]),
        stages=stages,
        analytic_kpc_to_hard_delay_myr=analytic,
        maximum_relative_energy_error=float(values[0]),
        maximum_relative_angular_momentum_error=float(values[1]),
        minimum_transition_radius_cells=float(values[2]),
        integration_time_myr=float(values[3]),
    )


@dataclass(frozen=True)
class ZoomConvergenceResult:
    status: str
    reference: ZoomRunResult
    comparison: ZoomRunResult
    maximum_stage_delay_systematic_fraction: float | None
    reasons: tuple[str, ...]


def compare_zoom_resolution_pair(
    reference: ZoomRunResult,
    comparison: ZoomRunResult,
    *,
    maximum_stage_delay_systematic_fraction: float = 0.20,
    maximum_conservation_error: float = 1.0e-3,
    minimum_transition_radius_cells: float = 4.0,
) -> ZoomConvergenceResult:
    controls = np.asarray(
        [
            maximum_stage_delay_systematic_fraction,
            maximum_conservation_error,
            minimum_transition_radius_cells,
        ]
    )
    if np.any(~np.isfinite(controls)) or np.any(controls < 0.0):
        raise ValueError("zoom convergence gates are invalid")
    if reference.case.physics != comparison.case.physics or (
        reference.case.replicate != comparison.case.replicate
    ):
        raise ValueError("zoom resolution pair does not share one physical realization")
    if (
        reference.case.numerics.finest_cell_size_pc
        >= comparison.case.numerics.finest_cell_size_pc
    ):
        raise ValueError("zoom reference must have a finer cell size than comparison")
    reasons = []
    for run, label in ((reference, "reference"), (comparison, "comparison")):
        if run.maximum_relative_energy_error > maximum_conservation_error:
            reasons.append(f"{label} energy error exceeds the gate")
        if run.maximum_relative_angular_momentum_error > maximum_conservation_error:
            reasons.append(f"{label} angular-momentum error exceeds the gate")
        if run.minimum_transition_radius_cells < minimum_transition_radius_cells:
            reasons.append(f"{label} transition radii are underresolved")
        if run.stage("hard_binary").status != "complete":
            reasons.append(f"{label} did not reach a hard binary")

    systematics = []
    for name in ZOOM_STAGE_NAMES[1:]:
        fine = reference.stage(name).elapsed_since_capture_myr
        coarse = comparison.stage(name).elapsed_since_capture_myr
        if fine is None or coarse is None:
            continue
        scale = max(abs(fine), np.finfo(float).tiny)
        systematics.append(abs(coarse - fine) / scale)
    maximum_systematic = max(systematics) if systematics else None
    if maximum_systematic is None:
        reasons.append("no complete stage delay can be compared")
    elif maximum_systematic > maximum_stage_delay_systematic_fraction:
        reasons.append("stage-delay resolution systematic exceeds the gate")
    return ZoomConvergenceResult(
        status=("accepted" if not reasons else "rejected"),
        reference=reference,
        comparison=comparison,
        maximum_stage_delay_systematic_fraction=(
            None if maximum_systematic is None else float(maximum_systematic)
        ),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class KpcDelayCalibrationRow:
    physics: ZoomPhysicsPoint
    simulated_kpc_to_hard_delay_myr: float
    analytic_kpc_to_hard_delay_myr: float
    multiplicative_delay_correction: float
    resolution_systematic_fraction: float
    source_case_id: str
    source_sha256: str


def accepted_kpc_delay_row(convergence: ZoomConvergenceResult) -> KpcDelayCalibrationRow:
    if convergence.status != "accepted":
        raise ValueError("only accepted zoom convergence results can calibrate delays")
    reference = convergence.reference
    simulated = reference.stage("hard_binary").elapsed_since_capture_myr
    analytic = reference.analytic_kpc_to_hard_delay_myr
    if simulated is None or simulated <= 0.0 or analytic is None:
        raise ValueError("accepted zoom lacks a positive simulated/analytic delay pair")
    assert convergence.maximum_stage_delay_systematic_fraction is not None
    return KpcDelayCalibrationRow(
        physics=reference.case.physics,
        simulated_kpc_to_hard_delay_myr=float(simulated),
        analytic_kpc_to_hard_delay_myr=float(analytic),
        multiplicative_delay_correction=float(simulated / analytic),
        resolution_systematic_fraction=(
            convergence.maximum_stage_delay_systematic_fraction
        ),
        source_case_id=reference.case.case_id,
        source_sha256=reference.source_sha256,
    )


class KpcDelayCalibrationTable:
    """Exact-point lookup until the validation grid supports interpolation."""

    def __init__(self, rows: tuple[KpcDelayCalibrationRow, ...]) -> None:
        if not rows:
            raise ValueError("kpc delay calibration table cannot be empty")
        ids = [row.physics.physics_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("kpc delay calibration physics points are duplicated")
        for row in rows:
            values = np.asarray(
                [
                    row.simulated_kpc_to_hard_delay_myr,
                    row.analytic_kpc_to_hard_delay_myr,
                    row.multiplicative_delay_correction,
                    row.resolution_systematic_fraction,
                ],
                dtype=float,
            )
            if (
                np.any(~np.isfinite(values))
                or np.any(values[:3] <= 0.0)
                or values[3] < 0.0
            ):
                raise ValueError("kpc delay calibration row has invalid numeric values")
            expected = (
                row.simulated_kpc_to_hard_delay_myr
                / row.analytic_kpc_to_hard_delay_myr
            )
            if not np.isclose(
                row.multiplicative_delay_correction,
                expected,
                rtol=1.0e-12,
                atol=0.0,
            ):
                raise ValueError("kpc delay calibration correction is inconsistent")
            if (
                not isinstance(row.source_case_id, str)
                or not row.source_case_id.strip()
                or not isinstance(row.source_sha256, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", row.source_sha256) is None
            ):
                raise ValueError("kpc delay calibration provenance is incomplete")
        self.rows = rows

    def lookup(self, physics: ZoomPhysicsPoint) -> KpcDelayCalibrationRow:
        matches = [row for row in self.rows if row.physics == physics]
        if len(matches) != 1:
            raise ValueError(
                "requested kpc delay point is uncalibrated; extrapolation is prohibited"
            )
        return matches[0]

    def calibrated_delay_segment(
        self,
        physics: ZoomPhysicsPoint,
        analytic_baseline_delay_myr: float,
        *,
        name: str = "kpc_to_pc",
    ) -> DelaySegment:
        """Apply an accepted exact-point correction without extrapolation.

        A point absent from the accepted table is an explicit censored physical
        result. Malformed baseline inputs are caller errors and raise
        ``ValueError`` rather than being confused with missing calibration
        support.
        """

        if isinstance(analytic_baseline_delay_myr, bool):
            raise ValueError("analytic baseline delay must be finite and positive")
        try:
            baseline = float(analytic_baseline_delay_myr)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "analytic baseline delay must be finite and positive"
            ) from error
        if not np.isfinite(baseline) or baseline <= 0.0:
            raise ValueError("analytic baseline delay must be finite and positive")
        try:
            row = self.lookup(physics)
        except ValueError:
            return DelaySegment(
                name,
                "censored",
                None,
                reason=(
                    "requested physical point is outside accepted kpc-delay "
                    "support; extrapolation is prohibited"
                ),
            )
        return DelaySegment(
            name,
            "complete",
            baseline * row.multiplicative_delay_correction,
            reason="accepted exact-point galaxy-merger zoom correction",
            source_case_id=row.source_case_id,
            source_sha256=row.source_sha256,
        )


def apply_kpc_delay_calibration(
    table: KpcDelayCalibrationTable,
    *,
    physics: ZoomPhysicsPoint,
    analytic_baseline_delay_myr: float,
    name: str = "kpc_to_pc",
) -> DelaySegment:
    """Public functional wrapper for exact-point kpc-delay calibration."""

    return table.calibrated_delay_segment(
        physics,
        analytic_baseline_delay_myr,
        name=name,
    )
