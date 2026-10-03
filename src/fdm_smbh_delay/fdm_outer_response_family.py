"""Source-bound q/e interpolation of candidate outer FDM drift/diffusion.

Interpolation is a numerical operation on supplied measurements, not evidence
that the resulting response is a validated physical delay law.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np

from .fdm_outer_response import FDMOuterResponseTable


def _axis_weights(nodes: tuple[float, ...], value: float) -> tuple[tuple[int, float], ...]:
    if value < nodes[0] or value > nodes[-1]:
        raise ValueError("q/e query lies outside measured grid")
    right = int(np.searchsorted(nodes, value, side="left"))
    if right < len(nodes) and value == nodes[right]:
        return ((right, 1.0),)
    left = right - 1
    weight = (value - nodes[left]) / (nodes[right] - nodes[left])
    return ((left, 1.0 - weight), (right, weight))


@dataclass(frozen=True)
class FDMOuterResponseFamily:
    """Rectangular measured q/e grid, with per-cell source identity.

    A query needs every corner of its enclosing cell.  Missing, uncalibrated,
    or radially unsupported corners censor the entire query.  Exact grid-node
    queries only require that node, so an unrelated bad cell does not poison it.
    """

    mass_ratios_q: tuple[float, ...]
    eccentricities: tuple[float, ...]
    tables: tuple[tuple[FDMOuterResponseTable, ...], ...]
    source_sha256: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        q = tuple(float(value) for value in self.mass_ratios_q)
        e = tuple(float(value) for value in self.eccentricities)
        tables = tuple(tuple(row) for row in self.tables)
        source_sha256 = tuple(tuple(row) for row in self.source_sha256)
        if (
            not q or not e
            or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in q)
            or any(not math.isfinite(value) or not 0.0 <= value < 1.0 for value in e)
            or any(b <= a for a, b in zip(q, q[1:]))
            or any(b <= a for a, b in zip(e, e[1:]))
        ):
            raise ValueError("FDM response q/e nodes must be strictly increasing and physical")
        if len(tables) != len(q) or len(source_sha256) != len(q):
            raise ValueError("FDM response q/e grid shape is inconsistent")
        for table_row, hash_row in zip(tables, source_sha256):
            if len(table_row) != len(e) or len(hash_row) != len(e):
                raise ValueError("FDM response q/e grid shape is inconsistent")
            for table, digest in zip(table_row, hash_row):
                if not isinstance(table, FDMOuterResponseTable):
                    raise ValueError("FDM response grid requires response tables")
                if table.component_frame != "orbital_rtn":
                    raise ValueError("FDM response grid requires an orbital RTN frame")
                if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    raise ValueError("FDM response source SHA-256 is required at every node")
        object.__setattr__(self, "mass_ratios_q", q)
        object.__setattr__(self, "eccentricities", e)
        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "source_sha256", source_sha256)

    def decision(self, mass_ratio_q: float, eccentricity: float, radius_pc: float) -> dict[str, Any]:
        """Return a convex q/e mixture only where all source nodes apply."""

        if (
            not math.isfinite(mass_ratio_q) or not 0.0 < mass_ratio_q <= 1.0
            or not math.isfinite(eccentricity) or not 0.0 <= eccentricity < 1.0
            or not math.isfinite(radius_pc) or radius_pc <= 0.0
        ):
            raise ValueError("FDM response query must be finite and physical")
        try:
            q_weights = _axis_weights(self.mass_ratios_q, mass_ratio_q)
            e_weights = _axis_weights(self.eccentricities, eccentricity)
        except ValueError as error:
            return {"status": "censored", "reason": str(error)}
        drift = np.zeros(3, dtype=float)
        diffusion = np.zeros((3, 3), dtype=float)
        sources: list[str] = []
        conventions: set[str] = set()
        for qi, qw in q_weights:
            for ei, ew in e_weights:
                response = self.tables[qi][ei].decision(radius_pc)
                if response["status"] != "available":
                    return {
                        "status": "censored",
                        "reason": f"q={self.mass_ratios_q[qi]}, e={self.eccentricities[ei]}: {response['reason']}",
                    }
                weight = qw * ew
                drift += weight * response["drift_acceleration_pc_myr2"]
                diffusion += weight * response["diffusion_tensor_pc2_myr3"]
                sources.append(self.source_sha256[qi][ei])
                conventions.add(self.tables[qi][ei].diffusion_convention)
        if len(conventions) != 1:
            return {"status": "censored", "reason": "q/e response corners use incompatible diffusion conventions"}
        diffusion = 0.5 * (diffusion + diffusion.T)
        if (
            np.any(~np.isfinite(drift)) or np.any(~np.isfinite(diffusion))
            or np.min(np.linalg.eigvalsh(diffusion)) < -1.0e-12
        ):
            return {"status": "censored", "reason": "q/e response mixture violated finite/PSD contract"}
        return {
            "status": "interpolated_candidate_pending_physical_validation",
            "drift_acceleration_pc_myr2": drift,
            "diffusion_tensor_pc2_myr3": diffusion,
            "source_sha256": tuple(sources),
            "diffusion_convention": conventions.pop(),
        }
