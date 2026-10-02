"""Source-bound FDM density–Poisson-potential moment on owned AMR leaves.

``integral rho*phi dV`` is gauge dependent and may include sink gravity.  It
must not be labelled gravitational self-energy or a conserved Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .fdm_gravity_shard import inspect_fdm_gravity_shard_pair
from .fdm_leaf_mass import FDMLeafMassIdentity, check_fdm_leaf_mass_identity
from .fdm_shard_format import (
    OwnedLeafAmplitudeSummary,
    _RecordReader,
    inspect_fdm_amr_shard_pair,
)
from .lagramses_fdm_provenance import LagRamsesFDMOuterWaveProvenance


@dataclass(frozen=True)
class OwnedLeafPotentialCouplingSummary:
    amplitude: OwnedLeafAmplitudeSummary
    gravity_path: Path
    density_times_potential_sum_by_level: tuple[float, ...]


@dataclass(frozen=True)
class FDMPotentialCoupling:
    status: str
    provenance_path: Path
    coarse_cells_per_box: int
    leaf_mass_identity: FDMLeafMassIdentity
    density_times_potential_code_by_level: tuple[float, ...] | None
    integrated_density_times_potential_code: float | None
    interpretation: str


def summarize_owned_leaf_potential_coupling(
    wave_path: str | Path,
    amr_path: str | Path,
    gravity_path: str | Path,
    *,
    owner_rank: int,
    simple_boundary: bool,
    fdm_use_hjm: bool,
    fdm_first_wave_level: int,
    particle_density_included: bool = False,
    expected_ncpu: int | None = None,
    byte_order: str = "<",
    maximum_array_bytes: int = 64 * 1024 * 1024,
) -> OwnedLeafPotentialCouplingSummary:
    """Read aligned wave, AMR son, and potential arrays one block at a time."""

    if (
        not isinstance(owner_rank, int) or isinstance(owner_rank, bool)
        or not isinstance(fdm_use_hjm, bool)
        or not isinstance(fdm_first_wave_level, int)
        or fdm_first_wave_level < 1
        or not isinstance(particle_density_included, bool)
        or maximum_array_bytes < 8
    ):
        raise ValueError("FDM potential-coupling extraction parameters are invalid")
    pair = inspect_fdm_amr_shard_pair(
        wave_path, amr_path, simple_boundary=simple_boundary,
        expected_ncpu=expected_ncpu, byte_order=byte_order,
    )
    gravity_pair = inspect_fdm_gravity_shard_pair(
        wave_path, gravity_path,
        particle_density_included=particle_density_included,
        expected_ncpu=pair.wave.ncpu, byte_order=byte_order,
    )
    wave, amr, gravity = pair.wave, pair.amr, gravity_pair.gravity
    if (
        owner_rank < 1 or owner_rank > wave.ncpu
        or any(
            not path.name.endswith(f"{owner_rank:05d}")
            for path in (wave.path, amr.path, gravity.path)
        )
    ):
        raise ValueError("FDM/AMR/gravity owner rank identity is invalid")
    integer_dtype = np.dtype(byte_order + "i4")
    real_dtype = np.dtype(byte_order + "f8")
    counts: list[int] = []
    density_sums: list[float] = []
    potential_sums: list[float] = []
    with (
        wave.path.open("rb") as wave_stream,
        amr.path.open("rb") as amr_stream,
        gravity.path.open("rb") as gravity_stream,
    ):
        wave_reader = _RecordReader(wave_stream, byte_order=byte_order)
        amr_reader = _RecordReader(amr_stream, byte_order=byte_order)
        gravity_reader = _RecordReader(gravity_stream, byte_order=byte_order)
        for _ in range(4):
            wave_reader.scalar()
            gravity_reader.scalar()
        amr_stream.seek(amr.fine_payload_offset)
        for level in range(1, wave.nlevelmax + 1):
            level_count = 0
            level_density = 0.0
            level_potential = 0.0
            for domain, ncache in enumerate(
                wave.grid_counts_by_level_and_domain[level - 1], start=1
            ):
                if (
                    wave_reader.scalar() != level
                    or wave_reader.scalar() != ncache
                    or gravity_reader.scalar() != level
                    or gravity_reader.scalar() != ncache
                ):
                    raise ValueError("FDM/gravity block identity changed during extraction")
                if not ncache:
                    continue
                owned = domain == owner_rank
                if owned and 8 * ncache > maximum_array_bytes:
                    raise ValueError("FDM/gravity block exceeds declared array memory bound")
                for _ in range(3):
                    amr_reader.record(4 * ncache)
                for _ in range(amr.ndim):
                    amr_reader.record(8 * ncache)
                for _ in range(1 + 2 * amr.ndim):
                    amr_reader.record(4 * ncache)
                masks = []
                for _ in range(1 << amr.ndim):
                    payload = amr_reader.record(4 * ncache, read_payload=owned)
                    if owned:
                        sons = np.frombuffer(payload, dtype=integer_dtype)
                        if np.any(sons < 0):
                            raise ValueError("AMR son index must be non-negative")
                        masks.append(sons == 0)
                for _ in range(2 * (1 << amr.ndim)):
                    amr_reader.record(4 * ncache)
                for child in range(1 << wave.ndim):
                    real_payload = wave_reader.record(8 * ncache, read_payload=owned)
                    imag_payload = wave_reader.record(8 * ncache, read_payload=owned)
                    if particle_density_included:
                        gravity_reader.record(8 * ncache)
                    phi_payload = gravity_reader.record(8 * ncache, read_payload=owned)
                    for _ in range(wave.ndim):
                        gravity_reader.record(8 * ncache)
                    if not owned:
                        continue
                    real = np.frombuffer(real_payload, dtype=real_dtype)
                    imaginary = np.frombuffer(imag_payload, dtype=real_dtype)
                    phi = np.frombuffer(phi_payload, dtype=real_dtype)
                    if (
                        np.any(~np.isfinite(real)) or np.any(~np.isfinite(imaginary))
                        or np.any(~np.isfinite(phi))
                    ):
                        raise ValueError("FDM amplitude or Poisson potential is non-finite")
                    mask = masks[child]
                    level_count += int(np.count_nonzero(mask))
                    density = (
                        np.maximum(real[mask], 0.0)
                        if fdm_use_hjm and level < fdm_first_wave_level
                        else real[mask] ** 2 + imaginary[mask] ** 2
                    )
                    level_density += float(np.sum(density, dtype=np.float64))
                    level_potential += float(np.sum(density * phi[mask], dtype=np.float64))
            if not math.isfinite(level_density) or not math.isfinite(level_potential):
                raise ValueError("FDM leaf potential moment is non-finite")
            counts.append(level_count)
            density_sums.append(level_density)
            potential_sums.append(level_potential)
        if wave_stream.read(1) or amr_stream.read(1) or gravity_stream.read(1):
            raise ValueError("FDM/AMR/gravity source changed during extraction")
    return OwnedLeafPotentialCouplingSummary(
        amplitude=OwnedLeafAmplitudeSummary(
            wave_path=wave.path, amr_path=amr.path,
            owner_rank=owner_rank, ncpu=wave.ncpu, ndim=wave.ndim,
            boxlen_code=amr.boxlen_code, fdm_use_hjm=fdm_use_hjm,
            fdm_first_wave_level=fdm_first_wave_level,
            leaf_cells_by_level=tuple(counts),
            density_sum_by_level=tuple(density_sums),
        ),
        gravity_path=gravity.path,
        density_times_potential_sum_by_level=tuple(potential_sums),
    )


def reconstruct_fdm_potential_coupling(
    summaries: Sequence[OwnedLeafPotentialCouplingSummary],
    provenance: LagRamsesFDMOuterWaveProvenance,
    *,
    coarse_cells_per_box: int,
) -> FDMPotentialCoupling:
    """Require writer mass identity before integrating the density–phi moment."""

    identity = check_fdm_leaf_mass_identity(
        [item.amplitude for item in summaries], provenance,
        coarse_cells_per_box=coarse_cells_per_box,
    )
    interpretation = (
        "raw integral rho*phi dV only; potential zero-point and sink/external "
        "components are not separated; this is not self-energy or Hamiltonian"
    )
    if identity.status != "raw_leaf_mass_reconstruction_matches_writer":
        return FDMPotentialCoupling(
            status="censored_potential_coupling_mass_identity",
            provenance_path=provenance.source_path,
            coarse_cells_per_box=coarse_cells_per_box,
            leaf_mass_identity=identity,
            density_times_potential_code_by_level=None,
            integrated_density_times_potential_code=None,
            interpretation=interpretation,
        )
    ordered = sorted(summaries, key=lambda item: item.amplitude.owner_rank)
    first = ordered[0].amplitude
    expected_prefix = "grav_" + provenance.psi_snapshot_prefix.removeprefix("fdm_")
    output_directory = provenance.source_path.parent
    for item in ordered:
        parent = item.gravity_path.parent
        if (
            item.gravity_path.name != f"{expected_prefix}{item.amplitude.owner_rank:05d}"
            or not (
                parent == output_directory
                or (
                    parent.parent == output_directory
                    and parent.name.startswith("group_")
                    and len(parent.name) == 11
                    and parent.name[6:].isdigit()
                )
            )
            or len(item.density_times_potential_sum_by_level) != len(first.leaf_cells_by_level)
            or any(not math.isfinite(value) for value in item.density_times_potential_sum_by_level)
        ):
            raise ValueError("FDM potential-coupling summary differs from bound gravity source")
    level_values = []
    for level in range(1, len(first.leaf_cells_by_level) + 1):
        dx = 0.5**level * first.boxlen_code / coarse_cells_per_box
        level_sum = math.fsum(
            item.density_times_potential_sum_by_level[level - 1]
            for item in ordered
        )
        value = level_sum * dx**first.ndim
        if not math.isfinite(value):
            raise ValueError("FDM potential-coupling integral is non-finite")
        level_values.append(value)
    return FDMPotentialCoupling(
        status="potential_coupling_measured_pending_full_hamiltonian",
        provenance_path=provenance.source_path,
        coarse_cells_per_box=coarse_cells_per_box,
        leaf_mass_identity=identity,
        density_times_potential_code_by_level=tuple(level_values),
        integrated_density_times_potential_code=math.fsum(level_values),
        interpretation=interpretation,
    )
