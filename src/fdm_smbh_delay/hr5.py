"""Numerical-population utilities for the legacy Horizon Run 5 sink tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.constants import G, M_sun, c
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


NSTEP_MAX = 296
HEADER_DTYPE = np.dtype(
    [
        ("redshift", "<f4", (NSTEP_MAX,)),
        ("output_number", "<f4", (NSTEP_MAX,)),
        ("omega_m", "<f4"),
        ("omega_lambda", "<f4"),
        ("h0", "<f4"),
        ("nstep", "<i4"),
        ("nsink", "<i4"),
        ("legacy_pointer", "<u8"),
    ],
    align=True,
)
SINK_DTYPE = np.dtype(
    [
        ("sink_id", "<i4"),
        ("state", "<f4", (NSTEP_MAX, 7)),
        ("receiver_id", "<i4"),
        ("capture_index", "<i4"),
    ],
    align=True,
)

# Native double-precision AGNType written by SRC(AGN)/SRC(MkAGN)/mkagn.c.
# The first 39 floating-point fields precede four 32-bit integer fields and
# four final floating-point host-galaxy fields.  The record size is 360 bytes.
MKAGN_DTYPE = np.dtype(
    [
        ("x", "<f8"),
        ("y", "<f8"),
        ("z", "<f8"),
        ("vx", "<f8"),
        ("vy", "<f8"),
        ("vz", "<f8"),
        ("mass", "<f8"),
        ("tbirth", "<f8"),
        ("Jx", "<f8"),
        ("Jy", "<f8"),
        ("Jz", "<f8"),
        ("Sx", "<f8"),
        ("Sy", "<f8"),
        ("Sz", "<f8"),
        ("dMsmbh", "<f8"),
        ("dMBH_coarse", "<f8"),
        ("dMEd_coarse", "<f8"),
        ("Esave", "<f8"),
        ("Smag", "<f8"),
        ("eps", "<f8"),
        ("dtnew", "<f8"),
        ("dMBHoverdt", "<f8"),
        ("dMEdoverdt", "<f8"),
        ("EAGN", "<f8"),
        ("Lbol", "<f8"),
        ("LhX", "<f8"),
        ("LsX", "<f8"),
        ("L15um", "<f8"),
        ("LB", "<f8"),
        ("LR", "<f8"),
        ("LUV", "<f8"),
        ("NHIxm", "<f8"),
        ("NHIym", "<f8"),
        ("NHIzm", "<f8"),
        ("NHIxp", "<f8"),
        ("NHIyp", "<f8"),
        ("NHIzp", "<f8"),
        ("NHId", "<f8"),
        ("mdisk", "<f8"),
        ("sink_id", "<i4"),
        ("mode", "<i4"),
        ("gid", "<i4"),
        ("global_gid", "<i4"),
        ("Mstar", "<f8"),
        ("Mgas", "<f8"),
        ("Mtot", "<f8"),
        ("Mdm", "<f8"),
    ],
    align=True,
)
MKAGN_ID_OFFSETS = {200: 160, 336: 288, 360: 312}


@dataclass(frozen=True)
class RedshiftRateFit:
    """Parameters of the redshift distribution used in the original draft."""

    phi_star: float
    z_star: float
    alpha: float
    beta: float
    success: bool
    n_bin: int


def read_tree_header(path: Path) -> np.void:
    """Read and validate the native C-structure header of an HR5 sink tree."""

    header = np.fromfile(path, dtype=HEADER_DTYPE, count=1)
    if header.size != 1:
        raise ValueError(f"Could not read the HR5 header from {path}")
    result = header[0]
    expected_size = HEADER_DTYPE.itemsize + int(result["nsink"]) * SINK_DTYPE.itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"Unexpected file size for {path}. Expected {expected_size} bytes and found "
            f"{path.stat().st_size} bytes."
        )
    return result


def read_mkagn_snapshot(path: Path) -> tuple[float, float, np.ndarray]:
    r"""Read one ``agn.NNNNN.dat`` snapshot created by the legacy MkAGN code.

    The returned masses are in :math:`h^{-1} M_\odot`, coordinates are in
    :math:`h^{-1}\,\mathrm{cMpc}`, and velocities are physical km/s.
    """

    with path.open("rb") as stream:
        redshift = np.fromfile(stream, dtype="<f8", count=1)
        local_timestep_yr = np.fromfile(stream, dtype="<f8", count=1)
        count = np.fromfile(stream, dtype="<i4", count=1)
        if redshift.size != 1 or local_timestep_yr.size != 1 or count.size != 1:
            raise ValueError(f"Could not read the MkAGN header from {path}")
        payload_size = path.stat().st_size - 20
        if int(count[0]) <= 0 or payload_size % int(count[0]) != 0:
            raise ValueError(
                f"The MkAGN payload in {path} is incompatible with its particle count"
            )
        record_size = payload_size // int(count[0])
        if record_size not in MKAGN_ID_OFFSETS:
            raise ValueError(
                f"Unsupported MkAGN record size {record_size} bytes in {path}. "
                f"Known sizes are {sorted(MKAGN_ID_OFFSETS)}."
            )
        if record_size == MKAGN_DTYPE.itemsize:
            dtype = MKAGN_DTYPE
        else:
            dtype = np.dtype(
                {
                    "names": ("x", "y", "z", "vx", "vy", "vz", "mass", "sink_id"),
                    "formats": ("<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<i4"),
                    "offsets": (0, 8, 16, 24, 32, 40, 48, MKAGN_ID_OFFSETS[record_size]),
                    "itemsize": record_size,
                }
            )
        records = np.fromfile(stream, dtype=dtype, count=int(count[0]))
    return float(redshift[0]), float(local_timestep_yr[0]), records


def infer_capture_receivers(
    minor_id: np.ndarray,
    minor_mass: np.ndarray,
    minor_position: np.ndarray,
    current_id: np.ndarray,
    current_mass: np.ndarray,
    current_position: np.ndarray,
    box_size_cmpc_over_h: float = 1048.5,
    mass_factor: float = 2.0,
    radius_increment_cmpc_over_h: float = 0.002,
    maximum_radius_cmpc_over_h: float = 0.5,
) -> np.ndarray:
    """Reproduce the receiver selection used by legacy ``mkmerging.c``.

    For every sink that disappears between adjacent outputs, the search starts
    at the distance to the nearest surviving sink.  The radius grows in fixed
    increments until it contains a survivor at least ``mass_factor`` times as
    massive as the disappearing sink.  The most massive eligible object inside
    that radius is assigned as the receiver.
    """

    minor_id = np.asarray(minor_id, dtype=np.int64)
    minor_mass = np.asarray(minor_mass, dtype=np.float64)
    minor_position = np.asarray(minor_position, dtype=np.float64)
    current_id = np.asarray(current_id, dtype=np.int64)
    current_mass = np.asarray(current_mass, dtype=np.float64)
    current_position = np.mod(np.asarray(current_position, dtype=np.float64), box_size_cmpc_over_h)
    if current_id.size == 0:
        return np.zeros(minor_id.size, dtype=np.int64)
    tree = cKDTree(current_position, boxsize=box_size_cmpc_over_h)
    receiver = np.zeros(minor_id.size, dtype=np.int64)
    for event_number, (sink_id, mass, position) in enumerate(
        zip(minor_id, minor_mass, minor_position)
    ):
        wrapped_position = np.mod(position, box_size_cmpc_over_h)
        nearest_distance = float(tree.query(wrapped_position, k=1)[0])
        radius = nearest_distance
        while radius <= maximum_radius_cmpc_over_h + 1.0e-12:
            neighbour = np.asarray(tree.query_ball_point(wrapped_position, radius), dtype=np.int64)
            if neighbour.size:
                eligible = neighbour[
                    (current_id[neighbour] != sink_id)
                    & (current_mass[neighbour] >= mass_factor * mass)
                ]
                if eligible.size:
                    receiver[event_number] = current_id[
                        eligible[np.argmax(current_mass[eligible])]
                    ]
                    break
            radius += radius_increment_cmpc_over_h
    return receiver


def find_dual_agn_pairs(
    records: np.ndarray,
    redshift: float,
    dimensionless_hubble: float,
    luminosity_threshold_erg_s: float = 1.0e43,
    minimum_separation_pkpc: float = 0.5,
    maximum_separation_pkpc: float = 30.0,
    box_size_cmpc_over_h: float = 717.229040,
) -> dict[str, np.ndarray]:
    """Select three-dimensional dual AGN pairs from one MkAGN snapshot."""

    required = {"sink_id", "mass", "x", "y", "z", "Lbol"}
    if records.dtype.names is None or not required.issubset(records.dtype.names):
        raise ValueError("The MkAGN record does not contain the luminosity fields")
    active = np.isfinite(records["Lbol"]) & (records["Lbol"] >= luminosity_threshold_erg_s)
    active_record = records[active]
    empty = {
        "active_count": np.array(active_record.size, dtype=np.int64),
        "id_1": np.empty(0, dtype=np.int64),
        "id_2": np.empty(0, dtype=np.int64),
        "separation_pkpc": np.empty(0),
        "mass_1_msun": np.empty(0),
        "mass_2_msun": np.empty(0),
        "luminosity_1_erg_s": np.empty(0),
        "luminosity_2_erg_s": np.empty(0),
        "eddington_ratio_1": np.empty(0),
        "eddington_ratio_2": np.empty(0),
    }
    if active_record.size < 2:
        return empty
    position = np.column_stack(
        [active_record["x"], active_record["y"], active_record["z"]]
    )
    maximum_comoving_distance = (
        maximum_separation_pkpc * dimensionless_hubble * (1.0 + redshift) / 1000.0
    )
    tree = cKDTree(np.mod(position, box_size_cmpc_over_h), boxsize=box_size_cmpc_over_h)
    pair = tree.query_pairs(maximum_comoving_distance, output_type="ndarray")
    if pair.size == 0:
        return empty
    delta = np.abs(position[pair[:, 0]] - position[pair[:, 1]])
    delta = np.minimum(delta, box_size_cmpc_over_h - delta)
    separation_pkpc = (
        np.linalg.norm(delta, axis=1)
        * 1000.0
        / (dimensionless_hubble * (1.0 + redshift))
    )
    selected = separation_pkpc >= minimum_separation_pkpc
    pair = pair[selected]
    separation_pkpc = separation_pkpc[selected]
    if pair.size == 0:
        return empty

    first = active_record[pair[:, 0]]
    second = active_record[pair[:, 1]]
    mass_first = first["mass"] / dimensionless_hubble
    mass_second = second["mass"] / dimensionless_hubble
    first_is_primary = mass_first >= mass_second
    primary = np.where(first_is_primary, pair[:, 0], pair[:, 1])
    secondary = np.where(first_is_primary, pair[:, 1], pair[:, 0])
    primary_record = active_record[primary]
    secondary_record = active_record[secondary]
    primary_mass = primary_record["mass"] / dimensionless_hubble
    secondary_mass = secondary_record["mass"] / dimensionless_hubble
    eddington_coefficient = 1.26e38
    return {
        "active_count": np.array(active_record.size, dtype=np.int64),
        "id_1": primary_record["sink_id"].astype(np.int64),
        "id_2": secondary_record["sink_id"].astype(np.int64),
        "separation_pkpc": separation_pkpc,
        "mass_1_msun": primary_mass,
        "mass_2_msun": secondary_mass,
        "luminosity_1_erg_s": primary_record["Lbol"].astype(np.float64),
        "luminosity_2_erg_s": secondary_record["Lbol"].astype(np.float64),
        "eddington_ratio_1": primary_record["Lbol"] / (eddington_coefficient * primary_mass),
        "eddington_ratio_2": secondary_record["Lbol"] / (eddington_coefficient * secondary_mass),
    }


def find_agn_pair_population(
    records: np.ndarray,
    redshift: float,
    dimensionless_hubble: float,
    luminosity_threshold_erg_s: float = 1.0e43,
    luminosity_field: str = "Lbol",
    minimum_separation_pkpc: float = 0.5,
    maximum_separation_pkpc: float = 30.0,
    minimum_mass_msun: float = 0.0,
    box_size_cmpc_over_h: float = 717.229040,
) -> dict[str, np.ndarray]:
    """Select dual and offset AGN among three-dimensional SMBH pairs.

    Every retained pair contains at least one member above the supplied
    luminosity threshold.  ``is_dual`` identifies pairs for which both members
    pass the threshold, while ``is_offset`` identifies pairs with one active
    member.  Primary and secondary labels follow SMBH mass rather than
    luminosity.
    """

    required = {
        "sink_id",
        "mass",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "Lbol",
        luminosity_field,
    }
    if records.dtype.names is None or not required.issubset(records.dtype.names):
        raise ValueError("The MkAGN record does not contain the requested AGN fields")
    if dimensionless_hubble <= 0.0:
        raise ValueError("dimensionless_hubble must be positive")
    if minimum_separation_pkpc < 0.0 or maximum_separation_pkpc <= minimum_separation_pkpc:
        raise ValueError("The separation bounds must be positive and ordered")

    mass_msun = np.asarray(records["mass"], dtype=np.float64) / dimensionless_hubble
    usable = (
        np.isfinite(mass_msun)
        & (mass_msun >= minimum_mass_msun)
        & np.isfinite(records["x"])
        & np.isfinite(records["y"])
        & np.isfinite(records["z"])
    )
    population = records[usable]
    population_mass = mass_msun[usable]
    luminosity = np.asarray(population[luminosity_field], dtype=np.float64)
    active = np.isfinite(luminosity) & (luminosity >= luminosity_threshold_erg_s)

    empty = {
        "active_count": np.array(np.count_nonzero(active), dtype=np.int64),
        "id_1": np.empty(0, dtype=np.int64),
        "id_2": np.empty(0, dtype=np.int64),
        "position_1_cmpc_over_h": np.empty((0, 3)),
        "position_2_cmpc_over_h": np.empty((0, 3)),
        "velocity_1_kms": np.empty((0, 3)),
        "velocity_2_kms": np.empty((0, 3)),
        "separation_pkpc": np.empty(0),
        "mass_1_msun": np.empty(0),
        "mass_2_msun": np.empty(0),
        "luminosity_1_erg_s": np.empty(0),
        "luminosity_2_erg_s": np.empty(0),
        "lbol_1_erg_s": np.empty(0),
        "lbol_2_erg_s": np.empty(0),
        "lhx_1_erg_s": np.empty(0),
        "lhx_2_erg_s": np.empty(0),
        "eddington_ratio_1": np.empty(0),
        "eddington_ratio_2": np.empty(0),
        "active_1": np.empty(0, dtype=bool),
        "active_2": np.empty(0, dtype=bool),
        "is_dual": np.empty(0, dtype=bool),
        "is_offset": np.empty(0, dtype=bool),
    }
    if population.size < 2 or not np.any(active):
        return empty

    position = np.column_stack([population["x"], population["y"], population["z"]])
    maximum_comoving_distance = (
        maximum_separation_pkpc * dimensionless_hubble * (1.0 + redshift) / 1000.0
    )
    tree = cKDTree(np.mod(position, box_size_cmpc_over_h), boxsize=box_size_cmpc_over_h)
    pair = tree.query_pairs(maximum_comoving_distance, output_type="ndarray")
    if pair.size == 0:
        return empty

    delta = position[pair[:, 1]] - position[pair[:, 0]]
    delta -= box_size_cmpc_over_h * np.rint(delta / box_size_cmpc_over_h)
    separation_pkpc = (
        np.linalg.norm(delta, axis=1)
        * 1000.0
        / (dimensionless_hubble * (1.0 + redshift))
    )
    selected = (
        (separation_pkpc >= minimum_separation_pkpc)
        & (active[pair[:, 0]] | active[pair[:, 1]])
    )
    pair = pair[selected]
    separation_pkpc = separation_pkpc[selected]
    if pair.size == 0:
        return empty

    first_is_primary = population_mass[pair[:, 0]] >= population_mass[pair[:, 1]]
    primary = np.where(first_is_primary, pair[:, 0], pair[:, 1])
    secondary = np.where(first_is_primary, pair[:, 1], pair[:, 0])
    primary_record = population[primary]
    secondary_record = population[secondary]
    primary_mass = population_mass[primary]
    secondary_mass = population_mass[secondary]
    active_primary = active[primary]
    active_secondary = active[secondary]
    eddington_coefficient = 1.26e38
    result = {
        "active_count": np.array(np.count_nonzero(active), dtype=np.int64),
        "id_1": primary_record["sink_id"].astype(np.int64),
        "id_2": secondary_record["sink_id"].astype(np.int64),
        "position_1_cmpc_over_h": np.column_stack(
            [primary_record["x"], primary_record["y"], primary_record["z"]]
        ).astype(np.float64),
        "position_2_cmpc_over_h": np.column_stack(
            [secondary_record["x"], secondary_record["y"], secondary_record["z"]]
        ).astype(np.float64),
        "velocity_1_kms": np.column_stack(
            [primary_record["vx"], primary_record["vy"], primary_record["vz"]]
        ).astype(np.float64),
        "velocity_2_kms": np.column_stack(
            [secondary_record["vx"], secondary_record["vy"], secondary_record["vz"]]
        ).astype(np.float64),
        "separation_pkpc": separation_pkpc,
        "mass_1_msun": primary_mass,
        "mass_2_msun": secondary_mass,
        "luminosity_1_erg_s": np.asarray(primary_record[luminosity_field], dtype=np.float64),
        "luminosity_2_erg_s": np.asarray(secondary_record[luminosity_field], dtype=np.float64),
        "lbol_1_erg_s": np.asarray(primary_record["Lbol"], dtype=np.float64),
        "lbol_2_erg_s": np.asarray(secondary_record["Lbol"], dtype=np.float64),
        "lhx_1_erg_s": np.asarray(primary_record["LhX"], dtype=np.float64),
        "lhx_2_erg_s": np.asarray(secondary_record["LhX"], dtype=np.float64),
        "eddington_ratio_1": np.asarray(primary_record["Lbol"], dtype=np.float64)
        / (eddington_coefficient * primary_mass),
        "eddington_ratio_2": np.asarray(secondary_record["Lbol"], dtype=np.float64)
        / (eddington_coefficient * secondary_mass),
        "active_1": active_primary,
        "active_2": active_secondary,
        "is_dual": active_primary & active_secondary,
        "is_offset": np.logical_xor(active_primary, active_secondary),
    }
    return result


def pair_component_multiplicity(
    first_id: np.ndarray,
    second_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the connected-system multiplicity associated with every pair.

    The second and third arrays contain the unique member identifiers and the
    multiplicity of the connected system that contains each member.
    """

    first = np.asarray(first_id, dtype=np.int64)
    second = np.asarray(second_id, dtype=np.int64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Pair identifiers must be matching one-dimensional arrays")
    if first.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    member, inverse = np.unique(np.r_[first, second], return_inverse=True)
    edge = inverse.reshape(2, first.size).T
    row = np.r_[edge[:, 0], edge[:, 1]]
    column = np.r_[edge[:, 1], edge[:, 0]]
    graph = coo_matrix(
        (np.ones(row.size, dtype=np.int8), (row, column)),
        shape=(member.size, member.size),
    )
    _, label = connected_components(graph, directed=False)
    size = np.bincount(label)
    member_multiplicity = size[label]
    pair_multiplicity = member_multiplicity[edge[:, 0]]
    return pair_multiplicity.astype(np.int64), member, member_multiplicity.astype(np.int64)


def fibonacci_sightlines(count: int) -> np.ndarray:
    """Return nearly uniform deterministic directions on the unit sphere."""

    if count < 1:
        raise ValueError("count must be positive")
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    azimuth = np.pi * (3.0 - np.sqrt(5.0)) * index
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))


def project_pair_observables(
    position_1_cmpc_over_h: np.ndarray,
    position_2_cmpc_over_h: np.ndarray,
    velocity_1_kms: np.ndarray,
    velocity_2_kms: np.ndarray,
    sightlines: np.ndarray,
    redshift: float,
    dimensionless_hubble: float,
    hubble_kms_mpc: float,
    box_size_cmpc_over_h: float = 717.229040,
) -> tuple[np.ndarray, np.ndarray]:
    """Project physical pairs and include Hubble flow in line-of-sight velocity."""

    position_1 = np.asarray(position_1_cmpc_over_h, dtype=np.float64)
    position_2 = np.asarray(position_2_cmpc_over_h, dtype=np.float64)
    velocity_1 = np.asarray(velocity_1_kms, dtype=np.float64)
    velocity_2 = np.asarray(velocity_2_kms, dtype=np.float64)
    direction = np.asarray(sightlines, dtype=np.float64)
    if position_1.shape != position_2.shape or position_1.ndim != 2 or position_1.shape[1] != 3:
        raise ValueError("Pair positions must have shape (N, 3)")
    if velocity_1.shape != position_1.shape or velocity_2.shape != position_1.shape:
        raise ValueError("Pair velocities must match the position arrays")
    if direction.ndim != 2 or direction.shape[1] != 3:
        raise ValueError("sightlines must have shape (M, 3)")
    norm = np.linalg.norm(direction, axis=1)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
        raise ValueError("sightlines must be finite nonzero vectors")
    direction = direction / norm[:, None]

    delta_comoving = position_2 - position_1
    delta_comoving -= box_size_cmpc_over_h * np.rint(delta_comoving / box_size_cmpc_over_h)
    delta_physical_mpc = delta_comoving / (dimensionless_hubble * (1.0 + redshift))
    delta_velocity = velocity_2 - velocity_1
    line_of_sight_distance = delta_physical_mpc @ direction.T
    separation_squared = np.sum(delta_physical_mpc**2, axis=1)[:, None]
    projected_separation_pkpc = 1000.0 * np.sqrt(
        np.maximum(0.0, separation_squared - line_of_sight_distance**2)
    )
    peculiar_velocity = delta_velocity @ direction.T
    line_of_sight_velocity_kms = np.abs(
        peculiar_velocity + hubble_kms_mpc * line_of_sight_distance
    )
    return projected_separation_pkpc, line_of_sight_velocity_kms


def interval_censored_cumulative_bounds(
    event_lower_gyr: np.ndarray,
    event_upper_gyr: np.ndarray,
    time_grid_gyr: np.ndarray,
    followup_gyr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound cumulative incidence for interval events and common right censoring."""

    lower = np.asarray(event_lower_gyr, dtype=np.float64)
    upper = np.asarray(event_upper_gyr, dtype=np.float64)
    grid = np.asarray(time_grid_gyr, dtype=np.float64)
    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("Event bounds must be matching one-dimensional arrays")
    if grid.ndim != 1 or np.any(np.diff(grid) < 0.0) or np.any(grid < 0.0):
        raise ValueError("time_grid_gyr must be non-negative and ordered")
    if followup_gyr < 0.0:
        raise ValueError("followup_gyr must be non-negative")
    certain = np.isfinite(upper)
    possible = np.isfinite(lower)
    cumulative_lower = np.mean(certain[:, None] & (upper[:, None] <= grid[None, :]), axis=0)
    cumulative_upper = np.mean(possible[:, None] & (lower[:, None] <= grid[None, :]), axis=0)
    beyond_followup = grid > followup_gyr
    cumulative_lower[beyond_followup] = np.nan
    cumulative_upper[beyond_followup] = np.nan
    return cumulative_lower, cumulative_upper


def redshift_rate_model(
    redshift: np.ndarray | float,
    phi_star: float,
    z_star: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    r"""Return :math:`\phi_* e^{-(z/z_*)^\beta}(z/z_*)^\alpha`."""

    z = np.asarray(redshift, dtype=np.float64)
    scaled = np.maximum(z / z_star, np.finfo(float).tiny)
    return phi_star * np.exp(-(scaled**beta)) * scaled**alpha


def fit_redshift_rate(
    redshift: np.ndarray,
    rate: np.ndarray,
    count: np.ndarray | None = None,
) -> RedshiftRateFit:
    """Fit the four-parameter redshift-rate form in logarithmic rate."""

    z = np.asarray(redshift, dtype=np.float64)
    y = np.asarray(rate, dtype=np.float64)
    selected = np.isfinite(z) & np.isfinite(y) & (z > 0.0) & (y > 0.0)
    if count is not None:
        selected &= np.asarray(count) >= 3
    z = z[selected]
    y = y[selected]
    if z.size < 5:
        return RedshiftRateFit(np.nan, np.nan, np.nan, np.nan, False, int(z.size))

    peak = int(np.argmax(y))
    initial = np.array(
        [np.log(max(y[peak] * np.e, 1.0e-30)), np.log(max(z[peak], 0.2)), 1.0, 4.0]
    )
    lower = np.array([np.log(1.0e-20), np.log(0.03), 0.05, 0.2])
    upper = np.array([np.log(1.0), np.log(20.0), 4.0, 20.0])

    def residual(parameters: np.ndarray) -> np.ndarray:
        log_phi, log_z_star, alpha, beta = parameters
        scaled = z / np.exp(log_z_star)
        model_log = log_phi - scaled**beta + alpha * np.log(scaled)
        return model_log - np.log(y)

    result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=5000)
    log_phi, log_z_star, alpha, beta = result.x
    return RedshiftRateFit(
        float(np.exp(log_phi)),
        float(np.exp(log_z_star)),
        float(alpha),
        float(beta),
        bool(result.success),
        int(z.size),
    )


def bootstrap_redshift_rate(
    redshift: np.ndarray,
    count: np.ndarray,
    exposure_cmpc3_gyr: np.ndarray,
    realizations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Refit the redshift-rate model after Poisson resampling bin counts.

    Successful rows contain ``phi_star``, ``z_star``, ``alpha``, and ``beta``
    in that order. Failed fits are omitted from the returned array.
    """

    z = np.asarray(redshift, dtype=np.float64)
    observed_count = np.asarray(count, dtype=np.int64)
    exposure = np.asarray(exposure_cmpc3_gyr, dtype=np.float64)
    if z.shape != observed_count.shape or z.shape != exposure.shape:
        raise ValueError("redshift, count, and exposure must have matching shapes")
    if np.any(observed_count < 0):
        raise ValueError("count must be non-negative")
    if np.any(~np.isfinite(exposure)) or np.any(exposure <= 0.0):
        raise ValueError("exposure must be finite and positive")
    if realizations < 1:
        raise ValueError("realizations must be positive")

    samples: list[tuple[float, float, float, float]] = []
    for _ in range(realizations):
        resampled_count = rng.poisson(observed_count)
        fit = fit_redshift_rate(z, resampled_count / exposure, resampled_count)
        parameters = (fit.phi_star, fit.z_star, fit.alpha, fit.beta)
        if fit.success and np.all(np.isfinite(parameters)):
            samples.append(parameters)
    if not samples:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray(samples, dtype=np.float64)


def delayed_redshift(
    capture_time_gyr: np.ndarray,
    delay_gyr: float,
    cosmology: FlatLambdaCDM,
    maximum_redshift: float = 20.0,
    grid_size: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a source-frame fixed delay to redshift and flag future events."""

    capture_time = np.asarray(capture_time_gyr, dtype=np.float64)
    delayed_time = capture_time + delay_gyr
    present_age = float(cosmology.age(0.0).value)
    censored = delayed_time > present_age

    redshift_grid = np.expm1(np.linspace(0.0, np.log1p(maximum_redshift), grid_size))
    age_grid = np.asarray(cosmology.age(redshift_grid).value)
    redshift = np.interp(
        np.minimum(delayed_time, present_age),
        age_grid[::-1],
        redshift_grid[::-1],
    )
    redshift[censored] = np.nan
    return redshift, censored


def binned_source_rate(
    event_redshift: np.ndarray,
    redshift_edges: np.ndarray,
    volume_cmpc3: float,
    cosmology: FlatLambdaCDM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure a source-frame event rate in redshift bins."""

    edges = np.asarray(redshift_edges, dtype=np.float64)
    count, _ = np.histogram(np.asarray(event_redshift), bins=edges)
    age = np.asarray(cosmology.age(edges).value)
    interval_gyr = age[:-1] - age[1:]
    rate = count / (volume_cmpc3 * interval_gyr)
    error = np.sqrt(count) / (volume_cmpc3 * interval_gyr)
    return count, rate, error


def bootstrap_binned_source_rate(
    count: np.ndarray,
    exposure_cmpc3_gyr: np.ndarray,
    realizations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate binned-rate quantiles by Poisson resampling event counts.

    The returned rows contain the 16th, 50th, and 84th percentiles. Each
    column corresponds to one redshift bin.
    """

    observed_count = np.asarray(count, dtype=np.int64)
    exposure = np.asarray(exposure_cmpc3_gyr, dtype=np.float64)
    if observed_count.ndim != 1 or observed_count.shape != exposure.shape:
        raise ValueError("count and exposure must be matching one-dimensional arrays")
    if np.any(observed_count < 0):
        raise ValueError("count must be non-negative")
    if np.any(~np.isfinite(exposure)) or np.any(exposure <= 0.0):
        raise ValueError("exposure must be finite and positive")
    if realizations < 1:
        raise ValueError("realizations must be positive")

    resampled_count = rng.poisson(observed_count, size=(realizations, observed_count.size))
    resampled_rate = resampled_count / exposure[None, :]
    return np.quantile(resampled_rate, (0.16, 0.5, 0.84), axis=0)


def cumulative_active_sources(
    redshift: np.ndarray,
    source_rate_cmpc3_gyr: np.ndarray,
    residence_time_yr: float,
    cosmology: FlatLambdaCDM,
    solid_angle_sr: float = 4.0 * np.pi,
) -> np.ndarray:
    """Count active sources on the past light cone to each survey depth."""

    z = np.asarray(redshift, dtype=np.float64)
    rate = np.asarray(source_rate_cmpc3_gyr, dtype=np.float64)
    distance = np.asarray(cosmology.comoving_distance(z).value)
    hubble = np.asarray(cosmology.H(z).value)
    speed_of_light_kms = 299792.458
    shell_cmpc3_per_z = solid_angle_sr * distance**2 * speed_of_light_kms / hubble
    integrand = rate * (residence_time_yr / 1.0e9) * shell_cmpc3_per_z
    return np.r_[0.0, cumulative_trapezoid(integrand, z)]


def circular_gw_background_contributions(
    chirp_mass_msun: np.ndarray,
    redshift: np.ndarray,
    volume_cmpc3: float,
    observed_frequency_hz: float,
) -> np.ndarray:
    r"""Return each event's contribution to :math:`h_c^2(f)`.

    The expression applies the discrete form of the Phinney theorem to a
    circular population whose frequency evolution is driven only by
    gravitational radiation.  The input events must represent one comoving
    volume over the sampled cosmic history.
    """

    mass = np.asarray(chirp_mass_msun, dtype=np.float64)
    event_redshift = np.asarray(redshift, dtype=np.float64)
    if mass.shape != event_redshift.shape:
        raise ValueError("chirp_mass_msun and redshift must have matching shapes")
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("chirp masses must be finite and positive")
    if np.any(~np.isfinite(event_redshift)) or np.any(event_redshift < 0.0):
        raise ValueError("redshifts must be finite and non-negative")
    if not np.isfinite(volume_cmpc3) or volume_cmpc3 <= 0.0:
        raise ValueError("volume_cmpc3 must be finite and positive")
    if not np.isfinite(observed_frequency_hz) or observed_frequency_hz <= 0.0:
        raise ValueError("observed_frequency_hz must be finite and positive")

    mass_kg = mass * M_sun.value
    volume_m3 = volume_cmpc3 * u.Mpc.to(u.m) ** 3
    coefficient = (
        4.0
        * G.value ** (5.0 / 3.0)
        / (3.0 * np.pi ** (1.0 / 3.0) * c.value**2)
        * observed_frequency_hz ** (-4.0 / 3.0)
        / volume_m3
    )
    return coefficient * mass_kg ** (5.0 / 3.0) / (1.0 + event_redshift) ** (1.0 / 3.0)


def histogram_quantiles(
    count: np.ndarray,
    edges: np.ndarray,
    probabilities: tuple[float, ...],
) -> np.ndarray:
    """Approximate quantiles from counts in adjacent scalar bins."""

    histogram = np.asarray(count, dtype=np.float64)
    bin_edges = np.asarray(edges, dtype=np.float64)
    if histogram.shape[-1] + 1 != bin_edges.size:
        raise ValueError("The final histogram dimension must match the supplied bin edges")
    flat = histogram.reshape(-1, histogram.shape[-1])
    result = np.full((flat.shape[0], len(probabilities)), np.nan)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for row_number, row in enumerate(flat):
        total = np.sum(row)
        if total <= 0.0:
            continue
        cumulative = np.cumsum(row) / total
        result[row_number] = np.interp(probabilities, cumulative, centers)
    return result.reshape(histogram.shape[:-1] + (len(probabilities),))
