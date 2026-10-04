"""Experimental periodic wave--compact-mass coupling on a TSC particle mesh.

The energy-gradient and spectral-momentum force candidates are distinct.
Neither is an accepted calibration mode.
"""

from __future__ import annotations

import numpy as np
import torch
from numba import njit


def _axis_stencil(grid_position: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nearest = int(np.floor(grid_position + 0.5))
    indices = nearest + np.array([-1, 0, 1], dtype=np.int64)
    separation = grid_position - indices.astype(np.float64)
    distance = np.abs(separation)
    weights = np.where(
        distance <= 0.5,
        0.75 - distance**2,
        0.5 * (1.5 - distance) ** 2,
    )
    derivatives = np.where(
        distance <= 0.5,
        -2.0 * separation,
        -np.sign(separation) * (1.5 - distance),
    )
    return indices, weights, derivatives


def tsc_stencil(
    *, positions: np.ndarray, resolution: int, box_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return periodic grid indices, weights, and physical position derivatives.

    Arrays have shapes ``(bodies,27,3)``, ``(bodies,27)``, and
    ``(bodies,27,3)``. Position derivatives are for the three-dimensional
    assignment weight, not for a separately interpolated finite-difference
    force. Each body has unit weight and zero summed weight derivative.
    """

    centres = np.asarray(positions, dtype=np.float64)
    if (
        centres.ndim != 2 or centres.shape[1] != 3 or centres.shape[0] < 1
        or not np.all(np.isfinite(centres)) or type(resolution) is not int
        or resolution < 4 or not np.isfinite(box_length) or box_length <= 0.0
    ):
        raise ValueError("TSC stencil requires finite positions and a periodic grid")
    indices = np.empty((centres.shape[0], 27, 3), dtype=np.int64)
    weights = np.empty((centres.shape[0], 27), dtype=np.float64)
    derivatives = np.empty((centres.shape[0], 27, 3), dtype=np.float64)
    cell_inverse = resolution / box_length
    for body, position in enumerate(centres):
        coordinate = np.remainder(
            position / box_length + 0.5, 1.0
        ) * resolution
        axes = [_axis_stencil(float(value)) for value in coordinate]
        slot = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    indices[body, slot] = (
                        axes[0][0][i] % resolution,
                        axes[1][0][j] % resolution,
                        axes[2][0][k] % resolution,
                    )
                    wx, wy, wz = (
                        axes[0][1][i], axes[1][1][j], axes[2][1][k]
                    )
                    weights[body, slot] = wx * wy * wz
                    derivatives[body, slot] = cell_inverse * np.array([
                        axes[0][2][i] * wy * wz,
                        wx * axes[1][2][j] * wz,
                        wx * wy * axes[2][2][k],
                    ])
                    slot += 1
    return indices, weights, derivatives


def tsc_source_density(
    *, masses: np.ndarray, positions: np.ndarray, resolution: int,
    box_length: float, device: torch.device,
) -> torch.Tensor:
    """Deposit compact masses as a physical density for periodic Poisson."""

    mass = np.asarray(masses, dtype=np.float64)
    indices, weights, _ = tsc_stencil(
        positions=positions, resolution=resolution, box_length=box_length
    )
    if mass.shape != (indices.shape[0],) or not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("TSC source requires one positive finite mass per body")
    flat_indices = (
        (indices[..., 0] * resolution + indices[..., 1]) * resolution
        + indices[..., 2]
    ).reshape(-1)
    cell_volume = (box_length / resolution) ** 3
    values = (mass[:, None] * weights / cell_volume).reshape(-1)
    unique_indices, inverse = np.unique(flat_indices, return_inverse=True)
    unique_values = np.zeros(unique_indices.size, dtype=np.float64)
    np.add.at(unique_values, inverse, values)
    source = torch.zeros(
        (resolution, resolution, resolution), dtype=torch.float64, device=device
    )
    source.view(-1).index_copy_(
        0,
        torch.as_tensor(unique_indices, device=device),
        torch.as_tensor(unique_values, dtype=torch.float64, device=device),
    )
    return source


def tsc_interaction_and_force(
    *, wave_potential: torch.Tensor, masses: np.ndarray,
    positions: np.ndarray, box_length: float,
) -> tuple[float, np.ndarray]:
    """Return ``sum(M W Phi_wave)`` and its negative position gradient."""

    if (
        wave_potential.ndim != 3
        or len(set(wave_potential.shape)) != 1
        or wave_potential.dtype != torch.float64
    ):
        raise ValueError("TSC force requires a finite cubic float64 potential")
    mass = np.asarray(masses, dtype=np.float64)
    indices, weights, derivatives = tsc_stencil(
        positions=positions, resolution=wave_potential.shape[0],
        box_length=box_length,
    )
    if mass.shape != (indices.shape[0],) or not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("TSC force requires one positive finite mass per body")
    device_indices = torch.as_tensor(indices, device=wave_potential.device)
    sampled = wave_potential[
        device_indices[..., 0], device_indices[..., 1], device_indices[..., 2]
    ].detach().cpu().numpy()
    if not np.all(np.isfinite(sampled)):
        raise ValueError("TSC force sampled a non-finite wave potential")
    energy = float(np.sum(mass[:, None] * weights * sampled))
    force = -mass[:, None] * np.sum(derivatives * sampled[..., None], axis=1)
    return energy, force


def periodic_spectral_gradient(
    *, potential: torch.Tensor, box_length: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a skew-adjoint periodic gradient with zero Nyquist derivative."""

    if (
        potential.ndim != 3 or len(set(potential.shape)) != 1
        or potential.dtype != torch.float64 or not torch.isfinite(potential).all()
        or not np.isfinite(box_length) or box_length <= 0.0
    ):
        raise ValueError("spectral gradient requires a finite cubic potential")
    resolution = potential.shape[0]
    wave_number = 2.0 * torch.pi * torch.fft.fftfreq(
        resolution, d=box_length / resolution,
        dtype=torch.float64, device=potential.device,
    )
    if resolution % 2 == 0:
        wave_number[resolution // 2] = 0.0
    spectrum = torch.fft.fftn(potential)
    gradients = []
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = resolution
        derivative = torch.fft.ifftn(
            spectrum * (1j * wave_number.reshape(shape))
        ).real
        gradients.append(derivative)
    return tuple(gradients)


def tsc_spectral_momentum_force(
    *, wave_potential: torch.Tensor, masses: np.ndarray,
    positions: np.ndarray, box_length: float,
) -> np.ndarray:
    """Interpolate minus spectral wave gradient with TSC weights.

    This force is conjugate to the grid's spectral momentum operator, not to
    the position derivative of the TSC interaction energy. It is a separate
    experimental candidate and must not replace the energy-gradient force
    without a measured total-energy error budget.
    """

    if (
        wave_potential.ndim != 3
        or len(set(wave_potential.shape)) != 1
        or wave_potential.dtype != torch.float64
        or not torch.isfinite(wave_potential).all()
        or not np.isfinite(box_length) or box_length <= 0.0
    ):
        raise ValueError("spectral TSC force requires a finite cubic wave potential")
    mass = np.asarray(masses, dtype=np.float64)
    indices, weights, _ = tsc_stencil(
        positions=positions, resolution=wave_potential.shape[0],
        box_length=box_length,
    )
    if (
        mass.shape != (indices.shape[0],) or np.any(~np.isfinite(mass))
        or np.any(mass <= 0.0)
    ):
        raise ValueError("spectral TSC force requires positive finite masses")
    device_indices = torch.as_tensor(indices, device=wave_potential.device)
    force = np.empty((mass.size, 3), dtype=np.float64)
    resolution = wave_potential.shape[0]
    wave_number = 2.0 * torch.pi * torch.fft.fftfreq(
        resolution, d=box_length / resolution,
        dtype=torch.float64, device=wave_potential.device,
    )
    if resolution % 2 == 0:
        wave_number[resolution // 2] = 0.0
    spectrum = torch.fft.fftn(wave_potential)
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = resolution
        gradient = torch.fft.ifftn(
            spectrum * (1j * wave_number.reshape(shape))
        ).real
        sampled = gradient[
            device_indices[..., 0], device_indices[..., 1],
            device_indices[..., 2],
        ].detach().cpu().numpy()
        force[:, axis] = -mass * np.sum(weights * sampled, axis=1)
        del sampled, gradient
    return force


def kick_binary_tsc(
    *, state: np.ndarray, masses: np.ndarray, wave_potential: torch.Tensor,
    box_length: float, plummer_radius: float, time_step: float,
    force_scheme: str = "energy_gradient",
) -> np.ndarray:
    """Exact velocity subflow at fixed wave density and SMBH positions.

    The default force is the position gradient of the TSC interaction energy.
    The alternative spectral-momentum force is experimental and is not an
    energy gradient. A signed step permits a reversibility check.
    """

    body = np.asarray(state, dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    if (
        body.shape != (2, 6) or mass.shape != (2,)
        or not np.all(np.isfinite(body)) or not np.all(np.isfinite(mass))
        or np.any(mass <= 0.0) or not np.isfinite(plummer_radius)
        or plummer_radius <= 0.0 or not np.isfinite(time_step)
        or force_scheme not in {"energy_gradient", "spectral_momentum"}
    ):
        raise ValueError("TSC kick requires a finite two-body state")
    if np.any(np.abs(body[:, :3]) >= 0.5 * box_length):
        raise ValueError("SMBH left the nonperiodic direct-binary force domain")
    if force_scheme == "energy_gradient":
        _, wave_force = tsc_interaction_and_force(
            wave_potential=wave_potential, masses=mass,
            positions=body[:, :3], box_length=box_length,
        )
    else:
        wave_force = tsc_spectral_momentum_force(
            wave_potential=wave_potential, masses=mass,
            positions=body[:, :3], box_length=box_length,
        )
    displacement = body[1, :3] - body[0, :3]
    if np.linalg.norm(displacement) >= 0.5 * box_length:
        raise ValueError("SMBH separation exceeds the direct-binary force domain")
    inverse_cube = (np.dot(displacement, displacement) + plummer_radius**2) ** -1.5
    accelerated = body.copy()
    accelerated[:, 3:] += time_step * wave_force / mass[:, None]
    accelerated[0, 3:] += time_step * mass[1] * displacement * inverse_cube
    accelerated[1, 3:] -= time_step * mass[0] * displacement * inverse_cube
    if not np.all(np.isfinite(accelerated)):
        raise ValueError("TSC kick produced a non-finite state")
    return accelerated


def drift_binary_tsc(
    *, state: np.ndarray, box_length: float, time_step: float,
) -> np.ndarray:
    """Exact SMBH kinetic subflow, with direct-binary boundary protection."""

    body = np.asarray(state, dtype=np.float64)
    if (
        body.shape != (2, 6) or not np.all(np.isfinite(body))
        or not np.isfinite(box_length) or box_length <= 0.0
        or not np.isfinite(time_step)
    ):
        raise ValueError("TSC drift requires a finite two-body state")
    advanced = body.copy()
    advanced[:, :3] += time_step * body[:, 3:]
    if not np.all(np.isfinite(advanced)) or np.any(
        np.abs(advanced[:, :3]) >= 0.5 * box_length
    ):
        raise ValueError("SMBH left the nonperiodic direct-binary force domain")
    if np.linalg.norm(advanced[1, :3] - advanced[0, :3]) >= 0.5 * box_length:
        raise ValueError("SMBH separation exceeds the direct-binary force domain")
    return advanced


def validate_binary_tsc_timestep(
    *, state: np.ndarray, masses: np.ndarray, box_length: float,
    resolution: int,
    plummer_radius: float, time_step: float,
) -> None:
    """Fail closed when one step undersamples an orbit or a TSC cell crossing."""

    body = np.asarray(state, dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    if (
        body.shape != (2, 6) or mass.shape != (2,)
        or not np.all(np.isfinite(body)) or not np.all(np.isfinite(mass))
        or np.any(mass <= 0.0) or not np.isfinite(box_length)
        or box_length <= 0.0 or type(resolution) is not int
        or resolution < 4 or not np.isfinite(plummer_radius)
        or plummer_radius <= 0.0 or not np.isfinite(time_step)
        or time_step <= 0.0
    ):
        raise ValueError("TSC time-step gate requires finite positive inputs")
    separation = np.linalg.norm(body[1, :3] - body[0, :3])
    local_frequency = np.sqrt(
        mass.sum() / (separation**2 + plummer_radius**2) ** 1.5
    )
    cell_size = box_length / resolution
    if time_step * local_frequency > 0.1:
        raise ValueError("SMBH orbit undersampled by TSC time step")
    if np.max(np.abs(body[:, 3:])) * time_step / cell_size > 0.1:
        raise ValueError("SMBH TSC cell crossing undersampled by time step")


def periodic_tsc_patches(
    *, potential: torch.Tensor, positions: np.ndarray,
    box_length: float, width: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Copy tiny periodic wave-potential patches for all RK4 stages."""

    centres = np.asarray(positions, dtype=np.float64)
    if (
        potential.ndim != 3 or len(set(potential.shape)) != 1
        or potential.dtype != torch.float64
        or centres.ndim != 2 or centres.shape[1] != 3
        or centres.shape[0] < 1 or not np.all(np.isfinite(centres))
        or not np.isfinite(box_length) or box_length <= 0.0
        or type(width) is not int or width < 6
    ):
        raise ValueError("periodic TSC patches require a valid wave grid and bodies")
    resolution = potential.shape[0]
    grid_positions = (centres / box_length + 0.5) * resolution
    starts = np.floor(grid_positions + 0.5).astype(np.int64) - 3
    patches = []
    for start in starts:
        axes = [
            torch.as_tensor(
                (start[axis] + np.arange(width)) % resolution,
                device=potential.device,
            )
            for axis in range(3)
        ]
        patch = potential[
            axes[0][:, None, None], axes[1][None, :, None],
            axes[2][None, None, :],
        ]
        patches.append(patch.detach().cpu().numpy())
    return np.stack(patches), starts


@njit(cache=True)
def _tsc_axis_weights(grid_position: float):
    nearest = int(np.floor(grid_position + 0.5))
    indices = np.empty(3, dtype=np.int64)
    weights = np.empty(3, dtype=np.float64)
    derivatives = np.empty(3, dtype=np.float64)
    for slot in range(3):
        node = nearest + slot - 1
        distance_signed = grid_position - node
        distance = abs(distance_signed)
        indices[slot] = node
        if distance <= 0.5:
            weights[slot] = 0.75 - distance * distance
            derivatives[slot] = -2.0 * distance_signed
        else:
            weights[slot] = 0.5 * (1.5 - distance) ** 2
            derivatives[slot] = -np.sign(distance_signed) * (1.5 - distance)
    return indices, weights, derivatives


@njit(cache=True)
def _tsc_patched_wave_acceleration(
    position: np.ndarray, patch: np.ndarray, start: np.ndarray,
    box_length: float, resolution: int,
) -> np.ndarray:
    grid_position = (position / box_length + 0.5) * resolution
    x_index, x_weight, x_derivative = _tsc_axis_weights(grid_position[0])
    y_index, y_weight, y_derivative = _tsc_axis_weights(grid_position[1])
    z_index, z_weight, z_derivative = _tsc_axis_weights(grid_position[2])
    acceleration = np.zeros(3, dtype=np.float64)
    inverse_cell = resolution / box_length
    for i in range(3):
        ix = x_index[i] - start[0]
        for j in range(3):
            iy = y_index[j] - start[1]
            for k in range(3):
                iz = z_index[k] - start[2]
                if (
                    ix < 0 or ix >= patch.shape[0]
                    or iy < 0 or iy >= patch.shape[1]
                    or iz < 0 or iz >= patch.shape[2]
                ):
                    raise ValueError("SMBH left its periodic TSC force patch")
                value = patch[ix, iy, iz]
                acceleration[0] -= (
                    value * x_derivative[i] * y_weight[j] * z_weight[k]
                    * inverse_cell
                )
                acceleration[1] -= (
                    value * x_weight[i] * y_derivative[j] * z_weight[k]
                    * inverse_cell
                )
                acceleration[2] -= (
                    value * x_weight[i] * y_weight[j] * z_derivative[k]
                    * inverse_cell
                )
    return acceleration


@njit(cache=True)
def _tsc_binary_derivative(
    state: np.ndarray, masses: np.ndarray, patches: np.ndarray,
    starts: np.ndarray, box_length: float, resolution: int,
    plummer_radius: float,
) -> np.ndarray:
    result = np.zeros_like(state)
    for body in range(2):
        result[body, :3] = state[body, 3:]
        result[body, 3:] = _tsc_patched_wave_acceleration(
            state[body, :3], patches[body], starts[body],
            box_length, resolution,
        )
    displacement = state[1, :3] - state[0, :3]
    denominator = (np.dot(displacement, displacement) + plummer_radius**2) ** 1.5
    result[0, 3:] += masses[1] * displacement / denominator
    result[1, 3:] -= masses[0] * displacement / denominator
    return result


@njit(cache=True)
def _advance_binary_rk4_tsc_impl(
    state: np.ndarray, masses: np.ndarray, patches: np.ndarray,
    starts: np.ndarray, box_length: float, resolution: int,
    plummer_radius: float, time_step: float, substeps: int,
) -> np.ndarray:
    advanced = state.copy()
    substep = time_step / substeps
    for _ in range(substeps):
        k1 = _tsc_binary_derivative(
            advanced, masses, patches, starts, box_length, resolution,
            plummer_radius,
        )
        k2 = _tsc_binary_derivative(
            advanced + 0.5 * substep * k1, masses, patches, starts,
            box_length, resolution, plummer_radius,
        )
        k3 = _tsc_binary_derivative(
            advanced + 0.5 * substep * k2, masses, patches, starts,
            box_length, resolution, plummer_radius,
        )
        k4 = _tsc_binary_derivative(
            advanced + substep * k3, masses, patches, starts,
            box_length, resolution, plummer_radius,
        )
        advanced += substep * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return advanced


def advance_binary_rk4_tsc(
    *, state: np.ndarray, masses: np.ndarray, patches: np.ndarray,
    patch_starts: np.ndarray, box_length: float, resolution: int,
    plummer_radius: float, time_step: float, substeps: int,
) -> np.ndarray:
    """Advance a binary with TSC force exactly conjugate to its PM energy."""

    state_array = np.asarray(state, dtype=np.float64).reshape(2, 6)
    mass = np.asarray(masses, dtype=np.float64)
    cubes = np.asarray(patches, dtype=np.float64)
    starts = np.asarray(patch_starts, dtype=np.int64)
    if (
        mass.shape != (2,) or np.any(~np.isfinite(mass))
        or np.any(mass <= 0.0) or np.any(~np.isfinite(state_array))
        or cubes.ndim != 4 or cubes.shape[0] != 2
        or cubes.shape[1:] != (cubes.shape[1],) * 3
        or starts.shape != (2, 3) or cubes.shape[1] < 6
        or not np.isfinite(box_length) or box_length <= 0.0
        or type(resolution) is not int or resolution < 4
        or not np.isfinite(plummer_radius) or plummer_radius <= 0.0
        or not np.isfinite(time_step) or time_step <= 0.0
        or type(substeps) is not int or substeps < 1
    ):
        raise ValueError("TSC RK4 requires a finite two-body state and force patches")
    return _advance_binary_rk4_tsc_impl(
        state_array, mass, cubes, starts, box_length, resolution,
        plummer_radius, time_step, substeps,
    ).reshape(-1)
