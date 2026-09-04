"""CUDA operators for a live FDM wave coupled to two compact masses.

The operators use the same dimensionless Schrödinger--Poisson convention as
PyUL_NBody.  They are kept separate from the public solver so that the original
CPU calculation remains an independent numerical reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

try:
    import torch
    import torch.nn.functional as torch_functional
except ImportError as error:  # pragma: no cover - exercised without GPU extras
    raise ImportError(
        "the CUDA live-wave solver requires the optional torch dependency"
    ) from error


@dataclass(frozen=True)
class SpectralGrid:
    coordinate: torch.Tensor
    kinetic_phase: torch.Tensor
    kinetic_wavenumber_squared: torch.Tensor
    poisson_inverse_wavenumber_squared: torch.Tensor
    cell_size: float
    cell_volume: float


def spectral_grid(
    *,
    resolution: int,
    box_length: float,
    time_step: float,
    device: torch.device,
) -> SpectralGrid:
    """Construct spectral factors for one Strang-splitting wave step."""

    if resolution < 2 or box_length <= 0.0 or time_step <= 0.0:
        raise ValueError("resolution, box length, and time step must be positive")
    real_dtype = torch.float64
    coordinate = torch.linspace(
        -0.5 * box_length,
        0.5 * box_length,
        resolution + 1,
        dtype=real_dtype,
        device=device,
    )[:-1]
    cell_size = box_length / resolution
    wave_number = 2.0 * torch.pi * torch.fft.fftfreq(
        resolution, d=cell_size, dtype=real_dtype, device=device
    )
    wave_number_real = 2.0 * torch.pi * torch.fft.rfftfreq(
        resolution, d=cell_size, dtype=real_dtype, device=device
    )
    k_squared = (
        wave_number[:, None, None].square()
        + wave_number[None, :, None].square()
        + wave_number[None, None, :].square()
    )
    poisson_k_squared = (
        wave_number[:, None, None].square()
        + wave_number[None, :, None].square()
        + wave_number_real[None, None, :].square()
    )
    poisson_inverse = torch.zeros_like(poisson_k_squared)
    nonzero = poisson_k_squared > 0.0
    poisson_inverse[nonzero] = 1.0 / poisson_k_squared[nonzero]
    kinetic_phase = torch.polar(
        torch.ones_like(k_squared), -0.5 * time_step * k_squared
    )
    return SpectralGrid(
        coordinate=coordinate,
        kinetic_phase=kinetic_phase,
        kinetic_wavenumber_squared=k_squared,
        poisson_inverse_wavenumber_squared=poisson_inverse,
        cell_size=cell_size,
        cell_volume=cell_size**3,
    )


def periodic_poisson_torch(
    density: torch.Tensor, inverse_wavenumber_squared: torch.Tensor
) -> torch.Tensor:
    """Solve ``nabla^2 Phi = 4 pi rho`` with zero mean potential."""

    if density.ndim != 3 or len(set(density.shape)) != 1:
        raise ValueError("density must be a cubic three-dimensional tensor")
    expected = (
        density.shape[0],
        density.shape[1],
        density.shape[2] // 2 + 1,
    )
    if inverse_wavenumber_squared.shape != expected:
        raise ValueError("Poisson spectral factor has an incompatible shape")
    density_k = torch.fft.rfftn(density)
    density_k.mul_(-4.0 * torch.pi * inverse_wavenumber_squared)
    density_k[0, 0, 0] = 0.0
    return torch.fft.irfftn(density_k, s=density.shape)


def plummer_potential_torch(
    *,
    coordinate: torch.Tensor,
    masses: np.ndarray,
    positions: np.ndarray,
    plummer_radius: float,
) -> torch.Tensor:
    """Return the softened potential of compact masses on a Cartesian grid."""

    masses_array = np.asarray(masses, dtype=float)
    positions_array = np.asarray(positions, dtype=float)
    if positions_array.shape != (masses_array.size, 3):
        raise ValueError("positions must have shape (number of masses, 3)")
    if np.any(masses_array <= 0.0) or plummer_radius <= 0.0:
        raise ValueError("masses and Plummer radius must be positive")
    n = coordinate.numel()
    potential = torch.zeros(
        (n, n, n), dtype=torch.float64, device=coordinate.device
    )
    for mass, position in zip(masses_array, positions_array, strict=True):
        distance_squared = (coordinate - float(position[0])).square()[:, None, None]
        distance_squared = distance_squared + (
            coordinate - float(position[1])
        ).square()[None, :, None]
        distance_squared = distance_squared + (
            coordinate - float(position[2])
        ).square()[None, None, :]
        distance_squared.add_(plummer_radius**2)
        potential.add_(torch.rsqrt(distance_squared), alpha=-float(mass))
    return potential


def sample_potential_and_acceleration(
    *,
    potential: torch.Tensor,
    positions: np.ndarray,
    box_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Trilinearly sample potential and its centred-difference acceleration."""

    position_array = np.asarray(positions, dtype=float)
    if position_array.ndim != 2 or position_array.shape[1] != 3:
        raise ValueError("positions must have shape (number of masses, 3)")
    if potential.ndim != 3 or len(set(potential.shape)) != 1:
        raise ValueError("potential must be a cubic three-dimensional tensor")
    resolution = potential.shape[0]
    cell_size = box_length / resolution
    samples = []
    for position in position_array:
        samples.append(position)
        for axis in range(3):
            minus = position.copy()
            plus = position.copy()
            minus[axis] -= cell_size
            plus[axis] += cell_size
            samples.extend((minus, plus))
    samples_array = np.asarray(samples)
    grid_indices = (samples_array / box_length + 0.5) * resolution
    normalized = 2.0 * grid_indices / (resolution - 1.0) - 1.0
    # torch grid_sample orders the coordinates as W, H, D. The PyUL arrays use
    # x, y, z as their first, second, and third axes respectively.
    torch_grid = torch.as_tensor(
        normalized[:, ::-1].copy(), dtype=torch.float64, device=potential.device
    ).reshape(1, 1, 1, -1, 3)
    sampled = torch_functional.grid_sample(
        potential.reshape(1, 1, *potential.shape),
        torch_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(-1)
    sampled_host = sampled.detach().cpu().numpy()
    values = []
    accelerations = []
    offset = 0
    for _position in position_array:
        values.append(sampled_host[offset])
        offset += 1
        gradient = []
        for _axis in range(3):
            minus, plus = sampled_host[offset : offset + 2]
            gradient.append((plus - minus) / (2.0 * cell_size))
            offset += 2
        accelerations.append(-np.asarray(gradient))
    return np.asarray(values), np.asarray(accelerations)


def advance_binary_rk4(
    *,
    state: np.ndarray,
    masses: np.ndarray,
    external_acceleration: np.ndarray,
    plummer_radius: float,
    time_step: float,
    substeps: int,
) -> np.ndarray:
    """Advance two particles while holding the wave acceleration per step."""

    state_array = np.asarray(state, dtype=float).reshape(2, 6).copy()
    masses_array = np.asarray(masses, dtype=float)
    external = np.asarray(external_acceleration, dtype=float)
    if masses_array.shape != (2,) or external.shape != (2, 3):
        raise ValueError("binary masses and external accelerations have invalid shape")
    if plummer_radius <= 0.0 or time_step <= 0.0 or substeps < 1:
        raise ValueError("Plummer radius, time step, and substeps must be positive")

    def derivative(stage: np.ndarray) -> np.ndarray:
        result = np.zeros_like(stage)
        result[:, :3] = stage[:, 3:]
        result[:, 3:] = external
        displacement = stage[1, :3] - stage[0, :3]
        denominator = (
            float(displacement @ displacement) + plummer_radius**2
        ) ** 1.5
        result[0, 3:] += masses_array[1] * displacement / denominator
        result[1, 3:] -= masses_array[0] * displacement / denominator
        return result

    substep = time_step / substeps
    for _ in range(substeps):
        k1 = derivative(state_array)
        k2 = derivative(state_array + 0.5 * substep * k1)
        k3 = derivative(state_array + 0.5 * substep * k2)
        k4 = derivative(state_array + substep * k3)
        state_array += substep * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return state_array.reshape(-1)


def potential_patches(
    *,
    potential: torch.Tensor,
    positions: np.ndarray,
    box_length: float,
    width: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Copy compact local potential cubes for particle RK4 interpolation."""

    if width < 6:
        raise ValueError("potential patches must contain at least six cells per axis")
    position_array = np.asarray(positions, dtype=float)
    resolution = potential.shape[0]
    grid_indices = (position_array / box_length + 0.5) * resolution
    base = np.floor(grid_indices).astype(int)
    starts = base - 2
    if np.any(starts < 0) or np.any(starts + width > resolution):
        raise ValueError("a compact mass is too close to the grid boundary")
    device_patches = torch.stack(
        [
            potential[
                start[0] : start[0] + width,
                start[1] : start[1] + width,
                start[2] : start[2] + width,
            ]
            for start in starts
        ]
    )
    return device_patches.detach().cpu().numpy(), starts


@njit(cache=True)
def _patched_wave_acceleration(
    position: np.ndarray,
    patch: np.ndarray,
    start: np.ndarray,
    box_length: float,
    resolution: int,
) -> np.ndarray:
    cell_size = box_length / resolution
    grid_position = (position / box_length + 0.5) * resolution
    base = np.floor(grid_position).astype(np.int64)
    remainder = grid_position - base
    local = base - start
    acceleration = np.zeros(3, dtype=np.float64)
    for corner_x in range(2):
        weight_x = remainder[0] if corner_x else 1.0 - remainder[0]
        for corner_y in range(2):
            weight_y = remainder[1] if corner_y else 1.0 - remainder[1]
            for corner_z in range(2):
                weight_z = remainder[2] if corner_z else 1.0 - remainder[2]
                weight = weight_x * weight_y * weight_z
                ix = local[0] + corner_x
                iy = local[1] + corner_y
                iz = local[2] + corner_z
                gradient_x = (patch[ix + 1, iy, iz] - patch[ix - 1, iy, iz]) / (
                    2.0 * cell_size
                )
                gradient_y = (patch[ix, iy + 1, iz] - patch[ix, iy - 1, iz]) / (
                    2.0 * cell_size
                )
                gradient_z = (patch[ix, iy, iz + 1] - patch[ix, iy, iz - 1]) / (
                    2.0 * cell_size
                )
                acceleration[0] -= weight * gradient_x
                acceleration[1] -= weight * gradient_y
                acceleration[2] -= weight * gradient_z
    return acceleration


@njit(cache=True)
def _patched_binary_derivative(
    state: np.ndarray,
    masses: np.ndarray,
    patches: np.ndarray,
    starts: np.ndarray,
    box_length: float,
    resolution: int,
    plummer_radius: float,
) -> np.ndarray:
    result = np.zeros_like(state)
    for body in range(2):
        result[body, 0:3] = state[body, 3:6]
        result[body, 3:6] = _patched_wave_acceleration(
            state[body, 0:3],
            patches[body],
            starts[body],
            box_length,
            resolution,
        )
    displacement = state[1, 0:3] - state[0, 0:3]
    denominator = (
        np.dot(displacement, displacement) + plummer_radius**2
    ) ** 1.5
    result[0, 3:6] += masses[1] * displacement / denominator
    result[1, 3:6] -= masses[0] * displacement / denominator
    return result


@njit(cache=True)
def _advance_binary_rk4_patched_impl(
    state: np.ndarray,
    masses: np.ndarray,
    patches: np.ndarray,
    starts: np.ndarray,
    box_length: float,
    resolution: int,
    plummer_radius: float,
    time_step: float,
    substeps: int,
) -> np.ndarray:
    advanced = state.copy()
    substep = time_step / substeps
    for _ in range(substeps):
        k1 = _patched_binary_derivative(
            advanced, masses, patches, starts, box_length, resolution, plummer_radius
        )
        k2 = _patched_binary_derivative(
            advanced + 0.5 * substep * k1,
            masses,
            patches,
            starts,
            box_length,
            resolution,
            plummer_radius,
        )
        k3 = _patched_binary_derivative(
            advanced + 0.5 * substep * k2,
            masses,
            patches,
            starts,
            box_length,
            resolution,
            plummer_radius,
        )
        k4 = _patched_binary_derivative(
            advanced + substep * k3,
            masses,
            patches,
            starts,
            box_length,
            resolution,
            plummer_radius,
        )
        advanced += substep * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return advanced


def advance_binary_rk4_patched(
    *,
    state: np.ndarray,
    masses: np.ndarray,
    patches: np.ndarray,
    patch_starts: np.ndarray,
    box_length: float,
    resolution: int,
    plummer_radius: float,
    time_step: float,
    substeps: int,
) -> np.ndarray:
    """Match PyUL's changing wave-force interpolation through each RK4 stage."""

    state_array = np.asarray(state, dtype=np.float64).reshape(2, 6)
    return _advance_binary_rk4_patched_impl(
        state_array,
        np.asarray(masses, dtype=np.float64),
        np.asarray(patches, dtype=np.float64),
        np.asarray(patch_starts, dtype=np.int64),
        float(box_length),
        int(resolution),
        float(plummer_radius),
        float(time_step),
        int(substeps),
    ).reshape(-1)


def wave_energy_components(
    *,
    wavefunction: torch.Tensor,
    density: torch.Tensor,
    wave_potential: torch.Tensor,
    compact_potential: torch.Tensor,
    kinetic_wavenumber_squared: torch.Tensor,
    cell_volume: float,
) -> tuple[float, float, float, float]:
    """Return kinetic, self-gravity, compact-interaction energy, and mass."""

    wavefunction_k = torch.fft.fftn(wavefunction)
    cells = wavefunction.numel()
    kinetic = (
        0.5
        * cell_volume
        / cells
        * torch.sum(kinetic_wavenumber_squared * wavefunction_k.abs().square())
    )
    self_gravity = 0.5 * cell_volume * torch.sum(wave_potential * density)
    interaction = cell_volume * torch.sum(compact_potential * density)
    mass = cell_volume * torch.sum(density)
    return tuple(
        float(value.detach().cpu())
        for value in (kinetic, self_gravity, interaction, mass)
    )
