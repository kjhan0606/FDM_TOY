"""Reduced diagnostics for the response of a live FDM wave to SMBHs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WaveDifferentialFields:
    density: np.ndarray
    gradients: tuple[np.ndarray, np.ndarray, np.ndarray]
    laplacian: np.ndarray
    kinetic_energy_density: np.ndarray
    mass_current: tuple[np.ndarray, np.ndarray, np.ndarray]
    schrodinger_energy_current: tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class MultipoleAmplitudes:
    mass: float
    l1_fraction: float
    l2_fraction: float


def periodic_centre_of_mass(density: np.ndarray, box_size: float) -> np.ndarray:
    """Return the circular centre of mass of a compact periodic density field."""

    rho = np.asarray(density, dtype=float)
    if rho.ndim != 3 or len(set(rho.shape)) != 1:
        raise ValueError("density must be a cubic 3D array")
    if box_size <= 0.0 or np.any(~np.isfinite(rho)) or np.any(rho < 0.0):
        raise ValueError("box size and density must be finite and non-negative")
    if not np.sum(rho) > 0.0:
        raise ValueError("density must have positive total mass")
    n = rho.shape[0]
    coordinate = np.linspace(-0.5 * box_size, 0.5 * box_size, n, endpoint=False)
    angle = 2.0 * np.pi * (coordinate / box_size + 0.5)
    centre = []
    for axis in range(3):
        weights = np.sum(rho, axis=tuple(index for index in range(3) if index != axis))
        moment = np.sum(weights * np.exp(1j * angle))
        phase = np.angle(moment) % (2.0 * np.pi)
        centre.append(box_size * (phase / (2.0 * np.pi) - 0.5))
    return np.asarray(centre)


def centred_grid(
    resolution: int, box_size: float, centre: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return minimum-image Cartesian displacements and radius."""

    if resolution < 2 or box_size <= 0.0:
        raise ValueError("resolution and box size must be positive")
    centre_array = np.asarray(centre, dtype=float)
    if centre_array.shape != (3,):
        raise ValueError("centre must have shape (3,)")
    coordinate = np.linspace(
        -0.5 * box_size, 0.5 * box_size, resolution, endpoint=False
    )
    components = []
    for axis in range(3):
        displacement = coordinate - centre_array[axis]
        components.append(
            displacement - box_size * np.floor(displacement / box_size + 0.5)
        )
    x = components[0][:, None, None]
    y = components[1][None, :, None]
    z = components[2][None, None, :]
    radius = np.sqrt(x * x + y * y + z * z)
    return x, y, z, radius


def spectral_wave_fields(
    wavefunction: np.ndarray,
    total_potential: np.ndarray,
    box_size: float,
) -> WaveDifferentialFields:
    """Evaluate spectral derivatives and Schrödinger mass and energy currents.

    Units follow ``i dpsi/dt = (-laplacian/2 + Phi) psi``.  The returned energy
    current is the local Schrödinger-field current.  Gravitational field energy
    is non-local and remains accounted for by the global Hamiltonian ledger.
    """

    psi = np.asarray(wavefunction, dtype=complex)
    potential = np.asarray(total_potential, dtype=float)
    if psi.shape != potential.shape or psi.ndim != 3 or len(set(psi.shape)) != 1:
        raise ValueError("wavefunction and potential must be matching cubic arrays")
    if box_size <= 0.0:
        raise ValueError("box size must be positive")
    n = psi.shape[0]
    cell_size = box_size / n
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=cell_size)
    psi_k = np.fft.fftn(psi)
    gradients = []
    kinetic_density = np.zeros(psi.shape, dtype=float)
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = n
        gradient = np.fft.ifftn(1j * k.reshape(shape) * psi_k)
        gradients.append(gradient)
        kinetic_density += 0.5 * np.abs(gradient) ** 2
    k_squared = (
        k[:, None, None] ** 2
        + k[None, :, None] ** 2
        + k[None, None, :] ** 2
    )
    laplacian = np.fft.ifftn(-k_squared * psi_k)
    density = np.abs(psi) ** 2
    mass_current = tuple(
        np.imag(np.conjugate(psi) * gradient) for gradient in gradients
    )
    psi_time_derivative = -1j * (-0.5 * laplacian + potential * psi)
    energy_current = tuple(
        -np.real(np.conjugate(psi_time_derivative) * gradient)
        for gradient in gradients
    )
    return WaveDifferentialFields(
        density=density,
        gradients=tuple(gradients),
        laplacian=laplacian,
        kinetic_energy_density=kinetic_density,
        mass_current=mass_current,
        schrodinger_energy_current=energy_current,
    )


def periodic_poisson_code(density: np.ndarray, box_size: float) -> np.ndarray:
    """Solve code-unit periodic Poisson gravity with zero mean potential."""

    rho = np.asarray(density, dtype=float)
    if rho.ndim != 3 or len(set(rho.shape)) != 1 or box_size <= 0.0:
        raise ValueError("density must be cubic and box size must be positive")
    n = rho.shape[0]
    cell_size = box_size / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=cell_size)[:, None, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=cell_size)[None, :, None]
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=cell_size)[None, None, :]
    k_squared = kx * kx + ky * ky + kz * kz
    k_squared[0, 0, 0] = np.inf
    potential_k = -4.0 * np.pi * np.fft.rfftn(rho) / k_squared
    return np.fft.irfftn(potential_k, s=rho.shape, axes=(0, 1, 2)).real


def plummer_potential_code(
    *,
    shape: tuple[int, int, int],
    box_size: float,
    masses: np.ndarray,
    positions: np.ndarray,
    plummer_radius: float,
) -> np.ndarray:
    """Return the SMBH potential used by the public PyUL_NBody calculation."""

    if len(set(shape)) != 1 or len(shape) != 3:
        raise ValueError("shape must describe a cubic grid")
    masses_array = np.asarray(masses, dtype=float)
    positions_array = np.asarray(positions, dtype=float)
    if positions_array.shape != (masses_array.size, 3):
        raise ValueError("positions must have shape (number of SMBHs, 3)")
    if np.any(masses_array <= 0.0) or plummer_radius <= 0.0:
        raise ValueError("masses and Plummer radius must be positive")
    n = shape[0]
    coordinate = np.linspace(-0.5 * box_size, 0.5 * box_size, n, endpoint=False)
    potential = np.zeros(shape, dtype=float)
    for mass, position in zip(masses_array, positions_array, strict=True):
        dx = coordinate[:, None, None] - position[0]
        dy = coordinate[None, :, None] - position[1]
        dz = coordinate[None, None, :] - position[2]
        potential -= mass / np.sqrt(dx * dx + dy * dy + dz * dz + plummer_radius**2)
    return potential


def multipole_amplitudes(
    density: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    radius: np.ndarray,
    selection: np.ndarray,
    cell_volume: float,
) -> MultipoleAmplitudes:
    """Return rotationally invariant normalized l=1 and l=2 amplitudes."""

    rho = np.asarray(density, dtype=float)
    mask = np.asarray(selection, dtype=bool)
    if rho.shape != radius.shape or rho.shape != mask.shape:
        raise ValueError("density, radius, and selection must have matching shapes")
    weights = rho[mask] * cell_volume
    mass = float(np.sum(weights))
    if mass <= 0.0:
        return MultipoleAmplitudes(0.0, np.nan, np.nan)
    selected_radius = radius[mask]
    safe_radius = np.where(radius > 0.0, radius, 1.0)
    xhat = np.broadcast_to(x, rho.shape)[mask] / safe_radius[mask]
    yhat = np.broadcast_to(y, rho.shape)[mask] / safe_radius[mask]
    zhat = np.broadcast_to(z, rho.shape)[mask] / safe_radius[mask]

    a10 = np.sqrt(3.0 / (4.0 * np.pi)) * np.sum(weights * zhat)
    a11 = -np.sqrt(3.0 / (8.0 * np.pi)) * np.sum(
        weights * (xhat + 1j * yhat)
    )
    l1_power = abs(a10) ** 2 + 2.0 * abs(a11) ** 2

    y20_shape = np.where(selected_radius > 0.0, 3.0 * zhat**2 - 1.0, 0.0)
    a20 = np.sqrt(5.0 / (16.0 * np.pi)) * np.sum(weights * y20_shape)
    a21 = -np.sqrt(15.0 / (8.0 * np.pi)) * np.sum(
        weights * zhat * (xhat + 1j * yhat)
    )
    a22 = np.sqrt(15.0 / (32.0 * np.pi)) * np.sum(
        weights * (xhat + 1j * yhat) ** 2
    )
    l2_power = abs(a20) ** 2 + 2.0 * abs(a21) ** 2 + 2.0 * abs(a22) ** 2
    normalization = np.sqrt(4.0 * np.pi) / mass
    return MultipoleAmplitudes(
        mass=mass,
        l1_fraction=float(normalization * np.sqrt(l1_power)),
        l2_fraction=float(normalization * np.sqrt(l2_power)),
    )
