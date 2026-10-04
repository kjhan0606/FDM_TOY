"""Conjugate-force checks for the gridded SMBH--wave interaction energy."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fdm_smbh_delay.force_energy_audit import plummer_grid_conjugate_force
from scripts.audit_wave_force_energy import (
    _binary_energy,
    _interaction_at_fixed_density,
    _rk4_wave_work,
    _verify_measured_energy_conversion,
)
from fdm_smbh_delay.torch_wave import (
    advance_binary_rk4_patched,
    plummer_potential_torch,
    potential_patches,
    wave_potential_energy_components,
)


def test_grid_force_is_negative_position_derivative_of_plummer_energy() -> None:
    axis = np.linspace(-2.0, 2.0, 16, endpoint=False)
    rng = np.random.default_rng(4152)
    density = rng.uniform(0.1, 1.5, size=(16, 16, 16))
    masses = np.array([0.7, 1.3])
    positions = np.array([[0.18, -0.23, 0.31], [-0.37, 0.14, -0.11]])
    softening = 0.2
    volume = (axis[1] - axis[0]) ** 3

    def interaction(body: int, centre: np.ndarray) -> float:
        rho = torch.from_numpy(density)
        compact = plummer_potential_torch(
            coordinate=torch.from_numpy(axis),
            masses=masses[body : body + 1],
            positions=centre.reshape(1, 3),
            plummer_radius=softening,
        )
        _, energy, _ = wave_potential_energy_components(
            density=rho,
            wave_potential=torch.zeros_like(rho),
            compact_potential=compact,
            cell_volume=volume,
        )
        return energy

    force = plummer_grid_conjugate_force(
        density=density,
        coordinate=axis,
        masses=masses,
        positions=positions,
        plummer_radius=softening,
        cell_volume=volume,
        slab_depth=3,
    )
    epsilon = 1.0e-5
    for body in range(2):
        for direction in range(3):
            plus = positions[body].copy()
            minus = positions[body].copy()
            plus[direction] += epsilon
            minus[direction] -= epsilon
            derivative = (
                interaction(body, plus) - interaction(body, minus)
            ) / (2.0 * epsilon)
            assert force[body, direction] == pytest.approx(
                -derivative, rel=2.0e-8, abs=2.0e-8
            )


def test_force_audit_rejects_invalid_density_and_shapes() -> None:
    arguments = dict(
        density=np.ones((4, 4, 4)),
        coordinate=np.arange(4, dtype=float),
        masses=np.array([1.0]),
        positions=np.zeros((1, 3)),
        plummer_radius=0.1,
        cell_volume=1.0,
    )
    bad = dict(arguments)
    bad["density"] = -np.ones((4, 4, 4))
    with pytest.raises(ValueError, match="finite physical inputs"):
        plummer_grid_conjugate_force(**bad)
    bad = dict(arguments)
    bad["positions"] = np.zeros((2, 3))
    with pytest.raises(ValueError, match="incompatible"):
        plummer_grid_conjugate_force(**bad)


def test_fixed_density_interaction_change_matches_conjugate_work() -> None:
    axis = np.linspace(-2.0, 2.0, 16, endpoint=False)
    rng = np.random.default_rng(4153)
    density = rng.uniform(0.1, 1.5, size=(16, 16, 16))
    masses = np.array([0.7, 1.3])
    positions = np.array([[0.18, -0.23, 0.31], [-0.37, 0.14, -0.11]])
    displacement = np.array([[1.0, -2.0, 0.5], [-0.5, 1.0, 2.0]]) * 1e-5
    volume = (axis[1] - axis[0]) ** 3
    grid = type("Grid", (), {
        "coordinate": torch.from_numpy(axis), "cell_volume": volume,
    })()
    rho = torch.from_numpy(density)
    old, _ = _interaction_at_fixed_density(
        density=rho, grid=grid, masses=masses, positions=positions,
        radius=0.2,
    )
    new, _ = _interaction_at_fixed_density(
        density=rho, grid=grid, masses=masses,
        positions=positions + displacement, radius=0.2,
    )
    force = plummer_grid_conjugate_force(
        density=density, coordinate=axis, masses=masses,
        positions=positions + 0.5 * displacement,
        plummer_radius=0.2, cell_volume=volume,
    )
    assert new - old == pytest.approx(
        -float(np.sum(force * displacement)), rel=2e-5, abs=2e-9
    )


def test_binary_energy_is_kinetic_plus_softened_mutual() -> None:
    state = np.array([
        [0.0, 0.0, 0.0, 1.0, 2.0, 0.0],
        [3.0, 0.0, 0.0, -1.0, 0.0, 0.0],
    ])
    expected = 0.5 * 2.0 * 5.0 + 0.5 * 3.0 - 6.0 / np.sqrt(9.25)
    assert _binary_energy(state, np.array([2.0, 3.0]), 0.5) == pytest.approx(expected)


def test_rk4_wave_work_tracks_orbital_energy_with_constant_wave_force() -> None:
    axis = torch.linspace(-2.0, 2.0, 17, dtype=torch.float64)[:-1]
    potential = -0.3 * axis[:, None, None].expand(16, 16, 16).contiguous()
    state = np.array([
        [0.10, 0.12, 0.13, 0.2, 0.1, 0.0],
        [-0.35, -0.18, -0.11, -0.1, 0.05, 0.0],
    ])
    masses = np.array([0.7, 1.3])
    patches, starts = potential_patches(
        potential=potential, positions=state[:, :3], box_length=4.0
    )
    advanced, work = _rk4_wave_work(
        state=state, masses=masses, patches=patches, starts=starts,
        box_length=4.0, resolution=16, radius=0.1, step=1e-3, substeps=9,
    )
    runtime = advance_binary_rk4_patched(
        state=state, masses=masses, patches=patches, patch_starts=starts,
        box_length=4.0, resolution=16, plummer_radius=0.1,
        time_step=1e-3, substeps=9,
    ).reshape(2, 6)
    assert np.allclose(advanced, runtime, rtol=1e-14, atol=1e-14)
    orbital_change = _binary_energy(advanced, masses, 0.1) - _binary_energy(
        state, masses, 0.1
    )
    assert orbital_change == pytest.approx(work, rel=1e-8, abs=1e-12)


def test_measured_hamiltonian_must_match_physical_conversion(
    tmp_path,
) -> None:
    outputs = tmp_path / "Outputs"
    (outputs / "NBody").mkdir(parents=True)
    for name, values in (
        ("ekandqlist", [10.0, 11.0]),
        ("egpsilist", [-4.0, -4.5]),
        ("egpcmlist", [-2.0, -2.5]),
    ):
        np.save(outputs / f"{name}.npy", np.asarray(values))
    initial = np.array([[0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=float)
    np.save(outputs / "NBody/NTM_#000.npy", initial)
    np.save(outputs / "NBody/NTM_#001.npy", initial)
    masses = np.array([2.0, 3.0])
    mutual = -6.0 / np.sqrt(1.0 + 0.1**2)
    series = np.zeros(2, dtype=[("combined_energy", float)])
    series["combined_energy"] = np.array([5.0 + mutual, 5.0 + mutual]) * 7.0
    _verify_measured_energy_conversion(
        trace=tmp_path, series=series, masses=masses,
        plummer_radius=0.1, energy_conversion=7.0,
    )
    series["combined_energy"][1] += 1.0
    with pytest.raises(ValueError, match="physical-unit solver budget"):
        _verify_measured_energy_conversion(
            trace=tmp_path, series=series, masses=masses,
            plummer_radius=0.1, energy_conversion=7.0,
        )
