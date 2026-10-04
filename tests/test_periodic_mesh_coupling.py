"""Mass, periodicity, reciprocity, and force checks for TSC wave coupling."""

import numpy as np
import pytest
import json
import sys

torch = pytest.importorskip("torch")

from fdm_smbh_delay.periodic_mesh_coupling import (
    _tsc_patched_wave_acceleration,
    advance_binary_rk4_tsc,
    periodic_tsc_patches,
    tsc_interaction_and_force,
    tsc_source_density,
    tsc_stencil,
)
from fdm_smbh_delay.torch_wave import periodic_poisson_torch, spectral_grid
from fdm_smbh_delay.pyul import (
    PYUL_MYR_S, PYUL_PARSEC_M, PYUL_SOLAR_MASS_KG,
)
from scripts import run_torch_wave_case


def test_tsc_deposition_preserves_mass_and_wraps_periodically() -> None:
    masses = np.array([0.7, 1.3])
    positions = np.array([[1.98, -1.99, 0.23], [-0.42, 0.55, -0.82]])
    indices, weights, derivatives = tsc_stencil(
        positions=positions, resolution=16, box_length=4.0
    )
    assert indices.shape == (2, 27, 3)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        derivatives.sum(axis=1), 0.0, rtol=0, atol=1e-13
    )
    source = tsc_source_density(
        masses=masses, positions=positions, resolution=16,
        box_length=4.0, device=torch.device("cpu"),
    )
    assert float(source.sum()) * (4.0 / 16) ** 3 == pytest.approx(
        masses.sum(), rel=1e-15
    )
    shifted = tsc_source_density(
        masses=masses, positions=positions + np.array([4.0, -4.0, 8.0]),
        resolution=16, box_length=4.0, device=torch.device("cpu"),
    )
    torch.testing.assert_close(source, shifted, rtol=0, atol=1e-13)
    overlapping = np.array([[0.01, 0.02, 0.03], [0.02, 0.02, 0.03]])
    first_overlap = tsc_source_density(
        masses=masses, positions=overlapping, resolution=16,
        box_length=4.0, device=torch.device("cpu"),
    )
    second_overlap = tsc_source_density(
        masses=masses, positions=overlapping, resolution=16,
        box_length=4.0, device=torch.device("cpu"),
    )
    torch.testing.assert_close(first_overlap, second_overlap, rtol=0, atol=0)
    assert float(first_overlap.sum()) * (4.0 / 16) ** 3 == pytest.approx(
        masses.sum(), rel=1e-15
    )


def test_tsc_poisson_cross_energy_is_reciprocal_and_force_is_its_gradient() -> None:
    grid = spectral_grid(
        resolution=16, box_length=4.0, time_step=1e-3,
        device=torch.device("cpu"),
    )
    rng = np.random.default_rng(9902)
    density = torch.as_tensor(rng.uniform(0.2, 1.5, size=(16, 16, 16)))
    masses = np.array([0.7, 1.3])
    positions = np.array([[1.93, -1.89, 0.23], [-0.42, 0.55, -0.82]])
    wave_potential = periodic_poisson_torch(
        density, grid.poisson_inverse_wavenumber_squared
    )
    source = tsc_source_density(
        masses=masses, positions=positions, resolution=16,
        box_length=4.0, device=torch.device("cpu"),
    )
    compact_potential = periodic_poisson_torch(
        source, grid.poisson_inverse_wavenumber_squared
    )
    grid_interaction = float(grid.cell_volume * torch.sum(density * compact_potential))
    particle_interaction, force = tsc_interaction_and_force(
        wave_potential=wave_potential, masses=masses, positions=positions,
        box_length=4.0,
    )
    assert grid_interaction == pytest.approx(
        particle_interaction, rel=2e-13, abs=2e-13
    )
    patches, starts = periodic_tsc_patches(
        potential=wave_potential, positions=positions,
        box_length=4.0,
    )
    for body in range(2):
        acceleration = _tsc_patched_wave_acceleration(
            positions[body], patches[body], starts[body], 4.0, 16
        )
        np.testing.assert_allclose(
            acceleration, force[body] / masses[body], rtol=2e-13, atol=2e-13
        )
    epsilon = 1e-5
    for body in range(2):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[body, axis] += epsilon
            minus[body, axis] -= epsilon
            energy_plus, _ = tsc_interaction_and_force(
                wave_potential=wave_potential, masses=masses,
                positions=plus, box_length=4.0,
            )
            energy_minus, _ = tsc_interaction_and_force(
                wave_potential=wave_potential, masses=masses,
                positions=minus, box_length=4.0,
            )
            assert force[body, axis] == pytest.approx(
                -(energy_plus - energy_minus) / (2.0 * epsilon),
                rel=2e-8, abs=2e-9,
            )
    periodic_energy, periodic_force = tsc_interaction_and_force(
        wave_potential=wave_potential, masses=masses,
        positions=positions + np.array([4.0, 0.0, -4.0]),
        box_length=4.0,
    )
    assert periodic_energy == pytest.approx(particle_interaction, rel=1e-13)
    np.testing.assert_allclose(periodic_force, force, rtol=1e-12, atol=1e-12)


def test_tsc_rk4_conserves_fixed_wave_interaction_with_refinement() -> None:
    grid = spectral_grid(
        resolution=16, box_length=4.0, time_step=1e-3,
        device=torch.device("cpu"),
    )
    rng = np.random.default_rng(9903)
    density = torch.as_tensor(rng.uniform(0.2, 1.5, size=(16, 16, 16)))
    wave_potential = periodic_poisson_torch(
        density, grid.poisson_inverse_wavenumber_squared
    )
    masses = np.array([0.7, 1.3])
    initial = np.array([
        [1.93, -1.89, 0.23, 0.12, 0.08, -0.04],
        [-0.42, 0.55, -0.82, -0.06, 0.07, 0.03],
    ])
    patches, starts = periodic_tsc_patches(
        potential=wave_potential, positions=initial[:, :3], box_length=4.0
    )

    def hamiltonian(state: np.ndarray) -> float:
        kinetic = 0.5 * np.sum(masses[:, None] * state[:, 3:] ** 2)
        distance = state[1, :3] - state[0, :3]
        mutual = -np.prod(masses) / np.sqrt(distance @ distance + 0.1**2)
        interaction, _ = tsc_interaction_and_force(
            wave_potential=wave_potential, masses=masses,
            positions=state[:, :3], box_length=4.0,
        )
        return float(kinetic + mutual + interaction)

    initial_h = hamiltonian(initial)
    errors = []
    for substeps in (1, 4):
        final = advance_binary_rk4_tsc(
            state=initial, masses=masses, patches=patches,
            patch_starts=starts, box_length=4.0, resolution=16,
            plummer_radius=0.1, time_step=0.04, substeps=substeps,
        ).reshape(2, 6)
        errors.append(abs(hamiltonian(final) - initial_h))
    assert errors[1] < errors[0] / 50.0


def test_tsc_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="finite positions"):
        tsc_stencil(
            positions=np.array([[np.nan, 0.0, 0.0]]),
            resolution=16, box_length=4.0,
        )
    with pytest.raises(ValueError, match="positive finite mass"):
        tsc_source_density(
            masses=np.array([0.0]), positions=np.zeros((1, 3)),
            resolution=16, box_length=4.0, device=torch.device("cpu"),
        )


def test_tsc_force_is_continuous_at_cell_edge_and_patch_exit_fails() -> None:
    potential = torch.as_tensor(
        np.random.default_rng(9904).normal(size=(16, 16, 16)),
        dtype=torch.float64,
    )
    edge = np.array([[0.125, 0.0, 0.0]])
    left = edge.copy()
    right = edge.copy()
    left[0, 0] -= 1e-9
    right[0, 0] += 1e-9
    _, left_force = tsc_interaction_and_force(
        wave_potential=potential, masses=np.array([1.0]),
        positions=left, box_length=4.0,
    )
    _, right_force = tsc_interaction_and_force(
        wave_potential=potential, masses=np.array([1.0]),
        positions=right, box_length=4.0,
    )
    np.testing.assert_allclose(left_force, right_force, rtol=1e-7, atol=1e-7)
    patches, starts = periodic_tsc_patches(
        potential=potential, positions=edge, box_length=4.0,
    )
    outside = edge[0] + np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="left its periodic TSC force patch"):
        _tsc_patched_wave_acceleration(outside, patches[0], starts[0], 4.0, 16)


def test_experimental_tsc_runner_restarts_without_changing_coupling(
    tmp_path, monkeypatch,
) -> None:
    reference = tmp_path / "seed"
    (reference / "Outputs/3Wfn").mkdir(parents=True)
    (reference / "Outputs/NBody").mkdir()
    axis = np.arange(16, dtype=float) * 4.0 / 16.0 - 2.0
    density = 1.0 + 0.1 * np.cos(2.0 * np.pi * axis[:, None, None] / 4.0)
    wave = np.broadcast_to(np.sqrt(density), (16, 16, 16)).astype(np.complex128)
    state = np.array([
        [-0.5, 0.0, 0.0, 0.0, -0.08, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.08, 0.0],
    ])
    np.save(reference / "Outputs/3Wfn/P3D_#000.npy", wave)
    np.save(reference / "Outputs/NBody/NTM_#000.npy", state)
    energy_j = PYUL_SOLAR_MASS_KG * (PYUL_PARSEC_M / PYUL_MYR_S) ** 2
    (reference / "fdm_adapter_metadata.json").write_text(json.dumps({
        "adapter_revision": "test", "box_size_pc": 4.0, "resolution": 16,
        "case_id": "reciprocal_test", "pyul_length_unit_m": PYUL_PARSEC_M,
        "pyul_time_unit_s": PYUL_MYR_S,
        "pyul_mass_unit_kg": PYUL_SOLAR_MASS_KG,
        "pyul_energy_unit_j": energy_j,
    }))
    (reference / "config.uldm").write_text(json.dumps({
        "Matter Particles": {
            "Plummer Radius": 0.1,
            "Condition": [[0.7, [-0.5, 0, 0], [0, -0.08, 0]],
                          [1.3, [0.5, 0, 0], [0, 0.08, 0]]],
        },
        "Duration": {"Time Duration": 0.01},
        "Save Options": {"Number": 4},
    }))
    (reference / "reproducibility.uldm").write_text("test\n")

    def run(output, *extra):
        monkeypatch.setattr(sys, "argv", [
            "run_torch_wave_case.py", str(reference), "--output", str(output),
            "--duration-myr", "0.01", "--save-number", "4",
            "--save-3d-number", "0", "--movie-frame-number", "0",
            "--checkpoint-every-saves", "1", "--device", "cpu", *extra,
        ])
        return run_torch_wave_case.main()

    full = tmp_path / "full"
    split = tmp_path / "split"
    mode = ("--wave-smbh-coupling", "periodic_tsc_reciprocal")
    assert run(full, *mode) == 0
    assert run(split, *mode, "--diagnostic-stop-after-save", "2") == 0
    assert run(split, *mode, "--resume") == 0
    full_summary = json.loads((full / "torch_run_summary.json").read_text())
    split_summary = json.loads((split / "torch_run_summary.json").read_text())
    assert full_summary["status"] == split_summary["status"] == "diagnostic_complete"
    metadata = json.loads((full / "fdm_adapter_metadata.json").read_text())
    assert metadata["experimental_coupling_not_a_calibration_release"] is True
    np.testing.assert_allclose(
        np.load(full / "Outputs/egpcmlist.npy"),
        np.load(full / "Outputs/egpcmMlist.npy"), rtol=1e-11, atol=1e-9,
    )
    np.testing.assert_allclose(
        np.load(full / "Outputs/NBody/NTM_#004.npy"),
        np.load(split / "Outputs/NBody/NTM_#004.npy"), rtol=0, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.load(full / "Checkpoints/wave_000004.npy"),
        np.load(split / "Checkpoints/wave_000004.npy"), rtol=0, atol=1e-12,
    )
    wave_components = [
        np.load(full / "Outputs" / f"{name}.npy")
        for name in ("ekandqlist", "egpsilist", "egpcmlist")
    ]
    binary = []
    masses = np.array([0.7, 1.3])
    for index in range(5):
        saved = np.load(full / f"Outputs/NBody/NTM_#{index:03d}.npy").reshape(2, 6)
        displacement = saved[1, :3] - saved[0, :3]
        binary.append(
            0.5 * np.sum(masses[:, None] * saved[:, 3:] ** 2)
            - np.prod(masses) / np.sqrt(displacement @ displacement + 0.1**2)
        )
    components = wave_components + [np.asarray(binary)]
    hamiltonian = sum(components)
    transfer_scale = max(abs(series[-1] - series[0]) for series in components)
    assert transfer_scale > 0.0
    assert abs(hamiltonian[-1] - hamiltonian[0]) / transfer_scale < 0.01
    with pytest.raises(ValueError, match="wave_smbh_coupling"):
        run(split, "--resume")
