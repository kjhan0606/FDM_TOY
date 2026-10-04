"""Mass, periodicity, reciprocity, and force checks for TSC wave coupling."""

import numpy as np
import pytest
import json
import sys

torch = pytest.importorskip("torch")

from fdm_smbh_delay.periodic_mesh_coupling import (
    _tsc_patched_wave_acceleration,
    advance_binary_rk4_tsc,
    drift_binary_tsc,
    kick_binary_tsc,
    periodic_tsc_patches,
    tsc_interaction_and_force,
    tsc_source_density,
    tsc_stencil,
    validate_binary_tsc_timestep,
)
from fdm_smbh_delay.torch_wave import (
    apply_kinetic_phase_in_place,
    apply_potential_half_kick_in_place,
    periodic_poisson_torch,
    spectral_grid,
    wave_density,
)
from fdm_smbh_delay.pyul import (
    PYUL_MYR_S, PYUL_PARSEC_M, PYUL_SOLAR_MASS_KG,
)
from scripts import analyze_pyul_wave_run, run_torch_wave_case


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


def test_tsc_binary_kick_drift_is_reversible_and_second_order() -> None:
    potential = torch.as_tensor(
        np.random.default_rng(9905).normal(scale=0.01, size=(16, 16, 16)),
        dtype=torch.float64,
    )
    masses = np.array([0.7, 1.3])
    initial = np.array([
        [-0.45, 0.01, 0.02, 0.01, -0.08, 0.0],
        [0.45, -0.01, -0.02, -0.01, 0.06, 0.0],
    ])

    def step(state: np.ndarray, dt: float) -> np.ndarray:
        first = kick_binary_tsc(
            state=state, masses=masses, wave_potential=potential,
            box_length=4.0, plummer_radius=0.1, time_step=0.5 * dt,
        )
        middle = drift_binary_tsc(
            state=first, box_length=4.0, time_step=dt,
        )
        return kick_binary_tsc(
            state=middle, masses=masses, wave_potential=potential,
            box_length=4.0, plummer_radius=0.1, time_step=0.5 * dt,
        )

    forward = step(initial, 0.02)
    np.testing.assert_allclose(step(forward, -0.02), initial, rtol=0, atol=1e-15)
    reference = initial.copy()
    for _ in range(64):
        reference = step(reference, 0.00125)
    errors = []
    for count in (4, 8, 16):
        state = initial.copy()
        for _ in range(count):
            state = step(state, 0.08 / count)
        errors.append(np.max(np.abs(state - reference)))
    assert errors[0] / errors[1] > 3.7
    assert errors[1] / errors[2] > 3.5
    with pytest.raises(ValueError, match="direct-binary force domain"):
        drift_binary_tsc(
            state=np.array([[1.99, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]]),
            box_length=4.0, time_step=0.02,
        )
    with pytest.raises(ValueError, match="separation exceeds"):
        kick_binary_tsc(
            state=np.array([[-1.2, 0, 0, 0, 0, 0], [1.2, 0, 0, 0, 0, 0]]),
            masses=masses, wave_potential=potential, box_length=4.0,
            plummer_radius=0.1, time_step=0.01,
        )
    with pytest.raises(ValueError, match="orbit undersampled"):
        validate_binary_tsc_timestep(
            state=initial, masses=masses, box_length=4.0,
            resolution=16, plummer_radius=0.1, time_step=0.2,
        )


def test_full_wave_binary_strang_step_reverses() -> None:
    axis = np.arange(8, dtype=float) * 0.5 - 2.0
    density = 1.0 + 0.1 * np.cos(2.0 * np.pi * axis[:, None, None] / 4.0)
    initial_wave = torch.as_tensor(
        np.broadcast_to(np.sqrt(density), (8, 8, 8)).astype(np.complex128)
    )
    initial_body = np.array([
        [-0.45, 0, 0, 0, -0.08, 0],
        [0.45, 0, 0, 0, 0.06, 0],
    ], dtype=float)
    masses = np.array([0.7, 1.3])
    grid = spectral_grid(
        resolution=8, box_length=4.0, time_step=0.01,
        device=torch.device("cpu"),
    )

    def fields(wave, body):
        wave_potential = periodic_poisson_torch(
            wave_density(wave), grid.poisson_inverse_wavenumber_squared
        )
        compact_density = tsc_source_density(
            masses=masses, positions=body[:, :3], resolution=8,
            box_length=4.0, device=torch.device("cpu"),
        )
        compact_potential = periodic_poisson_torch(
            compact_density, grid.poisson_inverse_wavenumber_squared
        )
        return wave_potential, compact_potential

    def step(wave, body, dt):
        wave = wave.clone()
        wave_potential, compact_potential = fields(wave, body)
        apply_potential_half_kick_in_place(
            wave, wave_potential + compact_potential, dt
        )
        body = kick_binary_tsc(
            state=body, masses=masses, wave_potential=wave_potential,
            box_length=4.0, plummer_radius=0.1, time_step=0.5 * dt,
        )
        spectrum = torch.fft.fftn(wave)
        phase = grid.kinetic_axis_phase
        apply_kinetic_phase_in_place(
            spectrum, phase if dt > 0 else phase.conj()
        )
        wave = torch.fft.ifftn(spectrum)
        body = drift_binary_tsc(state=body, box_length=4.0, time_step=dt)
        wave_potential, compact_potential = fields(wave, body)
        apply_potential_half_kick_in_place(
            wave, wave_potential + compact_potential, dt
        )
        body = kick_binary_tsc(
            state=body, masses=masses, wave_potential=wave_potential,
            box_length=4.0, plummer_radius=0.1, time_step=0.5 * dt,
        )
        return wave, body

    forward_wave, forward_body = step(initial_wave, initial_body, 0.01)
    reversed_wave, reversed_body = step(forward_wave, forward_body, -0.01)
    torch.testing.assert_close(reversed_wave, initial_wave, rtol=0, atol=1e-13)
    np.testing.assert_allclose(reversed_body, initial_body, rtol=0, atol=1e-14)


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


def _small_reference(reference) -> None:
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


@pytest.mark.parametrize("coupling", ["periodic_tsc_reciprocal", "periodic_tsc_strang"])
def test_experimental_tsc_runner_restarts_without_changing_coupling(
    tmp_path, monkeypatch, coupling,
) -> None:
    reference = tmp_path / "seed"
    _small_reference(reference)

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
    mode = ("--wave-smbh-coupling", coupling)
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
    wave_mass = np.load(full / "Outputs/ULDMass.npy")
    np.testing.assert_allclose(wave_mass, wave_mass[0], rtol=1e-12, atol=1e-12)
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
    if coupling == "periodic_tsc_strang":
        phase_jump = json.loads(
            (full / "initial_compact_phase_jump.json").read_text()
        )["max_neighbour_phase_difference_over_pi"]
        assert np.isfinite(phase_jump) and phase_jump > 0.0
        np.testing.assert_allclose(
            np.load(full / "Outputs/binary_hamiltonian.npy"), binary,
            rtol=1e-13, atol=1e-13,
        )
        np.testing.assert_allclose(
            np.load(full / "Outputs/total_hamiltonian.npy"), hamiltonian,
            rtol=1e-13, atol=1e-13,
        )
        np.testing.assert_allclose(
            np.load(full / "Outputs/total_hamiltonian.npy"),
            np.load(split / "Outputs/total_hamiltonian.npy"),
            rtol=0, atol=1e-12,
        )
    transfer_scale = max(abs(series[-1] - series[0]) for series in components)
    assert transfer_scale > 0.0
    assert abs(hamiltonian[-1] - hamiltonian[0]) / transfer_scale < 0.01
    with pytest.raises(ValueError, match="wave_smbh_coupling"):
        run(split, "--resume")
    if coupling == "periodic_tsc_strang":
        config_path = reference / "config.uldm"
        original_config = config_path.read_text()
        config = json.loads(original_config)
        config["Matter Particles"]["Plummer Radius"] = 0.2
        config_path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="reference_config_sha256"):
            run(split, *mode, "--resume")
        config_path.write_text(original_config)
        metadata_path = split / "fdm_adapter_metadata.json"
        saved = json.loads(metadata_path.read_text())
        saved["solver_source_sha256"]["scripts/run_torch_wave_case.py"] = "changed"
        metadata_path.write_text(json.dumps(saved))
        with pytest.raises(ValueError, match="solver_source_sha256"):
            run(split, *mode, "--resume")


def test_strang_coupled_binary_separation_refines_quadratically(
    tmp_path, monkeypatch,
) -> None:
    reference = tmp_path / "seed"
    _small_reference(reference)
    final_separation = []
    steps = []
    for factor in (1.0, 0.5, 0.25, 0.125):
        output = tmp_path / f"factor_{factor}"
        monkeypatch.setattr(sys, "argv", [
            "run_torch_wave_case.py", str(reference), "--output", str(output),
            "--duration-myr", "0.075", "--save-number", "1",
            "--save-3d-number", "0", "--movie-frame-number", "0",
            "--checkpoint-every-saves", "0", "--device", "cpu",
            "--wave-smbh-coupling", "periodic_tsc_strang",
            "--time-step-factor", str(factor),
        ])
        assert run_torch_wave_case.main() == 0
        metadata = json.loads((output / "fdm_adapter_metadata.json").read_text())
        assert metadata["binary_integrator"] == "joint_kick_drift_kick_v1"
        steps.append(metadata["actual_wave_steps"])
        body = np.load(output / "Outputs/NBody/NTM_#001.npy").reshape(2, 6)
        final_separation.append(np.linalg.norm(body[1, :3] - body[0, :3]))
    assert steps == [4, 8, 16, 32]
    differences = np.abs(np.diff(final_separation))
    assert differences[0] / differences[1] > 3.5
    assert differences[1] / differences[2] > 3.5


def test_strang_timestep_failure_records_terminal_status(tmp_path, monkeypatch) -> None:
    reference = tmp_path / "seed"
    _small_reference(reference)
    initial_path = reference / "Outputs/NBody/NTM_#000.npy"
    body = np.load(initial_path)
    body[:, 3] = 100.0
    np.save(initial_path, body)
    output = tmp_path / "stalled"
    arguments = [
        "run_torch_wave_case.py", str(reference), "--output", str(output),
        "--duration-myr", "0.01", "--save-number", "4",
        "--save-3d-number", "0", "--movie-frame-number", "0",
        "--checkpoint-every-saves", "1", "--device", "cpu",
        "--wave-smbh-coupling", "periodic_tsc_strang",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(ValueError, match="cell crossing undersampled"):
        run_torch_wave_case.main()
    summary = json.loads((output / "torch_run_summary.json").read_text())
    assert summary["status"] == "stalled_timestep_resolution"
    assert summary["failed_wave_step"] == 1
    monkeypatch.setattr(sys, "argv", arguments + ["--resume"])
    with pytest.raises(ValueError, match="cannot resume terminal diagnostic"):
        run_torch_wave_case.main()
    body[:, 3] = 0.0
    body[0, 0] = -1.2
    body[1, 0] = 1.2
    np.save(initial_path, body)
    spatial_output = tmp_path / "spatial"
    spatial_arguments = arguments.copy()
    spatial_arguments[spatial_arguments.index("--output") + 1] = str(spatial_output)
    monkeypatch.setattr(sys, "argv", spatial_arguments)
    with pytest.raises(ValueError, match="separation exceeds"):
        run_torch_wave_case.main()
    spatial_summary = json.loads(
        (spatial_output / "torch_run_summary.json").read_text()
    )
    assert spatial_summary["status"] == "stalled_spatial_domain"


def test_strang_analyser_rejects_changed_hamiltonian_ledger(
    tmp_path, monkeypatch,
) -> None:
    reference = tmp_path / "seed"
    _small_reference(reference)
    output = tmp_path / "partial"
    monkeypatch.setattr(sys, "argv", [
        "run_torch_wave_case.py", str(reference), "--output", str(output),
        "--duration-myr", "0.01", "--save-number", "4",
        "--save-3d-number", "0", "--movie-frame-number", "0",
        "--checkpoint-every-saves", "1", "--device", "cpu",
        "--wave-smbh-coupling", "periodic_tsc_strang",
        "--diagnostic-stop-after-save", "2",
    ])
    assert run_torch_wave_case.main() == 0
    monkeypatch.setattr(sys, "argv", [
        "analyze_pyul_wave_run.py", str(output), "--diagnostic-partial",
    ])
    assert analyze_pyul_wave_run.main() == 0
    ledger_path = output / "Outputs/total_hamiltonian.npy"
    ledger = np.load(ledger_path)
    ledger[1] += 1e-3
    np.save(ledger_path, ledger)
    with pytest.raises(ValueError, match="Strang Hamiltonian ledger"):
        analyze_pyul_wave_run.main()


def test_strang_integer_cell_translation_and_fractional_offset(
    tmp_path, monkeypatch,
) -> None:
    box_length = 4.0
    resolution = 16
    cell = box_length / resolution

    def shifted_reference(label: str, shift: float):
        reference = tmp_path / label
        _small_reference(reference)
        axis = np.arange(resolution) * cell - box_length / 2.0
        density = 1.0 + 0.1 * np.cos(
            2.0 * np.pi * (axis - shift)[:, None, None] / box_length
        )
        wave = np.broadcast_to(
            np.sqrt(density), (resolution, resolution, resolution)
        ).astype(np.complex128)
        np.save(reference / "Outputs/3Wfn/P3D_#000.npy", wave)
        particle_path = reference / "Outputs/NBody/NTM_#000.npy"
        particle = np.load(particle_path)
        particle[:, 0] += shift
        np.save(particle_path, particle)
        config_path = reference / "config.uldm"
        config = json.loads(config_path.read_text())
        for condition in config["Matter Particles"]["Condition"]:
            condition[1][0] += shift
        config_path.write_text(json.dumps(config))
        return reference

    references = {
        label: shifted_reference(f"seed_{label}", shift)
        for label, shift in (
            ("base", 0.0), ("cell", cell), ("half", cell / 2.0)
        )
    }
    outputs = {}
    for factor, label in (
        (0.25, "base"), (0.25, "half"),
        (0.125, "base"), (0.125, "cell"), (0.125, "half"),
    ):
        reference = references[label]
        output = tmp_path / f"run_{label}_{factor}"
        monkeypatch.setattr(sys, "argv", [
            "run_torch_wave_case.py", str(reference), "--output", str(output),
            "--duration-myr", "0.075", "--save-number", "1",
            "--save-3d-number", "0", "--movie-frame-number", "0",
            "--checkpoint-every-saves", "1", "--device", "cpu",
            "--wave-smbh-coupling", "periodic_tsc_strang",
            "--time-step-factor", str(factor),
        ])
        assert run_torch_wave_case.main() == 0
        body = np.load(output / "Outputs/NBody/NTM_#001.npy").reshape(2, 6)
        wave = np.load(output / "Checkpoints/wave_000001.npy")
        outputs[factor, label] = (body, wave)

    base_body, base_wave = outputs[0.125, "base"]
    integer_body, integer_wave = outputs[0.125, "cell"]
    translated_body = integer_body.copy()
    translated_body[:, 0] -= cell
    np.testing.assert_allclose(translated_body, base_body, rtol=0, atol=2e-12)
    np.testing.assert_allclose(
        integer_wave, np.roll(base_wave, 1, axis=0), rtol=0, atol=2e-12
    )
    separations = {
        key: np.linalg.norm(body[1, :3] - body[0, :3])
        for key, (body, _wave) in outputs.items()
    }
    offset_sensitivity = {
        factor: separations[factor, "half"] - separations[factor, "base"]
        for factor in (0.25, 0.125)
    }
    base_signal = separations[0.125, "base"] - 1.0
    temporal_difference = (
        separations[0.25, "base"] - separations[0.125, "base"]
    )
    assert all(np.isfinite(value) for value in offset_sensitivity.values())
    assert np.isfinite(base_signal) and np.isfinite(temporal_difference)
    assert abs(offset_sensitivity[0.125] - offset_sensitivity[0.25]) < (
        0.1 * abs(offset_sensitivity[0.125])
    )
    assert abs(offset_sensitivity[0.125]) > 10.0 * abs(temporal_difference)
    print(json.dumps({
        "scope": "n16 toy only; not a calibration or systematic-error estimate",
        "half_cell_offset_sensitivity_pc": offset_sensitivity,
        "base_separation_change_pc": base_signal,
        "base_16_minus_32_step_difference_pc": temporal_difference,
    }, sort_keys=True))
