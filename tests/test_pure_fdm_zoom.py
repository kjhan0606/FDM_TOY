from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from fdm_smbh_delay.pure_fdm_zoom import (
    NestedZoomCheckpointContract,
    bind_nested_zoom_checkpoint,
    preflight_pure_fdm_outer_zoom,
)
from fdm_smbh_delay.zoom_calibration import build_zoom_grid, load_zoom_grid


def _specification() -> dict:
    return yaml.safe_load(
        Path("configs/pure_fdm_outer_zoom_grid.yaml").read_text(encoding="utf-8")
    )


def test_pure_fdm_outer_preflight_creates_checkpoint_bound_nested_requests() -> None:
    grid = load_zoom_grid("configs/pure_fdm_outer_zoom_grid.yaml")
    preflight = preflight_pure_fdm_outer_zoom(grid)
    assert preflight.outer_case_count == 20
    assert preflight.baryon_mode == "baryon_free"
    assert len(preflight.nested_requests) == 10
    assert all(
        request.required_maximum_cell_size_pc == pytest.approx(0.25)
        for request in preflight.nested_requests
    )
    assert preflight.as_dict()["execution"]["gpu"] == "Slurm submission only when required"


def test_pure_fdm_preflight_rejects_a_cdm_case() -> None:
    specification = _specification()
    variation = {
        "dark_matter_model": "cdm",
        "fdm_particle_mass_ev": None,
        "fdm_core_radius_pc": None,
        "fdm_soliton_mass_msun": None,
    }
    specification["variations"] = [variation]
    with pytest.raises(ValueError, match="contains a CDM case"):
        preflight_pure_fdm_outer_zoom(build_zoom_grid(specification))


def test_pure_fdm_preflight_requires_phase_replicates() -> None:
    specification = deepcopy(_specification())
    specification["replicates"] = 1
    with pytest.raises(ValueError, match="lacks phase replicates"):
        preflight_pure_fdm_outer_zoom(build_zoom_grid(specification))


def _nested_contract(case_id: str) -> NestedZoomCheckpointContract:
    return NestedZoomCheckpointContract(
        outer_manifest_sha256="a" * 64,
        outer_case_id=case_id,
        checkpoint_path="/gpfs/example/outer/checkpoint-001",
        checkpoint_sha256="b" * 64,
        capture_event_uid="capture-1",
        force_ledger_sha256="c" * 64,
        wave_ledger_sha256="d" * 64,
        fdm_particle_mass_ev=1.0e-21,
        soliton_mass_msun=1.0e10,
        core_radius_pc=200.0,
        eta_sp=0.001,
        checkpoint_separation_pc=20.0,
        target_separation_pc=1.0,
        finest_cell_size_pc=0.25,
        minimum_softening_pc=0.125,
        maximum_wake_extent_pc=2.0,
        de_broglie_wavelength_pc=1.0,
        hjm_wave_seam_clearance_pc=3.0,
        boundary_clearance_pc=3.0,
    )


def test_nested_contract_binds_only_to_its_exact_outer_checkpoint() -> None:
    preflight = preflight_pure_fdm_outer_zoom(
        load_zoom_grid("configs/pure_fdm_outer_zoom_grid.yaml")
    )
    request = preflight.nested_requests[0]
    contract = _nested_contract(request.outer_case_id)
    assert bind_nested_zoom_checkpoint(request, contract) is contract
    with pytest.raises(ValueError, match="different outer case"):
        bind_nested_zoom_checkpoint(request, _nested_contract("other-case"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("finest_cell_size_pc", 0.26, "<= 0.25"),
        ("target_separation_pc", 0.9, "four-cell"),
        ("hjm_wave_seam_clearance_pc", 2.0, "HJM/wave seam"),
        ("boundary_clearance_pc", 2.0, "boundary lies"),
    ],
)
def test_nested_contract_rejects_unresolved_or_contaminated_setup(
    field: str, value: float, message: str
) -> None:
    values = _nested_contract("outer-case").as_dict()
    values.pop("schema_version")
    values[field] = value
    with pytest.raises(ValueError, match=message):
        NestedZoomCheckpointContract(**values)
