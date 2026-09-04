"""Small real-file fixtures for provenance-bound CDM orbit tests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from astropy import units as u

from fdm_smbh_delay.capture_ledger import read_capture_ledger
from fdm_smbh_delay.cdm_zoom_materialization import (
    assess_cdm_noncompacting_zoom_run_inputs,
    materialize_cdm_noncompacting_zoom_run_contract,
)
from fdm_smbh_delay.cdm_zoom_plan import load_cdm_noncompacting_zoom_plan
from fdm_smbh_delay.cdm_zoom_runtime_identity import (
    assess_cdm_noncompacting_zoom_runtime_identity,
)
from fdm_smbh_delay.constants import G_INTERNAL
from fdm_smbh_delay.lagramses_cdm_orbit import extract_lagramses_cdm_pair_orbit_track


_PC_CGS = float((1.0 * u.pc).to_value(u.cm))
_MSUN_CGS = float((1.0 * u.Msun).to_value(u.g))
_RUN_COUNTER = 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_output(
    root: Path,
    number: int,
    *,
    time_code: float,
    separation_pc: float,
    namelist: str,
    include_unit_d: bool,
    density_msun_pc3: float,
    capture_ledger_file: str = "zoom_capture.jsonl",
    build_git_hash: str = "a" * 40,
) -> Path:
    # Keep the synthetic periodic box comfortably larger than the most
    # expanded point in the log-linear track.  Otherwise the extractor's
    # minimum-image convention would fold an early sample onto the opposite
    # side of the box and corrupt the intended geometric block means.
    boxlen = 100000.0
    label = f"{number:05d}"
    directory = root / f"output_{label}"
    directory.mkdir(parents=True)
    (directory / "COMPLETE").write_text(label + "\n", encoding="utf-8")
    (directory / f"dm_run_provenance_{label}.txt").write_text(
        "# dm_run_provenance_v1\n"
        "dark_matter_model = cdm\n"
        "pic_enabled = .true.\n"
        "sidm_enabled = .false.\n"
        "fdm_enabled = .false.\n"
        f"nstep_coarse = {number}\n"
        f"time_code = {time_code:.12f}d0\n"
        "aexp = 5.0d-1\n"
        f"build_git_hash = {build_git_hash}\n"
        "namelist_copy = namelist.txt\n"
        "compilation_copy = compilation.txt\n"
        "smbh_capture_ledger_enabled = .true.\n"
        f"smbh_capture_ledger_file = {capture_ledger_file}\n"
        "smbh_merge_radius_cells = 0.0d0\n"
        "smbh_compaction_mode = no_finite_radius_rmerge_zero\n"
        "dm_transport = collisionless_nbody\n"
        "force_accounting = resolved_collisionless_only\n",
        encoding="utf-8",
    )
    info = (
        f"time = {time_code:.12f}d0\n"
        "aexp = 5.0d-1\n"
        f"unit_l = {_PC_CGS:.16e}\n"
        "unit_t = 3.15576e13\n"
        f"boxlen = {boxlen:.1f}d0\n"
    )
    if include_unit_d:
        info += f"unit_d = {_MSUN_CGS * density_msun_pc3 / _PC_CGS**3:.16e}\n"
    (directory / f"info_{label}.txt").write_text(info, encoding="utf-8")
    # Keep both coordinates away from the periodic boundary.  This makes the
    # extracted minimum-image difference numerically stable even for the
    # final sub-parsec samples (where subtracting two nearly box-sized
    # coordinates would otherwise lose significant digits).
    primary_x = 1000.0
    secondary_x = primary_x + separation_pc
    (directory / f"sink_{label}.csv").write_text(
        f"7,1.0d8,{primary_x:.15f},0.0,0.0,0.0,0.0,0.0,0.0,0.0\n"
        f"9,5.0d7,{secondary_x:.15f},0.0,0.0,0.0,0.0,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    (directory / "namelist.txt").write_text(namelist, encoding="utf-8")
    (directory / "compilation.txt").write_text("build provenance\n", encoding="utf-8")
    return directory


def _write_capture_binding(
    root: Path,
    *,
    capture_event_uid: str = "capture-7-9",
) -> Path:
    ledger_path = root / "smbh_capture_ledger_v1.jsonl"
    unit_velocity = (1.0 * u.pc / u.Myr).to_value(u.cm / u.s)
    speed = math.sqrt(G_INTERNAL * 2.0e8)
    rows = [
        {
            "schema_version": 1,
            "record_type": "event_begin",
            "event_uid": capture_event_uid,
            "classification": "BINARY",
            "nstep_coarse": 10,
            "ilevel": 1,
            "nmember": 2,
            "expected_pairs": 1,
            "aexp": 0.5,
            "redshift": 1.0,
            "t_code": 0.2,
            "texp": 0.4,
            "merge_radius_code": 2.0,
            "unit_length_cgs": _PC_CGS,
            "unit_velocity_cgs": unit_velocity,
            "unit_mass_cgs": _MSUN_CGS,
            "boxlen": 100.0,
            "complete": False,
        },
        {
            "schema_version": 1,
            "record_type": "member",
            "event_uid": capture_event_uid,
            "member_index": 1,
            "sink_id": 7,
            "mass_code": 1.0e8,
            "position_code": [0.5, 0.0, 0.0],
            "velocity_code": [0.0, 0.5 * speed, 0.0],
            "formation_time_code": 0.0,
            "accreted_mass_code": 0.0,
            "spin_magnitude": 0.5,
            "spin_direction": [0.0, 0.0, 1.0],
            "gas_angular_momentum_code": [0.0, 0.0, 0.0],
        },
        {
            "schema_version": 1,
            "record_type": "member",
            "event_uid": capture_event_uid,
            "member_index": 2,
            "sink_id": 9,
            "mass_code": 1.0e8,
            "position_code": [-0.5, 0.0, 0.0],
            "velocity_code": [0.0, -0.5 * speed, 0.0],
            "formation_time_code": 0.0,
            "accreted_mass_code": 0.0,
            "spin_magnitude": 0.5,
            "spin_direction": [0.0, 0.0, 1.0],
            "gas_angular_momentum_code": [0.0, 0.0, 0.0],
        },
        {
            "schema_version": 1,
            "record_type": "pair",
            "event_uid": capture_event_uid,
            "pair_index": 1,
            "sink_id_1": 7,
            "sink_id_2": 9,
            "within_rmerge": True,
            "two_body_bound": True,
            "legacy_pair_bound": True,
        },
        {
            "schema_version": 1,
            "record_type": "event_end",
            "event_uid": capture_event_uid,
            "nmember": 2,
            "npair": 1,
            "complete": True,
        },
    ]
    ledger_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    event = read_capture_ledger(ledger_path).events[0]
    binding_path = root / "capture_dm_run_binding.json"
    binding_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "capture_dm_run_provenance_bound",
                "interpretation": "capture-to-run provenance only",
                "capture_event": {
                    "event_uid": event.event_uid,
                    "event_sha256": event.event_sha256,
                    "ledger_path": str(ledger_path),
                },
                "run_provenance": {"dark_matter_model": "cdm"},
                "reasons": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return binding_path


def _write_capture_binding_for_ledger(
    root: Path,
    ledger_path: Path,
    *,
    capture_event_uid: str,
) -> Path:
    """Create the same binding schema for an already materialized output ledger."""

    event = next(
        item for item in read_capture_ledger(ledger_path).events
        if item.event_uid == capture_event_uid
    )
    binding_path = root / "runtime_capture_binding.json"
    binding_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "capture_dm_run_provenance_bound",
                "interpretation": "capture-to-run provenance only",
                "capture_event": {
                    "event_uid": event.event_uid,
                    "event_sha256": event.event_sha256,
                    "ledger_path": str(ledger_path),
                },
                "run_provenance": {"dark_matter_model": "cdm"},
                "reasons": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return binding_path


def _runtime_identity(
    root: Path,
    binding: Path,
    outputs: list[Path],
    *,
    capture_event_uid: str,
    case_id: str | None = None,
    case_input_artifact_paths: dict[str, Path] | None = None,
    expected_build_git_hash: str = "a" * 40,
    capture_ledger_file: str = "zoom_capture.jsonl",
    compilation_path: Path | None = None,
) -> Path:
    plan = load_cdm_noncompacting_zoom_plan("configs/cdm_noncompacting_zoom_grid.yaml")
    if case_id is None:
        case_id = plan.grid.cases[0].case_id
    selected_cases = [case for case in plan.grid.cases if case.case_id == case_id]
    if len(selected_cases) != 1:
        raise ValueError(f"unknown CDM fixture case_id: {case_id}")
    selected_case = selected_cases[0]
    namelist = root / "zoom.nml"
    base_namelist = (
        "&PHYSICS_PARAMS\n"
        f"levelmax={selected_case.numerics.levelmax}\n"
        "smbh=.true.\n"
        "rmerge=0.0d0\n"
        "smbh_capture_ledger=.true.\n"
        f"smbh_capture_ledger_file='{capture_ledger_file}'\n"
        "/\n"
    )
    namelist.write_text(base_namelist, encoding="utf-8")
    compilation = compilation_path or (root / "compilation_reference.txt")
    if compilation_path is None:
        compilation.write_text("build provenance\n", encoding="utf-8")
    artifact_names = (
        "host_orbit_initial_conditions",
        "initial_conditions",
        "baryon_configuration",
        "sink_initial_conditions",
    )
    artifact_paths: dict[str, Path] = {}
    for name in artifact_names:
        if case_input_artifact_paths is not None and name in case_input_artifact_paths:
            artifact_paths[name] = Path(case_input_artifact_paths[name]).expanduser().resolve()
        else:
            artifact_paths[name] = root / f"{name}.dat"
            artifact_paths[name].write_text(name + "\n", encoding="utf-8")
    arguments = dict(
        specification_path="configs/cdm_noncompacting_zoom_grid.yaml",
        case_id=selected_case.case_id,
        capture_binding_path=binding,
        capture_event_uid=capture_event_uid,
        primary_sink_id=7,
        secondary_sink_id=9,
        run_namelist_path=namelist,
        capture_ledger_file=capture_ledger_file,
        expected_build_git_hash=expected_build_git_hash,
        expected_compilation_path=compilation,
        case_input_artifact_paths=artifact_paths,
    )
    provisional, _ = assess_cdm_noncompacting_zoom_run_inputs(**arguments)
    assignments = {**provisional.execution_identity, **provisional.model_execution_identity}
    finalized_namelist = base_namelist.replace(
        "/\n",
        "".join(f"{name}='{value}'\n" for name, value in sorted(assignments.items()))
        + "/\n",
    )
    namelist.write_text(finalized_namelist, encoding="utf-8")
    for output in outputs:
        (output / "namelist.txt").write_text(finalized_namelist, encoding="utf-8")
        (output / "compilation.txt").write_text(
            compilation.read_text(encoding="utf-8"), encoding="utf-8"
        )
        label = output.name.removeprefix("output_")
        provenance = output / f"dm_run_provenance_{label}.txt"
        existing = provenance.read_text(encoding="utf-8") if provenance.exists() else ""
        additions = []
        if "cdm_zoom_execution_identity_status" not in existing:
            additions.append("cdm_zoom_execution_identity_status = available")
        if "model_zoom_execution_identity_status" not in existing:
            additions.append("model_zoom_execution_identity_status = available")
        if "model_zoom_levelmax" not in existing:
            additions.append(f"model_zoom_levelmax = {selected_case.numerics.levelmax}")
        for name, value in sorted(assignments.items()):
            if f"{name} =" not in existing:
                additions.append(f"{name} = {value}")
        if additions:
            existing = existing.rstrip() + "\n" + "\n".join(additions) + "\n"
        provenance.write_text(existing, encoding="utf-8")
    contract_directory = root / "contract"
    materialize_cdm_noncompacting_zoom_run_contract(
        **arguments,
        output_directory=contract_directory,
    )
    decision = assess_cdm_noncompacting_zoom_runtime_identity(
        contract_directory / "cdm_noncompacting_zoom_run_contract.json", outputs
    )
    if not decision.verified:
        raise AssertionError(f"fixture runtime identity is not verified: {decision.reasons}")
    identity = root / "runtime_identity.json"
    identity.write_text(json.dumps(decision.as_dict(), indent=2, sort_keys=True) + "\n")
    return identity


def make_attested_raw_track(
    tmp_path: Path,
    *,
    start_pc: float = 100.0,
    end_pc: float = 1.0,
    sample_count: int = 15,
    stalled: bool = False,
    include_unit_d: bool = True,
    density_msun_pc3: float = 1.0e12,
    capture_event_uid: str = "capture-7-9",
    case_id: str | None = None,
    model_physics_input: Path | None = None,
) -> Path:
    """Create a raw track by running the real metadata-only extractor."""

    global _RUN_COUNTER
    _RUN_COUNTER += 1
    specialized = model_physics_input is not None
    if specialized:
        physics_input = Path(model_physics_input).expanduser().resolve()
        physics_record = json.loads(physics_input.read_text(encoding="utf-8"))
        ensemble_path = Path(physics_record["capture_ensemble_path"])
        if not ensemble_path.is_absolute():
            ensemble_path = physics_input.parent / ensemble_path
        ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
        cdm_binding = ensemble["capture_bindings"]["cdm"]
        run_reference = cdm_binding["run_provenance"]["source"]["path"]
        run_path = Path(run_reference)
        if not run_path.is_absolute():
            run_path = ensemble_path.parent / run_path
        anchor = run_path.resolve().parent
        root = anchor.parent
        ledger_reference = cdm_binding["capture_event"]["ledger_path"]
        ledger_path = Path(ledger_reference)
        if not ledger_path.is_absolute():
            ledger_path = ensemble_path.parent / ledger_path
        binding = _write_capture_binding_for_ledger(
            root,
            ledger_path.resolve(),
            capture_event_uid=capture_event_uid,
        )
        family = ensemble["smoke"]["preflight"]["family"]
        family_path = Path(family["manifest_path"])
        if not family_path.is_absolute():
            family_path = ensemble_path.parent / family_path
        family_record = json.loads(family_path.read_text(encoding="utf-8"))
        shared_inputs = family_record["shared_inputs"]
        case_input_artifact_paths = {
            name: (
                Path(item["path"])
                if Path(item["path"]).is_absolute()
                else family_path.parent / item["path"]
            ).resolve()
            for name, item in shared_inputs.items()
        }
        case_input_artifact_paths["sink_initial_conditions"] = case_input_artifact_paths.pop(
            "smbh_seed_catalog"
        )
        host_orbit_path = physics_input.parent / "shared" / "host_orbit_initial_conditions.dat"
        if not host_orbit_path.exists():
            raise AssertionError("CDM fixture physics input lacks host-orbit identity artifact")
        case_input_artifact_paths["host_orbit_initial_conditions"] = host_orbit_path
        compilation_path = root / "compilation_reference.txt"
        compilation_path.write_text("model-specific fixture compilation\n", encoding="utf-8")
        expected_build_git_hash = cdm_binding["run_provenance"]["build_git_hash"]
        capture_ledger_file = cdm_binding["run_provenance"].get(
            "smbh_capture_ledger_file", "smbh_capture_ledger_v1.jsonl"
        )
        # The generic inventory fixture has one normal output at 00042.  Keep
        # it as the rate-ledger anchor and prepend synthetic complete outputs
        # so the same output set also provides an orbit track.
        if anchor.name != "output_00042":
            raise AssertionError("CDM fixture normal-output anchor must be output_00042")
        (anchor / "namelist.txt").unlink(missing_ok=True)
        (anchor / "compilation.txt").unlink(missing_ok=True)
        (anchor / "sink_00042.csv").unlink(missing_ok=True)
    else:
        root = tmp_path / f"attested-cdm-run-{_RUN_COUNTER}"
        root.mkdir(parents=True)
        binding = _write_capture_binding(root, capture_event_uid=capture_event_uid)
        case_input_artifact_paths = None
        expected_build_git_hash = "a" * 40
        capture_ledger_file = "zoom_capture.jsonl"
        compilation_path = None
    selected_levelmax = 21
    if case_id is not None:
        plan = load_cdm_noncompacting_zoom_plan("configs/cdm_noncompacting_zoom_grid.yaml")
        selected = [case for case in plan.grid.cases if case.case_id == case_id]
        if len(selected) != 1:
            raise ValueError(f"unknown CDM fixture case_id: {case_id}")
        selected_levelmax = selected[0].numerics.levelmax
    namelist = (
        "&PHYSICS_PARAMS\n"
        f"levelmax={selected_levelmax}\n"
        "smbh=.true.\n"
        "rmerge=0.0d0\n"
        "smbh_capture_ledger=.true.\n"
        f"smbh_capture_ledger_file='{capture_ledger_file}'\n"
        "/\n"
    )
    outputs: list[Path] = []
    time_step = 1.0e-4
    center_index = 2.0
    last_center_index = float(sample_count - 3)
    if specialized:
        output_numbers = list(range(1, sample_count)) + [42]
        output_times = [
            1.25 - time_step * (sample_count - 1 - index)
            for index in range(sample_count - 1)
        ] + [1.25]
        center_time = output_times[int(center_index)]
        # Match the original fixture convention: the requested end point is
        # the centre of the final five-sample regression block.  The final
        # two samples therefore provide a short post-end tail while output
        # 00042 remains the normal-output/rate-ledger anchor.
        last_time = output_times[-3]
        log_slope = (
            0.0
            if stalled or start_pc == end_pc or last_time <= center_time
            else math.log(end_pc / start_pc) / (last_time - center_time)
        )
    else:
        output_numbers = list(range(1, sample_count + 1))
        output_times = [1.0 + time_step * index for index in range(sample_count)]
        log_slope = (
            0.0
            if stalled or start_pc == end_pc
            else (
                0.0
                if last_center_index <= center_index
                else math.log(end_pc / start_pc)
                / ((last_center_index - center_index) * time_step)
            )
        )
    for index, number in enumerate(output_numbers):
        time_code = output_times[index]
        separation = start_pc * math.exp(
            log_slope * (time_code - output_times[int(center_index)])
        )
        if specialized and number == 42:
            anchor = root / "output_00042"
            info = (
                f"time = {time_code:.12f}d0\n"
                "aexp = 5.0d-1\n"
                f"unit_l = {_PC_CGS:.16e}\n"
                "unit_t = 3.15576e13\n"
                "boxlen = 100000.0d0\n"
            )
            if include_unit_d:
                info += f"unit_d = {_MSUN_CGS * density_msun_pc3 / _PC_CGS**3:.16e}\n"
            (anchor / "COMPLETE").write_text("00042\n", encoding="utf-8")
            (anchor / "info_00042.txt").write_text(info, encoding="utf-8")
            (anchor / "sink_00042.csv").write_text(
                f"7,1.0d8,1000.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n"
                f"9,5.0d7,{1000.0 + 2.0 * separation:.15f},0.0,0.0,0.0,0.0,0.0,0.0,0.0\n",
                encoding="utf-8",
            )
            outputs.append(anchor)
        else:
            outputs.append(
                _write_output(
                    root,
                    number,
                    time_code=time_code,
                    separation_pc=2.0 * separation,
                    namelist=namelist,
                    include_unit_d=include_unit_d,
                    density_msun_pc3=density_msun_pc3,
                    capture_ledger_file=capture_ledger_file,
                    build_git_hash=expected_build_git_hash,
                )
            )
    if specialized:
        ledger_source = ledger_path if ledger_path.exists() else root / capture_ledger_file
        ledger_bytes = ledger_source.read_bytes()
        for output in outputs:
            ledger_copy = output / capture_ledger_file
            if not ledger_copy.exists():
                ledger_copy.write_bytes(ledger_bytes)
    identity = _runtime_identity(
        root,
        binding,
        outputs,
        capture_event_uid=capture_event_uid,
        case_id=case_id,
        case_input_artifact_paths=case_input_artifact_paths,
        expected_build_git_hash=expected_build_git_hash,
        capture_ledger_file=capture_ledger_file,
        compilation_path=compilation_path,
    )
    track = extract_lagramses_cdm_pair_orbit_track(identity)
    raw_path = root / "raw_relative_orbit_track.json"
    raw_path.write_text(json.dumps(track.as_dict(), indent=2, sort_keys=True) + "\n")
    return raw_path


def unique_rate_path(tmp_path: Path, stage: str) -> Path:
    global _RUN_COUNTER
    _RUN_COUNTER += 1
    return tmp_path / f"{stage}_rate_track_{_RUN_COUNTER}.json"
