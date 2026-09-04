# CDM capture-to-coalescence interface

This is a CDM-only composition contract.  It is not a reimplementation of the
HR5 disappearance-history analysis, and it does not accept its fixed
population delay as an environmental hardening measurement.

The physical chain is split into three non-overlapping intervals:

```text
numerical capture -> hard binary -> GW-dominated binary -> coalescence
```

The first two intervals are separate `capture_to_hard_binary` and
`hard_binary_to_gw_regime` JSON records.  A completed record must name
`dark_matter_model: cdm`, give a decreasing separation interval, and reference
an immutable `accepted_model_specific_phase_ensemble` record for the same CDM
`physics_id`.  Its only accepted calibration method is
`resolved_cdm_rate_integration`.  The calibration also hashes the exact CDM
logarithmic-separation-rate track that was integrated.  This makes a
fixed-delay population model or an FDM/SIDM result ineligible by construction.

The accepted phase ensemble is schema version 2.  It records the common zoom
manifest and capture-event identity, plus the canonical zoom-specification
path/SHA and source/hash descriptors for both results in every fine/coarse
comparison.  Each resolved result is schema version 5 and carries a
hash-checked `runtime_identity` artifact.  A CDM stage reader
reopens those result files, physics-input records, normal-output inventories,
run sidecars, and diagnosed rate ledgers, then recomputes the resolution and
two-replica gates.  For the member that supplies the integrated rate, it also
reopens the strict CDM non-compacting runtime-identity record, checks its case,
manifest, and capture event against the result, and requires the diagnosed
rate ledger's normal-output provenance sidecar to be one of that identity's
attested outputs.  The integrated rate track's exact runtime-identity path/SHA
must be one of the accepted phase members.  A status-only or empty `members`
list, a case not present in the zoom specification, or a rate from an unlisted
runtime output set is therefore not a valid phase ensemble for a completed
delay.

For a completed stage, the schema is version 3.  In addition to the measured
delay it preserves the first-order rate uncertainty and the capture event
identity inherited from the raw orbit track:

```json
{
  "schema_version": 3,
  "status": "complete",
  "dark_matter_model": "cdm",
  "stage": "capture_to_hard_binary",
  "delay_myr": 20.0,
  "delay_uncertainty_myr": 0.3,
  "elapsed_lower_bound_myr": 0.0,
  "reason": "paired-resolution and phase-accepted rate integration",
  "interval_pc": {
    "start_separation_pc": 1000.0,
    "end_separation_pc": 1.0
  },
  "calibration": {
    "method": "resolved_cdm_rate_integration",
    "phase_ensemble_path": "accepted_cdm_phase_ensemble.json",
    "phase_ensemble_sha256": "<64 hexadecimal characters>",
    "physics_id": "<same CDM physics id>",
    "capture_event_uid": "<same capture event UID as the raw track>",
    "capture_event_sha256": "<64 hexadecimal characters>",
    "rate_track_path": "capture_to_hard_rates.json",
    "rate_track_sha256": "<64 hexadecimal characters>"
  }
}
```

The raw orbit-track input has exactly the same `physics_id` and carries the
capture event UID/SHA.  Every sample must include a positive
`orbital_period_myr`; the lagRamses extractor derives an instantaneous
two-body Kepler period estimate (a sampling-coverage proxy, not a measured
orbital period) from the RAMSES `unit_d` and `unit_l` fields (or a directly
supplied `unit_m`) when the info file provides them.  The sample records the
method as `instantaneous_two_body_kepler_estimate`.  This estimate does not
prove that the binary completed a physical orbit in a time-dependent host
potential.  A track without that physical period is retained as an
instantaneous diagnostic but is censored by the secular estimator.

The rate-track schema is version 3.  Its estimator records the raw-track
runtime-identity artifact in addition to the raw-track path/SHA.  It contains
at least three regression
points, each with a negative measured `dln_separation_dt_per_myr`, a
`slope_standard_error_per_myr`, an `r_squared`, and an `orbit_count`.  Each
regression block spans at least one Kepler-period sampling proxy by default
(the threshold is recorded as `minimum_orbits_per_block`); blocks with a
slope uncertainty at least 50% of the fitted decay rate or with `r_squared`
below 0.5 are censored.  Unused tail samples are recorded as
`discarded_sample_count`.  The integrated delay uncertainty is propagated
from the block slope errors.  The downstream reader recomputes all points,
coverage, and uncertainty from the raw track, so hand-edited rates are
rejected.  It is integrated only between the first and last measured
separations; no support is extrapolated.  Build a completed stage record
rather than entering its delay by hand:

```bash
python scripts/build_cdm_delay_stage.py \
  capture_to_hard_rates.json accepted_cdm_phase_ensemble.json \
  capture_to_hard.json
```

For a lagRamses CDM zoom, keep the selected SMBH pair from numerical
compaction with the native `&PHYSICS_PARAMS rmerge = 0`.  A later output must
record both `smbh_merge_radius_cells = 0` and
`smbh_compaction_mode = no_finite_radius_rmerge_zero` in its DM provenance sidecar;
the preflight refuses a compacting or ambiguous output:

```bash
python scripts/preflight_noncompacting_cdm_zoom.py \
  output_00042/dm_run_provenance_00042.txt cdm_noncompacting_preflight.json
```

This removes only finite-radius numerical compaction; exact coordinate
coincidence still requires explicit inspection.  It does not establish that the chosen
resolution, force sources, cadence, or hardening physics are adequate.

From a verified runtime-identity record containing at least three complete
non-compacting outputs, extract the raw relative orbit before attempting an
orbit average.  The extractor consumes precisely the identity-record output
set: it does not accept caller-substituted case IDs, coordinates, capture
bindings, or output directories.  The coordinate convention and proper-time
interpretation remain the contracted plan values and are never guessed from
the sink CSV.

```bash
python scripts/extract_lagramses_cdm_pair_orbit.py \
  --output cdm_raw_pair_orbit.json \
  cdm_zoom_runtime_identity.json
```

The runtime identity rechecks that the contract's capture binding identifies
the exact binary event in the original CDM capture ledger.  The resulting
`raw_relative_orbit_track` preserves that binding, the runtime-identity hash,
snapshot hashes, periodic minimum-image separations, and instantaneous
relative states.  It is not yet
a monotonic rate table: eccentric oscillations, stalled phases, and force
source attribution must be assessed before building a CDM delay stage.

The supplied `configs/cdm_noncompacting_zoom_grid.yaml` fixes the first
resolution-paired, two-phase CDM pilot: both meshes retain four cells at
1 pc, `rmerge=0`, output cadence resolves the target Kepler period by at
least 16 samples, and 15 complete outputs are required before a three-block
secular-rate candidate exists.  It is preflighted without submission:

```bash
python scripts/preflight_cdm_noncompacting_zoom_plan.py \
  configs/cdm_noncompacting_zoom_grid.yaml cdm_zoom_preflight.json
```

Before a manual Slurm submission, materialize one immutable input contract.
It records one selected resolution/phase case, verifies the original ledger
event and sink IDs again, checks the complete operator namelist for the
selected `levelmax`, `smbh=.true.`, `rmerge=0`, and an enabled, separately
named capture ledger.  It also requires hashes for the expected build,
compilation manifest, and the host/orbit, collisionless, and sink initial
condition inputs.  It then writes a new directory containing the audit record
and a reference-only `&PHYSICS_PARAMS` fragment.  It never edits the supplied
namelist, overwrites the original capture ledger, or submits a calculation.

```bash
python scripts/materialize_cdm_noncompacting_zoom_run.py \
  --case-id zoomphys-...-l21-r00 \
  --capture-binding capture_dm_run_binding.json \
  --capture-event-uid capture-11799-11801 \
  --primary-sink-id 11799 --secondary-sink-id 11801 \
  --run-namelist cdm_zoom.nml \
  --capture-ledger-file zoom_capture_11799_11801.jsonl \
  --expected-build-git-hash <40-character-lagRamses-revision> \
  --expected-compilation compilation.txt \
  --input-artifact host_orbit_initial_conditions=host_orbit_ic.dat \
  --input-artifact initial_conditions=collisionless_ic.dat \
  --input-artifact sink_initial_conditions=ic_sink \
  configs/cdm_noncompacting_zoom_grid.yaml cdm_zoom_contract
```

Only `ready_for_operator_submission` authorizes the operator to submit this
byte-identified input package.  The operator must copy the five
`cdm_zoom_*_sha256` assignments from the generated
`required_smbh_controls.nml` into the one `&PHYSICS_PARAMS` group of the
complete namelist; they attest the plan, capture event, and all three IC
roles.  A lagRamses build containing this provenance extension records those
values in every CDM output sidecar, and the runtime validator rejects an
output without that attestation or with any different value.  The explicit
hashes do not interpret an opaque IC binary format; the contract is
configuration provenance, not an independent physical validation of the
host, softening, particle loading, phase, convergence, or delay calibration.

The CDM consumer also requires the sidecar token
`force_accounting = resolved_collisionless_only`.  Existing lagRamses output
sidecars that lack this token are intentionally rejected.  The current
checked-out lagRamses writer does not yet emit the CDM/SIDM token, so its
outputs are not production-eligible for this reader until a separately
authorized writer patch and rebuild provide the exact runtime-derived token.

Before a submission, the operator may run the non-submitting source audit:

```bash
python scripts/audit_lagramses_writer_force_accounting.py \
  /path/to/lagRamses/patch/lagRamses/output_amr.kjhan.f90 \
  results/lagramses_writer_force_accounting_audit.json
```

`source_token_missing` is an expected blocker for the current checkout.  Even
`source_tokens_present_requires_runtime_test` is only a prerequisite: it does
not attest a compiled executable or prove that the emitted values were
derived from active runtime controls.  The normal-output sidecar and its
consumer remain the authoritative production gate.

After completion, check every output before extracting a pair orbit.  This
requires the `COMPLETE` marker, native no-finite-radius provenance, the same
ledger setting, the five attested execution-input digests, the contracted
build revision, a matching `compilation.txt`, and a copied `namelist.txt`
whose hash is exactly the one in the ready contract.  It rejects mixed run
roots, build/compilation identities,
non-monotone output time/step sequences, and cadence above the plan limit.
The result reports only the listed output count relative to the plan; it does
not itself accept a secular rate.

```bash
python scripts/validate_cdm_noncompacting_zoom_runtime.py \
  --output cdm_zoom_runtime_identity.json \
  cdm_zoom_contract/cdm_noncompacting_zoom_run_contract.json \
  output_00042 output_00043 output_00044
```

For a strictly shrinking, sufficiently sampled candidate, form a fixed-block
secular table with a documented regression rather than manually entering a
rate.  The builder requires at least three complete blocks and five samples in
each block; every block must span at least one Kepler-period sampling proxy,
have a negative slope whose standard error is below half its magnitude, and
have `r_squared >= 0.5`.  A block with a non-negative slope or either quality
failure is censored.

```bash
python scripts/derive_cdm_secular_rate_track.py cdm_raw_pair_orbit.json \
  --stage capture_to_hard_binary --samples-per-block 8 \
  --minimum-orbits-per-block 1 \
  --output cdm_secular_rates.json
```

The resulting rate table hashes the exact raw orbit track and its estimator
settings.  If fewer than three complete orbit-covered blocks exist, or any
block fails the slope, uncertainty, or regression-quality gates, the command
writes an explicit `status: censored` record with a reason and no delay.  This establishes
reproducibility, not adequacy: the accepted CDM resolution/phase ensemble and
force/conservation evidence remain required before the stage-delay builder can
use a completed track.

An uncalibrated, unresolved, stalled, or over-cosmic-time interval instead has
a non-complete status, no delay or interval, no `calibration`, and an explicit
reason.  It remains censored in the composed result.

The final interval is calculated only from the explicitly supplied initial
Peters orbit.  It must start at exactly the terminal separation of the second
CDM stage.  The interface invalidates a discontinuous handoff instead of
adding incompatible delays.

```bash
cdm-smbh-compose \
  --z-sink 1.0 \
  --capture-to-hard-summary capture_to_hard.json \
  --hard-to-gw-summary hard_to_gw.json \
  --gw-mass1 "1e8 Msun" \
  --gw-mass2 "3e7 Msun" \
  --gw-start-separation "0.01 pc" \
  --gw-eccentricity 0.2 \
  --output cdm_coalescence.json
```

This command launches no simulation.  It is allowed to return `censored`,
`incomplete`, or `invalid`; only a completed result is a physical CDM
coalescence-time estimate.
