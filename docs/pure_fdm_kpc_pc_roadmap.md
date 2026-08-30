# Pure-FDM kpc-to-pc roadmap

This roadmap defines the path from a pure-FDM galaxy-merger capture to the
parsec-scale live-wave calculation.  It is deliberately separate from the
PTA population synthesis: a physical delay enters the population only after
every interval below has passed its validation gates.

## Scientific scope

`pure-FDM` means that the CDM density and force are identically zero.  Stars
and gas are optional baryonic channels and must be marked `available` or
`absent`; they are never silently replaced by CDM.  A completely baryon-free
experiment marks both channels `absent`.

The few-kpc interval is not a one-body decay in a static spherical core.  It
contains two moving nuclei (and, in general, two evolving FDM cores) before a
common core forms.  The existing fixed-primary spherical solver is retained
only as an analytic baseline.  A physical value requires a pure-FDM merger
zoom or a closure calibrated against one.

The common numerical-capture inventory spans separate CDM, SIDM, and FDM run
families with matched macroscopic initial conditions; they are not mixed in a
single realization.  Each normal lagRamses output carries a model-specific
`dm_run_provenance_<output>.txt` sidecar.  Bind a complete capture event to a
later sidecar before placing it in any comparative ensemble:

```bash
python scripts/bind_capture_dm_run.py \
  smbh_capture_ledger_v1.jsonl capture-event-uid \
  output_00042/dm_run_provenance_00042.txt \
  results/capture_dm_run_binding.json
```

This check requires the sidecar ledger path to resolve to the exact event
ledger and a non-earlier output step.
It preserves the active model and SIDM/FDM controls, but does not assert that
different model runs reach capture simultaneously or that any capture is a
physical coalescence.  Only the FDM branch proceeds to the all-wave seed and
outer-wave workflow below; CDM and SIDM remain comparison baselines with their
own resolved-environment analyses.

### Comparison-family acceptance before model-specific analysis

The CDM, SIDM, and FDM branches use one explicitly matched comparison family;
they do not share particles, forces, or a mixed dark-matter realization.  The
family manifest pins SHA-256 values for the common initial conditions, baryon
configuration, and SMBH seed catalogue, plus one normal-output DM sidecar for
each model.  It must not be used to substitute a CDM density for an FDM
calculation.

The first four comparison records are deliberately non-submitting and have a
strict order:

1. `preflight_dm_comparison_family.py` verifies the three common files, the
   expected `cdm`/`sidm`/`fdm` labels, one concrete common build revision, and
   FDM resolved-wave-only force accounting.
2. `assess_dm_comparison_smoke.py` verifies that each selected normal output
   has a matching `COMPLETE` marker.  The marker is an operator-written
   completion record containing the five-digit output number, created only
   after the normal output is durable.
3. `register_dm_comparison_capture_ensemble.py` binds one complete numerical
   capture event from each model to that model's later sidecar.  It records
   provenance only; different event times, sink identifiers, and capture
   classifications are not forced to match, and no event is labelled a
   physical coalescence.
4. `assess_dm_comparison_physics_inputs.py` verifies the hashes of the
   model-specific resolved-environment evidence: profile, force, and
   conservation ledgers for all models; SIDM scattering ledger; FDM wave
   ledger and field-snapshot index.  A verified record means only that the
   inputs are ready for their separate analyses.

For example, with operator-prepared records:

```bash
python scripts/preflight_dm_comparison_family.py family.json results/family_preflight.json
python scripts/assess_dm_comparison_smoke.py family.json results/family_smoke.json
python scripts/register_dm_comparison_capture_ensemble.py \
  capture_registration.json results/capture_ensemble.json
python scripts/assess_dm_comparison_physics_inputs.py \
  physics_inputs.json results/physics_input_assessment.json
```

Any missing file, hash mismatch, incomplete output, unsupported sidecar, or
unbound capture yields a non-ready record.  These checks launch no simulation,
do not estimate a delay, and do not alter the FDM-only outer-to-inner gates.

### Resolved model-specific analysis contract

After the four comparison records pass, each completed run may be registered
as a resolved model-physics result.  Its zoom point now supports exactly one
of `cdm`, `sidm`, or `fdm`: SIDM points require cross section, velocity scale,
power law, interaction type/angular law, and inelastic flag; FDM points require
the particle mass, core radius, and soliton mass.  A CDM point cannot silently
carry either SIDM or FDM controls.

Every result preserves three environment channels: `stars`, `gas`, and
`dark_matter`.  A channel is explicitly `available`, with profile and force
ledger hashes, or `absent`, with no invented zero-valued ledger.  Positive
stellar mass or gas fraction requires the respective channel to be available.
The model evidence is then distinct:

- CDM uses only the accepted profile/force/conservation evidence;
- SIDM additionally preserves the accepted scattering ledger and measured
  maximum scatter probability;
- FDM additionally preserves a full, time-resolved wave ledger, its separate
  one-output raw wave-provenance record, and the field-snapshot index.  The
  full ledger is cross-checked against the registered force, profile, and
  field-index hashes; the compact raw provenance is not promoted into the
  full ledger.  Until a separately hash-bound residual-force decomposition is
  implemented, only `live_wave_only` force accounting is accepted.  An
  analytic FDM drag label is rejected.

The result supplies at least three measured, separation-ordered orbital-power,
torque, and eccentricity points.  A finer/coarser pair is accepted only if it
has common support over a factor of two in separation, three matched points,
conservation error below `1e-3`, four cells per orbital scale (and, for FDM,
per de Broglie and wake scale), no rate sign reversal, rate disagreement at
most 20 percent, and eccentricity disagreement at most 0.02.  A zero measured
power or torque is unresolved rather than treated as agreement.  Two or more
independent phase replicas are required before a model is ready for its own
physical interpretation.

These commands only inspect registered JSON evidence and never submit work:

```bash
python scripts/validate_resolved_model_physics_run.py \
  comparison_zoom_grid.yaml fdm_fine_result.json results/fdm_fine_checked.json
python scripts/compare_model_specific_resolution.py \
  comparison_zoom_grid.yaml fdm_fine_result.json fdm_coarse_result.json \
  results/fdm_resolution.json
python scripts/assess_model_specific_phase_ensemble.py comparison_zoom_grid.yaml \
  --pair fdm_fine_phase0.json fdm_coarse_phase0.json \
  --pair fdm_fine_phase1.json fdm_coarse_phase1.json \
  results/fdm_phase_ensemble.json
```

The CDM/SIDM counterparts use the same command sequence but remain separate
model analyses.  None of these records substitute a CDM/SIDM force for FDM,
combine overlapping intervals, or create a coalescence time.

## Staged work

### 0. Contract and scope freeze — complete

- CDM is excluded from the pure-FDM path.
- Missing data are not interpreted as zero force or zero delay.
- Inner live-wave results that lack accepted support remain `uncalibrated` or
  `censored`.

### 1. Outer-halo data contract — complete in the toy repository

`FDMOuterHaloClosure` stores no-extrapolation radial diagnostics:

- FDM mass current;
- coherence time and de Broglie wavelength;
- velocity-space diffusion coefficient;
- density-gradient scale;
- calibration status.

`EnvironmentProfileBundle` can preserve this closure with the source case,
SHA-256, and path.  These fields are diagnostics only; they do not define a
drag law.

### 2. Outer response and handoff contract — complete in the toy repository

`FDMOuterResponseTable` accepts a measured vector drift and a symmetric
positive-semidefinite velocity-diffusion tensor.  It returns `censored` when
the table is uncalibrated or a radius lies outside support.

`validate_outer_inner_handoff` requires:

- a positive-width overlap, at least a factor of two in separation;
- at least three matched rate points;
- orbital-power and torque disagreement no larger than 20 percent;
- eccentricity mismatch no larger than 0.02;
- dimensionless similarity mismatch no larger than 2 percent;
- no sign change in the measured power or torque.

The validator never adds overlapping delays.  It chooses a single handoff
surface only after both calculations pass.

### 3. Pure-FDM outer merger zoom — operator execution stage

The first explicit outer-grid specification is
`configs/pure_fdm_outer_zoom_grid.yaml`: four physical points, two phase
replicates, and two outer-stage resolutions (20 manifest cases).  It can be
materialized without submitting jobs with:

```bash
python scripts/generate_galaxy_zoom_grid.py \
  configs/pure_fdm_outer_zoom_grid.yaml \
  results/pure_fdm_outer_zoom_manifest.json
```

This manifest deliberately stops at the outer/common-core stage.  It is not
evidence that a 1 pc transition is resolved.

Before registration, generate the non-submitting preflight record:

```bash
python scripts/preflight_pure_fdm_outer_zoom.py \
  configs/pure_fdm_outer_zoom_grid.yaml \
  results/pure_fdm_outer_zoom_preflight.json
```

The current specification produces 20 outer cases and 10 deferred nested
requests.  Each request is attached to the finest outer run for one physical
point and phase replicate, and requires `dx <= 0.25 pc` plus the exact outer
checkpoint, checkpoint SHA-256, wave-seam provenance, and force ledger before
it can become a real inner run.

`NestedZoomCheckpointContract` enforces the registration gate: the requested
outer case ID must match exactly, the 1 pc target must retain at least four
cells, softening cannot exceed the finest cell, and both the HJM/wave seam and
the numerical boundary must lie beyond the protected wake/coherence extent.
It also carries separate force and wave-ledger SHA-256 values.  A failure is a
censored setup, not a reason to enlarge the force or reuse a neighbouring
checkpoint.

The operator must prepare a pure-FDM run family, submitted through Slurm, with
the following provenance fixed in the manifest:

- exact code revision, namelist, units, `m_fdm`, and initial wave phase;
- box size, refinement mask, `dx`, `dt`, softening, and HJM/wave seam;
- two un-compacted SMBH states and the complete pre-compaction ledger;
- FDM-only density, potential, current, mass/energy/angular-momentum fluxes;
- core centres, core radii, dipole/quadrupole modes, and granule coherence;
- boundary flux and the time at which a wake reaches a numerical boundary;
- explicit star/gas channel status if baryons are present.

For lagRamses FDM runs, setting `fdm_outer_ledger=.true.` in `&fdm_params`
adds a compact `fdm_outer_wave_provenance_<output>.txt` record beside each
normal `fdm_<output>.out*` wave snapshot.  It preserves the output epoch,
code-unit leaf mass/current integral, HJM seam settings, current-stencil
coverage, and the explicit resolved-wave/no-analytic-drag force accounting.
The current V2 record also preserves the active dual-soliton switch and both
runtime component parameter vectors, so a controlled two-core output can be
bound back to its materialized seed.
It is raw provenance only: the full snapshots remain mandatory for local
profiles, core motion/modes, granules, force work, and a complete outer wave
ledger.  The lagRamses patch document is
`patch/lagRamses/FDM_OUTER_WAVE_PROVENANCE.md`.

The raw text record can be checked and converted to an atomic JSON diagnostic
without re-running a simulation:

```bash
python scripts/validate_lagramses_fdm_outer_wave_provenance.py \
  output_00042/fdm_outer_wave_provenance_00042.txt \
  results/fdm_outer_wave_provenance_00042.json
```

`available_raw_provenance` means only that the summary is self-consistent and
that it declares resolved-wave-only FDM force accounting.  It is deliberately
not a calibrated response, an accepted outer closure, or a delay estimate.

Each completed normal output also carries
`resolved_physics_inventory_<output>.txt`.  This companion record names the
explicit stars, gas, and active dark-matter channel states; the potential
checkpoint; the sink state file; and the force/conservation ledgers.  FDM
outputs additionally name the field snapshot and wave-provenance availability.
It is a fail-closed pre-registration diagnostic:

```bash
python scripts/assess_lagramses_resolved_physics_inventory.py \
  --stars-required --gas-required \
  output_00042/resolved_physics_inventory_00042.txt \
  results/resolved_physics_inventory_00042.json
```

The normal writer currently declares the force and conservation ledgers
`unavailable`; therefore this command correctly returns `censored`.  It does
not assign a zero force, zero torque, or zero delay.  An FDM record also
remains censored until its wave provenance is available.  A particle dump
whose stars have not been classified is censored even when a stellar channel
is not requested, because the baryonic composition has not been stated.
Subsequent force-work, conservation, mode, profile, and resolution records
must still be measured and registered separately.  A schema-v2
model-physics input now carries one hash-bound inventory-assessment record for
each of CDM, SIDM, and FDM.  The downstream reader re-parses each assessment
and its raw output before accepting a model-specific result; its force,
conservation, SIDM-scattering, and FDM-wave ledger digests must match the
corresponding registered artifacts.  Legacy schema-v1 physics-input records
are deliberately not ready for model-specific registration.

For a controlled two-core experiment, define a YAML seed with exactly two
FDM solitons and two SMBH sinks in code units, then materialize the matching
all-wave `&FDM_PARAMS` fragment and the two-row lagRamses `ic_sink` input:

```bash
python scripts/materialize_dual_soliton_ic.py \
  dual_soliton_seed.yaml \
  initial_conditions/pure_fdm_case
```

The seed requires `dark_matter_model: fdm`, explicit star/gas availability,
zero sink CDM fraction, and each SMBH within its assigned soliton core.  It
sets `fdm_use_hjm=.false.`, `fdm_dual_soliton_ic=.true.`, and
`fdm_outer_ledger=.true.`; its `box_length_code` must equal the lagRamses
`&AMR_PARAMS boxlen` used with the generated `ic_sink`.  A coherent two-soliton overlap is not a
single-stream HJM state, so an HJM seam is prohibited at initialization.  The
materialized files are initial data only; relaxation, paired resolution,
phase replicas, and the outer-wave evidence gates remain mandatory.

When a complete pre-compaction capture ledger exists, first derive the two
local sink rows with `export_capture_fdm_sink_pair.py`.  The frame
specification binds the exact capture-event UID, local code units, member
order, and angular-momentum vectors to an external `mass_projection` record:

```bash
python scripts/export_capture_fdm_sink_pair.py \
  smbh_capture_ledger_v1.jsonl \
  capture_fdm_seed_frame.json \
  results/capture_fdm_sink_pair.json
```

The ledger supplies the numerical sink masses and kinematics.  The projection
must separately supply two authoritative SMBH masses plus its source case ID,
SHA-256, and path; neither total sink mass nor an accreted-mass diagnostic is
silently reinterpreted as an SMBH mass.  The target zero CDM fraction is an
explicit definition of the new pure-FDM experiment, not a claim about the
source HR5 run.  The export verifies that the declared SHA-256 matches the
projection file (with relative paths resolved beside the frame specification).
It is a provenance-bound sink-pair input, not a soliton fit or an accepted
physical delay.

To avoid manually copying those rows into a seed, supply the two soliton
components independently (with no `sinks` field) and assemble both the
materializer-ready seed and the capture-pair evidence together:

```bash
python scripts/assemble_capture_dual_soliton_seed.py \
  smbh_capture_ledger_v1.jsonl \
  capture_fdm_seed_frame.json \
  explicit_soliton_configuration.yaml \
  capture_derived_seed.yaml \
  results/capture_fdm_sink_pair.json
```

This command does not fit a soliton, select a phase, or alter the capture
orbit.  It accepts only explicitly declared two-core FDM parameters and
injects the provenance-bound sink rows.  The resulting YAML is then passed to
`materialize_dual_soliton_ic.py`.

Once the soliton parameters are independently specified and the seed is
materialized, bind its two `ic_sink` rows back to that capture pair before run
preflight:

```bash
python scripts/verify_capture_dual_soliton_seed.py \
  results/capture_fdm_sink_pair.json \
  initial_conditions/pure_fdm_case/dual_soliton_seed_manifest.json \
  results/capture_seed_materialization_binding.json
```

Only `capture_seed_materialization_identity_verified` preserves the same
capture-derived numerical sink mass, projected SMBH mass, position, velocity,
angular momentum, and target zero CDM fraction.  It is an input-identity
check; it does not validate the soliton fit, wave relaxation, conservation,
or any physical delay.

Before the operator submits a completed run namelist, verify that its scalar
FDM/AMR switches, all two-soliton components, and the run-directory
`ic_sink` remain identical to the materialized seed:

```bash
python scripts/preflight_dual_soliton_run.py \
  initial_conditions/pure_fdm_case/dual_soliton_seed_manifest.json \
  run.nml \
  ic_sink \
  results/pure_fdm_dual_soliton_preflight.json
```

Only `ready_for_operator_submission` means that this configuration identity
check passed.  It neither submits a job nor certifies the relaxation,
conservation, paired-resolution, phase-replica, boundary, or calibration
gates.  A mismatch is `not_ready_for_operator_submission`, not a physical
delay or a scientific censoring result.

After the first normal FDM output, verify the runtime V2 provenance against
the same materialized seed before treating the output as a member of that
controlled case:

```bash
python scripts/validate_dual_soliton_runtime_identity.py \
  initial_conditions/pure_fdm_case/dual_soliton_seed_manifest.json \
  output_00042/fdm_outer_wave_provenance_00042.txt \
  results/pure_fdm_dual_soliton_runtime_identity.json
```

`runtime_seed_identity_verified` establishes only that the raw V2 output
declares the same all-wave two-core configuration and passes the raw
force/current coverage gate.  It does not relax, calibrate, or otherwise
accept the physical merger calculation.

The bounded initial relaxation window is assessed separately from the full
time-resolved snapshots.  Its evidence table must preserve both core masses,
radii, periodic centres, and the independently measured wave-mass,
Hamiltonian, and angular-momentum error series.  It is accepted only after a
verified runtime identity record exists:

```bash
python scripts/assess_dual_soliton_relaxation.py \
  results/pure_fdm_dual_soliton_relaxation_evidence.json \
  results/pure_fdm_dual_soliton_relaxation_assessment.json
```

The declared window thresholds are evaluated without re-reading or modifying
the wave fields.  A pass identifies a stable initial two-core window; it is
not an outer-merger result, and a failed window cannot be relabelled as zero
pairing delay.

No direct syn101 execution is permitted.  GPU execution is Slurm-only.  When
CPU preprocessing is needed, it is run manually on Lageunha with one process,
one numerical-library thread, a memory bound, and restartable outputs.  The
operator owns job submission, cancellation on collision, registration, and
commit/push; code and scientific acceptance remain separate responsibilities.

### 4. Nested inner zoom — operator execution stage

The existing 1--2 pc cell-size galaxy zoom cannot satisfy a four-cell gate at
1 pc.  A nested inner zoom must therefore use `dx <= 0.25 pc`, with the final
choice set by the measured pericentre, de Broglie wavelength, softening, and
wake extent.  The inner run must share an exact checkpoint with the outer run
and preserve the wave Hamiltonian ledger.

The current Koo/Boey live-wave anchors and a galaxy-scale outer halo are not
automatically in the same similarity class.  Matching requires, at minimum,
`m_fdm`, soliton mass and core radius, `eta_SP`, binary mass fraction,
`a/r_c`, eccentricity, local density/current, mode phase, and external tidal
field.

### 5. Calibration and acceptance — evaluation stage

The evaluator compares orbit-averaged `dE_orb/dt`, `dL_orb/dt`, eccentricity,
and their phase distributions on the positive-width overlap.  Paired
resolution and phase-replicate runs are mandatory.  A live wake supplies its
own work and torque; a subgrid response may receive only a measured residual.

The outer stochastic model is a calibrated drift-plus-diffusion consumer.  No
analytic response is generated when `eta_nl` is outside calibrated support,
the de Broglie scale is unresolved, a granule/wake reaches a seam or boundary,
or the response is dominated by a sign-changing torque or stochastic stall.

`PureFDMOuterRunResult` is the required outer-run result record.  It binds the
exact grid case and manifest SHA-256 to the outer closure, conservation
diagnostics, stages (`outer_start`, `common_core`, `nested_checkpoint`), and
the nested checkpoint contract.  Its paired-resolution evaluator requires
both the de Broglie scale and wake to have at least four cells and returns only
`accepted_outer_uncalibrated`: it permits inner registration but does not
publish a physical kpc-to-pc delay.

Every registered outer result also carries an `FDMOuterWaveLedger`: hashes of
the time-resolved field and profile snapshot indices, force-source ledger,
core centres, FDM current, coherence and de Broglie scales, granule statistic,
complex l=1/l=2 modes, wake extent, seam/boundary clearance, and wave
Hamiltonian/mass/angular-momentum diagnostics.  It rejects an analytic FDM
drag force and cannot accept a seam or boundary inside the protected wave
extent.  The nested checkpoint's force and wave-ledger hashes must match this
outer evidence exactly.

`assess_pure_fdm_outer_phase_ensemble` then retains the common-core timing
mean and sample standard deviation across independently converged phase
replicates.  This dispersion is a stochastic outer-halo observable, not a
resolution correction and not a coalescence-time prediction.

After manual Lageunha postprocessing has registered each available result, the
operator writes a small index with the exact manifest hash and case-ID-to-JSON
paths.  It may contain only the cases that have finished; omitted, unreadable,
or provenance-invalid records become explicit censored realizations:

```json
{
  "schema_version": 1,
  "manifest_sha256": "<exact hash from the generated manifest>",
  "results": {
    "<case id>": "outer-results/<case id>.json"
  }
}
```

Evaluate that index without submitting a job or rerunning postprocessing:

```bash
python scripts/evaluate_pure_fdm_outer_ensemble.py \
  configs/pure_fdm_outer_zoom_grid.yaml \
  results/pure_fdm_outer_result_index.json \
  results/pure_fdm_outer_ensemble_evaluation.json
```

The evaluator checks only the finest adjacent resolution pair for each phase
replicate (the two available levels in the initial grid), stores every source
path and SHA-256, and reports an ensemble as
`accepted_outer_ensemble_uncalibrated` only after all its phase pairs pass.
That status identifies outer checkpoints eligible for the subsequent overlap
evaluation; it never authorizes a physical delay, nested submission, or PTA
catalogue entry by itself.

Only an accepted outer phase ensemble can be converted into a checkpoint-bound
registration record.  The following command independently re-reads the index
and source records, checks that every checkpoint retains the exact FDM mass,
soliton mass, core radius, case ID, and manifest, and writes no scheduler
request:

```bash
python scripts/prepare_pure_fdm_nested_zoom_registration.py \
  configs/pure_fdm_outer_zoom_grid.yaml \
  results/pure_fdm_outer_result_index.json \
  results/pure_fdm_nested_zoom_registration.json
```

The operator may use only `ready_for_operator_registration` entries to create
the corresponding nested zoom inputs.  The generated record is intentionally
not executable: each new run still requires the operator's Slurm submission
and the already specified collision policy.

### 6. End-to-end delay and PTA handoff — final stage

Only after the previous gates pass do we compose:

```text
numerical capture -> pure-FDM outer merger -> common core/1 pc
-> accepted inner FDM response -> 0.01 pc -> GW evolution
```

The population/ PTA layer receives the per-event delay and its provenance,
uncertainty, and censoring status.  A missing or censored interval is never
replaced with zero and is not silently converted into a physical coalescence
time.

## Censor conditions

The event remains censored for incomplete provenance, `MULTIPLE` capture,
unbound re-escape, unsupported radius or similarity class, insufficient
overlap, failed Hamiltonian or resolution gates, HJM/wave seam contamination,
periodic-boundary wake contamination, unresolved core/de Broglie/orbit scales,
static-core energy injection above the validated binding-energy fraction, or
failure to reach the target within the available cosmic time.

## Current implementation status

The repository currently contains the bridge, radial profile bundle, outer
closure diagnostics, calibrated response-table boundary, overlap validator,
and the typed dual-soliton/two-SMBH seed plus run-input identity preflight.
The lagRamses patch implements the opt-in all-wave seed and raw outer-wave
provenance writer; it does not alter ordinary dynamics when disabled.  The
full toy-repository suite passes (`423 passed` at the latest check).  No
pure-FDM outer zoom has been submitted and no end-to-end physical delay is yet
accepted.
