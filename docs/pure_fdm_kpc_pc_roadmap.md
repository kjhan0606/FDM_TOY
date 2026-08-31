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
A saved registered capture ensemble is only a pointer: downstream readers
rebuild its family smoke check and every capture-to-run binding from the
source ledger event, event digest, and current output sidecar before using it.

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

Schema-v2 result records do not accept a free-standing array of rate points.
They must name a separately hash-bound diagnosed rate ledger whose case,
phase/replicate, model, capture UID, channels, diagnostics, and rate values
match exactly. That ledger must in turn name the exact inventory and
dm_run_provenance sidecar from the registered normal output; their model,
output directory, step, time, and scale factor are rechecked. This prevents a
result JSON from being relabelled with another output or edited rate values.
lagRamses now emits a common CDM/SIDM/FDM model-zoom execution identity when
the materialized namelist supplies the manifest SHA-256, exact case ID,
capture-event SHA-256, and initial-condition/baryon-configuration/sink-input
SHA-256 values. The reader
requires that identity from the same normal-output sidecar and compares it
with the requested case, registered capture, and shared inputs; an unavailable
identity is censored. The sidecar does not by itself recompute a rate from raw
time series, so a production estimator must still bind its source series and
settings before its diagnosed rates become a scientific acceptance.

For an FDM result, the selected manifest case must also declare
`fdm_use_hjm` and `fdm_first_wave_level`; the sidecar values emitted by the
solver are compared with those declarations. A grid without them remains
outside the resolved model-specific acceptance path rather than acquiring an
implicit wave prescription.

`materialize_model_zoom_execution_contract.py` verifies these assignments in
the one solver-consumed `&PHYSICS_PARAMS` group and writes a merge-only
assignment fragment for one CDM, SIDM, or FDM case after it re-reads one
complete bound binary capture event and hashes the initial conditions, baryon
configuration, and SMBH seed catalogue. It is deliberately non-submitting and
does not claim that the raw capture belongs to the selected model realization
or that the solver consumed those external files; model dynamics, FDM wave
controls, and operator Slurm launch remain separate explicit inputs.

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
The current V3 record also preserves the active dual-soliton switch and both
runtime component parameter vectors, so a controlled two-core output can be
bound back to its materialized seed.  It records the expected MPI rank count
for the per-rank FDM and AMR files, allowing a downstream source ledger to
reject an incomplete shard set rather than infer completeness from a subset.
V4 further records a solver-invocation execution-instance token; it can
separate adjacent restart segments but is not a globally unique lineage proof.
V5 binds a restarted child to the execution token read from its concrete
parent raw-provenance file, so a downstream output-set checker can reject a
same-number checkpoint from a sibling branch.
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
It also stores the exact capture-ledger and frame-specification digests.  A
later seed check re-reads both sources and re-derives every sink row, rather
than trusting the pair JSON as a kinematic authority.  It is a
provenance-bound sink-pair input, not a soliton fit or an accepted physical
delay.
Older v1 pair records lack these raw-ledger/frame attestations and are
therefore deliberately non-consumable by the current seed-binding gate;
re-export them from the original ledger and frame specification instead of
trying to upgrade the JSON by hand.

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

Before treating that seed as the `smbh_seed_catalog` input of a declared FDM
zoom case, join the saved capture-to-seed decision to the saved common zoom
contract:

```bash
python scripts/materialize_fdm_capture_seed_zoom_binding.py \
  results/model_zoom_execution_contract.json \
  results/capture_seed_materialization_binding.json \
  results/fdm_capture_seed_zoom_binding
```

The FDM contract must name that exact seed manifest as its sink-input artifact,
use the same capture-event UID and SHA-256, declare the same axion mass, and
state baryonic channels consistently with the seed.  A materialized dual
soliton seed is all-wave (`fdm_use_hjm=false`), so a case that requests HJM is
rejected instead of silently changing the initialization.  The wave-level
field remains explicit even for this all-wave initial state.  A verified
`fdm_capture_seed_zoom_identity_verified` record is still only a declaration
gate: it does not show that lagRamses consumed `ic_sink` or the seed fragment,
nor that the two solitons relaxed or that their wake is resolved.
It also does **not** yet equate the seed's code-unit soliton profile with the
zoom case's physical binary total mass, mass ratio, core radius, or soliton
mass: the materialized seed has no validated dimensional field-mass/core
diagnostic.  Those coordinates remain unbound until a relaxed wave-state
measurement and its normalization artifact are introduced; they must not be
treated as a similarity-class match merely because this declaration gate
passes.

Before the operator submits a completed run namelist, verify that its scalar
FDM/AMR switches, all two-soliton components, non-compacting SMBH controls
(`smbh=.true.`, `rmerge=0`, and the capture ledger), and the run-directory
`ic_sink` remain identical to the materialized seed/configuration:

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

For a declared FDM zoom case, retain one saved join between that ready
all-wave preflight and the earlier capture-to-seed-to-zoom declaration:

```bash
python scripts/materialize_fdm_declared_run_input_binding.py \
  --expected-build-git-hash <40-character-lagRamses-revision> \
  results/fdm_capture_seed_zoom_binding/fdm_capture_seed_zoom_binding.json \
  results/pure_fdm_dual_soliton_preflight.json \
  results/fdm_declared_run_input_binding
```

This re-reads both decisions, requires their seed manifest to be identical,
requires the checked run namelist to be byte-identical to the one in the
common zoom contract, and records the exact intended lagRamses revision.  Its
verified status proves only that the declared
`&PHYSICS_PARAMS` identity and the all-wave `&FDM_PARAMS`/`ic_sink` input are
one configuration.  It still does not prove solver consumption; that claim
starts only with a separately validated runtime provenance record.

After the first normal FDM output, verify the runtime V2/V3/V4/V5 provenance against
the same materialized seed:

```bash
python scripts/validate_dual_soliton_runtime_identity.py \
  initial_conditions/pure_fdm_case/dual_soliton_seed_manifest.json \
  output_00042/fdm_outer_wave_provenance_00042.txt \
  results/pure_fdm_dual_soliton_runtime_identity.json
```

`runtime_seed_identity_verified` establishes only that the raw V2/V3/V4/V5 output
declares the same all-wave two-core configuration and passes the raw
force/current coverage gate.  It does not relax, calibrate, or otherwise
accept the physical merger calculation.

It also does not itself attest that the solver consumed the exact declared
namelist or `ic_sink`, nor that an output belongs to the declared model-zoom
case.  Therefore, enumerate every normal output intended for a temporal
series and bind the copied namelist, compilation record, model-zoom sidecar,
`COMPLETE` marker, and root-level raw FDM provenance to the saved declared-run
input:

```bash
python scripts/validate_fdm_declared_zoom_runtime.py \
  --output results/fdm_declared_zoom_runtime_identity.json \
  results/fdm_declared_run_input_binding/fdm_declared_run_input_binding.json \
  output_00042 output_00043 output_00044
```

Only `fdm_runtime_output_identity_verified` permits these explicitly listed
normal outputs to be described as one declared FDM output set.  The verifier
requires the intended build revision to agree among the declaration, DM
sidecar, and compilation record; the explicit non-compacting `rmerge=0`
mode; a common run root; monotonic time, step, and scale factor; exact
copied-namelist identity; exact common model-zoom hashes; and agreement
between the raw FDM and DM run-provenance controls.  It supports the normal
`output_XXXXX/group_00001/` metadata layout without guessing group
directories.

V4 emits one execution-instance token per solver invocation.  V5 records the
parent token read from the concrete restart-parent raw provenance.  The
verifier requires each token to have one restart parent, rejects a first
listed segment with an unlisted parent, and permits a token transition only
when the child names both the immediately preceding output number and its
exact parent token.  V2/V3 output identity remains conditional because it has
no such discriminator.  V5 still does not supply a cryptographic run UUID or
a complete checkpoint lineage, so a suspected token collision or fork remains
conditional and cannot support a production delay result.

The relaxation consumer re-reads that saved decision, re-hashes its seed
manifest and raw provenance, and reconstructs the decision before use.  A
copied status field, a changed raw provenance file, or a stale seed manifest
therefore cannot authorize the relaxation window.

Before writing the relaxation evidence table, a manual Lageunha extractor
must list the raw V3/V4/V5 provenance path for every selected output in a small
JSON manifest (`raw_fdm_provenance_paths`).  The binding code reads each
record itself, verifies its two-soliton configuration against the seed, and
requires and hashes every per-CPU `fdm_*.outNNNNN` and corresponding
`amr_*.outNNNNN` topology shard from `00001` through the V3 `mpi_ncpu` value.
The raw runtime
provenance must appear exactly once in the temporal sequence.  Bind this
source set without reading full fields or launching an FFT:

```bash
python scripts/materialize_dual_soliton_relaxation_sample_ledger.py \
  --runtime-output-identity results/fdm_declared_zoom_runtime_identity.json \
  results/pure_fdm_dual_soliton_runtime_identity.json \
  results/dual_soliton_relaxation_sample_manifest.json \
  results/dual_soliton_relaxation_sample_ledger.json
```

The resulting ledger is required by the relaxation evidence.  It requires the
manifest's raw-provenance paths to be exactly the verified output-set paths;
an omitted, added, or substituted V3/V4/V5 output is rejected.  It is still source
identity only: the core properties and the wave mass, Hamiltonian, and
angular-momentum series must still be measured from those sources by a
declared extractor.  Bind that diagnostic series to both the exact sample
ledger and immutable extractor bytes before assessment:

```bash
python scripts/materialize_dual_soliton_relaxation_diagnostic_provenance.py \
  results/dual_soliton_relaxation_sample_ledger.json \
  results/dual_soliton_relaxation_diagnostic_manifest.json \
  results/dual_soliton_relaxation_diagnostic_provenance.json
```

The diagnostic manifest records the extractor path and version with the
measured series.  This provides reproducible source and method identity; it
does not yet attest that the extractor was executed, so it cannot replace the
later paired-resolution and conservation checks.  This declared-series path
remains useful for developing the extractor, but it must not be reported as an
executed relaxation measurement.

For an actual Lageunha run, use the bounded execution wrapper.  It verifies the
sample ledger before launching, invokes the command with `shell=False`, forces
the numerical-library thread environment to one, and appends the ledger and a
private temporary result path as the extractor's two positional arguments.  A
validated result is hard-linked to the requested `--result` path only after
the command exits successfully:

```bash
python scripts/run_dual_soliton_relaxation_extractor.py \
  --sample-ledger results/dual_soliton_relaxation_sample_ledger.json \
  --result results/dual_soliton_relaxation_extractor_result.json \
  --attestation results/dual_soliton_relaxation_extractor_attestation.json \
  --working-directory /path/to/lageunha/working/directory \
  --extractor python /path/to/lageunha_relaxation_extractor.py
```

The extractor must write a strict JSON result containing
`dual_soliton_relaxation_extractor_result`, its version, the exact sample-ledger
path and SHA-256, and the diagnostics arrays.  A non-zero exit, malformed or
ledger-mismatched result, pre-existing output, or timeout produces no
attestation.  The attestation re-hashes the command files, result, and sample
ledger and records the command, arguments, UTC interval, host, and one-thread
environment.  It is a wrapper-declared execution record, not an OS audit and
not proof that the extractor's internal implementation correctly measured the
physics.

Bind that execution record to a diagnostic provenance file before using it in
the relaxation assessment:

```bash
python scripts/materialize_dual_soliton_relaxation_executed_provenance.py \
  results/dual_soliton_relaxation_sample_ledger.json \
  results/dual_soliton_relaxation_extractor_attestation.json \
  results/dual_soliton_relaxation_executed_diagnostic_provenance.json
```

The assessment then emits an explicit
`relaxation_conservation_executed_series_*` status.  That status is still
conditional: it records a wrapper-declared command execution and source-bound threshold
diagnostics, not a converged relaxation pass, an outer-merger calibration, or
a physical coalescence delay.

The relaxation evidence schema requires both the source ledger and the
diagnostic provenance.  Earlier evidence files without those records are not
relaxation inputs and must be re-extracted rather than relabelled as verified.

V2 output identity remains useful for declared-case/run-root/namelist/build
provenance, but it cannot enter the relaxation source ledger because it lacks
the expected MPI-rank count.  V3 closes the declared shard-set completeness
check; V5 adds an exact parent-token check at each listed restart transition.
Neither proves a globally collision-free checkpoint lineage or the extractor's
internal physical method.  The manual diagnostic path remains a conditional
declared-series threshold result; the execution-attested path records the
command and result but is still not a converged relaxation or conservation
pass.

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
the wave fields.  A conditional within-threshold result is not an
outer-merger result or a physical relaxation pass, and an outside-threshold
result cannot be relabelled as zero pairing delay.

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

#### Runtime-versus-offline backreaction gate

The default subgrid path is an offline consumer of measured environment
response; it does not inject a second force into a live lagRamses run.  Before
using a frozen profile, analytic closure, or measured residual, compare a live
and a frozen calculation started from the same checkpoint.  The paired tracks
must carry separate source hashes, common checkpoint identity, conservation
errors, and orbital-resolution evidence.  The comparison is performed by
`assess_live_frozen_backreaction` and is deliberately model agnostic:

- `offline_acceptable`: CDM, SIDM, or FDM live/frozen orbital power and torque
  agree within 20 percent, eccentricity differs by at most 0.02, signs do not
  reverse, and both tracks pass the conservation/resolution gates.  The
  offline closure is allowed only over the measured overlap; it is not a
  global extrapolation.
- `runtime_required`: the paired tracks are resolved but live wake/scattering/
  wave backreaction changes a rate or eccentricity beyond those limits, or a
  rate changes sign.  The runtime treatment (or a source-decomposed live
  residual) must be retained; an analytic term must not be added on top of it.
- `censored`: the tracks do not share a checkpoint, lack a positive-width
  overlap or enough matched points, fail conservation/resolution, use the same
  source artifact, or contain an unresolved power/torque rate.  No zero force,
  zero delay, or runtime conclusion is inferred.

For FDM the live track must explicitly use `live_wave_only` accounting.  A
live wave wake and an analytic FDM drag are never summed.  For CDM and SIDM the
same gate tests whether resolved wake/kinetic-energy exchange or scattering
changes the offline closure; it does not assume that either effect is small.

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
relaxation stage now also has a bounded Lageunha execution wrapper: it binds a
verified sample ledger, uses a shell-free one-thread child environment, writes
the result through a private temporary path, and publishes a recheckable
wrapper-declared attestation only after a successful source-bound result.
The full toy-repository suite passes (`546 passed` at the latest check).

No pure-FDM outer zoom has been submitted, and no actual Lageunha relaxation
extractor command or end-to-end physical delay has been run or accepted.  The
next operator input is the real extractor command together with a verified
sample ledger and `fdm_runtime_output_identity_verified` output set; until
those inputs exist, the relaxation result remains unmeasured rather than being
filled with a proxy or a zero delay.
