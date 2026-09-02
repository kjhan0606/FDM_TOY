# FDM SMBH Delay

`fdm-smbh-delay` estimates the unresolved delay between a parsec-scale
supermassive-black-hole (SMBH) binary and a separation of `0.01 pc` inside a
fuzzy-dark-matter (FDM) soliton.

It is the delay-model component of the working paper **Pulsar Timing Array in
Fuzzy Dark Matter Model**, which connects SMBH coalescence delays, cosmological
event populations, and PTA signals. The manuscript is maintained separately in
the Overleaf repository recorded in
[`docs/project_links.md`](docs/project_links.md).

The package is a **parameterized toy model and unresolved inspiral
prescription**, not a first-principles prediction of physical coalescence.
Published separation curves provide comparison cases, but the adopted soliton
profiles still require calibration against the corresponding simulations. In
particular, a merger of sink particles in lagRamses is a resolution-scale
numerical event and must not be called an SMBH coalescence.

## v0.1 scope

The first model integrates

```text
1 pc --(static soliton + analytic FDM wave drag)--> 0.01 pc
```

It does not yet predict the preceding numerical-radius-to-parsec inspiral. A
Peters gravitational-wave time is available below the FDM interval. The orbit
calculation returns a crossing time or a censored timeout, an orbital time
series, energy and momentum transfer, and validity flags.

The scientific design and lagRamses context are preserved in
[`docs/HANDOFF.md`](docs/HANDOFF.md).
Repository locations, including the Overleaf paper remote, are recorded in
[`docs/project_links.md`](docs/project_links.md).

## Install

Python 3.11 or newer is required.

```bash
python -m pip install -e '.[dev]'
pytest
```

The public configuration interface requires explicit units. Bare numeric
values are rejected for dimensional quantities.

## Run

```bash
fdm-smbh-delay configs/lagramses_m22_example.yaml --output results/m22_example
```

or, without installing the console script:

```bash
python scripts/run_case.py configs/lagramses_m22_example.yaml \
  --output results/m22_example
```

The output directory contains:

- `summary.json`: status, delay, energy error, validity flags, and provenance
- `timeseries.csv`: orbit, energy, angular momentum, density, and drag metrics
- `config.yaml`: exact input configuration used for the calculation

`timeout` is a valid censored physical result, not a numerical failure.
The analytic-orbit command records the transfer without evolving the
wavefunction. Separate PyUL_NBody adapters now run fully coupled
Schrödinger–Poisson tests in which the FDM wake supplies the force and no
analytic FDM drag is applied. The coupling and double-counting rules are
documented in [`docs/wave_energy_coupling.md`](docs/wave_energy_coupling.md).

The numerical capture interface, corrected Koo and Boey profile definitions,
live-wave parameter grid, and first PyUL_NBody resolution tests are documented
in [`docs/merge_calibration.md`](docs/merge_calibration.md). These tools leave
the separate lagRamses working tree unchanged.

To combine a sink time with the three physical intervals, use only verified
records:

```bash
fdm-pta-delay \
  --z-sink 1.0 \
  --backreaction-decision results/case/backreaction_decision.json \
  --backreaction-delay-record results/case/kpc_to_pc_delay.json \
  --fdm-summary results/case/summary.json \
  --gw-record results/case/gravitational_wave_delay.json
```

The command returns no `true_merge_time_myr` if an interval is missing,
invalid, or censored, and exits with status 2 for a non-complete estimate.
Missing physics is never interpreted as zero delay. A bare `--kpc-to-pc-delay`
or `--gw-delay` value is retained only as an explicit censored/operator-asserted
record; it can never create a complete physical estimate. The kpc delay record
must be bound to a verified live/frozen decision and its integrated separation
interval must lie inside that decision's measured overlap.
`uncalibrated`, `outside-support`, and `stalled` FDM summaries are composed as
explicitly censored intervals, distinct from absent inputs. The JSON result
lists both `missing_segments` and `censored_segments`; its additive `segments`
records retain per-interval reasons and source provenance.

An accepted galaxy-merger zoom table can correct an analytic kpc-to-pc
baseline only at the exact physical point represented by a table row:

```python
from fdm_smbh_delay.zoom_calibration import apply_kpc_delay_calibration

kpc_segment = apply_kpc_delay_calibration(
    table,
    physics=event_physics,
    analytic_baseline_delay_myr=analytic_delay_myr,
)
```

The returned complete segment retains the zoom source case and SHA-256. An
unmeasured physical point returns a censored segment; the consumer never
interpolates or extrapolates the zoom correction.

CDM comparison runs use a separate, provenance-gated capture-to-hardening-to-
GW interface.  It requires accepted CDM resolution and phase evidence for
each environmental interval, explicit collisionless force accounting, and
orbit-covered secular blocks with regression-quality and uncertainty gates;
stalled or under-covered tracks remain `censored`.  The extractor's period is
an instantaneous two-body Kepler sampling proxy, not proof of a completed
orbit in the extended host potential.  It rejects HR5-style fixed population
delays:
[`docs/CDM_COALESCENCE_INTERFACE.md`](docs/CDM_COALESCENCE_INTERFACE.md).

Resolved model-physics results use schema version 5 and carry a hash-checked
`runtime_identity` artifact.  The phase cache records and re-hashes the exact
zoom specification file.  For the rate-track anchor, CDM stage composition
reopens the non-compacting runtime identity, checks its case/manifest/capture
contract, and requires the rate-ledger output sidecar to belong to that
attested output set; an identity path alone is not sufficient.  The generic
schema-v5 result reader treats `runtime_identity` as a hash-checked artifact;
semantic runtime validation is performed by the CDM rate-track anchor, not by
schema-v5 alone for arbitrary CDM/SIDM/FDM results.

## Example configuration

```yaml
model:
  name: wave_df_3d
  alpha_df: 0.341
  drag: true
  fdm_bulk_velocity: ["0 km/s", "0 km/s", "0 km/s"]

binary:
  M1: "5.0e7 Msun"
  M2: "5.0e7 Msun"
  separation: "1 pc"
  eccentricity: 0.0
  orbit: circular

fdm:
  particle_mass: "1.0e-21 eV"
  soliton_mass: "1.0e9 Msun"
  mass_definition: total_profile
  core_radius: "2 pc"
  profile: schive_fit

integration:
  stop_separation: "0.01 pc"
  max_time: "20 Myr"
  output_samples: 1000
  rtol: 1.0e-9
  atol: 1.0e-12
```

The example core radius is a schema demonstration, not a validated literature
fiducial value. Published comparisons require matching each paper's soliton
mass and core-radius definitions.

## Development status

The repository currently implements the static-soliton three-dimensional orbit
calculation, FDM energy and momentum transfer, Peters gravitational-wave
times, conservative composition of the physical coalescence time, and a source
study of the lagRamses conditions for a merger of sink particles. The Koo and
Boey profile definitions and fitted separation curves are included as reference
cases. Fully coupled PyUL_NBody calculations measure the complete Hamiltonian,
the osculating binary elements, orbit-averaged power and torque, correlated
uncertainties, central-density evolution, and sparse three-dimensional wave
diagnostics. Calibration against converged long calculations, the
accepted production q-e-separation table, and the cosmological PTA population
remain future work. The runtime table schema and sparse q-e-small-separation
pilot are implemented, but a pilot row is not published unless every numerical
acceptance gate passes.

The point-mass osculating semi-major axis is an optional diagnostic. It can be
undefined when the instantaneous state is not bound in the point-mass Kepler
approximation, even though the binary is evolving in the extended FDM and
softened-SMBH potential. Separation, angular momentum, and the complete
Hamiltonian exchange remain the primary live-wave diagnostics in that case.

## Fully coupled wave calculation

The live-wave calculation requires a separate checkout of the public
PyUL_NBody solver. A representative command is

```bash
python scripts/run_pyul_wave_case.py \
  --pyul-path /path/to/PyUL_NBody \
  --case-id koo_mbh1.0e8 \
  --resolution 128 \
  --duration-myr 1.0 \
  --save-number 2048 \
  --rk-steps 36 \
  --box-pc 40 \
  --output results/pyul_long_term/koo_1myr_s2048
```

The corresponding orbit-resolved measurements are generated with

```bash
python scripts/analyze_pyul_wave_run.py RUN
python scripts/analyze_pyul_secular_exchange.py RUN
python scripts/analyze_pyul_line_density.py RUN
python scripts/analyze_pyul_wave_response.py RUN
python scripts/build_wave_exchange_table.py RUN --output wave_exchange.csv
```

The `512^3` wave-response analysis has used about 37 GB of resident memory per
process. Run only one such analysis at a time, restrict numerical libraries to
one thread, and checkpoint one sparse three-dimensional sample per invocation:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python scripts/analyze_pyul_wave_response.py RUN \
  --resume --max-new-samples 1
```

Repeat the command until it produces `wave_response_timeseries.csv`,
`wave_radial_profiles.csv`, and `wave_response_summary.json`. An interrupted
analysis retains `wave_response_timeseries.partial.csv` and
`wave_radial_profiles.partial.csv`; both files are required for `--resume`, and
their sample indices and row counts are validated before another sample is
processed.

Generate the sparse q-e-small-separation pilot manifest with

```bash
python scripts/generate_wave_calibration_grid.py \
  configs/wave_calibration_qe_extension.yaml \
  --output results/wave_calibration_qe_extension
```

The manifest records the Kepler orbit-mean separation in cells and the
pericentre in both cells and Plummer radii. Generation fails before any solver
is launched when a requested resolution violates the configured two-cell
orbit-mean or two-Plummer-radius pericentre gate. At the fixed pilot box size,
the `a/r_c=0.05` pair therefore uses `n=512,768`, not `n=384,512`.

Long GPU calculations use the PyUL-compatible Torch backend. Install the
optional dependencies with `python -m pip install -e '.[gpu]'`, export one
initial three-dimensional state with PyUL, and launch the continuation through
the provenance-preserving wrapper:

```bash
python scripts/launch_torch_wave_case.py REFERENCE_RUN \
  --output results/torch_wave/koo_n512 \
  --duration-myr 1.0 \
  --save-number 2048 \
  --movie-frame-number 360 \
  --save-3d-number 32 \
  --checkpoint-every-saves 32 \
  --device cuda:0
```

The wrapper snapshots the uncommitted numerical source and its SHA-256 hashes
under `RUN/torch_solver_provenance`. A resumed calculation is rejected if that
source no longer matches, preventing two numerical implementations from being
joined into one trajectory.

For a manually managed shared GPU, place `FDM_SOLVER_PID_FILE` outside the new
run directory and attach the NVML guard to the recorded solver PID:

```bash
# Terminal 1
export FDM_SOLVER_PID_FILE=/path/to/logs/solver.pid
python scripts/launch_torch_wave_case.py REFERENCE_RUN \
  --output RUN --device cuda:0

# Terminal 2, after the marker appears
python scripts/guard_cuda_process.py \
  --gpu-index 0 --solver-pid "$(cat "$FDM_SOLVER_PID_FILE")" \
  --status-file /path/to/logs/gpu_guard_status.json
```

The guard calls NVML directly and interrupts only the owned solver if another
compute PID appears on that GPU. It does not launch repeated `ps`, `pgrep`, or
`nvidia-smi` scans.

The intermediate `384^3` spatial-convergence run on `syn101` uses a dedicated
manual tripwire:

```bash
tmux new-session -d -s fdm_n384_monitor \
  'exec bash scripts/monitor_syn101_n384.sh 0'
```

The monitor refuses an occupied GPU, stops the managed calculation when a
foreign Slurm job or GPU process appears, and does not retry a failed evolution
or finalizer. GPU ownership is checked through the solver PID marker and
`nvidia-smi`; no general process scan is used. After the evolution, the monitor
runs only conservation, orbit-averaged exchange, line-density, and three-level
spatial-convergence measurements. The memory-intensive three-dimensional wave
response remains a separate operation.

The Boey `384^3` spatial repeats use a separate guarded sequence:

```bash
tmux new-session -d -s fdm_boey_n384_guard \
  'exec python scripts/run_guarded_syn101_boey_n384.py --gpu-index 0'
```

The sequence evolves the 2, 5, and 10 percent cases in that order. A one-time
preflight requires an idle GPU and no foreign Slurm allocation on `syn101`.
The guard then keeps one persistent `nvidia-smi --query-compute-apps` stream
and one `squeue --iterate` stream instead of launching repeated process scans.
The compute-context stream also detects a competing process that has allocated
GPU memory but is momentarily running no kernel. An unmanaged GPU context, a
foreign Slurm allocation, a telemetry failure, or a termination signal stops
only the managed process group. A later invocation resumes an incomplete case
from its atomic Torch checkpoint and skips every completed case.

If a completed spatial repeat misses the resolved Hamiltonian gate, preserve
that run and give a named numerical recovery its own output directory. For the
Boey 10 percent `384^3` half-wave-timestep recovery, the guarded GPU driver and
file-only CPU finalizer are

```bash
bash scripts/run_boey_n384_dt05_recovery.sh
bash scripts/watch_boey_n384_dt05_recovery.sh
```

The GPU driver waits through one persistent memory telemetry stream until card
0 is idle, resumes after every guarded yield, and never replaces the original
`dt=1` result. The CPU tripwire wakes only on the recovery completion file,
then applies the bounded post-processing sequence, refreshes every accepted
source comparison, and requires the combined release verifier to succeed.

Sparse `384^3` wave-response analysis runs one snapshot at a time on `syntax`:

```bash
bash scripts/run_safe_n384_wave_response.sh /path/to/completed_n384_run
```

The runner holds one global lock, limits numerical libraries to one thread,
bounds virtual memory at 32 GiB, and resumes from the paired partial tables.
Every successful invocation adds at most one new three-dimensional sample.
The complete Boey sequence can be handed to the low-impact CPU tripwire with

```bash
tmux new-session -d -s fdm_boey_n384_tripwire \
  'exec bash scripts/watch_boey_n384_postprocess.sh'
```

The tripwire checks only completion files at five-minute intervals. It waits
for the Koo `384^3` response to release the global FFT lock and for all three
Boey evolutions, then runs the conservation, secular-exchange, line-density,
resumable wave-response, dimensionless-exchange, and matched-`512^3`
measurements sequentially. It finally builds the accepted combined Koo and
Boey calibration release. Final release verification requires accepted Koo
and Boey rows, accepted Boey rows on at least two binary-to-soliton mass
planes, and a positive-width accepted separation overlap where an actual mass
interpolation succeeds. If the repeats do not provide that overlap, the
pipeline fails the release gate and retains the rejected-bin diagnostics; it
does not relax the spatial-systematic threshold.

Add `--save-movie-plane` to the live-wave command when a movie is required.
This option writes the central FDM density plane at every diagnostic output
without retaining the full three-dimensional wavefunction at the same cadence.
After `analyze_pyul_wave_run.py` has generated the conservation history, render
the density, SMBH trajectories, binary separation, and Hamiltonian components
with

```bash
python scripts/render_pyul_movie.py RUN \
  --output results/pyul_convergence/koo_n512.mp4 \
  --poster results/pyul_convergence/koo_n512_poster.png
```

MP4 output uses the optional `video` dependencies. GIF output remains
available when the output path has a `.gif` suffix.

The full evolution and a selected interval can be combined in a four-panel
figure with

```bash
python scripts/plot_pair_separation.py RUN \
  --output pair_separation.pdf \
  --wave-density-panel \
  --mark-resolution-limit \
  --koo-reference \
  --time-zoom-myr 0.0 0.05
```

The live calculation is physically interpretable only while its spatial and
Hamiltonian acceptance tests pass. An orbit below two cell widths is retained
as a numerical result but is not used as a calibrated decay rate. Long
calibration runs should also use `--save-3d-number` to retain sparse fields for
radial energy transport and spherical-mode measurements.

Numerical variants can also be compared at matched physical separation. The
default comparison divides the common resolved separation range into eight
bins and retains a bin only when every calculation contributes at least eight
complete orbits:

```bash
python scripts/summarize_pyul_convergence.py \
  n512=/path/to/n512 \
  n384=/path/to/n384 \
  n256=/path/to/n256 \
  --separation-bins 8 \
  --minimum-orbits-per-separation-bin 8 \
  --output spatial_convergence.json
```

The JSON retains both the same-time comparison and the matched-separation
bootstrap intervals. The latter prevents a difference in binary separation or
orbital phase from being assigned directly to spatial resolution. Bootstrap
blocks are capped at half the selected cycles, so the minimum eight-orbit
sample retains at least two independent block lengths.

Subgrid release rows must also keep the Hamiltonian drift below one percent in
each run's initial contiguous spatially resolved interval, resolve the measured
half-density radius by at least two cells in both runs, keep the resolution-pair
absolute mean-eccentricity mismatch at or below 0.02, and keep each power,
torque, and total-wave rate spatial difference below 20 percent. The 0.02
eccentricity cap is one fifteenth of the q-e design's minimum plane spacing and
may only be tightened. The full-run
maximum Hamiltonian drift remains in the convergence JSON as a diagnostic but
does not reject a bin solely because a later, already underresolved tail drifts.
Rejected bins remain in the provenance summary but cannot be loaded by the
runtime interpolator. The builder publishes the CSV first and a checksum-bearing
`.summary.json` commit marker second:

```bash
python scripts/build_subgrid_calibration_table.py \
  --source koo2024=/path/to/koo_n512_n384.json \
  --source boey2025=/path/to/boey_n512_n384.json \
  --output fdm_subgrid_calibration.csv
```

Production code should load this pair with
`SubgridCalibrationTable.from_release`; it verifies the schema, provenance,
row count, profile list, input-derived release ID, and CSV SHA-256. The release
ID is stored in every CSV row, so a stop between the CSV and summary replaces
cannot validate against the previous summary even when the numerical rows are
unchanged. `from_csv` is reserved for
exploratory data and test fixtures. No interpolation is permitted outside the
accepted mass-ratio, eccentricity, binary-mass, and separation ranges or across
a rejected separation bin. Higher-resolution failure or insufficient joint
separation-eccentricity support remains unresolved/censored; it must not trigger
automatic n=1024 expansion or extrapolation. A v4 table interpolates piecewise
linearly in
`(q,e)` only when every bracketing plane independently supplies the required
mass-separation support. The outer half of an accepted edge bin uses that bin's
measured value, while every interpolated systematic is the maximum of all
bracketing rows. These rules are stored in the release summary and checked by
the production loader.
`advance_calibrated_exchange` accepts the orbital power and torque already
measured from a resolved wake and applies only the residual relative to the
calibrated target. Leave those arguments at zero only when the FDM response is
unresolved; applying the full target on top of a live wake would count the same
exchange twice.

## Horizon Run 5 comparison sample

The legacy HR5 sink tree can be regenerated as an explicit catalog of binary
captures with interval-censored times.
The two output boundaries are retained because a disappearing sink is last
resolved at output `i-1`, whereas the assigned surviving SMBH is selected from
the population at output `i`. The files retain `receiver_id` as a historical
field name.

```bash
python scripts/extract_hr5_capture_catalog.py \
  '/home/kjhan/BACKUP/GalFinder/SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/BinarySMBH/Sink_Merging_Tree.dat.Updated'

python scripts/reproduce_hr5_original_figures.py --rebuild-cache
python scripts/validate_hr5_capture_receivers.py
python scripts/analyze_hr5_dual_agn.py
```

The first command produces 576,278 binary-capture intervals from 1,688,677 sink
histories. The second command redraws Figures 1--13 of the earlier HR5 draft.
The third command tests the assigned surviving SMBHs against the phase-space
phase-space states and consecutive MkAGN outputs. The fourth command measures
spatially selected dual AGN candidates and single-AGN pairs, estimates spatial
variance with an eight-region jackknife, and constructs matched pure
two-member comparisons with interval and right censoring.

The disappearing sinks are measured directly, but the assigned surviving SMBHs
come from distance and mass criteria rather than direct records of the capture
partners. The possible binary captures are not physical SMBH coalescences. The event
definitions, validation of assigned companions, active-pair selection, and literature
comparison are documented in
[`docs/hr5_reproduction.md`](docs/hr5_reproduction.md).
