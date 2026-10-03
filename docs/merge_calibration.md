# Numerical capture and live-wave calibration

This document records the first three tasks required before a runtime subgrid
model can be added to lagRamses. No lagRamses source file is changed by these
calculations.

## 1. Numerical binary-capture boundary

The active lagRamses source links sinks at

\[
r_{\rm num}=r_{\rm merge}\frac{L_{\rm box}/h}{2^{\ell_{\max}}}.
\]

The scale is physical and independent of redshift. It is not universal across
runs. `info_*.txt` supplies `levelmax` and `H0`; the initial-condition header or
startup log must supply the comoving box length in Mpc/h. The archived namelist
supplies `rmerge`. If it omits `rmerge`, the active source default is one finest
cell. The feed-mode namelist generator often writes four cells, so the archived
run value must be used rather than assuming either value.

```bash
python scripts/inspect_lagramses_capture.py \
  --info /path/to/output_XXXXX/info_XXXXX.txt \
  --namelist /path/to/output_XXXXX/namelist.txt \
  --box-size-mpc-h 1.2
```

An exact initial orbit also requires a record written before the sink group is
compacted. For a two-member record, the inspection tool returns the relative
position and velocity, centre-of-mass state, two-body specific energy,
orbital energy, angular momentum, eccentricity vector, and osculating
semi-major axis. A group with more than two members remains `MULTIPLE` and is
not converted to an arbitrary sequence of binaries.
The FDM_TOY ledger reader checks that pair flags are actual JSON booleans,
that pair indices are complete, and that `within_rmerge` agrees with the
periodic source coordinates and recorded numerical merge radius.  When the
source-specific energy is present, its sign must also agree with the recorded
two-body binding flag.  A contradictory event is rejected before it can
seed post-capture dynamics; a completed ledger transaction alone is not a
physical-coalescence claim.
For full native writer records, the reader additionally recomputes total
sink mass, periodic centre of mass and velocity, maximum pair separation,
relative kinetic and potential terms, and pair angular momentum from the
member rows.  A partial native diagnostic block is rejected.  Older sparse
import records without that block remain readable, but do not acquire the
full native-conservation verification merely by being parseable.  Transfer
of a capture binary into a pure-FDM dual-SMBH seed requires the verified
native block; a sparse legacy record is censored at that seed boundary even
if its orbital elements can be inspected.

No production FDM namelist and pre-compaction event record are present in this
repository. Consequently, the interface and calculation are fixed, but a
single production value of `r_num` is not asserted yet.

## 2. Koo and Boey reference definitions

Koo et al. (2024) use a numerical Schrödinger--Poisson ground state with

\[
M_s=10^9\,M_\odot,\qquad
\rho_0=7.05\times10^6\,M_\odot\,{\rm pc}^{-3}
\]

for `m = 1e-21 eV`. A Schive profile constrained to the same total mass and
central density has `r_c = 2.3047283691 pc`. This is an equivalent analytic
profile, not the exact numerical ground state. The physical initial binary
separation is `0.9 pc`; the averaged fit uses `D0 = 0.8 pc`.

The implemented Koo coefficient follows their equation (18), including

\[
\widetilde M_s=M_s+2\gamma M_{\rm BH},\qquad\gamma=2.192.
\]

Boey et al. (2025) define the initial profile with their equation (4). At the
fiducial particle mass and `r_c = 2.2 pc`, this gives
`rho_0 = 8.1107848e6 Msun/pc^3` and a total Schive-profile mass of
`1.0006508e9 Msun`. The binary starts at `D = 3 pc`. The initializer gives
`584.69 km/s` for each `1e8 Msun` SMBH, within 0.1 percent of the reported
`584.14 km/s`.

The reference tables and plot are regenerated with

```bash
python scripts/reproduce_literature.py \
  --output results/literature_reproduction --plot
```

The Koo table contains equations (7) and (18). The Boey table contains equation
(26) with Table I. They reproduce the published fitted curves; they are not
digitized simulation histories.

## 3. Live-wave parameter space

The structured grid varies

- `q = 1.0, 0.3, 0.1`;
- `e = 0.0, 0.3, 0.6`;
- `M_bin/M_s = 0.04, 0.10, 0.20`;
- `a/r_c = 0.10, 0.40, 1.36`.

Each physical case also records
`hbar^2/(G m^2 M_s r_c)`. This coefficient measures the quantum term relative
to self-gravity after the equations are normalized by the soliton mass, core
radius, and dynamical time. Physical boson-mass variants do not require a
separate calculation when this coefficient and all other dimensionless state
variables agree. The Koo and Boey profile proxies give 0.37079 and 0.38819,
respectively, and therefore remain distinct initial states. A disturbed core
also requires its measured complex dipole and quadrupole amplitudes. Soliton
similarity scaling does not remove that dependence.

Six literature anchors form tier 0. Nine fiducial and one-axis variations form
tier 1. The remaining 72 interaction cases form tier 2. The resulting 87
physical cases expand to 108 resolution runs. Tier-0 anchors use effective
resolutions of 512, 1024, and 2048 cells per box. Tier 1 uses 512 and 1024;
tier 2 is delayed until the 1024-cell anchor calculation is validated.

```bash
python scripts/generate_wave_calibration_grid.py \
  configs/wave_calibration_grid.yaml \
  --output results/wave_calibration_grid
```

Every run requires a coupled Schrödinger--Poisson field and moving SMBHs.
Analytic FDM drag is disabled because the resolved wake supplies the force.

### Sparse q-e and small-separation extension

The production table schema is version 4. Each accepted row now carries the
SMBH mass ratio and the duration-weighted, orbit-averaged osculating
eccentricity in addition to profile, binary mass fraction, and separation.
Its separation coordinate is the orbit-mean physical SMBH distance, not the
semimajor axis. The bound-binary runtime maps its secular `(a,e)` state to
`<r>_t=a(1+e^2/2)` before lookup, matching the Kepler mean used in the run
manifest. This mapping is a Kepler approximation: a strongly non-Keplerian
orbit in a disturbed core requires an explicitly measured mean separation,
not silent use of `a` as a substitute. If the mapped mean lies outside
accepted bins, the binary is uncalibrated/censored.
The manifest's input eccentricity remains separate provenance because the
extended soliton potential shifts the point-mass osculating diagnostic.
Runtime lookup interpolates in `q` or `e` only when
every bracketing plane has accepted mass and separation support at the query.
Physical-rate evaluation also requires the FDM particle mass. It recomputes
`hbar^2/(G m^2 M_s r_c)` from that mass, soliton mass, and core radius, and
rejects a mismatch with the table's similarity class beyond relative `1e-6`.
The bound-binary consumer treats that mismatch as uncalibrated/censored rather
than applying a rate from another boson-mass regime.
The soliton mass and core radius must use the same profile definitions as the
calibration row; a catalogue fit with different normalization is not silently
treated as the same similarity class.
Mass interpolation likewise requires accepted separation support on both mass
planes, and separation interpolation may cross only contiguous accepted bins.
Missing support, gaps, and all extrapolation stop the bound-binary calculation
with an uncalibrated result.

The production-candidate sparse extension contains ten physical cases and 28
resolution runs. It preserves the original 20 pilot run IDs and adds one
adjacent finer resolution for each of the eight tier-1/2 cases. Six cases
sample distinct `(q,e)` planes at `a/r_c=0.20`. The circular
equal-mass plane and the `(q,e)=(0.3,0.3)` plane also reach `a/r_c=0.10` and
`0.05`, corresponding to `0.22` and `0.11 pc` for the Boey-profile anchor.

```bash
python scripts/generate_wave_calibration_grid.py \
  configs/wave_calibration_qe_extension.yaml \
  --output results/wave_calibration_qe_extension
```

The tier hierarchy is `128/256/512`, `256/512/768`, and, at the smallest
axis, `512/768`. These calculations remain candidates rather than accepted
table rows. The table builder requires at least eight complete orbits, a resolved
half-density radius, Hamiltonian error below the production limit, agreement
of power and torque across the resolution pair, an absolute resolution-pair
mean-eccentricity mismatch no larger than 0.02, and a minimum binary
separation greater than two Plummer radii. The compact 12-core-radius box also
requires a doubled-box control before these cases can be promoted to a
production release.

The eccentricity limit is absolute because eccentricity is an interpolation
axis and a fractional measure is undefined for circular binaries. The value
0.02 is one fifteenth of the design's minimum 0.3 spacing between eccentricity
planes, so a matched-separation bin cannot silently compare different local
eccentricity states. The eight-orbit moving-block bootstrap caps its block at
half the available cycles, guaranteeing at least two independent block
lengths instead of producing a zero-width interval when `N=8`.

No workflow may append `n=1024` automatically. If an adjacent higher-resolution
pair still fails a production gate, or if the calculations have insufficient
common `(separation,eccentricity)` support, the affected bin remains
`unresolved/censored`. A further calculation requires a separately reviewed
resource and resolution design; absent that evidence the runtime must not
interpolate or extrapolate through the missing bin.

The 28 designed q-e runs have completed their seed, Torch, and response stages
under `/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/`. The earlier
`qe_extension_evaluation.json` compares several obsolete coarse/fine pairs and
must not be used as the final acceptance decision. Reassess the finest adjacent
pair of every manifest case from the existing diagnostics with
`scripts/reassess_qe_extension.py`. Its output records candidate matched bins
but never promotes them to a production table without the doubled-box control:

```bash
python scripts/reassess_qe_extension.py \
  --manifest results/wave_calibration_qe_extension/run_manifest.csv \
  --torch-root /gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/torch \
  --output-dir /gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/reassessment_<unique-id> \
  --profile-id boey2025
```

The reassessment first writes into a unique sibling staging directory and
publishes the complete assessment by rename. A caught failure removes only
that owned staging directory, while an already published or concurrently
created output directory is preserved. This makes a failed assessment safe to
retry under a new output name; it does not make interrupted wave integrations
restartable by itself.

The completed 2026-10-03 assessment at
`/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/reassessment_20261003_finest_v1/assessment.json`
used source revision `a8a875d` and has SHA-256
`1c2f6b5617dfdd9cabdff2ba8b552607394c30a11375ba2d3aa1a1a14dec0c34`.
All ten finest-adjacent comparisons are censored under the prescribed eight
separation bins and eight complete orbits per run per bin: eight retain no
matched bin, and two equal-mass circular small-separation cases have no common
resolved-separation support. No candidate or production calibration row was
admitted. The `q=0.3, e=0.3, a/r_c=0.05` coarse run has no complete orbit before
its first instantaneous two-cell underresolution crossing. The seven other
zero-bin cases have only 11–12 initially resolved complete orbits per run,
distributed across eight separation bins; their Hamiltonian conservation
checks pass, but the matched-bin sample-size requirement does not.
The reproducible read-only occupancy audit is

```bash
python scripts/audit_qe_bin_occupancy.py \
  --assessment /gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/reassessment_20261003_finest_v1/assessment.json
```

It verifies the original diagnostic hashes before counting orbits. In the
seven common-support zero-bin cases, the largest count shared by both members
of any prescribed bin is only 1–3 orbits, versus eight required. Filling all
eight bins would require at least 64 resolved complete orbits **per run** even
before any rate, eccentricity, softening, conservation, or box-size gate. This
is a necessary counting bound, not a prediction that a longer integration
would keep every orbit spatially resolved or yield converged rates.

As a **design diagnostic only**, recomputing the same completed trajectories
with one broad separation bin retains a bin in four of the ten cases. This
post-hoc change does not alter the registered eight-bin assessment, does not
establish resolved dependence on separation, and cannot release table rows.
The reproducible exploratory command is

```bash
python scripts/audit_qe_bin_occupancy.py \
  --assessment /gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/reassessment_20261003_finest_v1/assessment.json \
  --exploratory-one-bin
```

Three of those four broad bins still fail the 20-percent spatial systematic
limit for orbital power or total wave-energy rate; the circular equal-mass
case also fails orbital torque. Only `qe_q100_e060_a020` passes those
single-bin numerical gates, and it remains **unreleased** because the binning
was selected after examining the pilot and no doubled-box control exists.
The next calibration design must specify its separation-bin widths and orbit
budget prospectively, demonstrate a shared resolved interval for both members
of every resolution pair, and include a doubled-box control. Until those gates
pass, the runtime must report these q-e-small-separation domains as
uncalibrated/censored.

For a new registered design, the resolution pair and same-cell-size doubled-box
control must use identical physical separation-bin edges. The comparison CLI
accepts `--separation-bin-edges-pc` (comma-separated numbers) together with a
matching `--separation-bins` count. It records the fixed edges and omits a bin
unless the sampled mean separations in both calculations span its edges with
the required complete orbits. This establishes common sampling support only;
it does not by itself establish doubled-box agreement or authorize a
production calibration release.
Once both summaries exist, compare them read-only with
`scripts/assess_qe_box_control.py --profile-id PROFILE --resolution-pair PAIR.json
--doubled-box BOX.json`. The reference of the box comparison must be the
twice-larger box at unchanged cell size and softening, and its other run must
be the resolution pair's fine run. Both comparisons must use the same fixed
edges, SMBH/soliton initial conditions, numerical settings, and measured
fine-run rates. The result reports candidate or censored bins only;
`production_calibration_row_admitted` stays false until release provenance and
the full calibration gates are implemented and verified.
The assessor requires matching timestep factor, particle RK4 substeps, backend,
kinetic-phase layout, and wave-buffer lifetime across the coarse, fine, and
doubled-box runs. A
missing legacy layout tag cannot be mixed with the current separable-axis
layout, even though the two implement the same spectral drift mathematically.
The command also recomputes each fixed-bin comparison from its named raw
orbit and conservation diagnostics, verifies their checksums before and after
the audit, and refuses any diagnostic larger than 16 MiB on the login node.
An edited or stale comparison JSON is therefore not sufficient evidence of
box agreement.
Adding `--candidate-output PATH.json` creates a non-production candidate
package with only box-supported rows, both comparison checksums, and the
verified raw-input checksums. It refuses to overwrite an existing path and
still does not produce a runtime-loadable calibration table.

The corresponding resource-design calculation (which launches no solver) is

```bash
python scripts/plan_qe_followup_resources.py \
  --manifest results/wave_calibration_qe_extension/run_manifest.csv \
  --cases results/wave_calibration_qe_extension/physical_cases.csv \
  --assessment /gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/reassessment_20261003_finest_v1/assessment.json
```

With `--assessment`, the planner rechecks the pilot's named raw-diagnostic
hashes and reports each case's initially resolved orbit count and common
separation support alongside the resource bound.  In the current pilot, seven
cases have common support but no eight-orbit fixed bin; two equal-mass
small-separation cases have no common support, and the `q=0.3, e=0.3,
a/r_c=0.05` coarse run has no initially resolved complete orbit.  Thus a
longer duration is not a sufficient prescription for any case.  The latter
three require a changed resolution/initial-condition design before duration
can matter.  The report is a design diagnostic, not a registered set of bin
edges or authorization to launch or release a calibration row.

An additional hash-verified audit of **all 18 adjacent pilot pairs**, not
only the finest pair per case, is recorded at
`/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/adjacent_pilot_audit_20261003_v1.json`
(SHA-256 `32ded31033798e4ea6601419e61baaca6c4f6fc5f8f2b944d6f7933d3e7c3281`).
It rechecked the 28 named completed runs and their diagnostic inputs before
and after the audit. Eight pairs have common resolved-separation support but
zero bins with eight complete orbits per run; six have no common support, and
four lack even two initially resolved complete orbits in one member. In
particular, the n=256/512 a/r_c=0.20 pairs with e=0 or 0.3 have common
support and about 12 initially resolved orbits in each run, whereas the
n=128/256 alternatives do not offer a usable common interval. A new n=256/384
pair with n=512/768 same-cell-size box controls is therefore a *prospective
resource design to test*, not an observed convergence result: its n=384
support and long-duration behavior are unmeasured, and n=512 wave-response
work remains subject to the project approval rule. The smaller-separation
and e=0.6 cases also need separate resolution/initial-condition redesign;
this audit does not justify promoting the n=256 route to them.

For a **new** q/e campaign, record reviewed fixed physical edges, the
resolution pair, the same-cell-size doubled-box control, and a necessary
duration before running any seed or wave calculation:

```bash
python scripts/register_qe_followup_design.py \
  --case-id qe_q100_e060_a020 \
  --cases results/wave_calibration_qe_extension/physical_cases.csv \
  --manifest results/wave_calibration_qe_extension/run_manifest.csv \
  --coarse-resolution 512 --fine-resolution 768 \
  --separation-bin-edges-pc '<reviewed-low>,<reviewed-high>' \
  --duration-myr '<reviewed-duration>' \
  --output '<new-design-path>.json'
```

The placeholders must be replaced by a reviewed design; this is not a
ready-to-run command. The writer refuses overwrite and hashes the exact
physical-case and run-manifest CSVs. Commit the design before launching new
calculations. The PyUL seed runner accepts `--qe-design`,
`--qe-design-manifest`, and `--qe-design-role` (`coarse`, `fine`, or
`doubled_box_control`); it refuses mismatched geometry and stamps the verified
design identity into the seed metadata. The short PyUL initial-state seed is
not required to cover the planned orbit budget; the Torch evolution is, and
is rejected if its requested duration is shorter than the registered plan.
For a separate follow-up run manifest, mark each row
`requires_qe_design=true` and invoke `plan_wave_calibration_runs.py` with
`--qe-design PATH` and the design-bound manifest and physical-case CSVs. The
planner then emits seed commands with the design and role arguments, uses the
registered duration rather than the pilot target duration for Torch evolution,
and refuses missing or mismatched design bindings on existing seeds and
restarts. It only prints commands; it does not submit jobs or certify their
resource use. Without `--qe-design`, a marked follow-up manifest fails closed.
The Torch runner re-verifies that binding before evolving the wave, including
on restart. For a new comparison, pass `--qe-design`, `--qe-design-cases`,
and `--qe-design-manifest` to `summarize_pyul_convergence.py`; it obtains the
bin edges and orbit minimum from the bound design instead of accepting
post-hoc CLI values. A design file and its hashes alone cannot establish
that it predates the calculations: retain the pre-run Git commit and the
run's stamped metadata. The doubled-box assessor rechecks the design against
both comparison summaries and their raw run metadata; a registered and an
unregistered comparison, or two different designs, cannot be combined. This
path still does not authorize a GPU run or a
calibration release. The present 128–432 GiB doubled-box estimates require
an independently reviewed resource/solver design.
The seed and comparison path refuse a doubled-box role when its uniform-grid
estimate exceeds the design's declared reference GPU memory. Raising that
declaration without an actual capacity and peak-memory review is not a
resource validation.

The first prospective follow-up triplet is registered in
`results/wave_calibration_qe_followup_q100e000/` (design SHA-256
`23ded7852f5d4b371257254c6b5b72ffdac2b15afff4051e5fdcd1ec1d413c0d`).
It tests `q=1, e=0, a/r_c=0.20` with a fresh `n=256/384` resolution pair and
an `n=768` same-cell-size doubled-box control. Its single fixed physical bin
is `[0.430, 0.438] pc`, prospectively chosen inside the pilot `n=256/512`
common resolved interval `[0.428703, 0.439566] pc`; that pilot interval is
planning evidence, not evidence that the unmeasured `n=384` or doubled-box
trajectories will cover the bin. The registered 0.10 Myr duration is 36.59
initial Kepler periods, compared with the necessary eight complete orbits in
the bin. Neither nominal period count nor elapsed duration guarantees eight
*resolved* orbits in that fixed bin. This one-plane feasibility triplet does
not complete the required q/e and smaller-separation table extension.

The `n=768` control has 52.8 pc box length and the same 0.06875 pc cell size
as the `n=384` fine run. The 16-array estimate is 54 GiB, below the design's
80 GiB reference but **not** a measured peak or a capacity clearance. Prior
`n=768` work measured 52.32 GiB of allocated device memory for another box;
reserved CUDA memory, FFT workspace, concurrent users, long-run peak, and
the CPU wave-response memory still require a new resource review. The
design-bound planner reports three missing seeds and launches nothing. No
seed, Torch, or response job is authorized by this registration; the project
approval rule for heavy wave-response work remains in force. A failed common
bin, conservation, spatial-rate, or doubled-box gate leaves this plane
uncalibrated/censored and cannot release a production row.

A bounded read of the completed pilot
`qe_q100_e060_a020_n768/torch_run_summary.json` (SHA-256
`18f5b62137871479998720faede8bc65df2836797d9a4075b8e2b15ce1868a76`)
records 170,880 wave steps in 55,069.24 s over 0.032799 Myr and
56,174,844,416 peak allocated GPU bytes (52.32 GiB). Linear projection to
the new 0.10 Myr plan is 46.64 h on the same effective solver/device setup,
uncomfortably close to the cluster's 48 h job limit; a bounded checkpointed
multi-job continuation is required, not one job assumed to finish. This is
only a timing projection: the doubled box changes physical boundary
conditions and the new solver revision's long-run cost is unmeasured.
The existing `n=768` complex-wave snapshot is 6.75 GiB by file size; 17 such
snapshots would occupy about 114.75 GiB before checkpoints, logs, or other
outputs. The planned save count and scratch availability must be checked
against the effective command before submission. Scaling the observed
`n=512` wave-response RSS of about 37 GB by cell count gives roughly 125 GB
for `n=768`, not a measured memory bound; require a single-threaded,
snapshot-at-a-time response preflight with an explicit Slurm memory request.

At the initial Kepler period, the **necessary** 64-orbit duration for complete
coverage of eight bins is 0.1749 Myr at `a/r_c=0.20`, 0.0618 Myr at 0.10, and
0.0219 Myr at 0.05. These are not sufficient durations: the smallest
eccentric case already has about 29 nominal periods in its 0.01 Myr pilot but
zero initially resolved coarse-grid orbits. Doubling the box while preserving
cell size raises the finest members of the present pairs from `n=512` to
`n=1024` or from `n=768` to `n=1536`. The existing 16-array uniform-grid model
estimates 128 and 432 GiB, respectively, above an 80 GiB reference device.
The hash-verified completed Torch summaries measure peak **allocated** device
memory of 14.50 GiB at `n=512` and 52.32 GiB at `n=768`. Cubic projection
to the doubled grids gives 116.0 and 418.5 GiB; the resource planner reports
these independently of the 16-array estimate. Neither projection includes
CUDA reserved memory, FFT workspace, or a newly measured large-grid peak.
These estimates are planning warnings, not measured peak GPU footprints. A
new Torch solver revision separates the kinetic phase by axis and releases
previous-step wave buffers before its FFT. Its large-grid peak has not yet
been measured, so the historical peak measurements and cubic projections
cannot be treated as a capacity clearance for that revision. A
new release therefore needs an independently reviewed solver/resource design
for its doubled-box controls; merely extending the current run duration or
submitting the same grids on an 80 GiB card cannot establish the required
calibration.

An A/B **n=256, one-wave-step memory probe** used the same
`qe_q100_e000_a020_n256` PyUL seed, 0.000001 Myr duration, nine RK4 substeps,
Torch 2.11.0+cu130, and an NVIDIA A10 on syn04. The baseline source at
`391f6e9` (Slurm job `411438`) peaked at 1,946,683,904 allocated device
bytes (1.813 GiB); the separable-phase and early-buffer-release source at
`aacbc15` (job `411436`) peaked at 1,544,036,864 bytes (1.438 GiB). The
matched reduction is 383.994 MiB, or 20.7% of the baseline peak. Both runs
completed exactly one wave step. Their source snapshots, run metadata, and
summaries are in
`/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/torch/memory_probe_n256_baseline_411438/`
and `/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/torch/memory_probe_n256_411436/`.
This is PyTorch peak **allocated** memory, not CUDA reserved memory or total
device occupancy. The saved one-step energy and mass arrays differ by at most
`3.9e-16` relative to each array's maximum absolute value; the saved SMBH
state differs by at most `1.01e-14` in its saved units and the radial-density
profile by at most `8.8e-16` relative to its maximum. These small diagnostics
do not validate the full wave state or longer trajectories. The one-step
n=256 probe also does not establish longer-run peaks or capacity at n=512,
768, or 1024; those larger grids remain unmeasured for the new solver revision.

A further allocation change computes `|psi|^2` directly from its real and
imaginary components, including the spectral-power diagnostic. With the same
one-step seed and A10, revision `93694f1` (Slurm job `411440`) peaked at
1,409,825,280 allocated bytes (1.313 GiB), 127.994 MiB below job `411436`.
The saved energy arrays agree to at most `2.6e-16` relative to their maximum
absolute values, and the saved SMBH state differs by at most `1.94e-18` in
its saved units. The source snapshot and summary are in
`/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/torch/memory_probe_n256_density_v2_411440/`.
The combined measured one-step reduction relative to baseline job `411438`
is 27.6%, but remains an n=256 allocation result only. Subsequent runs stamp
`wave_density_layout=real_imag_addcmul_v1`; restarts and q/e box controls
reject mismatched density layouts.

The compact-mass Plummer field now uses 32-cell x slabs and one in-place
inverse-square-root slab rather than full-grid distance and inverse-distance
temporaries. Revision `8650d9a` (Slurm job `411454`) completed the same n=256
one-step probe on syn04. Its saved energy, mass, SMBH-state, and radial-density
arrays match job `411440` exactly, while its peak allocated memory was
unchanged at 1,409,825,280 bytes. The source snapshot and summary are in
`/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/torch/memory_probe_n256_slab_v1_411454/`.
The equal global peak implies that some other stage dominates this particular
n=256 run; it does not show that the slab change clears larger grids.

Optional stage-level CUDA profiling identified the actual n=256 peaks in
revision `8b9b17a` (job `411465`): the initial and final saved-wave energy
evaluations reached 1,409,825,280 allocated bytes; the old potential kicks
were within 6,144 bytes of that peak. Forward FFT, inverse FFT, Poisson, and
compact-potential stages were below that peak. Revision `6e2cb3c` (job
`411477`) reduced each potential-kick
peak to 1,141,383,680 bytes by forming the complex phase in a single buffer,
but the saved-energy peak remained unchanged. Revision `ed252c2` (job
`411481`) released the unused total-potential array before each saved-energy
evaluation, reducing the measured whole-run peak to 1,275,607,552 bytes
(1.188 GiB). The saved energy, mass, SMBH-state, and radial-density arrays
match job `411477` exactly. Profiles and source snapshots are in the
respective `memory_probe_n256_stage_v1_411465`,
`memory_probe_n256_phase_v1_411477`, and
`memory_probe_n256_lifetime_v1_411481` directories under the Torch q/e
scratch root above. The present n=256 one-step result is 34.5% below the
`391f6e9` baseline peak, but neither cubic projection nor the one-step profile
is a capacity clearance for n=1024; FFT workspaces, CUDA reserved memory,
long-run behavior, and the actual large-grid allocation remain unmeasured.

Revision `7436839` (Slurm job `411487`) computes the potential-energy and
wave-mass scalars first, releases the full-grid density during the saved
kinetic-energy FFT, and rebuilds density for output. Its n=256 one-step peak
fell to 1,141,383,680 allocated bytes (1.063 GiB), now at the potential
kicks; the saved kinetic-energy stage peaked at 1,120,416,768 bytes. Energy,
mass, and SMBH-state arrays match job `411481` exactly, while the saved
radial-density profile differs by at most `1.8e-16` relative to its maximum.
The source snapshot, summary, and stage profile are in
`/gpfs/kjhan/FDM_TOY_RESULTS/qe_extension/torch/memory_probe_n256_energy_v1_411487/`.
The measured allocated-memory reduction from baseline job `411438` is 41.4%
for this one-step n=256 test. That ratio is not an n=1024 capacity prediction
or a q/e calibration result.

The guarded q-e queue uses observed device-memory profiles rather than the
uniform-grid estimate alone. An `n=512` stage requires at least 22 GiB total
and 21 GiB free: this admits a 23,028 MiB A10 while retaining more than 2 GiB
above the measured 19,326 MiB solver footprint. An `n=768` stage requires an
80 GiB-class device and at least 64 GiB free. It is therefore
`memory_capacity_censored` on an A10 and must be routed to an 80 GiB GPU.
Insufficient free memory on an otherwise supported GPU is instead the
retryable status `preflight_memory_busy`.

For a shared 80 GiB GPU, `--wait-until-gpu-empty` polls only the NVML compute
process list. It neither scans the host process table nor signals a process
that already owns the GPU. After the device becomes empty, the free-memory
floor is checked and the normal mid-run collision guard remains active. An
optional `--gpu-empty-timeout-seconds` converts an overlong wait into the
retryable `gpu_empty_wait_timeout` status.

```bash
python scripts/run_guarded_qe_plan.py \
  --manifest results/wave_calibration_qe_extension/run_manifest.csv \
  --cases results/wave_calibration_qe_extension/physical_cases.csv \
  --initial-root /path/to/pyul_initial \
  --torch-root /path/to/torch \
  --pyul-path /path/to/PyUL_NBody \
  --log-root /path/to/logs \
  --run-id qe_q100_e060_a020_n768 \
  --gpu-index 0 \
  --wait-until-gpu-empty \
  --gpu-empty-timeout-seconds 43200
```

New live-wave metadata records `mass_ratio_q`, `initial_eccentricity`,
`semi_major_axis_pc`, and `initial_separation_pc`. The eccentricity must come
from this provenance because the apocentre initializer includes the soliton's
differential force; a two-body osculating inversion would not reproduce the
input eccentricity in that extended potential.

## Public live-wave solver check

The public moving-particle solver cited by Koo et al. is
[`Sifyrena/PyUL_NBody`](https://github.com/Sifyrena/PyUL_NBody). The adapter
records its exact commit and keeps the checkout external to this repository.
It also supplies compatibility aliases required by current SciPy and IPython
without editing the upstream source.

Each run records the exact PyUL revision, the FDM_TOY revision, the adapter
arguments, and whether the FDM_TOY working tree was clean at launch. A dirty
launch remains identifiable and must be accompanied by the corresponding
local changes before the result enters a calibration release.

```bash
python scripts/run_pyul_wave_case.py \
  --pyul-path /path/to/PyUL_NBody \
  --case-id koo_mbh1.0e8 \
  --resolution 512 \
  --duration-myr 0.000001 \
  --box-pc 40 \
  --save-number 1 \
  --output results/pyul_smoke

python scripts/analyze_pyul_wave_run.py \
  results/pyul_smoke/koo_mbh1.0e8_n512
```

For a calculation that spans at least one complete orbit, the secular analysis
compares successive equal-phase boundaries and writes the orbit-resolved power,
torque, and energy ledger:

```bash
python scripts/analyze_pyul_secular_exchange.py \
  results/pyul_orbit_pilots/koo_orbit1/koo_mbh1.0e8_n128
```

The global ledger may be plotted separately from the orbital separation with

```bash
python scripts/plot_energy_exchange.py \
  results/pyul_orbit_pilots/koo_orbit1/koo_mbh1.0e8_n128 \
  --output figures/energy_exchange.pdf
```

The wave intrinsic energy, wave--SMBH interaction energy, SMBH centre-of-mass
kinetic energy, and combined-Hamiltonian residual remain separate columns. This
prevents a transient change in the interaction energy from being labelled as
irreversible wave heating.

The one-dimensional density states provide a low-volume diagnostic of the
long-term rearrangement of the wave:

```bash
python scripts/analyze_pyul_line_density.py \
  results/pyul_wave/koo_mbh1.0e8_n128
```

The saved line passes through the soliton centre along the simulation y axis.
It is neither a spherical density profile nor a measure of deposited energy.

A short calculation with frequent three-dimensional fields may be compared
against the same interval of a longer calculation with sparse orbital output:

```bash
python scripts/compare_pyul_overlap.py \
  results/pyul_orbit_pilots/koo_orbit1/koo_mbh1.0e8_n128 \
  results/pyul_long_term/koo_1myr_s2048/koo_mbh1.0e8_n128
```

The comparison interpolates the longer calculation to the short output times.
Its reported difference therefore includes cadence interpolation and does not
replace spatial or temporal convergence.

After several runs pass the Hamiltonian and spatial-resolution tests, their
orbit-resolved measurements can be expressed in common physical units with

```bash
python scripts/build_wave_exchange_table.py \
  /path/to/run_a /path/to/run_b \
  --output results/wave_exchange_table.csv
```

The table normalizes time by the dynamical time of the soliton, energy by
`G M1 M2 / r_core`, and angular momentum by
`mu sqrt(G M_binary r_core)`. The table retains every orbit and records the
separation in cell units. It does not select a fitting sample automatically.
When three-dimensional wave states are available, the same row includes the
soliton-to-binary centre offset, central density, measured half-density radius,
outer mass and intrinsic energy, complex density multipoles, and radial mass
and Schrödinger-energy fluxes. These quantities are normalized by the soliton
and orbital scales of that calculation. The measured half-density radius
receives a separate spatial-resolution flag and is not interpreted when it
spans fewer than two cell widths. The binary radial, tangential, and normal
unit vectors define the orbital frame at the matched wave state. The table
records the complex multipoles in both the simulation frame and this orbital
frame. Each invariant amplitude agrees before and after the rotation.

Long calculations may retain frequent SMBH and scalar-energy states while
writing the three-dimensional density and wavefunction less often. For example,
the following choice writes 2,049 orbital states and 33 three-dimensional field
states at matching times:

```bash
python scripts/run_pyul_wave_case.py \
  --pyul-path /path/to/PyUL_NBody \
  --case-id koo_mbh1.0e8 \
  --resolution 128 \
  --duration-myr 1 \
  --save-number 2048 \
  --save-3d-number 32 \
  --time-step-factor 0.5 \
  --output results/pyul_wave
```

The time-step factor multiplies the solver limit. A value below one increases
the number of wave steps and should be varied together with the spatial
resolution when the Hamiltonian error exceeds the adopted tolerance. PyUL
interprets `--rk-steps 36` as 36 Runge--Kutta stages per wave step, which gives
nine RK4 substeps for the SMBHs. The run metadata records both quantities. The
adapter limits FFT and NumExpr threads to the CPUs assigned to the batch task
and records this thread count with the numerical setup.

### Immediate convergence sequence

The first convergence comparison repeats the initial `0.1083984375 Myr` of
the Koo anchor. This interval ends just before the `128^3` trajectory falls
below two cell widths.

| calculation | cells | wave step factor | SMBH RK4 substeps | estimated wave steps |
|---|---:|---:|---:|---:|
| long-run prefix | 128 | 1.0 | 9 | 6,833 |
| wave-step repeat | 128 | 0.5 | 9 | 13,666 |
| particle-step repeat | 128 | 1.0 | 18 | 6,833 |
| spatial repeat | 256 | 1.0 | 9 | 27,330 |

The two `128^3` repeats separate the wave integrator from the SMBH integrator.
The `256^3` calculation then halves the cell width from 0.3125 to 0.15625 pc.
The orbital power, torque, Hamiltonian residual, separation history, and local
wave state must agree before any row enters the physical calibration.

Completed numerical variants are compared over the time interval for which
every calculation keeps the binary separation above two cell widths:

```bash
python scripts/summarize_pyul_convergence.py \
  baseline=/path/to/baseline_run \
  wave_dt_half=/path/to/wave_step_repeat \
  particle_rk18=/path/to/particle_step_repeat \
  spatial_n256=/path/to/spatial_repeat \
  --output results/pyul_convergence/koo_prefix_convergence_summary.json
```

The common-interval rates retain reversible orbital-phase and interaction-energy
variations. They are therefore assessed together with the orbit-averaged power
and torque. The comparison records numerical differences but does not assign an
automatic convergence decision.

The completed Koo-anchor smoke calculations give:

| resolution | cell size [pc] | duration [yr] | wave-mass error | total-energy error | error / transferred energy |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.3125 | 100 | `1.3e-15` | `3.7e-5` | `4.00e-2` |
| 256 | 0.15625 | 1 | `2.2e-16` | `6.9e-9` | `1.92e-2` |
| 512 | 0.078125 | 1 | `1.1e-15` | `3.1e-9` | `5.17e-3` |

The 512-cell calculation matches Koo's `0.08 pc` spatial resolution. It is a
short high-resolution conservation test, not a measurement of the decay time.

### Torch GPU continuation and convergence results

The Torch backend continues a three-dimensional PyUL initial state with the
same dimensionless Schrödinger--Poisson convention. It retains the resolved
wave force on the SMBHs and the SMBH force on the wave; analytic FDM drag
remains disabled. The output layout is compatible with the conservation,
orbit-averaging, line-density, sparse-wave, and movie diagnostics above.

```bash
python scripts/launch_torch_wave_case.py /path/to/pyul_initial_run \
  --output /path/to/torch_run \
  --duration-myr 1.0 \
  --save-number 2048 \
  --movie-frame-number 360 \
  --save-3d-number 32 \
  --checkpoint-every-saves 32 \
  --rk4-substeps 9 \
  --device cuda:0
```

The launcher writes `torch_solver_provenance/manifest.json` as soon as the run
metadata appears. The manifest preserves the uncommitted Torch runner and
operators, takes committed dependencies from the recorded adapter revision,
and records SHA-256 hashes. A restart fails before integration if these sources
do not match the preserved copies. Existing calculations whose uncommitted
sources predate their metadata can be frozen explicitly with

```bash
python scripts/snapshot_torch_provenance.py /path/to/torch_run
```

The long `512^3` Koo-anchor calculation completed `1 Myr` with analytic FDM
drag disabled. The separation changed from `0.9 pc` to `0.2874 pc`, and the
maximum Hamiltonian error was `0.2035%` of the transferred energy. Its binary
remained above two cell widths throughout the calculation. The long `256^3`
calculation first crossed that numerical boundary at `0.85498 Myr`.

The completed common-window comparisons are:

| comparison and reference | common interval [Myr] | separation difference / initial | mean orbital power | mean orbital torque | final-window orbital power | final-window orbital torque |
|---|---:|---:|---:|---:|---:|---:|
| `512^3`, half wave step vs unit step | 0.1000 | `-0.405%` | `-0.661%` | `-0.661%` | `-1.13%` | `-0.744%` |
| `512^3`, 18 vs 9 SMBH RK4 substeps | 0.1000 | `-0.0516%` | `-0.0093%` | `-0.0360%` | `-0.0215%` | `-0.0464%` |
| `256^3` vs `512^3`, unit step | 0.1000 | `+2.58%` | `-5.79%` | `-3.59%` | `-3.35%` | `-4.49%` |
| `384^3` vs `512^3`, unit step | 0.8550 | `-1.52%` | `+2.04%` | `+0.189%` | `-23.9%` | `-50.9%` |
| `256^3` vs `512^3`, unit step | 0.8550 | `+1.54%` | `+5.44%` | `+0.076%` | `+32.9%` | `+14.5%` |
| `256^3`, half vs unit wave step | 0.8550 | `-1.52%` | `-0.966%` | `-0.511%` | `-14.3%` | `-16.6%` |

The final-window estimates use complete orbits near the end of the common
resolved interval. Their broad variations exceed the cumulative separation
and interval-mean differences. The maximum Hamiltonian errors over transferred
energy are `0.6829%` for the unit-step `256^3` calculation and `0.2587%` for
its half-step repeat, both below the adopted one-percent limit. The numerical
ledger and cumulative inspiral therefore pass. The local common-time rates
remain sensitive to the orbital and wave phases because the trajectories reach
slightly different separations at the same time.

The completed `384^3` calculation reached `1 Myr` with a final separation of
`0.2923 pc`. The binary remained above two cell widths, and the Hamiltonian
error was `0.3581%` of the transferred energy. A state-matched comparison bins
complete orbits by physical separation and requires at least eight orbits from
every calculation in a retained bin. The `384^3` and `512^3` calculations give
the following absolute fractional differences across seven retained bins:

| matched-separation rate | median difference | maximum difference |
|---|---:|---:|
| orbital power | `2.82%` | `13.28%` |
| orbital torque | `3.66%` | `10.88%` |
| total wave-energy rate | `5.07%` | `16.97%` |

The corresponding median differences between `256^3` and `512^3` are
`16.51%`, `15.61%`, and `16.38%`. The high-resolution pair therefore supports
a provisional secular calibration. Each separation bin retains its measured
spatial systematic, whose maximum reaches about `17%`. The coarse calculation
does not set the local transfer coefficient.

The current dimensionless exchange tables retain all 81 Koo rows and all 454
Boey rows. The binary-separation, measured half-density-radius, and Hamiltonian
tests admit 81 Koo rows and 433 Boey rows to the provisional secular sample.
The wave-state timing test admits 48 rows from each family to the
phase-dependent sample, for 96 rows in total. These flags identify eligible
measurements rather than final fitted coefficients. The completed Koo
resolution comparison supplies a measured spatial systematic. The Boey mass
trend remains provisional until a matching spatial comparison is available.

## Interaction-energy convention

The coupled Hamiltonian is

\[
H_{\rm tot}=K_\psi+\frac{1}{2}\int\rho\Phi_\psi\,dV
+\int\rho\Phi_{\rm BH}\,dV+K_{\rm BH}+U_{\rm BH-BH}.
\]

The wave--SMBH cross term is therefore

\[
E_{\psi-{\rm BH}}=\int\rho\Phi_{\rm BH}\,dV,
\]

counted once and without a factor of one half. This grid integral is the
authoritative interaction energy for the calibration calculations. The
alternative sum `M_BH Phi_wave(x_BH)` is not included in the Hamiltonian.
PyUL fixes the mean periodic wave potential to zero, whereas its Plummer SMBH
potential is zero at infinity. The absolute values of the two estimators thus
use different potential zeros.

A controlled single-SMBH calculation separates this gauge offset from a force
error:

```bash
python scripts/audit_interaction_energy.py \
  --resolutions 64 128 256 512 \
  --output results/interaction_energy_audit
```

The unaligned central energies differ by 17--19 percent at every resolution.
After one constant offset is removed, the maximum difference in the energy
change decreases from 18.4 percent at 64 cells to 0.293 percent at 512 cells.
The maximum force difference decreases from 26.5 to 1.62 percent. The
non-convergent absolute difference is therefore a potential-gauge difference;
the remaining convergent difference comes from softening, finite differencing,
and interpolation.

The 512-cell audit passes the adopted limits of 0.5 percent for interaction-
energy changes and 2 percent for the force. Long calculations must preserve
the explicit ledger

\[
\Delta E_{\rm orb}+\Delta K_{\rm COM}+\Delta E_{\psi,\rm intrinsic}
+\Delta E_{\psi-{\rm BH}}=0,
\]

where `E_wave,intrinsic` contains wave kinetic and self-gravity energies. The
cross term remains a separate transient reservoir. Energy deposited in the
wave is measured from the intrinsic wave energy and outgoing fluxes, not by
assigning `-Delta E_orb` locally without the other ledger terms. The total
Hamiltonian error must remain below one percent of the measured energy
transfer, not merely below a tolerance normalized by the much larger soliton
binding energy.

Long calculations must additionally save the density or wavefunction needed
for radial fluxes, central-density evolution, the radial monopole profile, and
the complex `l = 1, 2` density multipoles. The individual complex coefficients
retain the phase that a rotationally invariant mode fraction discards. Only
after the literature anchors pass the energy, force, and resolution tests
should the 87-case physical grid be executed.
