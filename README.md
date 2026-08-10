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
The transfer ledger is not yet a live wavefunction update; the coupling and
double-counting rules are documented in
[`docs/wave_energy_coupling.md`](docs/wave_energy_coupling.md).

The numerical capture interface, corrected Koo and Boey profile definitions,
live-wave parameter grid, and first PyUL_NBody resolution tests are documented
in [`docs/merge_calibration.md`](docs/merge_calibration.md). These tools leave
the separate lagRamses working tree unchanged.

To combine a sink time with all three physical intervals, use:

```bash
fdm-smbh-compose \
  --z-sink 1.0 \
  --fdm-summary results/case/summary.json \
  --kpc-to-pc-delay "100 Myr" \
  --gw-delay "3 Myr"
```

The command returns no `true_merge_time_myr` if an interval is missing,
invalid, or censored. Missing physics is never interpreted as zero delay.

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
cases, and the first live-wave resolution tests have been completed. Calibration
against their full simulated density histories, the numerical-radius-to-parsec
inspiral, long live-wave calculations, and the cosmological PTA population
remain future work.

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
