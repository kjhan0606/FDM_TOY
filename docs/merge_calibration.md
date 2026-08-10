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

## Public live-wave solver check

The public moving-particle solver cited by Koo et al. is
[`Sifyrena/PyUL_NBody`](https://github.com/Sifyrena/PyUL_NBody). The adapter
records its exact commit and keeps the checkout external to this repository.
It also supplies compatibility aliases required by current SciPy and IPython
without editing the upstream source.

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

The completed Koo-anchor smoke calculations give:

| resolution | cell size [pc] | duration [yr] | wave-mass error | total-energy error | error / transferred energy |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.3125 | 100 | `1.3e-15` | `3.7e-5` | `4.00e-2` |
| 256 | 0.15625 | 1 | `2.2e-16` | `6.9e-9` | `1.92e-2` |
| 512 | 0.078125 | 1 | `1.1e-15` | `3.1e-9` | `5.17e-3` |

The 512-cell calculation matches Koo's `0.08 pc` spatial resolution. It is a
short high-resolution conservation test, not a measurement of the decay time.

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
for radial fluxes, central-density evolution, and `l = 0, 1, 2` mode amplitudes.
Only after the literature anchors pass the energy, force, and resolution tests
should the 87-case physical grid be executed.
