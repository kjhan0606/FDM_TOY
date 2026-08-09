# Horizon Run 5 SMBH capture and dual AGN analysis

## Data provenance

The comparison sample is derived from the legacy HR5 sink tree

```text
/home/kjhan/BACKUP/GalFinder/SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/BinarySMBH/Sink_Merging_Tree.dat.Updated
```

The file contains 1,688,677 fixed sink histories over 278 outputs. The extraction
finds 576,278 sink disappearances with assigned receivers. Of these events,
576,036 have positive masses for both objects at the last resolved output.

The receiver-selection calculation is reconstructed from `mkmerging.c`. A sink
that is present at output `i-1` and absent at output `i` is matched to a sink at
`i`. The search radius starts at the nearest-survivor distance, increases by
0.002 cMpc, and stops at 0.5 cMpc. Within each radius, the most massive survivor
with at least twice the disappearing sink mass is selected. The assigned
receiver is therefore a legacy numerical association, not a recorded physical
binary partner.

## Output-step convention

Every event is an interval. No single column silently mixes the two boundaries.

| Quantity | Output | Interpretation |
|---|---:|---|
| disappearing sink mass, position, velocity | `i-1` | last resolved state |
| receiver mass used for mass ratio and chirp mass | `i-1` | two-object mass estimate before disappearance |
| receiver identifier | selected at `i` | reconstructed from the surviving population |
| receiver mass at assigned output | `i` | post-disappearance diagnostic, not used in chirp mass |
| numerical capture time | between `i-1` and `i` | interval-censored event |
| assigned capture time and redshift | `i` | upper boundary of the interval |

The catalog stores both history indices, output numbers, redshifts, cosmic
times, and the interval width. A later delay condition such as capture within
1 Gyr is accepted only when the upper boundary meets the condition. If the
1 Gyr boundary falls inside the output interval, the case is recorded
separately.

## Regeneration

```bash
python scripts/extract_hr5_capture_catalog.py \
  '/home/kjhan/BACKUP/GalFinder/SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/BinarySMBH/Sink_Merging_Tree.dat.Updated'

python scripts/reproduce_hr5_original_figures.py --rebuild-cache
python scripts/analyze_hr5_dual_agn.py
```

The full tree can also be rebuilt from consecutive MkAGN snapshots with
`scripts/build_hr5_binary_capture_history.py`. The local MkAGN directory does
not contain all outputs from 19 through 296. It has a consecutive sequence from
20 through 26 and sparse later snapshots. The builder refuses missing output
intervals by default so that an incomplete sequence cannot be interpreted as a
physical disappearance history.

## Original figures

`scripts/reproduce_hr5_original_figures.py` redraws the 13 figures in the
earlier draft with external PDF graphics. It also writes
`hr5_original_figure_validation.json`. The validation compares the regenerated
values with numerical statements in the draft. The original point series are
not available in a machine-readable table, so an exact pointwise residual
cannot be claimed.

The regenerated seed-rate peak, massive-SMBH captured-mass fraction, and
fixed-delay rate normalization satisfy the reported or visually readable
benchmarks. The legacy cumulative count for the $10^{6}\,M_\odot$ chirp-mass
threshold does not satisfy the value stated in the earlier draft. The
validation file keeps the discrepancy visible rather than tuning a hidden
normalization.

Figures 12 and 13 use the physical all-sky differential comoving volume with
`Omega=4 pi sr`, equivalent to 41,253 square degrees. The earlier draft wrote an
additional factor of one third in the shell expression. The validation output
retains that legacy normalization for comparison, but it is not used in the new
PTA source counts. Counts for another solid angle follow
`N(Omega)=N_all-sky Omega/(4 pi)`.

## Dual AGN selection and measurements

The dual AGN sample uses the MkAGN snapshots at outputs 89, 117, and 296. Both
members must satisfy the adopted bolometric-luminosity threshold, and their
three-dimensional physical separation must lie between 0.5 and 30 pkpc. The
reported pair fraction is `N_pair/N_active_AGN`.

| redshift | luminosity threshold [erg s^-1] | active AGN | pairs | pair fraction | later direct capture | upper boundary within 1 Gyr |
|---:|---:|---:|---:|---:|---:|---:|
| 2.848 | 1e43 | 77,805 | 2,532 | 0.03254 | 2,147 | 2,005 |
| 2.848 | 1e44 | 21,864 | 410 | 0.01875 | 364 | 341 |
| 1.499 | 1e43 | 29,446 | 364 | 0.01236 | 295 | 248 |
| 1.499 | 1e44 | 13,254 | 126 | 0.00951 | 100 | 88 |
| 0.625 | 1e43 | 8,179 | 39 | 0.00477 | censored | censored |
| 0.625 | 1e44 | 2,624 | 12 | 0.00457 | censored | censored |

The output-296 linkage is right-censored because no later HR5 output is present.
The direct-capture columns describe the reconstructed numerical association.
They do not establish physical SMBH coalescence.

## Literature comparison

Published dual AGN fractions depend strongly on luminosity, projected or
three-dimensional separation, obscuration, and the denominator used for the
fraction. Direct numerical comparison therefore requires matched selections.

- [Liu et al. 2011](https://arxiv.org/abs/1104.0950) measured kpc-scale AGN
  pairs in SDSS and reported a few-percent pair fraction over 5--100 kpc.
- [Van Wassenhove et al. 2012](https://arxiv.org/abs/1111.0223) showed that
  simultaneous activity becomes more common at small separations, while the
  observable phase remains short.
- [Steinborn et al. 2016](https://arxiv.org/abs/1510.08465) measured a roughly
  one-percent dual AGN fraction at redshift two in Magneticum for a bolometric
  threshold of 1e43 erg s^-1.
- [Capelo et al. 2017](https://arxiv.org/abs/1611.09244) quantified how viewing
  angle, luminosity threshold, and separation change the observable duration
  in controlled galaxy-merger calculations.
- [Rosas-Guevara et al. 2019](https://arxiv.org/abs/1805.01479) connected dual
  AGN below 30 kpc to the EAGLE galaxy-merger population and found percent-level
  fractions for a hard X-ray selection.
- [Volonteri et al. 2022](https://arxiv.org/abs/2112.07193) followed Horizon-AGN
  duals to later black-hole mergers and found a strong but selection-dependent
  connection between 4--30 kpc pairs and later merger events.
- [Chen et al. 2023](https://arxiv.org/abs/2208.04970) found a three-percent
  dual fraction at redshifts two to three in ASTRID and showed that many bright,
  close pairs merge within 500 Myr.
- [Puerto-Sanchez et al. 2025](https://arxiv.org/abs/2411.15297) compared nine
  cosmological simulations with a common selection and found that predicted
  dual fractions and number densities retain substantial inter-simulation
  variation.

The HR5 values lie within the broad range of these studies. This agreement is a
selection-level comparison only. Projection, obscuration, host association,
and the numerical capture prescription still require matched tests before a
physical conclusion is drawn.
