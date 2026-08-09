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

`scripts/plot_hr5_capture_population.py` writes the capture-rate and chirp-mass
distribution to `hr5_capture_population.pdf`. Both horizontal axes use
`log10(1 + z_cap)`, and the two-dimensional counts use equal intervals in that
coordinate. The mass-ratio probability density is written separately to
`hr5_capture_mass_ratio.pdf`.

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

Figures 10 and 11 show count-statistical uncertainties in the fitted redshift
distribution. Each redshift-bin count is independently resampled from a
Poisson distribution 200 times. The four distribution parameters are refitted
for every realization. Symbols mark the bootstrap medians, and error bars span
the 16th through 84th percentiles. The calculation does not include cosmic
variance, output-time uncertainty, or ambiguity in the reconstructed capture
receiver.

Figures 8 and 9, which appear as Figures 11 and 12 in the current JKAS draft,
show the same count-statistical uncertainty for each measured redshift-bin
rate. The vertical bars span the 16th through 84th percentiles of 200
independent Poisson realizations. The measured rates remain at the symbol
positions. The numerical values are written to
`hr5_fixed_delay_rate_bootstrap.csv`.

All manuscript graphics use 10-point legend text, open geometric symbols, and
a color-vision-deficiency-safe palette. Line styles duplicate the color
encoding for grayscale reproduction. Legends remain inside the axes and occupy
regions separated from the measured curves and uncertainty intervals.

## Dual AGN selection and measurements

The dual AGN sample uses the MkAGN snapshots at outputs 89, 117, and 296. Both
members satisfy the adopted bolometric-luminosity threshold, and their
three-dimensional physical separation lies between 0.5 and 30 pkpc. Three
fractional measures are retained because published measurements use different
denominators. `N_pair/N_AGN` counts pair edges. `N_member/N_AGN` counts every
unique active SMBH with an active companion. The pure-dual measure removes
members of connected systems containing three or more active SMBHs.

| redshift | active AGN | dual pairs | number density [cMpc^-3] | pair/AGN | member/AGN | pure member/AGN |
|---:|---:|---:|---:|---:|---:|---:|
| 2.848 | 77,805 | 2,532 | 2.329e-4 | 0.03254 | 0.06110 | 0.05609 |
| 1.499 | 29,446 | 364 | 3.349e-5 | 0.01236 | 0.02343 | 0.02187 |
| 0.625 | 8,179 | 39 | 3.588e-6 | 0.00477 | 0.00954 | 0.00954 |

The table adopts `Lbol >= 1e43 erg s^-1`. The calculation also measures
`Lbol >= 1e44 erg s^-1` and hard-X-ray `L(2-10 keV) >= 1e42 erg s^-1`
selections. A separate comparison imposes `M_BH >= 1e6 Msun` on both members
before classifying dual and offset AGN. At redshift 2.848 this comparison
contains 387 dual and 574 offset pairs. At redshift 1.499 it contains 295 dual
and 1,336 offset pairs.

The projected-selection calculation follows each physically associated
three-dimensional pair over 128 deterministic sightlines. It applies
`0.5 <= r_p <= 30 pkpc` and includes peculiar velocity plus Hubble flow in
`Delta v_los`. For the fiducial dual population, the mean retained fractions
are 0.888 and 0.990 at redshift 2.848 for velocity limits of 300 and 600 km/s.
The corresponding values at redshift 1.499 are 0.869 and 0.986. These fractions
measure the viewing-angle retention of three-dimensional pairs. They do not
include unrelated foreground or background objects in an observational
cylinder.

The direct-capture distribution restricts the fiducial result to pure dual
systems. At redshift 2.848, 1,938 of 2,182 pure duals have a later direct
numerical capture. The interval-censored cumulative fraction at 1 Gyr lies
between 0.832 and 0.841. At redshift 1.499, 270 of 322 pure duals have a later
capture, and the 1 Gyr interval lies between 0.717 and 0.720. For the
mass-limited comparison, the 1 Gyr bounds are 0.845--0.855 for duals and
0.803--0.807 for offsets at redshift 2.848. They are 0.702--0.705 and
0.597--0.603 at redshift 1.499. The output-296 population is right-censored at
the selection time because no later HR5 output is present.

Neutral-hydrogen columns are available along six cardinal sightlines at
outputs 89 and 296. The analysis reports the fraction of active sightlines
above `N_H = 1e23` and `1e24 cm^-2`. It does not convert these columns into a
survey detection probability because that step requires a spectral and
instrument response model.

The local MkAGN products contain zero-valued galaxy identifiers and host mass
fields. The archived FoF/PSB paths point to a scratch directory that is no
longer present. The present sample therefore cannot separate same-galaxy and
distinct-galaxy pairs or construct a host-matched control population. The
number-density comparison with distinct-galaxy samples in the literature must
retain this limitation.

The output files are `hr5_dual_agn_summary.json`,
`hr5_dual_agn_capture_cdf.csv`, `hr5_dual_agn_pairs.csv`, and
`hr5_dual_offset_pairs_mbh_ge_1e6.csv`. The direct-capture columns describe the
reconstructed numerical association. They do not establish physical SMBH
coalescence.

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

The HR5 values lie within the broad range of these studies. The common
luminosity and separation cuts allow a selection-level comparison. The missing
host association prevents the distinct-galaxy cut used in the recent
multi-simulation comparison. Obscuration, survey response, and the numerical
capture prescription remain separate physical uncertainties.
