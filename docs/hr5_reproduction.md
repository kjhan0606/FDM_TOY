# Horizon Run 5 binary-capture and active SMBH pair analysis

## Data provenance

The comparison sample is derived from the legacy HR5 sink tree

```text
/home/kjhan/BACKUP/GalFinder/SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/BinarySMBH/Sink_Merging_Tree.dat.Updated
```

The file contains 1,688,677 fixed sink histories over 278 outputs. The extraction
finds 576,278 binary captures from disappearing SMBH particles. Of these events,
576,036 have positive masses for both objects at the last resolved output.

The companion-selection calculation is reconstructed from `mkmerging.c`. A sink
that is present at output `i-1` and absent at output `i` is matched to a sink at
`i`. The search radius starts at the nearest-survivor distance, increases by
0.002 cMpc, and stops at 0.5 cMpc. Within each radius, the most massive survivor
with at least twice the disappearing sink mass is selected. The assigned
surviving SMBH is therefore an inferred companion rather than the partner
recorded for the merger of two sink particles. The catalog keeps `receiver_id`
as the historical field name.

## Output-step convention

Every event is an interval. No single column silently mixes the two boundaries.

| Quantity | Output | Interpretation |
|---|---:|---|
| disappearing sink mass, position, velocity | `i-1` | last resolved state |
| mass of the assigned surviving SMBH used for mass ratio and chirp mass | `i-1` | two-object mass estimate before disappearance |
| identifier of the surviving SMBH | selected at `i` | inferred from the surviving population |
| mass of the surviving SMBH at the assigned output | `i` | post-disappearance diagnostic, not used in chirp mass |
| binary-capture time | between `i-1` and `i` | interval-censored event |
| assigned capture time and redshift | `i` | upper boundary of the interval |

The catalog stores both history indices, output numbers, redshifts, cosmic
times, and the interval width. A later delay condition such as capture within
1 Gyr is accepted only when the upper boundary meets the condition. If the
1 Gyr boundary falls inside the output interval, the case is recorded
separately.

## Validation of assigned companions

`scripts/validate_hr5_capture_receivers.py` reads the assigned surviving SMBH at both
interval boundaries and compares its last-resolved state with the disappearing
sink. The two-body diagnostic uses

```text
v_escape = sqrt(2 G (M_disappearing + M_surviving) / separation)
```

All 576,278 associations satisfy the 0.5 cMpc search limit and the companion
mass factor of two because those conditions define the legacy selection. Of
576,277 associations with finite last-resolved phase-space quantities, only 41
have `v_relative <= v_escape`. The fraction is `7.1146e-5`. The median
separation at the last common output is 4.614 pkpc, and the median relative
speed is 218.2 km/s. Also, 6.54 percent of the inferred captures assign the
same surviving SMBH to at least one other disappearance in the same output.

The locally available consecutive MkAGN outputs span 20 through 26. Only one
sink disappears across those intervals. Reconstructing its companion from the
two adjacent MkAGN snapshots reproduces the legacy tree assignment exactly.
This single event tests the implementation but does not independently validate
the full association catalogue.

The phase-space result separates two measurements. Sink disappearance and its
output interval come directly from the histories. The identity of the assigned companion, pair
mass, chirp mass, and relative orbit remain conditional on the legacy
association until direct records of the merger of two sink particles are restored. The validation
products are written under `results/hr5/receiver_validation/`.

## Regeneration

```bash
python scripts/extract_hr5_capture_catalog.py \
  '/home/kjhan/BACKUP/GalFinder/SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/BinarySMBH/Sink_Merging_Tree.dat.Updated'

python scripts/reproduce_hr5_original_figures.py --rebuild-cache
python scripts/validate_hr5_capture_receivers.py
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
variance, output-time uncertainty, or ambiguity in the assignment of the
surviving SMBH.

Figures 8 and 9, which appear as Figures 11 and 12 in the current JKAS draft,
show the same count-statistical uncertainty for each measured redshift-bin
rate. The vertical bars span the 16th through 84th percentiles of 200
independent Poisson realizations. The measured rates remain at the symbol
positions. The numerical values are written to
`hr5_fixed_delay_rate_bootstrap.csv`.

Figures 9 and 10 use 6-point labels, ticks, panel letters, and legends after the
requested 40 percent reduction. Figure 3 uses 6-point axis labels and tick
labels while retaining 10-point panel letters and legends. The remaining
manuscript graphics use 10-point text, open geometric symbols, and a
color-vision-deficiency-safe palette. Line styles duplicate the color encoding
for grayscale reproduction. Legends remain inside the axes and occupy regions
separated from the measured curves and uncertainty intervals.

## Active SMBH pair selection and measurements

The active-pair sample uses the MkAGN snapshots at outputs 89, 117, and 296.
Both members of a dual AGN candidate satisfy the adopted bolometric-luminosity
threshold. Exactly one member of a single-AGN pair satisfies the threshold.
Every pair has a three-dimensional physical separation between 0.5 and
30 pkpc. Three fractional measures are retained because published measurements
use different denominators. `N_pair/N_AGN` counts pair edges.
`N_member/N_AGN` counts every unique active SMBH with an active companion. The
pure-dual measure removes members of connected systems containing three or
more active SMBHs.

| redshift | active AGN | dual pairs | number density [cMpc^-3] | pair/AGN | member/AGN | pure member/AGN |
|---:|---:|---:|---:|---:|---:|---:|
| 2.848 | 77,805 | 2,532 | 2.329e-4 | 0.03254 | 0.06110 | 0.05609 |
| 1.499 | 29,446 | 364 | 3.349e-5 | 0.01236 | 0.02343 | 0.02187 |
| 0.625 | 8,179 | 39 | 3.588e-6 | 0.00477 | 0.00954 | 0.00954 |

The table adopts `Lbol >= 1e43 erg s^-1`. The calculation also measures
`Lbol >= 1e44 erg s^-1` and hard-X-ray `L(2-10 keV) >= 1e42 erg s^-1`
selections. Eight equal slabs along the long axis give spatial jackknife
errors on the fiducial number densities of `1.743e-5`, `3.874e-6`, and
`5.808e-7 cMpc^-3` at the three redshifts. A separate comparison imposes
`M_BH >= 1e6 Msun` on both members. At redshift 2.848 it contains 387
dual-active and 574 single-active pairs. At redshift 1.499 it contains 295
dual-active and 1,336 single-active pairs.

The projected-selection calculation follows each physically associated
three-dimensional pair over 128 deterministic sightlines. It applies
`0.5 <= r_p <= 30 pkpc` and includes peculiar velocity plus Hubble flow in
`Delta v_los`. For the fiducial dual population, the mean retained fractions
are 0.888 and 0.990 at redshift 2.848 for velocity limits of 300 and 600 km/s.
The corresponding values at redshift 1.499 are 0.869 and 0.986. These fractions
measure the viewing-angle retention of three-dimensional pairs. They do not
include unrelated foreground or background objects in an observational
cylinder.

The controlled binary-capture comparison retains systems with exactly two
SMBHs in the mass-limited population. It standardizes `log10(M_primary)`,
`log10(mass ratio)`, `log10(separation)`, and
`log10(relative speed + 10 km/s)`, then finds a unique minimum-distance
assignment. At redshift 2.848, all 326 dual AGN candidates are matched to
326 of 470 single-AGN pairs. The maximum absolute standardized
mean difference decreases from 0.328 to 0.088. The interval-censored 1 Gyr
binary-capture fractions are 0.893--0.902 and 0.859--0.862. Their difference is
bounded by 0.031--0.043.

At redshift 1.499, all 243 dual AGN candidates are matched to 243 of 1,135
single-AGN pairs. The maximum absolute standardized mean
difference decreases from 0.486 to 0.031. The corresponding binary-capture
fractions are 0.749--0.753 and 0.683--0.695. Their difference is bounded by
0.053--0.070. The output-296 population is right-censored at the selection time
because no later HR5 output is present. These are inferred binary captures,
not direct records of the partner or physical coalescences.

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
`hr5_dual_offset_pairs_mbh_ge_1e6.csv`. Matched systems are written to
`hr5_dual_offset_matched_pairs.csv`. Columns whose historical names begin with
`receiver_` identify the surviving SMBH assigned by the distance and mass
criteria. They do not identify the partner selected by the simulation or
establish physical SMBH coalescence.

Output 296 at redshift 0.625 contains 8,179 SMBHs above the fiducial bolometric
limit. Their median Eddington ratio is 0.0410, and the central 68 percent
interval is 0.0158--0.271. The fraction above an Eddington ratio of 0.1 is
0.288. The luminosity comparison is written to
`hr5_eddington_luminosity_z0p625.pdf`.

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
- [Saeedzadeh et al. 2024](https://arxiv.org/abs/2403.17076) followed
  Romulus25 dual AGNs and found rapidly evolving, slowly evolving, and
  nonmerging histories.
- [Buttigieg et al. 2025](https://arxiv.org/abs/2504.17549) showed that
  premature black hole mergers in FABLE can precede completion of the host
  merger by several Gyr and can alter the predicted massive-binary population.
- [Chen et al. 2025](https://arxiv.org/abs/2512.16844) applied observational
  selection functions to ASTRID and found that roughly 30--70 percent of the
  selected dual AGNs coalesce within about 1 Gyr.

The HR5 values lie within the broad range of these studies. The common
luminosity and separation cuts allow a selection-level comparison. The missing
host association prevents the distinct-galaxy cut used in the recent
multi-simulation comparison. Obscuration, survey response, and the numerical
capture prescription remain separate physical uncertainties.
