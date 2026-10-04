# SMBH orbital energy and FDM wave backreaction

The shrinking SMBH orbit loses both energy and momentum. A useful delay model
must specify the corresponding transfer to FDM. A decrease in binary energy
without a corresponding FDM gain does not describe a closed physical system.

## Energy and momentum conservation in v0.1

For the analytic force on SMBH `i`, `F_DF,i`, the opposite momentum transfer is

\[
\frac{d\mathbf P_{\rm FDM}}{dt}=-\sum_i\mathbf F_{{\rm DF},i}.
\]

Two energy rates are kept separately:

\[
\dot E_{\rm FDM,lab}=-\sum_i\mathbf F_{{\rm DF},i}\cdot\mathbf v_i,
\]

\[
\dot E_{\rm FDM,exc}=-\sum_i\mathbf F_{{\rm DF},i}\cdot
(\mathbf v_i-\mathbf u_{\rm FDM}).
\]

The first completes the mechanical energy balance in the simulation frame. The second measures the
energy delivered to wakes/internal wave excitations in the local FDM rest
frame. Their difference is the bulk-flow work associated with the transferred
momentum. They are identical for the v0 static background, `u_FDM=0`.

The orbital integration records the instantaneous power, the instantaneous
force, the cumulative energy in both frames, and the cumulative momentum. A
normalized Gaussian profile distributes the exchange over a numerical mesh
while preserving the volume integrated energy and momentum source terms.

## Static-background breakdown

The density profile supplies the virial binding energy estimate `|W|/2`. If cumulative
rest-frame injection exceeds 10% of this scale, the result receives
`STATIC_SOLITON_BACKREACTION`. Injection above the full binding energy also
receives `SOLITON_DISRUPTION_POSSIBLE`. The unchanged density and potential are
not self-consistent past the indicated limits.

This test is necessary but not sufficient. Deposited energy may rearrange the
central density locally before it becomes a large fraction of the global
binding energy.

## Coupling to an evolving wavefunction

The preferred fully coupled mode evolves the FDM wavefunction with the moving
SMBH potential,

\[
i\hbar\partial_t\psi=
\left[-\frac{\hbar^2\nabla^2}{2m}+m(\Phi_{\rm FDM}+\Phi_{\rm BH})\right]\psi,
\]

while the SMBHs respond to the same evolved FDM density. The wake,
gravitational cooling, soliton deformation, and energy transfer then arise from
one Hamiltonian. The coupled calculation must test conservation of SMBH kinetic
energy, SMBH mutual interaction energy, wave kinetic energy, FDM self-gravity,
and SMBH and FDM interaction energy.

The adopted live-wave Hamiltonian is

\[
H_{\rm tot}=K_\psi+\frac{1}{2}\int\rho\Phi_\psi\,dV
+\int\rho\Phi_{\rm BH}\,dV+K_{\rm BH}+U_{\rm BH-BH}.
\]

Thus `integral rho*Phi_BH dV` is the interaction energy and is counted once.
It has no factor of one half. A point value such as
`sum M_BH*Phi_wave(x_BH)` can differ by a potential-gauge constant and is used
only to test forces and resolution after one reference offset is removed.

For a binary, the energy exchange is diagnosed with

\[
\Delta E_{\rm orb}+\Delta K_{\rm COM}+\Delta E_{\rm FDM,intrinsic}
+\Delta E_{\rm FDM-BH}=0.
\]

The intrinsic FDM energy contains wave kinetic and self-gravity terms. The
interaction energy remains separate because it changes as the SMBHs move even
before energy propagates into a wake. Therefore `-Delta E_orb` must not be
identified directly with local wave heating. The deposition measurement uses
the intrinsic energy change together with radial energy flux and the evolving
wave modes. The controlled numerical audit and acceptance limits are recorded
in [`merge_calibration.md`](merge_calibration.md).

## Experimental time-centred periodic TSC diagnostic

The `periodic_tsc_strang` mode deposits each SMBH on the periodic mesh with
TSC weights and derives the force on that SMBH from the gradient of the same
wave–SMBH interaction energy. Each step applies a joint half potential kick
to the wave phase and SMBH velocities, a full spectral-wave and ballistic-SMBH
drift, and a second joint half kick at the updated density and positions.
This replaces the earlier diagnostic sequence in which SMBHs were advanced
through an end-of-step wave field. The direct SMBH–SMBH force remains Plummer
softened and nonperiodic; the mode stops if the pair leaves its stated spatial
domain or a step undersamples an orbit or a TSC cell crossing.

The runner records wave mass, the separate binary and wave energy terms, their
combined Hamiltonian, and the initial SMBH-potential phase jump between
neighbouring cells. It binds restarts to the reference configuration and
solver-source hashes. The small-grid tests cover interaction reciprocity,
forward/backward reversal, second-order refinement of a coupled short orbit,
energy bookkeeping, and restart equivalence. These checks justify a short
diagnostic prefix, not a calibrated orbital-decay row. A short n256 time-step
ladder now measures Hamiltonian drift, separation convergence, and total
momentum drift (below). Spatial-resolution convergence, physical n256
cell-offset sensitivity, full-orbit and pericentre coverage, and the
interaction-softening comparison remain open.
No result from this mode may enter the FDM delay table until those gates pass.

A toy translation check is a warning about the remaining spatial gate, not an
error bar for n256. At 16³, shifting the whole initial condition by exactly
one mesh cell reproduces the final wave and SMBH state to 2×10⁻¹² in
code-state values. Shifting by half a cell changes the final SMBH separation
by about −6.30×10⁻⁵ pc after 0.075 Myr. Reducing the wave step from 16 to 32
steps changes the unshifted separation by 1.99×10⁻⁷ pc, while the half-cell
difference changes by only 2.29×10⁻⁸ pc. The unshifted separation itself
decreases by 8.64×10⁻³ pc in this toy interval.

A common 64/128-step ladder at 16³, 32³, and 64³ gives half-cell-minus-node
differences of −6.305×10⁻⁵, −1.783×10⁻⁵, and −5.349×10⁻⁶ pc at 128 steps.
Their magnitudes fall by factors of 3.54 and 3.33 as the cell is halved.
At 64³, doubling the step count changes this offset difference by
3.66×10⁻⁸ pc, much less than the 32³-to-64³ offset difference. This verifies
short-prefix node-to-midpoint sensitivity along this toy sequence only. It
does not prove convergence of the separation itself, since TSC source
smoothing changes with cell size. Intermediate offsets, a wave-induced-signal
control, longer orbits, and physical n256 spatial checks remain required
before calibration.

The source-bound n256, q=0.3, e=0.3, a=0.2 pc Strang ladder from Slurm job
412096 covers only the first ten common saves (about 17 years). At wave-step
factors 1, 1/2, 1/4, and 1/8, the maximum prefix Hamiltonian error divided
by the component-transfer scale is respectively 0.01845, 0.00576, 0.00152,
and 0.000385. The coarsest case fails the registered 0.01 diagnostic limit;
the others pass that *short-prefix* energy check. The last-save separation
differences show approximately second-order time-step refinement. Wave mass
and grid/point interaction reciprocity close to numerical precision. The
trace summary is
`/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/periodic_tsc_strang_step_trace_v1/strang_step_trace_summary_v1.json`
(SHA-256 `4ea3b8d2143298af3f0d16b39fae91d2672fce69d48a6a3a5079f392819cce5f`).

The independent spectral-wave-plus-SMBH momentum audit does *not* converge
to zero as the time step shrinks: its quadratic-extrapolated total-momentum
residual is 0.02327 of the exchanged momentum, above the registered 0.001
diagnostic threshold. This is a short-prefix, fixed-spatial-grid inference,
not a claim about a long-time or spatial limit. The source-bound summary is
`/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/periodic_tsc_strang_step_trace_v1/momentum_refinement_summary_v1.json`.
The TSC interaction-energy gradient is not the force conjugate to the grid's
spectral momentum operator. `periodic_tsc_strang_momentum` therefore tests a
separate, spectral-gradient force candidate while retaining the same wave
source, energy ledger, and fail-closed domain checks. Its instantaneous
grid/particle momentum forces cancel in a controlled operator test, but the
discrete wave evolution may still have aliasing and the new force is not the
gradient of the interaction energy. A 16³ wave-only phase-kick control has
zero grid self-force to 10⁻¹² code units yet changes spectral wave momentum
by about 6.9×10⁻⁶ code units in one half step; this explicitly separates
the operator identity from the time-integrated result. Neither exact momentum
conservation nor acceptable energy drift is assumed; both must be measured
before this mode can contribute any FDM delay calibration.

The matched n256 spectral-momentum run (Slurm 412175) now provides that
short-prefix comparison. At wave-step factors 1, 1/2, 1/4, and 1/8, the
endpoint total-momentum residual divided by exchanged momentum is
1.58×10⁻⁵, 1.75×10⁻⁵, 1.79×10⁻⁵, and 1.80×10⁻⁵. Quadratic extrapolation from
the two finest steps gives 1.80×10⁻⁵, compared with 2.33×10⁻² for the
energy-gradient Strang force, an improvement by a factor of about 1,292 on
this fixed grid and interval. Successive momentum differences have measured
orders 1.98 and 2.03. The residual does not vanish with time-step refinement;
the operator-level cancellation has not produced exact wave-plus-SMBH
momentum conservation. The maximum prefix Hamiltonian error divided by
component transfer is 0.01844, 0.00575, 0.00151, and 0.000381; the coarsest
step still fails the registered 0.01 energy diagnostic. Peak allocated GPU
memory was 1.68 GB, versus 1.15 GB for the earlier energy-gradient run.
The source-bound comparison verifies identical initial-wave and SMBH-state
hashes, physical configuration, and the same maximum-prefix energy-error
definition in both runs. It is
`/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/periodic_tsc_strang_momentum_step_trace_v1/momentum_candidate_comparison_v2.json`
(SHA-256 `be3f363f07118771c8f1369effe3885cc5252258fe01b9fcfccdfa065a7a580d`).
This is evidence about a 17-year prefix, not an orbital-decay or coalescence
time. The momentum diagnostic samples only the endpoint, so it does not
bound intermediate drift. The cause of the residual floor is not established.
Physical n256 spatial, offset, softening, and full-orbit gates remain open;
no q/e/a calibration row is released.

An n256 wave-only null control (Slurm 412226) evolved the same initial wave
over the same 17-year interval and four time-step levels, but omitted the SMBH
potential. Its CPU-verified endpoint momentum changes are 2.85×10⁻⁸,
6.16×10⁻⁸, 1.22×10⁻⁷, and 2.61×10⁻⁷ code units, each below the approximately
1.14×10⁻⁶ code-unit spectral summation floor. The coupled residuals are about
4.67–5.30×10⁻⁴ code units. Thus the *unperturbed* wave-only drift is not a
resolved explanation for the coupled residual. The coupled wave acquires
SMBH-induced phase structure, however, so the null is neither an upper nor a
lower bound on its self-gravity error. The finest-level CPU audit is
`/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/wave_only_momentum_null_v1/f0125/cpu_audit_v1.json`
(SHA-256 `5fe370bcd8614c5b355ea45e2f08f98bbccdbee6cc94dfb7c80f25ee6300d660`).
A coupled-state attribution test was therefore required to measure the
self-gravity and compact-potential phase-kick contributions, including
intermediate maxima, before assigning a physical origin to the remaining
momentum floor.

The coupled-state factorized-kick replay is now complete for those four n256
levels (Slurm 412276 and 412288). It reproduces each saved binary state and
the final wave checkpoint; the maximum relative wave replay difference is
6.1×10⁻¹⁵. A separate CPU endpoint audit agrees with the replay's total
momentum change within 2.7×10⁻⁸ code units. The source-bound summary checks
each half-kick's *interaction-only* wave-plus-SMBH impulse separately from
the joint momentum change, verifies both phase-factor orderings, and refuses
calibration release. It is
`/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/coupled_kick_attribution_v1/summary_v1.json`
(SHA-256 `8073f6ed75be61cad81915a4e6cc578bd3f37de40eec4e9e2692c2f17f3f9c7c`).

| Step factor | Endpoint residual norm, code | Largest interaction-only half-kick defect, code | Largest relative vector gap between either interaction ordering and endpoint |
| --- | ---: | ---: | ---: |
| 1 | 4.668×10⁻⁴ | 3.479×10⁻⁵ | 1.12×10⁻⁴ |
| 1/2 | 5.151×10⁻⁴ | 1.791×10⁻⁵ | 1.93×10⁻⁴ |
| 1/4 | 5.274×10⁻⁴ | 8.774×10⁻⁶ | 3.27×10⁻⁴ |
| 1/8 | 5.304×10⁻⁴ | 4.321×10⁻⁶ | 5.89×10⁻⁴ |

The residual-norm refinement-difference ratios are 3.94 and 4.12, consistent
with a nonzero fixed-grid limit and a roughly second-order temporal correction
on this short interval. The interaction kick pair accounts for nearly all of
the measured endpoint residual in either factor order. This is an operator
attribution, **not** a demonstrated physical origin for the residual. The
individual summed self-kick and kinetic-drift contributions are at most
2.76×10⁻⁷ code units and lie below the approximately 5.69×10⁻⁷
single-evaluation spectral
summation heuristic; their signs and nonzero magnitudes are unresolved. The
older `maximum_half_kick_action_reaction_defect_code` fields in the four replay
JSON files include the self-kick and should be read as *joint* half-kick
momentum changes; the source-bound summary provides the interaction-only
values above. Alternate phase-factor ordering changes the attribution at the
1.5–3.3×10⁻⁸ code-unit level without changing this limited conclusion.
Independent Opus 5.5 code review found no algebraic or replay falsifier but
could not directly read the GPFS result files; its follow-up assessment used
the supplied numerical values and likewise did not certify the tiny terms or
full-orbit accuracy. A precision cross-check of stage momenta, physical n256
spatial/offset/softening gates, and a full-orbit conservation gate remain
required before any q/e/a calibration row can be released.

A follow-up n256 replay (Slurm 412319) changed only the order of the
floating-point summation used to evaluate spectral wave momentum at each
stage, using the same wave and SMBH trajectories at factors 1 and 1/8. The
original closure check was initially too strict when large individual wave
and SMBH momenta nearly cancelled; its tolerance is now scaled to the
individual operands. That correction changes no evolution operator or
saved trajectory. The final corrected replay has wave checkpoint agreement
better than 6.1×10⁻¹⁵ relative. The interaction-pair attribution changes
by 1.89×10⁻⁸ and 2.48×10⁻⁸ code units in the self-first ordering, respectively
4.1×10⁻⁵ and 4.7×10⁻⁵ of the endpoint residual. Across both phase orders,
the largest change is 1.50×10⁻⁷ code units, 2.83×10⁻⁴ of the finest-level
residual. Therefore the measured dominance of the interaction pair survives
this reduction-order perturbation; the much smaller individual self and
drift attributions remain unresolved. This test shares the same FFT and does
not establish a rigorous numerical error bound or spatial convergence. Its
source-bound comparison is
`/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/coupled_kick_precision_v3/comparison_v1.json`
(SHA-256 `f27a16b4f5603972a18039e1082494319e18b0d4b260ca1fb8baff1f4df290a6`).

An n256 spatial-anchoring check has now translated the *same* designated
q030/e030/a020 initial wave, SMBHs, and declared soliton centre along x by
one whole cell or half a cell. The one-cell seed uses an exact array roll;
the half-cell seed uses a periodic Fourier phase. Both parent and derived
arrays, units, source, and physical parameters are hash-bound. Four matching
17-year coupled prefixes at factors 1 and 1/8 ran under Slurm 412333. The
one-cell control reproduces the baseline SMBH trajectory after undoing the
translation to at most 2.3×10⁻¹³ code units and the final wave to 1.15×10⁻¹⁴
relative. Its maximum separation difference is 3.33×10⁻¹⁶ pc. This tests
discrete translation equivariance, not physical resolution.

The half-cell-minus-baseline final separation is −3.047×10⁻¹⁰ pc at factor 1
and −2.934×10⁻¹⁰ pc at factor 1/8; their difference is 1.13×10⁻¹¹ pc. The
finest-level offset is 4.91×10⁻⁶ of the *total* baseline separation change
over this prefix, which includes the direct SMBH orbit and cannot be called
an FDM-only decay uncertainty. The initial Nyquist-plane power fraction is
3.15×10⁻¹² in both seeds, and the high-frequency shell at or above 0.375 cycles
per cell contains 2.29×10⁻¹⁰ of their power. Yet the half-cell translation
changes the initial *discrete* Hamiltonian by +1.177×10⁴ code units, almost
entirely through the mesh wave–SMBH interaction; it is not an identical
discrete initial state. The maximum Hamiltonian-error/component-transfer
ratios are 0.01844 (baseline) versus 0.01898 (half-cell) at factor 1, and
0.0003806 versus 0.0003948 at factor 1/8. The coarse case fails the unchanged
0.01 diagnostic limit. All four wave-mass relative errors are below 2×10⁻¹⁴.
The source-bound CPU audit is
`/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/periodic_offset_step_trace_v1/audit_v2.json`
(SHA-256 `1e6237b68cb697c94c91468ca2fa4f9c4da54ef0268f36330ab71749749899d4`).
Independent Opus 5.5 review agreed that this is a fixed-grid, short-prefix
diagnostic only and identified parent-seed, unit, initial-energy and Nyquist
checks, which the second audit includes. It did not independently read the
GPFS outputs. Other sub-cell phases, multi-axis combinations, interaction
softening and spatial-resolution ladders, longer orbital coverage, and full conservation
remain open; this test does not release a calibration row.

The same source-bound half-cell shift was also applied separately along y
and z at the finest short-prefix step (Slurm 412348). Relative to the
unshifted f0125 run, the final separation changes by −2.934×10⁻¹⁰ pc for x,
+3.354×10⁻¹² pc for y, and +5.128×10⁻¹² pc for z. The x response is 57.2
times the larger transverse response in this one setup. The initial
discrete Hamiltonian shifts by about +1.177×10⁴ code units for x but only
+3.186×10³ for y and z. The three maximum Hamiltonian-error/transfer ratios
are respectively 0.000395, 0.000304, and 0.000289, all below the short-prefix
0.01 diagnostic threshold. These comparisons bind each translated input to
the common parent and check the same numerical solver, units, masses,
provenance and all saved SMBH states. The source-bound directional audit is
`/gpfs/kjhan/FDM_TOY_RESULTS/qe_followup_q030e030_a020_v1/periodic_transverse_offset_step_trace_v1/directional_audit_v1.json`
(SHA-256 `9d8ff34b8c8a33816dc7e07f9b217412eab6e7dad110067426ac8e06de4da1bb`).
The axis dependence does not define a statistical spatial-error bar: each
direction has only one sub-cell phase on one fixed grid and a 17-year prefix.

## Double-counting prohibition

Two calculations remain physically distinct.

1. The analytic drag calculation applies the calibrated drag and transfers the
   opposite energy and momentum to FDM within the stated backreaction limits.
2. The evolving wave calculation measures the force from the FDM density and
   omits the analytic drag and any additional energy injection.

Applying analytic drag to the SMBHs while also letting the same resolved wake
decelerate them counts the interaction twice. Likewise, injecting arbitrary
"heat" into `|psi|^2` is not acceptable. The evolution must preserve FDM mass,
apply the required momentum, add the required energy, and pass a total-energy
convergence test.

## Orbit-averaged element rates below the resolved scale

The calibration calculations will return the secular binary power
`dE_orb/dt` and the torque along the orbital angular momentum
`d|L_orb|/dt`. For a Keplerian internal binary,

\[
E_{\rm orb}=-\frac{G M_1 M_2}{2a},\qquad
L_{\rm orb}=\mu\sqrt{G(M_1+M_2)a(1-e^2)}.
\]

The corresponding rates are

\[
\dot a=\frac{2a^2}{G M_1 M_2}\dot E_{\rm orb},
\]

\[
\frac{d e^2}{dt}=(1-e^2)
\left(\frac{\dot a}{a}-2\frac{\dot L_{\rm orb}}{L_{\rm orb}}\right).
\]

The second expression remains regular at `e=0`. The scalar `de/dt` follows by
division by `2e` only for nonzero eccentricity. The wave receives
`-dE_orb/dt` and `-d|L_orb|/dt`.

A single perturbation that rotates at angular frequency `Omega` obeys

\[
\dot E_{\rm orb}=\Omega\dot L_{\rm orb}.
\]

The orbit-resolved analysis therefore records
`dot E_orb/(Omega dot L_orb)`. A value near unity permits the energy and
angular momentum to enter the wave through one rotating pattern. A significant
departure requires additional orbital harmonics, a radial response, or both.
The measured ratio constrains the wave source and is not imposed in advance.

The local FDM state also affects the secular response. Zhang et al. (2026,
<https://arxiv.org/abs/2602.11512>) find that an initially unperturbed
equal-mass binary mainly excites the quadrupole and does not show sustained
stone-skipping rebounds. A seeded dipole changes the long-term motion in the
same fully coupled setting. The calibration state must therefore retain the
complex dipole and quadrupole amplitudes, or an equivalent local wave
description, rather than assume that masses and orbital elements determine a
unique decay rate. The diagnostic records multipoles about both the wave centre
and the binary centre. Their difference separates an internal distortion of
the soliton from a displacement of the wave relative to the binary.

## Form of the calibrated transfer

The transfer model must retain a secular part and a coherent response. In
dimensionless variables, a suitable expansion is

\[
\mathcal P=\mathcal P_0(q,e,f_\mathrm{bin},a/r_c,\eta_\mathrm{SP})
+\sum_{\ell m}\mathop{\rm Re}
\left[C^P_{\ell m} A^{\rm orb}_{\ell m}\right],
\]

\[
\mathcal T=\mathcal T_0(q,e,f_\mathrm{bin},a/r_c,\eta_\mathrm{SP})
+\sum_{\ell m}\mathop{\rm Re}
\left[C^T_{\ell m} A^{\rm orb}_{\ell m}\right].
\]

Here `f_bin=M_binary/M_soliton` and
`eta_SP=hbar^2/(G m^2 M_soliton r_c)`. The coefficients
`A_lm^orb` denote the local complex density multipoles expressed relative to
the instantaneous orbital frame. Their phases are physical. Replacing them by
the invariant `l=1` and `l=2` amplitudes would erase whether a coherent mode
removes energy from the binary or returns energy to it.

The leading unperturbed term obeys the symmetry of the binary. An equal-mass
circular binary has no internal dipole about its centre of mass and first
drives a rotating quadrupole. A displaced soliton or an existing dipole breaks
that symmetry. The fitted dipole response must therefore be conditioned on the
measured wave state instead of being assigned to every equal-mass binary.

Only rows that satisfy the spatial and Hamiltonian tests enter a provisional
fit. A physical calibration additionally requires agreement between spatial
resolutions, wave time-step factors, and SMBH RK4 substep counts. The fit
returns signed orbital power and torque. It does not force either quantity to
remain negative because coherent wave modes may return energy or angular
momentum over part of their oscillation. Long-time decay follows only after
averaging over the relevant orbital and soliton-mode periods.

The exchange tables distinguish two row selections. A secular row requires
the binary separation and the measured half-density radius to remain above two
cell widths, and the initial resolved interval must pass the Hamiltonian limit.
A phase-dependent row must also have a saved three-dimensional wave state
within one half of the local orbital period. Sparse wave states that fail this
time-offset limit remain available for secular power and torque but do not
supply a complex mode phase.

The fitted power and torque determine the orbital update. The resolved
multipole potential supplies part of the opposite wave exchange. A separate
mode source supplies only the residual defined above. This sequence preserves
the Hamiltonian constraint and prevents the coherent response from being
applied twice.

The minimal mode decomposition assigns `Omega dot L_wave` to a pattern rotating
at the orbital frequency and assigns

\[
\dot E_{m=0}=\dot E_{\rm wave}-\Omega\dot L_{\rm wave}
\]

to a radial mode with no angular momentum. The function
`decompose_wave_mode_exchange` evaluates this split. A negative radial
remainder rules out one orbital-frequency pattern plus a positive-energy radial
mode and requires other orbital harmonics or an interval with energy returned
from the wave. The decomposition diagnoses the required modes but does not
alter the wavefunction.

## Multipole potential below the resolved binary scale

When the resolved calculation replaces the two SMBHs by one particle at their
centre of mass, the particle already supplies the monopole potential of the
binary. The internal orbit may expose its time-dependent multipoles through

\[
\delta\Phi_{\rm bin}=\Phi_1+\Phi_2-\Phi_{M_1+M_2}.
\]

The function `unresolved_binary_potential_correction` evaluates this difference
with the same Plummer length and periodic convention for all three terms. The
dipole vanishes when the internal positions are centred on the binary centre of
mass. The leading far-field term then scales as the rotating quadrupole.

Adding this Hermitian potential to the Schrödinger equation preserves FDM mass
and lets the wave respond through its phase. The resulting work and torque must
be measured. If the internal orbit also follows calibrated power and torque,
the work produced by the multipole correction counts toward the required wave
increment. Applying the full calibrated increment in addition would count the
same transfer twice.

For one finite interval, let `Delta E_res` and `Delta L_res` denote the work
and torque already received by the resolved wave from the multipole potential.
The remaining increments are

\[
\Delta E_{\rm rem}=-\Delta E_{\rm orb}-\Delta E_{\rm res},
\qquad
\Delta\mathbf L_{\rm rem}=-\Delta\mathbf L_{\rm orb}
-\Delta\mathbf L_{\rm res}.
\]

The function `residual_wave_exchange` evaluates these quantities and tests
closure. A calibrated mode source acts only on the remainder. If the resolved
multipole has already supplied the target exchange, the additional source is
zero. The remainder may change sign when a coherent wave mode returns energy
to the orbit.

These equations apply to orbit-averaged internal elements. The smooth soliton
potential may produce reversible changes in osculating Kepler elements over an
orbit. Such changes must not enter the fitted dissipative rates. The function
`keplerian_exchange_rates` implements the conversion after the resolved wake
and reversible cross-energy reservoir have been separated from the secular
exchange.

For a finite subgrid interval, `advance_keplerian_exchange` updates orbital
energy and angular momentum first and then recovers the new semi-major axis,
eccentricity, and orbital phase. The energy and angular momentum increments of
the wave are exactly opposite to the orbital increments. A step that produces
an unbound energy, non-positive angular momentum, or invalid eccentricity is
rejected rather than projected onto an artificial bound orbit.
