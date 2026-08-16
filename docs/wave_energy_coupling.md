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
