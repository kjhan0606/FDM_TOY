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
