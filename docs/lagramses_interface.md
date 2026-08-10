# Future lagRamses interface

No lagRamses source file is changed at the present stage. The interface below
defines the physical state that must be available before the calibrated model
is inserted.

## State at numerical binary capture

lagRamses should write a binary-capture record immediately before `merge_sink`
compacts the FOF group. Differences between snapshots do not recover the exact
state at that instant.

The record requires

- the source revision, run identifier, step, scale factor, and cosmic time
- member identifiers, dynamical masses, black hole masses, positions,
  velocities, and spins
- the separation vector, relative velocity, centre-of-mass state, orbital
  energy, angular momentum, eccentricity vector, and osculating semi-major axis
- local gas, stellar, and FDM densities and velocities
- the fitted mass, core radius, centre, and bulk velocity of the FDM soliton
- complex FDM dipole and quadrupole amplitudes measured about the soliton centre
- the numerical linking length, finest cell width, and gravitational softening

A group with more than two sinks retains the complete member list and receives
a `MULTIPLE` flag. The analysis does not convert such a group into an arbitrary
sequence of binaries.

## Internal binary state

After binary capture, one resolved particle represents the total monopole and
centre-of-mass motion. A separate internal state retains the two masses,
semi-major axis, eccentricity vector, orbital phase, orbital-plane orientation,
energy, and angular momentum. The two physical SMBHs remain distinct until
coalescence even though the resolved calculation advances one centre-of-mass
particle.

The runtime update performs the following physical operations.

1. The local FDM state, including coherent dipole and quadrupole amplitudes,
   and the internal binary elements select an accepted region of the
   calibration table.
2. The interpolated power and torque change internal orbital energy and angular
   momentum over the current interval.
3. The new energy and angular momentum determine the semi-major axis,
   eccentricity, and orbital phase. An invalid bound orbit rejects the step.
4. The resolved wave receives the opposite energy and angular momentum.
5. The binary multipole correction equals the potential of the two internal
   members minus the monopole potential already supplied by the centre-of-mass
   particle.
6. The Hamiltonian ledger tests the orbital, centre-of-mass, wave intrinsic,
   wave-SMBH interaction, and total energies.

The force from a resolved FDM wake and an analytic FDM drag never act on the
same orbital degree of freedom. Work and torque produced by the multipole
correction count toward the calibrated wave increment. Only a measured residual
may receive an additional source.

## Physical coalescence time

The model records three intervals separately. They extend from the numerical
binary-capture scale to 1 pc, from 1 pc to 0.01 pc through the calibrated FDM
regime, and from 0.01 pc through gravitational radiation. An interval that
exceeds the cosmic time available at capture remains censored. Missing or
invalid intervals never receive a zero duration.
