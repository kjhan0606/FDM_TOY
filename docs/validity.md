# Validity policy

This code is a calibrated toy/subgrid model. Every result must be interpreted
together with its `validity_flags`.

The current implementation warns when:

- the nonlinear-response estimate `eta_nl` reaches or exceeds unity;
- `M_enclosed / M_BH` leaves the point-perturber regime;
- the drag formula is evaluated below its velocity floor;
- the requested parameters lie outside documented literature calibration;
- the integration reaches its time budget without the target separation;
- the final crossing is not a bound binary crossing;
- the static-background energy ledger exceeds its configured tolerance.
- cumulative wave-excitation energy is no longer small compared with the
  soliton virial binding-energy proxy.

Soliton pinching, phase-dependent thrust, gravitational cooling, gas and
stellar drag, and gravitational radiation are not part of v0.1. A timeout is a
valid censored outcome and must not be removed from population statistics.

See [`wave_energy_coupling.md`](wave_energy_coupling.md) for the required
energy/momentum exchange and the analytic-drag versus live-wave separation.
