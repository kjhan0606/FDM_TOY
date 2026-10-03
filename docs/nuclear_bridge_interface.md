# lagRamses-to-nuclear bridge interface

`fdm_smbh_delay.nuclear_bridge` defines the first data boundary between a
lagRamses capture ledger and the kpc-to-pc inspiral model.  It does not add an
inspiral law.  It records the state and prevents an incomplete environment
from being interpreted as zero drag.

## Contract

`NuclearBridgeInput` contains:

- the lagRamses event UID, run ID, physical capture time, and redshift;
- the two-member `PairOrbitalState` at the pre-compaction capture boundary;
- a target semimajor axis, normally `1 pc` for the FDM handoff;
- an `EnvironmentSnapshot` with exactly one stellar, gas, and FDM channel;
- the source path and SHA-256 digest used to construct the record.

`EnvironmentChannel` uses one of these statuses:

- `available`: all closure quantities required by that channel are present;
- `absent`: the component is explicitly absent, not merely unmeasured;
- `missing`: the simulation did not provide the required state;
- `censored`: the state lies outside the validated support;
- `invalid`: the record cannot support physical integration.

The aggregate bridge status is `ready`, `missing_environment`, `censored`, or
`invalid`.  An unbound pair can still be a `ready` bridge input: the bridge
must then model dual-nucleus rebinding rather than silently treating the event
as a hard binary.

## FDM double-counting guard

An available FDM channel declares either `analytic_unresolved` or
`live_resolved`.  The former sets `resolved_wake=false`; the latter requires
`resolved_wake=true`.  The schema rejects inconsistent combinations, so an
analytic FDM drag term cannot be attached to a state whose live FDM wake is
already resolved.

## Serialization

```python
from fdm_smbh_delay.nuclear_bridge import NuclearBridgeInput

bridge = NuclearBridgeInput.read_json("capture_bridge.json")
if bridge.ready_for_integration:
    # Pass bridge.pair and the explicit environment channels to the model.
    ...
else:
    # Preserve bridge.status and bridge.reasons in the delay catalogue.
    ...
```

`NuclearBridgeInput.write_json` writes the versioned record atomically.  The
current schema version is `1`.  A future lagRamses exporter should use
`NuclearBridgeInput.from_capture_event` so that the ledger event digest and
source path remain attached to the handoff.

The Fortran lagRamses source is intentionally unchanged in this step.  The
Python boundary exporter below emits this contract after a complete `BINARY`
ledger transaction; `MULTIPLE` transactions require a separate member-
retention design before they can enter the same path.

## Ledger exporter

The Python boundary exporter now combines one complete JSONL ledger event with
one environment snapshot:

```bash
python scripts/export_nuclear_bridge.py \
  --ledger /path/to/smbh_capture_ledger.jsonl \
  --environment /path/to/environment_snapshot.json \
  --event-uid 10-1-7-9-2 \
  --run-id lagramses-run-001 \
  --capture-time-myr 4200.0 \
  --target-semimajor-axis-pc 1.0 \
  --output bridge-10-1-7-9-2.json
```

The environment file may contain one snapshot or a `snapshots` list.  The
exporter requires exactly one matching `event_uid`, rejects incomplete ledger
tails, and preserves separate source hashes for the ledger event and the
environment snapshot.

## Kpc-to-hard adapter

The bridge record does not invent radial profiles from one local snapshot.
The caller must provide a `KpcInspiralModel` whose profiles and friction
controls match the declared channels, then use the explicit member mapping:

```python
from fdm_smbh_delay.bridge_adapter import (
    BridgeMemberAssignment,
    initial_kpc_to_hard_state_from_bridge,
)

state = initial_kpc_to_hard_state_from_bridge(
    bridge,
    model=model,
    config=config,
    assignment=BridgeMemberAssignment(
        primary_member_id=11799,
        secondary_member_id=11800,
    ),
)
```

The adapter checks both SMBH masses, reverses the ledger displacement when
necessary to form secondary-minus-primary coordinates, and rejects a missing
environment or an FDM mode mismatch.  It returns the existing restartable
`KpcToHardState`; the hardening law remains the separately tested solver.

## What is used inside the unresolved 1 kpc region?

The inner kiloparsec is not assumed to be spatially resolved by the parent
simulation.  A local sink snapshot therefore supplies closure values, while a
separately versioned `EnvironmentProfileBundle` supplies shell-averaged radial
tables covering the integration interval (for example, 1 kpc down to 1 pc).
The model never turns one local value into a radial law and never extrapolates a
table beyond its support.  A missing table is `missing`; a table that does not
cover the requested radius is `censored`; only an explicitly absent component
is treated as zero.

| component | conservative term | dissipative term | required closure |
| --- | --- | --- | --- |
| stars | stellar enclosed mass in `host_potential` | Maxwellian collisionless drag, with explicit core-stall bracket | density, enclosed mass, bulk velocity, velocity dispersion |
| gas | gas enclosed mass in `host_potential` | Mach-dependent Ostriker drag | density, enclosed mass, sound speed, rotation fraction and axis |
| FDM | FDM profile in `host_potential` | analytic unresolved wake or live-resolved wake, never both | density, enclosed mass, bulk velocity, core radius and FDM mode |
| CDM/dark matter | absent in the pure-FDM model | no CDM drag term | do not add a CDM component unless a mixed-dark-sector model is explicitly selected |

Thus a visible particle count is not required at every radius, but the closure
must be supported by a shell profile and provenance.  In the pure-FDM study,
the dark-sector contribution at a few kpc must instead come from an *outer FDM
halo* profile (or a pure-FDM zoom), not from a CDM surrogate.  The current
builder accepts such an FDM profile through `host_potential`, but does not
invent an outer-halo continuation from the parsec-scale soliton.  If that
outer FDM profile is unavailable, the kpc-to-pc segment remains censored.

Stars and gas may still be included as baryons when the chosen galaxy model
contains them.  If the experiment is a completely baryon-free pure-FDM
model, those channels must be explicitly `absent` as well.

For `live_resolved` FDM, this static kpc solver deliberately applies no
analytic FDM drag; the resolved wake force must come from the corresponding
zoom/wave integration.  The flag is retained in the force budget so that a
live wake cannot be mistaken for a zero-density analytic model.

The construction helper makes these rules executable:

```python
from fdm_smbh_delay.bridge_adapter import build_kpc_model_from_profile_bundle

model = build_kpc_model_from_profile_bundle(
    bridge,
    profiles=profiles,
    host_potential=host_potential,  # pure-FDM halo + any explicitly selected baryons
    secondary_bh_mass_msun=secondary_mass,
    stellar_friction=stellar_controls,
    fdm_particle_mass_ev=m_fdm_ev,
    fdm_alpha_df=alpha_df,
)
```

If the profile support ends before the requested 1 pc handoff, the delay is
kept as censored and routed to a controlled zoom calibration; no numerical
value is manufactured from an unvalidated extrapolation.

### Outer-FDM wave closure

For a pure-FDM kpc model, `EnvironmentProfileBundle.outer_fdm_closure` may
carry radial diagnostics from a live outer-halo calculation:

- mass current in `Msun pc^-2 Myr^-1`;
- coherence time and de Broglie wavelength;
- velocity-space diffusion coefficient;
- density-gradient scale and a calibrated/uncalibrated status.

These fields are diagnostics, not an implicit drag prescription.  A closure
with status `uncalibrated` cannot produce a physical delay.  The eventual
drift-plus-diffusion model must be calibrated against a pure-FDM zoom and
validated over a finite overlap interval before it can hand off to the inner
live-wave calculation.

`FDMOuterResponseTable` is the corresponding consumer boundary for a measured
response: it stores a vector drift acceleration and a positive-semidefinite
velocity-diffusion tensor.  It returns `censored` for an uncalibrated table or
an out-of-support radius.  It does not integrate a stochastic trajectory or
turn a local profile into an analytic force law.
