# Galaxy-merger zoom calibration contract

The kpc-to-hard-binary model is calibrated only against controlled lagRamses
zoom pairs that share one physical realization and differ in numerical
resolution. The manifest is generated with:

```bash
python3 scripts/generate_galaxy_zoom_grid.py \
  configs/galaxy_merger_zoom_grid.yaml \
  galaxy_merger_zoom_manifest.json
```

The current designed grid contains 16 physical points and two resolutions per
point (32 runs). It varies mass ratio, initial eccentricity, host cusp/core,
gas fraction and rotation, nuclear-envelope mass, FDM particle mass, one CDM
control, one lower-mass host, and one coupled extreme point. The manifest does
not submit jobs and does not authorize simultaneous production runs.

Each zoom result is a version-1 JSON object containing the exact manifest case,
the lagRamses `capture_event_uid`, and the ordered stages
`numerical_capture`, `common_nucleus`, `bound_binary`, and `hard_binary`.
Every stage records status, elapsed time since numerical capture, and
separation. The capture boundary must be complete at zero elapsed time. Later
complete times must increase and separations must decrease.

The result diagnostics must include:

- maximum relative energy error;
- maximum relative angular-momentum error;
- the minimum number of cells across any reported transition radius;
- total integrated time;
- the uncalibrated analytic kpc-to-hard delay used for comparison.

`compare_zoom_resolution_pair` accepts a pair only when both runs reach a hard
binary, both conservation errors are at most `1e-3`, every transition radius
has at least four cells, and the maximum stage-delay resolution difference is
at most 20 percent. An accepted fine run supplies the multiplicative correction
to the analytic delay. The initial table performs exact physical-point lookup;
interpolation and extrapolation remain prohibited until the populated grid
demonstrates support between points.

## Static-host applicability boundary

`integrate_dual_nucleus_to_hard` supplies the analytic baseline that the zoom
grid will test. Its dynamical state is the secondary SMBH coordinate relative
to a primary fixed at the host centre. The primary mass must equal the central
point mass in the configured host potential. The relative equation includes
the primary reflex term, so the mutual two-body acceleration uses the sum of
the SMBH masses. Other host components must be spherical and time independent.

Crossing the configured common-nucleus radius is only a geometric state
transition. A pair with non-negative osculating two-body energy continues as
`COMMON_NUCLEUS_UNBOUND`; it reaches physical completion of this segment only
after becoming bound and crossing the event-specific hard-binary radius in
semi-major axis. An unbound passage that leaves the common nucleus is censored
as outside the static-host domain. A mismatch between the explicit primary
mass and central point mass is invalid rather than silently reinterpreted.

This baseline does not model a moving primary in an asymmetric potential,
triaxial or time-dependent galaxy-merger torques, evolving density profiles,
or non-spherical nuclear stripping. Those effects belong to the controlled
lagRamses zoom calculation. Until an accepted resolution pair supplies a
calibration row, its delay correction cannot be inferred or extrapolated.
