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
