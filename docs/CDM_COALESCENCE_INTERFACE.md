# CDM capture-to-coalescence interface

This is a CDM-only composition contract.  It is not a reimplementation of the
HR5 disappearance-history analysis, and it does not accept its fixed
population delay as an environmental hardening measurement.

The physical chain is split into three non-overlapping intervals:

```text
numerical capture -> hard binary -> GW-dominated binary -> coalescence
```

The first two intervals are separate `capture_to_hard_binary` and
`hard_binary_to_gw_regime` JSON records.  A completed record must name
`dark_matter_model: cdm`, give a decreasing separation interval, and reference
an immutable `accepted_model_specific_phase_ensemble` record for the same CDM
`physics_id`.  Its only accepted calibration method is
`resolved_cdm_rate_integration`.  The calibration also hashes the exact CDM
logarithmic-separation-rate track that was integrated.  This makes a
fixed-delay population model or an FDM/SIDM result ineligible by construction.

For a completed stage, the schema is:

```json
{
  "schema_version": 2,
  "status": "complete",
  "dark_matter_model": "cdm",
  "stage": "capture_to_hard_binary",
  "delay_myr": 20.0,
  "elapsed_lower_bound_myr": 0.0,
  "reason": "paired-resolution and phase-accepted rate integration",
  "interval_pc": {
    "start_separation_pc": 1000.0,
    "end_separation_pc": 1.0
  },
  "calibration": {
    "method": "resolved_cdm_rate_integration",
    "phase_ensemble_path": "accepted_cdm_phase_ensemble.json",
    "phase_ensemble_sha256": "<64 hexadecimal characters>",
    "physics_id": "<same CDM physics id>",
    "rate_track_path": "capture_to_hard_rates.json",
    "rate_track_sha256": "<64 hexadecimal characters>"
  }
}
```

The rate-track input has exactly the same `physics_id` and stage, at least
three separation-ordered points, and a negative measured
`dln_separation_dt_per_myr` at every point.  It is integrated only between its
first and last separations; no support is extrapolated.  Build a completed
stage record rather than entering its delay by hand:

```bash
python scripts/build_cdm_delay_stage.py \
  capture_to_hard_rates.json accepted_cdm_phase_ensemble.json \
  capture_to_hard.json
```

An uncalibrated, unresolved, stalled, or over-cosmic-time interval instead has
a non-complete status, no delay or interval, no `calibration`, and an explicit
reason.  It remains censored in the composed result.

The final interval is calculated only from the explicitly supplied initial
Peters orbit.  It must start at exactly the terminal separation of the second
CDM stage.  The interface invalidates a discontinuous handoff instead of
adding incompatible delays.

```bash
cdm-smbh-compose \
  --z-sink 1.0 \
  --capture-to-hard-summary capture_to_hard.json \
  --hard-to-gw-summary hard_to_gw.json \
  --gw-mass1 "1e8 Msun" \
  --gw-mass2 "3e7 Msun" \
  --gw-start-separation "0.01 pc" \
  --gw-eccentricity 0.2 \
  --output cdm_coalescence.json
```

This command launches no simulation.  It is allowed to return `censored`,
`incomplete`, or `invalid`; only a completed result is a physical CDM
coalescence-time estimate.
