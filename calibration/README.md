# v3 calibration bundle schema

Measured values belong under `calibration/<camera_id>/v3/`; the repository
does not manufacture a complete EVK4 calibration.  Every bundle must contain
`metadata.json` with:

```json
{
  "schema_version": "genesis-event-calibration-v3",
  "camera_id": "your-camera-id",
  "source_status": "measured",
  "measurement_session": "YYYY-MM-DD-or-lab-id",
  "units": {"radiance": "document-the-lab-unit", "time": "s"}
}
```

The remaining files are explicit artifacts:

- `intrinsics.json`: width/height, projection model, `K`, distortion and
  calibration target/procedure.
- `radiometry.json`: `mode` (`physical_log` or `calibrated_transfer`), gain,
  offset, floor, status (`uncalibrated`, `effective`, `calibrated`) and, for a
  measured transfer, a relative `transfer_path` to a JSON table containing
  `input` and `log_response` arrays.
- `contrast_threshold.json`: ON/OFF distributions, temperature/exposure and
  fitting method; do not collapse mismatch into one undocumented scalar.
- `background_noise.npz`: measured `rate_hz`, optional `on_probability` and
  `hot_pixel_mask`, with the exact sensor shape recorded in metadata.
- `temporal_response.json`: measured cutoff/impulse-response procedure and
  confidence intervals.
- `readout.json`: arbiter/dead-time/latency evidence.  Until this exists the
  simulator uses the explicitly ideal readout hook.

Load and validate a bundle with:

```python
from genesis_event_plugin.calibration import CalibrationBundle
bundle = CalibrationBundle.from_directory("calibration/evk4_unit_01/v3")
print(bundle.summary())
```

Missing artifacts are reported as missing.  A calibrated transfer table is
never silently clipped or replaced with the legacy video DN=20 transfer.
