# R1.5 archived photometric root-cause analysis

**Status:** CPU-only follow-up of the already archived episodes 004 and 005. No Genesis renderer, physics, GPU, training, or event-model generation was run. Frozen simulator core is unchanged.

## What was checked

The analysis reuses the ten exact Direct-HDR frames and the already generated ideal-DVS event streams from `r1_5_photometric_sampling_contract_20261024a`. It recreates the same fixed ground mask: visible as ground in both endpoints and all ten direct frames, two pixels from segmentation boundaries, with unchanged depth. It then asks whether the unexplained brightness is global or spatially localized, whether each ground pixel changes in one direction or reverses, and where saved events occur.

## Results

| Measure | Episode 004, center | Episode 005, +55 mm |
|---|---:|---:|
| Fixed ground pixels | 61,322 | 61,039 |
| Range of the ground's median log luminance over ten times | 0 | 0 |
| Ground pixels with endpoint-model residual above 0.2 log | 71 | 96 |
| Ground pixels whose sampled brightness changes direction at least once | 161 | 208 |
| Direct-DVS events / unique active ground pixels | 308 / 73 | 451 / 108 |
| Direct events on pixels with endpoint-model residual >0.05 log | 308/308 | 451/451 |
| Direct events on pixels with endpoint-model residual >0.20 log | 293/308 | 432/451 |
| Active pixels containing both ON and OFF events | 28 | 48 |
| Fraction of active pixels in the 3 largest 8-connected groups | 57.5% | 37.0% |

Every direct event in both clips lands on a pixel where the endpoint log-linear sequence differs from the actual intermediate radiance by more than 0.05 log; about 95% land on pixels with more than 0.20 log error. The event maps are spatially sparse and visually near dark moving-shadow boundaries in the cached frames. This is consistent with a moving cast shadow, but is not causal proof. The distribution does argue against a frame-wide exposure or gain shift as the main source. See [`episode_004_floor_photometric_map.png`](episode_004_floor_photometric_map.png) and [`episode_005_floor_photometric_map.png`](episode_005_floor_photometric_map.png).

## Root-cause assessment

1. **Most likely:** a moving shadow changes irradiance on a stationary floor. The robot scene sets only ambient light in `VisOptions`; Genesis 1.2.2 documents `shadow=True` by default and `plane_reflection=False`. The cached images visibly contain the robot shadow. A shadow moves in image brightness without any floor motion or depth change, so geometry-only brightness constancy and endpoint interpolation cannot represent its transient history.
2. **Not yet proven:** shadow is the dominant cause. Only a matched shadow-on/off render at the exact same cached states can establish that. Other renderer lighting or shadow-map sampling effects remain possible.
3. **Separate issue:** the HDR path resolves multisample color while segmentation is rendered in a separate single-sample pass. This can affect silhouette coverage and geometry/radiance agreement. It cannot by itself explain all the measured interior-floor temporal signal; those pixels are stationary, have identical endpoint depth, and are at least two pixels from geometry-label boundaries.
4. **Not indicated by this experiment:** a global radiance/gamma or exposure change. The native HDR shader writes scene-linear values to an RGBA16F attachment without gamma encoding or clipping, and the fixed-floor median is unchanged. This analysis does not establish that the archived renderer binary/source exactly matched the currently checked-out overlay.

This interpretation agrees with established simulator design limits. ESIM's adaptive sampling derives a brightness-change bound from image gradients and scene motion under a Lambertian brightness-constancy assumption; the paper also explains that a fixed frame rate can miss a faster brightness signal. A changing cast-shadow visibility term is outside that geometry-only bound unless the renderer exposes it or additional radiance samples observe it. V2E instead runs its DVS model on original/interpolated grayscale frames; interpolation can use observed image evidence, but cannot recover an unobserved transient that is absent from its source frames. These are method boundaries, not a claim that either simulator is universally better. Sources: [ESIM paper](https://proceedings.mlr.press/v87/rebecq18a/rebecq18a.pdf), [V2E source](https://github.com/SensorsINI/v2e), [Genesis 1.2.2 VisOptions](https://genesis-world.readthedocs.io/en/v1.2.2/_modules/genesis/options/vis.html).

## Next discriminating experiment

When a GPU is available, rerender the same archived source, midpoint, and endpoint states for both episodes under `shadow on/off × multisample on/off`, plus one high-contrast contour control. Do not advance physics or synthesize timestamps. Compare within each matched renderer condition:

- scene-linear HDR on the fixed ground and around contours;
- direct versus endpoint-predicted event count, polarity, and event-time residual;
- fixed-ground spatial residual maps and target/robot/ground partitions;
- source hash, requested and actual renderbuffer sample count, GL vendor/renderer/version, queried **shading** sample positions, and a separate empirical coverage-mask diagnostic;
- warm direct-render, pass, readback, and DVS costs separately.

If disabling shadows removes the ground temporal residual while leaving the contour result essentially unchanged, the missing term is dynamic illumination, not geometry coverage. If it persists, inspect shader material/specular terms and actual sample/resolve behavior before expanding the experiment. Keep the formal scene's shadow setting unchanged unless the research protocol is explicitly revised.

The renderer API documents shadow controls on `VisOptions`; the current benchmark scene can expose an explicit diagnostic-only override without changing frozen core. A future FramePacket should distinguish `geometry_support` from `radiance_temporal_support` and identify actual sample/resolve provenance. A small endpoint residual alone must not certify the interval as photometrically known.

## Reproduction and hashes

From the repository root:

```bash
python benchmark/robot_closed_loop/diagnostics/analyze_r1_5_cached_photometric_rootcause.py \
  --output benchmark/robot_closed_loop/results/r1_5_photometric_rootcause_cpu_20261024b_rerun1
```

The script refuses to overwrite an existing run directory. Its output is captured in `rootcause_analysis.json`; input hashes are recorded there and `SHA256SUMS` covers the JSON and both maps. The run used repository commit `7a19b4d824e7ed40d92c5cf8f58b3a3ef0c0e95c` plus the then-uncommitted analysis script SHA256 `e41a1168e61a67e51cc6691378a666d2b252c69e02f7c0d1e5b677a113a692d2`. No experiment configuration was changed or needed; the exact cache inputs are identified by their SHA256 values in the JSON.
