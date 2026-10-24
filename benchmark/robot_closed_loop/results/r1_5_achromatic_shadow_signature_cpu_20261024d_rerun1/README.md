# R1.5 cached achromatic-lighting signature

CPU-only analysis of already archived episode 004/005 Direct-HDR frames, segmentation, poses, and fixed-floor events. No renderer, physics, GPU, training, model inference, or frozen-core edit was used.

The analysis asks whether fixed-floor log-RGB changes are approximately a common multiplicative change across channels, whether visible geometry moves during the same 10 ms, and whether floor event pixels lie unusually close to projected robot/target masks or an analytic target-shadow boundary. The exact statistics and input hashes are in `achromatic_shadow_signature.json`; overlays are in the two PNGs.

The unchanged ideal DVS model is also run on the saved direct luminance and Ours reconstructed intermediate luminance, with the same initial frame and an evaluator-only fixed-floor mask. This isolates temporal input differences; it does not test a learned estimator.

## Main measurements

| Episode | Direct-HDR floor events | Ours-reconstruction floor events | Median RGB chromatic residual / common change | Target / robot translation (mm) | Events / floor area within 5 px of predicted target-shadow edge |
|---|---:|---:|---:|---:|---:|
| 004 | 308 over 10 calls | 189 in first frame | 0.646% | 3.425 / 0.007 | 58.4% / 1.2% |
| 005 | 451 over 10 calls | 216 in first frame | 0.573% | 3.426 / 0.110 | 44.1% / 1.3% |

The direct DVS replay matches the archived direct floor events exactly (after canonical ordering). Direct events occupy all 10 1 ms DVS calls; the Ours reconstructed luma puts all its floor events in the first queried subframe. Direct floor changes are almost achromatic across RGB, the robot silhouette is almost unchanged, and the target moves about 3.43 mm. Under the documented default light direction, 44–58% of event samples fall within 5 px of the analytic target-shadow boundary, compared with about 1.2% of fixed floor pixels. This supports a target-shadow contribution, but does not prove it by intervention.

The projected target-shadow polygon assumes an opaque cube, support plane z=0, and Genesis 1.2.2's default directional-light ray direction (-1,-1,-1); the magenta outline is an analytic hypothesis, not a rendered shadow buffer. Direct events are enriched near that moving boundary. Near-achromatic changes also support illumination/shadow variation. The robot silhouette is nearly fixed while the target translates about 3.43 mm. This makes target-driven shadow motion the leading specific hypothesis, but same-state repeat renders and a shadow on/off intervention remain necessary for causal attribution.

The segmentations and poses are offline diagnostics only. They are not fed into event generation or downstream perception.

## Reproduction

From the repository root, use a fresh output directory:

```bash
python benchmark/robot_closed_loop/diagnostics/analyze_r1_5_cached_achromatic_shadow_signature.py --output benchmark/robot_closed_loop/results/r1_5_achromatic_shadow_signature_cpu_20261024d_rerun1
```

The output JSON records the input hashes, scene-config hash, execution commit, and script hash. No additional experiment config was changed.

## Method references

- ESIM's adaptive sampling derives a brightness-change bound under Lambertian brightness constancy; that assumption does not account for an unobserved moving-shadow visibility term: [ESIM paper](https://proceedings.mlr.press/v87/rebecq18a.html).
- A path-tracing event simulator samples log luminance and uses statistical event detection; this is a principled but substantially heavier alternative if the raster shadow path remains unobservable: [Monte Carlo Path Tracing and Statistical Event Detection](https://arxiv.org/abs/2408.07996).
- A recent event-adaptive path-tracing method refines time around event decisions and uses stream compaction; it is a possible later reference for a high-fidelity fallback, not a proposed change to the current benchmark: [Path-Tracing-Based Event Camera Simulation via Event-Adaptive Time Refinement](https://tatsylab.github.io/projects/manabe2026eapt/).
- Genesis 1.2.2 documents the default light direction and `shadow=True`: [VisOptions source](https://genesis-world.readthedocs.io/en/v1.2.2/_modules/genesis/options/vis.html).
