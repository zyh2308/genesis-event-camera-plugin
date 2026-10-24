# R1.5 target-shadow spatial translation null

CPU-only post hoc analysis on the existing episode 004/005 cached direct-HDR events. The actual statistic is the fraction of fixed-floor event pixels within 5 px of the analytically projected, time-unioned target-shadow edge. The null exhaustively translates the same contour by every integer offset inside a fixed window of one third of each image dimension. It retains placements only when the number of contour pixels inside the fixed floor region is 0.8–1.2 times the actual visible support. This conditions on visible contour length because much of the projected edge falls outside the fixed floor mask or behind segmented objects. The contour and its 5 px neighborhood must remain inside the image.

The observed placement lies in the upper tail of these translations in both episodes, but some other placements can score similarly. This is a spatial specificity diagnostic, not a renderer intervention, an independent test set, a formal inferential p-value, or causal proof. The shadow polygon is an analytic hypothesis based on the documented default light direction. See the JSON for input/config/code hashes and exact translation ranges.

Reproduce from the repository root:

```bash
python benchmark/robot_closed_loop/diagnostics/check_r1_5_shadow_edge_spatial_null.py \
  --output benchmark/robot_closed_loop/results/r1_5_shadow_edge_spatial_null_cpu_20261024g
```
