# Genesis renderer overlay status (v3)

The optional overlay is a small, source-visible patch against the Genesis
1.2.2 files listed in its `README.md`; it is not a redistributable copy of the
Genesis source tree.

## Available

- Headless/offscreen rasterizer RGB forward pass can allocate an `RGBA16F`
  main framebuffer and expose the unclipped floating-point scene-linear
  buffer through `camera.render(radiance=True)`.
- Depth and segmentation can be requested alongside that pass.
- The public batch renderer and ray-tracer paths remain unchanged and reject
  `radiance=True` explicitly.

## Not available yet

- `camera.render(motion_vectors=True)` intentionally raises. A valid renderer
  motion vector needs the current and previous clip-space positions for every
  visible primitive, plus a visibility/depth convention. The 1.2.2 public
  camera/rasterizer boundary does not retain that previous state, and a
  camera-pose difference alone would be wrong for articulated Genesis links.
- The physics plugin therefore keeps its NumPy projected-SE(3) reference and
  records the approximation. It can consume native vectors once a matching
  Genesis attachment is implemented, without silently switching paths.

The source-level capability check is part of the safety contract: an
experiment report must state whether it used the HDR overlay, display RGB, or
projected motion, and cannot call the latter “renderer-native motion vectors.”
