# True-unknown region smoke check (local development only)

This extra one-pixel control was added after the g1 five-control and cached-replay runs. It verifies that a pixel unsupported by both endpoint surfaces remains NaN and requests exactly one render. This is a local CPU smoke check, not a g1 result and not evidence that real-clip unknowns are handled efficiently. The full g1 control record remains in `r1_5_surface_raster_controls_20261021a/`.
