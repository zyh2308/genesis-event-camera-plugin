# Genesis renderer HDR v2 overlay

This is the versioned source overlay for the independent Genesis 1.2.2 runtime.
It contains changes below `genesis/vis/` and `genesis/ext/pyrender/`, including
the float framebuffer and its shaders. The active conda installation must not
be edited in place.

On the research server, build a project-local runtime by copying the installed
`genesis` package and applying the complete `genesis/` subtree from this
directory:

```bash
python benchmark/sim2real/scripts/prepare_hdr_runtime.py \
  --runtime-root "$PWD/runtime/sim2real/genesis_hdr_v2"
```

The script verifies `genesis-world==1.2.2`, verifies every overlay file after
copy, and writes `runtime_manifest.json`. Put that runtime first on
`PYTHONPATH` and use `PYOPENGL_PLATFORM=egl` for headless rasterization.

Default `camera.render()` remains uint8; only explicit `radiance=True` returns
scene-linear float32. The RGBA16F buffer is an HDR numeric interface, not an
absolute EVK4 photon/exposure calibration. `camera.render(motion_vectors=True)`
fails explicitly because this overlay does not have a verifiable previous
clip-space attachment.

The smoke tests exercise the renderer and adapter contracts. Full historical
implementation and performance comparison notes remain in the original
research workspace.
