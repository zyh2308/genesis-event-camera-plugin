# HDR renderer smoke attempt

- Host/worktree: `g1:/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real`
- Branch/commit during run: `exp/public-sim2real-v1` / `75217b0cba3c655aa8491ae0a381af79a72844b9`
- Scope: 64x48 Genesis HDR renderer contract smoke only; no trajectory or task model.
- Selected device: physical GPU 1 (RTX 3090), after an immediate preflight showed no compute process on that GPU. GPU 3 had other-user CARLA processes and was excluded.
- Environment: `Eventbased_WAM`, Genesis 1.2.2, `CUDA_VISIBLE_DEVICES=1`, `PYOPENGL_PLATFORM=egl`.
- Runtime: project-local copy at `/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2`; the first assembly applied only the two `genesis/vis/` files instead of all eight overlay files.
- Result: exit code 1 after about 21 seconds. The copied `Camera` called `OffscreenRenderer.render(radiance=True)`, but the unpatched installed `genesis/ext/pyrender/offscreen.py` did not accept that argument. This is an incomplete runtime assembly, not evidence that the full source overlay fails.
- Shared cache: Quadrants warned that it could not acquire `/home/zhaoyuhan/.cache/quadrants/qdcache/kernel_compilation_manager/ticache.lock`. No shared cache files were cleaned or modified.
- Raw log: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s0-hdr-render-smoke/stdout.log`; command record is in the same server run directory.
- Corrective implementation: `benchmark/sim2real/scripts/prepare_hdr_runtime.py` copies the installed package into an isolated project runtime, applies the entire overlay subtree, and verifies its hashes. Rerun remains pending.
