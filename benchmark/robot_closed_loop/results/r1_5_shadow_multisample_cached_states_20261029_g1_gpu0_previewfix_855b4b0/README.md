# R1.5 matched shadow x multisample cached-state render

This diagnostic restores exact logged Panda and target states from cached development episodes 004 and 005. Physics is not stepped. It renders actual start, middle, and end timestamps under the four `shadow on/off x GL_MULTISAMPLE on/off` conditions. The same ideal physical-log DVS response is applied to each condition. A separate simple-control scene renders only a plane and high-contrast box; the box is moved by render-only pose updates of about one image pixel per 5 ms. That control is explicitly synthetic and is not a new robot rollout.

Genesis allocates an HDR multisample framebuffer using `min(GL_MAX_SAMPLES, 4)`. The experiment changes only the actual `GL_MULTISAMPLE` capability during the HDR draw. It does not change framebuffer allocation or the radiance resolve/readback pass. Every HDR pass records actual GL enable state, draw FBO identity, sample-buffer count, sample count, and implementation sample positions. Each radiance readback also records the color and, when depth is requested, depth resolve blits and checks their framebuffer identities. Genesis runs segmentation as a separate camera pass. The fixed floor mask is evaluator-only; the DVS consumes the full rendered image.

This is a direct-render factorial to identify shadow and raster-multisample contributions. It is not yet candidate-versus-reference reconstruction fidelity, not a full 1 kHz trace, and not a change to the official Direct-HDR reference. Any later candidate/reference comparison must use matching shadow and multisample settings within each pair.

Run on g1 only after confirming an idle GPU and checking ownership. From the repository root:

```bash
ROBOT_PYTHON="$HOME/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python"
GPU_ID=${GPU_ID:?set to a confirmed free GPU ID}
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES= EGL_DEVICE_ID="$GPU_ID" PYOPENGL_PLATFORM=egl \
  "$ROBOT_PYTHON" benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_shadow_multisample_cached_states.py \
  --cache-root benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059 \
  --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --experiment-config benchmark/robot_closed_loop/configs/r1_5_shadow_multisample_cached_states_20261025b.json \
  --output benchmark/robot_closed_loop/results/r1_5_shadow_multisample_cached_states_20261025b
```
