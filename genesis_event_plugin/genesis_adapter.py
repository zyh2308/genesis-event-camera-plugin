"""Small single-environment adapter for the Genesis 1.2.x camera/link API.

The default Genesis 1.2.x camera API exposes display RGB/depth/segmentation.
The independent ``genesis_renderer_hdr_v2`` copy adds an opt-in
``Camera.render(radiance=True)`` path; this adapter can consume it without
changing the legacy direct runner. The Torch kernels support batch tensors,
but this Genesis adapter still selects the first environment from batched
poses and is not an end-to-end Genesis B>1 integration.
"""

from typing import Callable, Mapping, Optional

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def camera_to_world_cv(camera) -> np.ndarray:
    """Return camera pose in the OpenCV convention expected by v2 flow code.

    Genesis stores camera transforms in its OpenGL convention.  The public
    ``extrinsics`` property applies the same y/z sign conversion but is a
    cached property, so we derive the current matrix directly from
    ``camera.transform`` on every call. If Genesis returns a batched pose,
    this compatibility adapter intentionally selects environment 0.
    """
    T = _array(camera.transform).astype(np.float64, copy=True)
    if T.ndim == 3:
        T = T[0]
    if T.shape != (4, 4):
        raise ValueError(f"camera.transform must be 4x4, got {T.shape}")
    T[:3, 1:3] *= -1.0
    return T


def link_to_world_cv(link, envs_idx=None) -> np.ndarray:
    """Build one world pose from a Genesis RigidLink position/quaternion.

    Batched position/quaternion outputs are currently reduced to environment
    0. This is not a complete B>1 adapter.
    """
    try:
        pos = link.get_pos(envs_idx=envs_idx, relative=False)
        quat = link.get_quat(envs_idx=envs_idx, relative=False)
    except TypeError:
        # Compatibility with wrappers that expose positional-only env index.
        pos = link.get_pos(envs_idx, relative=False)
        quat = link.get_quat(envs_idx, relative=False)
    pos = _array(pos).astype(np.float64)
    quat = _array(quat).astype(np.float64)
    if pos.ndim > 1:
        pos = pos[0]
    if quat.ndim > 1:
        quat = quat[0]
    if pos.shape != (3,) or quat.shape != (4,):
        raise ValueError(f"unexpected link pose shapes: {pos.shape}, {quat.shape}")
    # Genesis' utility uses the same quaternion convention as the rigid solver.
    from genesis.utils.geom import trans_quat_to_T
    T = _array(trans_quat_to_T(pos, quat)).astype(np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Genesis pose conversion returned {T.shape}")
    return T


def make_motion_state_provider(camera, links_by_seg_id: Mapping[int, object]):
    """Create a single-environment provider returning camera/link poses.

    ``links_by_seg_id`` must use the exact integer IDs emitted by Genesis
    segmentation at ``VisOptions.segmentation_level='link'``. This provider
    does not return a batch of environments.
    """
    def provider(scene=None, cam=None):
        active_camera = cam if cam is not None else camera
        return {
            "camera_to_world": camera_to_world_cv(active_camera),
            "object_to_world": {
                int(seg_id): link_to_world_cv(link)
                for seg_id, link in links_by_seg_id.items()
            },
        }
    return provider


def make_frame_provider(
    camera,
    radiance_hook: Optional[Callable] = None,
    native_radiance: bool = False,
):
    """Adapt Genesis 1.2.2 ``Camera.render`` to the v3 frame-provider API.

    ``radiance_hook`` is deliberately optional.  Without either hook the
    returned mapping contains RGB/depth/segmentation only and the v2 plugin
    must be configured with ``input_space='srgb'``.  ``native_radiance=True``
    requests the independent renderer's float32 linear output in the same
    render call, avoiding a second RGB render.  It is intentionally explicit
    so an unpatched Genesis installation fails rather than silently claiming
    HDR. Batched Genesis render outputs are not converted into an end-to-end
    B>1 provider here.
    """
    if native_radiance and radiance_hook is not None:
        raise ValueError("native_radiance and radiance_hook are mutually exclusive")

    def provider(scene=None, cam=None):
        active_camera = cam if cam is not None else camera
        if native_radiance:
            rgb, depth, seg, _ = active_camera.render(
                rgb=True, depth=True, segmentation=True, radiance=True
            )
        else:
            rgb, depth, seg, _ = active_camera.render(
                rgb=True, depth=True, segmentation=True
            )
        payload = {
            "rgb": _array(rgb),
            "depth": _array(depth),
            "seg": _array(seg),
        }
        if native_radiance:
            payload["radiance"] = _array(rgb)
        elif radiance_hook is not None:
            payload["radiance"] = _array(radiance_hook(active_camera, scene))
        return payload
    return provider
