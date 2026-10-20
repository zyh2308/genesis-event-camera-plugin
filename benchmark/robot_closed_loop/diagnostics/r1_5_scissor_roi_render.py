"""Full-camera projection with scissored HDR draw and partial readback.

This is a diagnostic-only optimization for exact repair pixels. It leaves the
registered Genesis camera, projection, full-scene depth test, and frozen core
untouched. The GL scissor clips raster writes to one integer ROI, and the
readback returns only that ROI. The main camera therefore keeps exactly the
same pixel rays and occlusion ordering as the full Direct-HDR render.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def _validate_crop(crop_xyxy: Sequence[int], width: int, height: int) -> tuple[int, int, int, int]:
    if len(crop_xyxy) != 4:
        raise ValueError("crop_xyxy must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (int(v) for v in crop_xyxy)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"crop {crop_xyxy} lies outside {(width, height)}")
    return x0, y0, x1, y1


class MainCameraScissorROI:
    """Temporarily scissor the existing Genesis camera and read back one ROI."""

    def __init__(self, camera: Any):
        self.camera = camera
        self.width, self.height = (int(v) for v in camera.res)
        self._target = None
        self._original_configure = None
        self._original_forward = None
        self._original_read = None
        self._read_rgb_float_crop = None
        self._installed = False
        self._crop = None
        self._gl = None
        self._pr_renderer = None

    def install(self) -> Dict[str, Any]:
        if self._installed:
            raise RuntimeError("scissor renderer is already installed")
        try:
            from genesis.ext.pyrender import renderer as pyrender_renderer
            from genesis.ext.pyrender import jit_render
            from genesis.ext.pyrender.constants import RenderFlags
            from OpenGL import GL
            import numba as nb
        except ImportError as exc:
            raise RuntimeError("Genesis offscreen OpenGL/Numba renderer API is unavailable") from exc

        rasterizer = getattr(self.camera, "_rasterizer", None)
        if rasterizer is None:
            raise RuntimeError("camera has no Genesis rasterizer")
        self._target = rasterizer._camera_targets.get(self.camera.uid)
        if self._target is None:
            raise RuntimeError("camera has no registered Genesis rasterizer target")
        if (int(self._target.viewport_width), int(self._target.viewport_height)) != (self.width, self.height):
            raise RuntimeError("scissored path requires the native full-resolution camera viewport")
        if self._target.dpscale != 1:
            raise RuntimeError("scissored path requires dpscale=1 to preserve source pixels")
        self._gl = GL
        self._pr_renderer = pyrender_renderer
        if self._target.jit._read_color_buf_float is None:
            self._target.jit.gen_func_ptr()

        gl_rgb = jit_render.GL_RGB
        gl_rgba = jit_render.GL_RGBA
        gl_float = jit_render.GL_FLOAT
        gl_unsigned_byte = jit_render.GL_UNSIGNED_BYTE
        address_to_ptr = jit_render.address_to_ptr

        @nb.jit(
            nb.float32[:, :, :] (
                nb.int32, nb.int32, nb.int32, nb.int32, self._target.jit.gl.wrapper_type,
            ),
            cache=False,
        )
        def read_rgb_float_crop(gl_x: int, gl_y: int, crop_w: int, crop_h: int, gl: Any):
            buf = np.empty((crop_h, crop_w, 3), np.float32)
            gl.glReadPixels(
                gl_x, gl_y, crop_w, crop_h,
                gl_rgb, gl_float,
                address_to_ptr(buf.ctypes.data),
            )
            return buf[::-1, :, :]

        @nb.jit(
            nb.uint8[:, :, :] (
                nb.int32, nb.int32, nb.int32, nb.int32, nb.boolean, self._target.jit.gl.wrapper_type,
            ),
            cache=False,
        )
        def read_rgb_uint8_crop(gl_x: int, gl_y: int, crop_w: int, crop_h: int,
                                rgba: bool, gl: Any):
            channels = 4 if rgba else 3
            buf = np.empty((crop_h, crop_w, channels), np.uint8)
            fmt = gl_rgba if rgba else gl_rgb
            gl.glReadPixels(
                gl_x, gl_y, crop_w, crop_h,
                fmt, gl_unsigned_byte,
                address_to_ptr(buf.ctypes.data),
            )
            return buf[::-1, :, :]

        self._read_rgb_float_crop = read_rgb_float_crop
        self._read_rgb_uint8_crop = read_rgb_uint8_crop
        self._original_configure = self._target._configure_forward_pass_viewport
        self._original_forward = self._target._forward_pass
        self._original_read = self._target._read_main_framebuffer
        saved_state: dict[str, Any] = {}

        def configure_forward(flags):
            self._original_configure(flags)
            if self._crop is None:
                return
            saved_state["enabled"] = bool(GL.glIsEnabled(GL.GL_SCISSOR_TEST))
            saved_state["box"] = tuple(int(v) for v in GL.glGetIntegerv(GL.GL_SCISSOR_BOX))
            x0, y0, x1, y1 = self._crop
            GL.glEnable(GL.GL_SCISSOR_TEST)
            GL.glScissor(x0, self.height - y1, x1 - x0, y1 - y0)

        def read_main_framebuffer(scene, flags):
            if self._crop is None:
                return self._original_read(scene, flags)
            if flags & (RenderFlags.RET_DEPTH | RenderFlags.SEG | RenderFlags.DEPTH_ONLY):
                if not (flags & RenderFlags.SEG) or flags & (RenderFlags.RET_DEPTH | RenderFlags.DEPTH_ONLY):
                    raise RuntimeError("scissored ROI supports HDR or segmentation-only readback")
            if not flags & RenderFlags.OFFSCREEN:
                raise RuntimeError("scissored ROI requires Genesis offscreen rendering")
            if self._target._main_fb_ms is None or self._target._main_fb is None:
                raise RuntimeError("Genesis multisample HDR framebuffer was not initialized")
            x0, y0, x1, y1 = self._crop
            gl_y0, gl_y1 = self.height - y1, self.height - y0
            crop_h, crop_w = y1 - y0, x1 - x0
            if flags & RenderFlags.SEG:
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._target._main_fb)
                color = self._read_rgb_uint8_crop(
                    np.int32(x0), np.int32(gl_y0), np.int32(crop_w), np.int32(crop_h),
                    bool(flags & RenderFlags.RGBA), self._target.jit.gl.wrapper_instance,
                )
            else:
                if not flags & RenderFlags.RADIANCE:
                    raise RuntimeError("scissored non-segmentation readback requires scene-linear HDR")
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._target._main_fb_ms)
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._target._main_fb)
                GL.glBlitFramebuffer(
                    x0, gl_y0, x1, gl_y1,
                    x0, gl_y0, x1, gl_y1,
                    GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR,
                )
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._target._main_fb)
                color = self._read_rgb_float_crop(
                    np.int32(x0), np.int32(gl_y0), np.int32(crop_w), np.int32(crop_h),
                    self._target.jit.gl.wrapper_instance,
                )
            return (color,)

        def forward_pass(scene, flags, seg_node_map=None, env_idx=-1):
            try:
                return self._original_forward(scene, flags, seg_node_map=seg_node_map, env_idx=env_idx)
            finally:
                if "enabled" in saved_state:
                    if saved_state["enabled"]:
                        GL.glEnable(GL.GL_SCISSOR_TEST)
                        GL.glScissor(*saved_state["box"])
                    else:
                        GL.glDisable(GL.GL_SCISSOR_TEST)
                    saved_state.clear()

        self._target._configure_forward_pass_viewport = configure_forward
        self._target._forward_pass = forward_pass
        self._target._read_main_framebuffer = read_main_framebuffer
        self._installed = True
        return {
            "renderer": "Genesis full-camera projection + GL_SCISSOR_TEST + HDR partial framebuffer blit/readback",
            "full_resolution_wh": [self.width, self.height],
            "camera_projection_changed": False,
            "camera_node_changed": False,
            "full_scene_depth_test_retained": True,
            "candidate_channel": "scene-linear HDR RGB",
            "optional_audit_channel": "link segmentation only; excluded from candidate cost",
            "diagnostic_only": True,
            "frozen_simulator_core_modified": False,
        }

    def set_crop(self, crop_xyxy: Sequence[int]) -> tuple[int, int, int, int]:
        self._crop = _validate_crop(crop_xyxy, self.width, self.height)
        return self._crop

    def render_hdr_crop(self, time_s: float, crop_xyxy: Sequence[int]) -> np.ndarray:
        if not self._installed:
            raise RuntimeError("install scissor renderer before rendering")
        self.set_crop(crop_xyxy)
        try:
            image, = self.camera.render(
                rgb=True, depth=False, segmentation=False,
                radiance=True, force_render=True,
            )
        finally:
            self._crop = None
        image = np.asarray(image, dtype=np.float32)
        x0, y0, x1, y1 = self._crop
        expected = (y1 - y0, x1 - x0, 3)
        if image.shape != expected:
            raise RuntimeError(f"scissor readback shape {image.shape} != expected {expected} at t={time_s}")
        if not np.isfinite(image).all():
            raise RuntimeError(f"non-finite scissor HDR at t={time_s}")
        return image.copy()

    def render_segmentation_crop(self, time_s: float, crop_xyxy: Sequence[int]) -> np.ndarray:
        """Render link labels for an optional same-state occlusion audit."""
        if not self._installed:
            raise RuntimeError("install scissor renderer before rendering")
        self.set_crop(crop_xyxy)
        try:
            _, _, segmentation, _ = self.camera.render(
                rgb=False, depth=False, segmentation=True, force_render=True,
            )
        finally:
            self._crop = None
        segmentation = np.asarray(segmentation, dtype=np.int32)
        x0, y0, x1, y1 = self._crop
        expected = (y1 - y0, x1 - x0)
        if segmentation.shape != expected:
            raise RuntimeError(
                f"scissor segmentation shape {segmentation.shape} != expected {expected} at t={time_s}"
            )
        return segmentation.copy()

    def close(self) -> None:
        if not self._installed:
            return
        self._target._configure_forward_pass_viewport = self._original_configure
        self._target._forward_pass = self._original_forward
        self._target._read_main_framebuffer = self._original_read
        self._installed = False


def roi_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Smallest integer crop containing every requested pixel; no GT involved."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("request mask must be two-dimensional")
    ys, xs = np.nonzero(values)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
