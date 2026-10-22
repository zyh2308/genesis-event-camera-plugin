"""Diagnostic OpenGL triangle rasterizer for R1.5 surface reprojection.

This uses the active Genesis headless EGL context, a private floating-point
framebuffer, hardware triangle coverage, perspective-correct varying
interpolation, and the hardware depth test. It is intentionally kept in the
robot benchmark layer; it does not alter Genesis or the frozen event core.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmark.robot_closed_loop.diagnostics.r1_5_surface_raster_candidate import (
    _build_surface_triangles,
    prepare_surface_raster_source,
)


_VERTEX_SHADER = r"""#version 330 core
uniform sampler2D u_projected_uv;
uniform sampler2D u_projected_depth;
uniform sampler2D u_source_luma;
uniform ivec2 u_size;
uniform float u_near;
uniform float u_far;

out float v_luma;
out float v_camera_z;

void main() {
    int id = gl_VertexID;
    ivec2 p = ivec2(id % u_size.x, id / u_size.x);
    float z = texelFetch(u_projected_depth, p, 0).r;
    vec2 uv = texelFetch(u_projected_uv, p, 0).rg;
    v_luma = texelFetch(u_source_luma, p, 0).r;
    v_camera_z = z;

    if (!(z > u_near) || z >= u_far || any(isnan(uv)) || any(isinf(uv))) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }

    // Genesis/benchmark UVs use top-left image coordinates whose integer
    // positions are pixel centers. OpenGL rasterizes half-integer centers.
    float x_ndc = 2.0 * (uv.x + 0.5) / float(u_size.x) - 1.0;
    float y_ndc = 1.0 - 2.0 * (uv.y + 0.5) / float(u_size.y);
    float a = (u_far + u_near) / (u_far - u_near);
    float b = -2.0 * u_far * u_near / (u_far - u_near);
    gl_Position = vec4(x_ndc * z, y_ndc * z, a * z + b, z);
}
"""


_FRAGMENT_SHADER = r"""#version 330 core
uniform int u_label;
in float v_luma;
in float v_camera_z;
layout(location = 0) out float out_luma;
layout(location = 1) out float out_camera_z;
layout(location = 2) out int out_label;
void main() {
    out_luma = v_luma;
    out_camera_z = v_camera_z;
    out_label = u_label;
}
"""


def _scalar(gl: Any, value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


class GLContinuousSurfaceRasterizer:
    """Rasterize prepared same-label surface triangles in Genesis' GL context."""

    def __init__(self, camera: Any):
        try:
            from OpenGL import GL
            from OpenGL.GL import shaders
        except Exception as exc:  # pragma: no cover - server renderer runtime.
            raise RuntimeError("PyOpenGL is required for the diagnostic GL backend") from exc

        self.camera = camera
        self.GL = GL
        self.width, self.height = (int(v) for v in camera.res)
        self.near = max(float(camera.near), 1e-4)
        self.far = float(camera.far)
        if not self.far > self.near:
            raise ValueError(f"invalid camera clip range [{self.near}, {self.far}]")
        self._rasterizer = getattr(camera, "_rasterizer", None)
        self._renderer = getattr(self._rasterizer, "_renderer", None)
        if self._renderer is None:
            raise RuntimeError("GL surface backend requires Genesis offscreen rasterizer")
        self._program = None
        self._vao = None
        self._fbo = None
        self._color_textures: list[int] = []
        self._depth_rbo = None
        self._uv_texture = None
        self._z_texture = None
        self._active = False
        self._prepared: list[dict] = []
        self._uniforms: dict[str, int] = {}
        self._allocate()

    def _current(self):
        self._renderer.make_current()
        self._active = True

    def _uncurrent(self):
        if self._active:
            self._renderer.make_uncurrent()
            self._active = False

    def _tex2d(self, internal_format: int, fmt: int, dtype: int,
               data: np.ndarray | None = None) -> int:
        gl = self.GL
        tex = _scalar(gl, gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, self.width, self.height,
                        0, fmt, dtype, data)
        return tex

    def _allocate(self):
        gl = self.GL
        self._current()
        saved = {
            "draw_fbo": _scalar(gl, gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING)),
            "read_fbo": _scalar(gl, gl.glGetIntegerv(gl.GL_READ_FRAMEBUFFER_BINDING)),
            "renderbuffer": _scalar(gl, gl.glGetIntegerv(gl.GL_RENDERBUFFER_BINDING)),
            "active_texture": _scalar(gl, gl.glGetIntegerv(gl.GL_ACTIVE_TEXTURE)),
        }
        saved["texture_2d"] = _scalar(gl, gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D))
        try:
            from OpenGL.GL import shaders
            self._program = int(shaders.compileProgram(
                shaders.compileShader(_VERTEX_SHADER, gl.GL_VERTEX_SHADER),
                shaders.compileShader(_FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER),
                validate=False,
            ))
            self._uniforms = {
                name: int(gl.glGetUniformLocation(self._program, name))
                for name in ("u_projected_uv", "u_projected_depth", "u_source_luma",
                             "u_size", "u_near", "u_far", "u_label")
            }
            self._vao = _scalar(gl, gl.glGenVertexArrays(1))
            self._uv_texture = self._tex2d(gl.GL_RG32F, gl.GL_RG, gl.GL_FLOAT)
            self._z_texture = self._tex2d(gl.GL_R32F, gl.GL_RED, gl.GL_FLOAT)

            self._fbo = _scalar(gl, gl.glGenFramebuffers(1))
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
            formats = (
                (gl.GL_R32F, gl.GL_RED, gl.GL_FLOAT),
                (gl.GL_R32F, gl.GL_RED, gl.GL_FLOAT),
                (gl.GL_R32I, gl.GL_RED_INTEGER, gl.GL_INT),
            )
            for i, (internal, fmt, dtype) in enumerate(formats):
                tex = self._tex2d(internal, fmt, dtype)
                self._color_textures.append(tex)
                gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0 + i,
                                          gl.GL_TEXTURE_2D, tex, 0)
            self._depth_rbo = _scalar(gl, gl.glGenRenderbuffers(1))
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._depth_rbo)
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT32F,
                                     self.width, self.height)
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                         gl.GL_RENDERBUFFER, self._depth_rbo)
            draw_buffers = [gl.GL_COLOR_ATTACHMENT0 + i for i in range(3)]
            gl.glDrawBuffers(3, draw_buffers)
            status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
            if status != gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"diagnostic surface FBO incomplete: 0x{status:x}")
        finally:
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, saved["draw_fbo"])
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, saved["read_fbo"])
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, saved["renderbuffer"])
            gl.glBindTexture(gl.GL_TEXTURE_2D, saved["texture_2d"])
            gl.glActiveTexture(saved["active_texture"])
            self._uncurrent()

    def prepare_sources(self, sources: list[dict], K: np.ndarray,
                        moving_ids: set[int], max_pixel_scale: float = 4.0) -> list[dict]:
        """Upload source radiance and continuity-filtered topology once per view."""
        gl = self.GL
        prepared = []
        self._current()
        old_active = _scalar(gl, gl.glGetIntegerv(gl.GL_ACTIVE_TEXTURE))
        old_vao = _scalar(gl, gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING))
        gl.glActiveTexture(gl.GL_TEXTURE0 + 2)
        old_unit2_texture = _scalar(gl, gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D))
        try:
            gl.glBindVertexArray(self._vao)
            for source in sources:
                image = np.asarray(source["radiance_luma"], dtype=np.float32)
                depth = np.asarray(source["depth"], dtype=np.float32)
                labels = np.asarray(source["segmentation"], dtype=np.int64)
                base = prepare_surface_raster_source(
                    image, depth, labels, K, max_pixel_scale=max_pixel_scale,
                    raster_labels=moving_ids, raster_backend="numba_serial",
                )
                triangles, tri_labels = _build_surface_triangles(
                    labels, base["base_valid"] & np.isin(labels, base["labels_to_process"]),
                    base["edges"], base["labels_to_process"],
                )
                used_vertices = np.unique(triangles.reshape(-1)) if triangles.size else np.empty(0, np.int64)
                gl.glActiveTexture(gl.GL_TEXTURE0 + 2)
                luma_tex = self._tex2d(gl.GL_R32F, gl.GL_RED, gl.GL_FLOAT,
                                       np.ascontiguousarray(image))
                index_buffers = {}
                for label in np.unique(tri_labels):
                    indices = np.ascontiguousarray(
                        triangles[tri_labels == label].reshape(-1), dtype=np.uint32,
                    )
                    buffer = _scalar(gl, gl.glGenBuffers(1))
                    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, buffer)
                    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices,
                                    gl.GL_STATIC_DRAW)
                    index_buffers[int(label)] = (buffer, int(indices.size))
                prepared.append({
                    **base,
                    "luma_texture": luma_tex,
                    "index_buffers": index_buffers,
                    "triangle_count": int(len(triangles)),
                    "used_vertices": used_vertices,
                    "source_time_s": float(source.get("time_s", np.nan)),
                })
            self._prepared.extend(prepared)
        finally:
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
            gl.glBindVertexArray(old_vao)
            gl.glBindTexture(gl.GL_TEXTURE_2D, old_unit2_texture)
            gl.glActiveTexture(old_active)
            self._uncurrent()
        return prepared

    def activate(self):
        if self._active:
            raise RuntimeError("Genesis GL context is already active")
        self._current()

    def deactivate(self):
        self._uncurrent()

    def warp_current(self, uv: np.ndarray, z: np.ndarray, projected_valid: np.ndarray,
                     prepared_source: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, dict]:
        """Rasterize one projected source while this object's GL context is current."""
        if not self._active:
            raise RuntimeError("call activate() before GL raster calls")
        gl = self.GL
        uv = np.asarray(uv, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        projected_valid = np.asarray(projected_valid, dtype=bool)
        expected = (self.height, self.width)
        if uv.shape != (2, *expected) or z.shape != expected or projected_valid.shape != expected:
            raise ValueError("projected uv/z/valid shapes do not match the camera")
        # The validated robot clips never move these cached surfaces behind the
        # camera. Fail closed if one does, rather than allowing invalid
        # triangle vertices to create partially clipped invented geometry.
        used_vertices = prepared_source["used_vertices"]
        if used_vertices.size and not projected_valid.reshape(-1)[used_vertices].all():
            raise RuntimeError("GL raster path encountered a prepared surface vertex behind the camera")

        uv_upload = np.ascontiguousarray(np.moveaxis(uv, 0, -1))
        z_upload = np.ascontiguousarray(z)
        saved = {
            "draw_fbo": _scalar(gl, gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING)),
            "read_fbo": _scalar(gl, gl.glGetIntegerv(gl.GL_READ_FRAMEBUFFER_BINDING)),
            "viewport": tuple(int(v) for v in gl.glGetIntegerv(gl.GL_VIEWPORT)),
            "program": _scalar(gl, gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)),
            "vao": _scalar(gl, gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)),
            "active_texture": _scalar(gl, gl.glGetIntegerv(gl.GL_ACTIVE_TEXTURE)),
            "depth_func": _scalar(gl, gl.glGetIntegerv(gl.GL_DEPTH_FUNC)),
            "depth_mask": bool(gl.glGetBooleanv(gl.GL_DEPTH_WRITEMASK)),
            "depth_test": bool(gl.glIsEnabled(gl.GL_DEPTH_TEST)),
            "blend": bool(gl.glIsEnabled(gl.GL_BLEND)),
            "cull": bool(gl.glIsEnabled(gl.GL_CULL_FACE)),
            "scissor": bool(gl.glIsEnabled(gl.GL_SCISSOR_TEST)),
            "color_mask": tuple(bool(v) for v in gl.glGetBooleanv(gl.GL_COLOR_WRITEMASK)),
        }
        texture_bindings = []
        for unit in range(3):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            texture_bindings.append(_scalar(gl, gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D)))
        unpack_alignment = _scalar(gl, gl.glGetIntegerv(gl.GL_UNPACK_ALIGNMENT))
        try:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
            gl.glViewport(0, 0, self.width, self.height)
            gl.glDrawBuffers(3, [gl.GL_COLOR_ATTACHMENT0 + i for i in range(3)])
            gl.glDisable(gl.GL_BLEND)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glDisable(gl.GL_SCISSOR_TEST)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LESS)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
            gl.glClearBufferfv(gl.GL_COLOR, 0, np.array([np.nan], np.float32))
            gl.glClearBufferfv(gl.GL_COLOR, 1, np.array([np.inf], np.float32))
            gl.glClearBufferiv(gl.GL_COLOR, 2, np.array([-1], np.int32))
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._uv_texture)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                               gl.GL_RG, gl.GL_FLOAT, uv_upload)
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._z_texture)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                               gl.GL_RED, gl.GL_FLOAT, z_upload)
            gl.glActiveTexture(gl.GL_TEXTURE2)
            gl.glBindTexture(gl.GL_TEXTURE_2D, prepared_source["luma_texture"])

            gl.glUseProgram(self._program)
            gl.glUniform1i(self._uniforms["u_projected_uv"], 0)
            gl.glUniform1i(self._uniforms["u_projected_depth"], 1)
            gl.glUniform1i(self._uniforms["u_source_luma"], 2)
            gl.glUniform2i(self._uniforms["u_size"], self.width, self.height)
            gl.glUniform1f(self._uniforms["u_near"], self.near)
            gl.glUniform1f(self._uniforms["u_far"], self.far)
            gl.glBindVertexArray(self._vao)
            draw_count = 0
            for label, (buffer, count) in sorted(prepared_source["index_buffers"].items()):
                if not count:
                    continue
                gl.glUniform1i(self._uniforms["u_label"], int(label))
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, buffer)
                gl.glDrawElements(gl.GL_TRIANGLES, count, gl.GL_UNSIGNED_INT, None)
                draw_count += 1

            luma = np.empty((self.height, self.width), np.float32)
            camera_z = np.empty_like(luma)
            labels = np.empty((self.height, self.width), np.int32)
            for attachment, output, fmt, dtype in (
                (0, luma, gl.GL_RED, gl.GL_FLOAT),
                (1, camera_z, gl.GL_RED, gl.GL_FLOAT),
                (2, labels, gl.GL_RED_INTEGER, gl.GL_INT),
            ):
                gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 + attachment)
                gl.glReadPixels(0, 0, self.width, self.height, fmt, dtype, output)
        finally:
            gl.glBindVertexArray(saved["vao"])
            gl.glUseProgram(saved["program"])
            for unit, texture in enumerate(texture_bindings):
                gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, unpack_alignment)
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, saved["draw_fbo"])
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, saved["read_fbo"])
            gl.glViewport(*saved["viewport"])
            gl.glDepthFunc(saved["depth_func"])
            gl.glDepthMask(gl.GL_TRUE if saved["depth_mask"] else gl.GL_FALSE)
            gl.glColorMask(*saved["color_mask"])
            for capability, enabled in ((gl.GL_DEPTH_TEST, saved["depth_test"]),
                                        (gl.GL_BLEND, saved["blend"]),
                                        (gl.GL_CULL_FACE, saved["cull"]),
                                        (gl.GL_SCISSOR_TEST, saved["scissor"])):
                (gl.glEnable if enabled else gl.glDisable)(capability)
            gl.glActiveTexture(saved["active_texture"])

        luma = luma[::-1].copy()
        camera_z = camera_z[::-1].copy()
        labels = labels[::-1].copy()
        support = (labels >= 0) & np.isfinite(luma) & np.isfinite(camera_z)
        luma[~support] = np.nan
        camera_z[~support] = np.inf
        diagnostics = {
            "backend": "opengl_hardware_triangles_depth_buffer",
            "triangle_count": int(prepared_source["triangle_count"]),
            "draw_calls": int(draw_count),
            "supported_pixels": int(support.sum()),
            "projected_valid_fraction": float(projected_valid.mean()),
        }
        return luma, camera_z, labels.astype(np.int64), support, diagnostics

    def close(self):
        gl = self.GL
        self._current()
        try:
            for source in self._prepared:
                textures = [source.get("luma_texture")]
                buffers = [b for b, _ in source.get("index_buffers", {}).values()]
                textures = [x for x in textures if x]
                if textures:
                    gl.glDeleteTextures(textures)
                if buffers:
                    gl.glDeleteBuffers(len(buffers), buffers)
            if self._color_textures:
                gl.glDeleteTextures(self._color_textures)
            if self._uv_texture:
                gl.glDeleteTextures([self._uv_texture, self._z_texture])
            if self._depth_rbo:
                gl.glDeleteRenderbuffers(1, [self._depth_rbo])
            if self._fbo:
                gl.glDeleteFramebuffers(1, [self._fbo])
            if self._vao:
                gl.glDeleteVertexArrays(1, [self._vao])
            if self._program:
                gl.glDeleteProgram(self._program)
        finally:
            self._uncurrent()

