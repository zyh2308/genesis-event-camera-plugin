# Genesis renderer HDR v2（独立副本说明）

这是 2026-09-27 为事件插件验证建立的 Genesis 1.2.2 renderer patch overlay（v3
主线继续使用）。

- 完整的独立 renderer 副本位于研究工作区；本目录只保留可分发的修改文件和 smoke tests。
- 本地目录保存本次修改的 renderer/vis 文件、适配器接口和 smoke tests；没有覆盖原始 Genesis 安装。
- 运行时应把远程完整副本放在 `PYTHONPATH` 最前面，并使用 `PYOPENGL_PLATFORM=egl` 的 headless rasterizer。
- 默认 `camera.render()` 保持 uint8；只有显式 `radiance=True` 才返回 scene-linear float32。
- 当前 RGBA16F buffer 是 HDR 数值接口，不等于 EVK4 的绝对光子/曝光标定。
- `camera.render(motion_vectors=True)` 当前会显式报错；1.2.2 overlay 没有
  可验证的 previous clip-space attachment，不能伪造 renderer-native flow。

完整实施、验证输出、速度与 EViS 对照见：
`research/reports/Genesis原生HDR渲染副本实施与EViS速度分析_20260927.md`
