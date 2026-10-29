#!/usr/bin/env python3
"""Capture the original Direct-HDR event source's frames without extra renders.

Run this driver against the pinned R0.5 runner checkout. It wraps only
DirectHdrEventSource.advance, copies the HDR array already rendered by the
ordinary event source at selected timestamps, and then runs the ordinary
runner unchanged. No extra physics step, control call, or camera render is
introduced by the capture hook.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

import benchmark.robot_closed_loop.event_state as event_state


def main() -> int:
    window_text = os.environ.get("R15_ONLINE_HDR_CAPTURE_WINDOW_S", "")
    output_text = os.environ.get("R15_ONLINE_HDR_CAPTURE_PATH", "")
    if not window_text or not output_text:
        raise RuntimeError("set R15_ONLINE_HDR_CAPTURE_WINDOW_S and R15_ONLINE_HDR_CAPTURE_PATH")
    start_s, end_s = (float(v) for v in window_text.split(","))
    if not 0 <= start_s < end_s:
        raise ValueError(f"invalid capture window: {window_text!r}")
    output = Path(output_text).resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)

    captures: list[tuple[float, np.ndarray]] = []
    original_advance = event_state.DirectHdrEventSource.advance

    def advance_and_copy(self: Any, camera: Any, t_s: float) -> None:
        result = original_advance(self, camera, t_s)
        if start_s - 1e-10 <= float(t_s) <= end_s + 1e-10:
            if self.last_hdr is None:
                raise RuntimeError(f"event source has no rendered HDR at t={t_s}")
            captures.append((float(t_s), np.asarray(self.last_hdr, dtype=np.float32).copy()))
        return result

    event_state.DirectHdrEventSource.advance = advance_and_copy
    import benchmark.robot_closed_loop.run as runner  # noqa: E402

    runner.main()

    times = np.asarray([t for t, _ in captures], dtype=np.float64)
    images = np.asarray([image for _, image in captures], dtype=np.float32)
    if len(times) < 2 or images.ndim != 4 or not np.isfinite(images).all():
        raise RuntimeError(f"invalid captured online HDR array: times={times.shape}, images={images.shape}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, times_s=times, hdr_rgb=images)
    expected = np.arange(start_s, end_s + 0.00025, 0.0005, dtype=np.float64)
    if len(times) != len(expected) or not np.allclose(times, expected, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"expected exact 0.5ms timestamps, got {len(times)} rows: {times.tolist()}")
    metadata = {
        "status": "PASS",
        "capture_window_s": [start_s, end_s],
        "capture_hook": "copy DirectHdrEventSource.last_hdr immediately after the original advance call",
        "extra_physics_steps": 0,
        "extra_controls": 0,
        "extra_camera_renders": 0,
        "sample_count": int(len(times)),
        "interval_s_min": float(np.min(np.diff(times))),
        "interval_s_max": float(np.max(np.diff(times))),
        "hdr_shape": list(images.shape),
        "hdr_dtype": str(images.dtype),
        "hdr_finite": bool(np.isfinite(images).all()),
        "capture_file": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"online_hdr_capture": metadata}, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
