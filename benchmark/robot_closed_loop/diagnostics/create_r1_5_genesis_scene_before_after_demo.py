#!/usr/bin/env python3
"""Create a same-trajectory Genesis scene demo of the legacy warp and coverage candidate.

The script replays archived Genesis camera renders and state logs. It does not
initialize Genesis or advance physics. The displayed Direct-HDR frames are
exact renders at saved states, not hidden-state ground truth between samples.
The coverage resolver remains diagnostic-only; frozen core files are untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from genesis_event_plugin.dvs_emulator import DvsEmulator
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator
from genesis_event_plugin.physics_warp import project_endpoint_to_alpha


DEFAULT_CACHE = ROOT / (
    "benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/"
    "native_hdr_cutoff059"
)
BASE_DIAGNOSTIC = ROOT / "benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_pixel_rule_attribution.py"
LOG_EPS = 1e-6
DEPTH_TIE_M = 1e-5
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_diagnostic():
    spec = importlib.util.spec_from_file_location("r1_5_pixel_rule_diagnostic", BASE_DIAGNOSTIC)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import diagnostic candidate: {BASE_DIAGNOSTIC}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_cache(ep: str, cache_root: Path):
    episode_dir = cache_root / f"episode_{ep}"
    source_path = episode_dir / "same_state_source_cache.h5"
    evidence_path = episode_dir / "rendered_frame_evidence.h5"
    with h5py.File(source_path, "r") as f:
        src = {
            name: np.asarray(f[name][:])
            for name in (
                "timestamps_s", "radiance_luma", "depth", "segmentation",
                "camera_to_world_cv", "object_poses", "segmentation_pose_ids",
            )
        }
        width, height = map(int, f.attrs["camera_resolution_wh"])
        fov = float(f.attrs["vertical_fov_deg"])
    with h5py.File(evidence_path, "r") as f:
        direct_times = np.asarray(f["dense_times_s"][:], dtype=np.float64)
        direct = np.asarray(f["dense_direct_hdr_luminance"][:], dtype=np.float32)
        direct_rgb = np.asarray(f["dense_direct_hdr_rgb"][:], dtype=np.float32)
        archived_legacy = np.asarray(f["ours_intermediate_luminance"][:], dtype=np.float32)
    return episode_dir, source_path, evidence_path, src, direct_times, direct, direct_rgb, archived_legacy, width, height, fov


def make_candidate_frames(diag, src, times, direct_times, archived_legacy, width, height, fov):
    left = int(np.flatnonzero(np.isclose(times, 0.5805, atol=1e-9, rtol=0))[0])
    right = int(np.flatnonzero(np.isclose(times, 0.5905, atol=1e-9, rtol=0))[0])
    if right != left + 1:
        raise RuntimeError("expected adjacent Genesis endpoint renders at 0.5805/0.5905 s")
    focal = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    K = np.asarray([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    object0 = {int(seg): src["object_poses"][left, j]
               for j, seg in enumerate(src["segmentation_pose_ids"])}
    object1 = {int(seg): src["object_poses"][right, j]
               for j, seg in enumerate(src["segmentation_pose_ids"])}
    T0, T1 = src["camera_to_world_cv"][left], src["camera_to_world_cv"][right]
    I0, I1 = src["radiance_luma"][left], src["radiance_luma"][right]

    # Recreate the frozen legacy path and require exact agreement with the
    # archived output before visualizing a before/after comparison.
    interpolator = PhysicsInterpolator(
        K,
        sampler_config=AdaptiveSamplingConfig(min_samples=10, max_samples=10),
        disocclusion_policy="hold",
        composite="next_primary",
        warp_backend="numpy",
    )
    legacy, sample_times = interpolator.interpolate(
        I0, src["depth"][left], src["segmentation"][left],
        I1, src["depth"][right], src["segmentation"][right],
        T0, T1, None, float(times[left]), float(times[right]),
        input_space="linear", object_world0=object0, object_world1=object1,
    )
    legacy = np.asarray(legacy, dtype=np.float32)
    legacy_replay_delta = float(np.max(np.abs(legacy - archived_legacy)))
    if legacy_replay_delta > 1e-6:
        raise RuntimeError(f"legacy replay differs from archived frame by {legacy_replay_delta:g}")
    if len(sample_times) != len(direct_times) or not np.allclose(sample_times, direct_times, atol=1e-8, rtol=0):
        raise RuntimeError("legacy and Direct-HDR query timestamps do not align")

    gy, gx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    grid = np.stack((gx, gy), axis=0).astype(np.float32)
    candidate_frames = []
    for qi, t in enumerate(direct_times):
        alpha = float((t - times[left]) / (times[right] - times[left]))
        p0 = project_endpoint_to_alpha(
            src["depth"][left], src["segmentation"][left], K, T0,
            T0, alpha, 0, object0, object1, None,
        )
        p1 = project_endpoint_to_alpha(
            src["depth"][right], src["segmentation"][right], K, T1,
            T0, alpha, 1, object0, object1, None,
        )
        cov0, valid0, _ = diag.coverage_splat(I0, src["depth"][left], *p0)
        cov1, valid1, _ = diag.coverage_splat(I1, src["depth"][right], *p1)
        valid = valid0 | valid1
        frame = np.full_like(cov0, np.nan, dtype=np.float32)
        frame[valid1] = cov1[valid1]
        frame[~valid1 & valid0] = cov0[~valid1 & valid0]
        frame = diag.hold_fill(frame, valid, alpha, I0, I1)
        # The exact observed endpoint is an algebraic identity. Preserve it
        # explicitly so this presentation does not obscure that invariant.
        if abs(alpha - 1.0) <= 1e-8:
            frame = I1.astype(np.float32, copy=True)
        candidate_frames.append(frame)
    return legacy, np.stack(candidate_frames), legacy_replay_delta


def make_dvs_config(values):
    return DvsNoiseConfig(
        pos_thres=float(values["positive_threshold"]),
        neg_thres=float(values["negative_threshold"]),
        sigma_thres=float(values["threshold_noise"]),
        cutoff_hz=float(values["cutoff_hz"]),
        leak_rate_hz=float(values["leak_rate_hz"]),
        leak_jitter_fraction=0.0,
        shot_noise_rate_hz=float(values["shot_noise_rate_hz"]),
        noise_rate_cov_decades=0.0,
        refractory_period_s=float(values["refractory_period_s"]),
        log_input=False, radiance_input=True,
        response_mode=values["response_mode"], radiometric_gain=1.0,
        radiometric_offset=0.0, radiometric_floor=float(values["radiometric_floor"]),
        radiometric_calibration_status="uncalibrated",
        readout_model="ideal_readout", seed=int(values["seed"]),
    )


def replay_events(frames, frame_times, initial_frame, dvs_values):
    dvs = DvsEmulator(res=initial_frame.shape, config=make_dvs_config(dvs_values), device="cpu")
    first_t = float(frame_times[0] - 0.001)
    dvs.initialize(initial_frame, t=first_t)
    pos = np.zeros(initial_frame.shape, dtype=np.uint16)
    neg = np.zeros(initial_frame.shape, dtype=np.uint16)
    cumulative = []
    per_step = []
    for frame, t in zip(frames, frame_times):
        events = np.asarray(dvs.generate_events(frame, float(t)))
        per_step.append(int(len(events)))
        if events.size:
            x = events[:, 1].astype(np.int64)
            y = events[:, 2].astype(np.int64)
            polarity = events[:, 3]
            pos_mask = polarity > 0
            np.add.at(pos, (y[pos_mask], x[pos_mask]), 1)
            np.add.at(neg, (y[~pos_mask], x[~pos_mask]), 1)
        cumulative.append((pos.copy(), neg.copy()))
    return cumulative, per_step


def tone_map(linear):
    x = np.maximum(np.asarray(linear, dtype=np.float32), 0.0) * 1.5
    x = x / (1.0 + x)
    x = np.power(np.clip(x, 0.0, 1.0), 1.0 / 2.2)
    return np.round(x * 255.0).astype(np.uint8)


def to_bgr_visual(linear):
    x = tone_map(linear)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=-1)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def event_image(counts):
    pos, neg = counts
    scale = np.maximum(np.log1p(pos.astype(np.float32) + neg.astype(np.float32)) / np.log(9.0), 0.0)
    scale = np.clip(scale, 0.0, 1.0)
    pos_scale = scale * (pos > 0)
    neg_scale = scale * (neg > 0)
    rgb = np.zeros((*pos.shape, 3), dtype=np.uint8)
    # ON: orange-red. OFF: cyan. The polarity colors are a visualization key.
    rgb[..., 0] = np.maximum((255 * pos_scale).astype(np.uint8), (0 * neg_scale).astype(np.uint8))
    rgb[..., 1] = np.maximum((100 * pos_scale).astype(np.uint8), (220 * neg_scale).astype(np.uint8))
    rgb[..., 2] = (255 * neg_scale).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_label(frame, x, y, text, font, color=(238, 242, 247)):
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def panel(frame, x, title, image, font, subline=""):
    frame = draw_label(frame, x + 6, 0, title, font)
    frame = draw_label(frame, x + 6, 27, subline, font, color=(148, 163, 184)) if subline else frame
    h, w = image.shape[:2]
    y0 = 58
    frame[y0:y0 + h, x + (384 - w) // 2:x + (384 - w) // 2 + w] = image
    return frame


def save_video(ep, label, times, direct_rgb, legacy, candidate, event_cumulative,
               event_counts, metrics, output, fps=30, frames_per_sample=3):
    width, height = 1280, 800
    path = output / f"episode_{ep}_{label.replace(' ', '_').replace('+', 'plus')}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer (mp4v)")
    font = ImageFont.truetype(str(FONT_PATH), 22)
    small = ImageFont.truetype(str(FONT_PATH), 17)
    x_positions = [32, 448, 864]

    for qi, t in enumerate(times):
        canvas = np.full((height, width, 3), (17, 24, 39), dtype=np.uint8)
        canvas = draw_label(canvas, 32, 10, "Genesis 事件相机：同一轨迹修复前后对照", font)
        canvas = draw_label(canvas, 32, 42, f"{label}  |  相同 Genesis 状态、相机与时间点；t={t:.4f} s", small, (148, 163, 184))

        top = 82
        # Panel titles above imagery: Genesis HDR reference / old / candidate.
        for x, title, sub, img in zip(
            x_positions,
            ["Genesis HDR 直接渲染", "修复前：原始重建", "修复候选：覆盖率合成"],
            ["saved exact state", "legacy next-primary + splat", "diagnostic-only candidate"],
            [to_bgr_visual(direct_rgb[qi]), to_bgr_visual(legacy[qi]), to_bgr_visual(candidate[qi])],
        ):
            local = np.full((316, 384, 3), (24, 34, 52), dtype=np.uint8)
            local = panel(local, 0, title, img, small, sub)
            canvas[top:top + local.shape[0], x:x + 384] = local

        bottom = 400
        event_titles = ["Direct-HDR 事件", "修复前事件", "修复候选事件"]
        event_sub = [f"累积 {sum(event_counts[0][:qi + 1]):,} 个",
                     f"累积 {sum(event_counts[1][:qi + 1]):,} 个",
                     f"累积 {sum(event_counts[2][:qi + 1]):,} 个"]
        event_sets = [event_cumulative[0][qi], event_cumulative[1][qi], event_cumulative[2][qi]]
        for x, title, sub, counts in zip(x_positions, event_titles, event_sub, event_sets):
            local = np.full((316, 384, 3), (24, 34, 52), dtype=np.uint8)
            local = panel(local, 0, title, event_image(counts), small, sub)
            canvas[bottom:bottom + local.shape[0], x:x + 384] = local

        footer = (
            f"10 ms窗口累计事件：Direct {metrics['event_counts']['direct']:,}  |  "
            f"旧版 {metrics['event_counts']['legacy']:,}  |  候选 {metrics['event_counts']['candidate']:,}"
        )
        canvas = draw_label(canvas, 32, 728, footer, small, (226, 232, 240))
        canvas = draw_label(canvas, 32, 758,
            "1 ms采样逐帧回放（约100倍慢放）  ·  候选仍有光度残差  ·  不是正式核心切换", small,
            (148, 163, 184))
        for _ in range(frames_per_sample):
            writer.write(canvas)

    # Short end card keeps the last event accumulation legible in presentations.
    last = canvas.copy()
    for _ in range(fps):
        writer.write(last)
    writer.release()
    return path


def save_overview(episode_data, output):
    width, height = 1920, 1120
    canvas = np.full((height, width, 3), (17, 24, 39), dtype=np.uint8)
    font = ImageFont.truetype(str(FONT_PATH), 36)
    sub = ImageFont.truetype(str(FONT_PATH), 24)
    small = ImageFont.truetype(str(FONT_PATH), 20)
    canvas = draw_label(canvas, 50, 24, "Genesis Event Camera｜重建修复前后", font)
    canvas = draw_label(canvas, 50, 74, "同一机器人拦截场景、同一相机与真实保存轨迹；只替换中间图像的覆盖率/深度竞争规则。", sub, (148, 163, 184))
    cols = [60, 680, 1300]
    row_y = [155, 620]
    titles = ["Genesis HDR 参考", "修复前：旧规则", "修复候选：新覆盖规则"]
    for row, ep in enumerate(("004", "005")):
        item = episode_data[ep]
        canvas = draw_label(canvas, 60, row_y[row] - 42,
                            f"{item['label']}  ·  0.5815–0.5905 s", sub)
        qi = -1
        imgs = [to_bgr_visual(item["direct_rgb"][qi]),
                to_bgr_visual(item["legacy"][qi]),
                to_bgr_visual(item["candidate"][qi])]
        for col, (x, title, img) in enumerate(zip(cols, titles, imgs)):
            canvas = draw_label(canvas, x, row_y[row], title, small)
            panel_w, panel_h = 560, 360
            resized = cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_CUBIC)
            canvas[row_y[row] + 34:row_y[row] + 34 + panel_h, x:x + panel_w] = resized
        met = item["metrics"]
        ytext = row_y[row] + 408
        line = (f"事件（Direct / 旧版 / 候选）：{met['event_counts']['direct']:,} / "
                f"{met['event_counts']['legacy']:,} / {met['event_counts']['candidate']:,}   ·   "
                f">0.2 log误差点：{met['legacy_bad_pixel_times_over_0p2']:,} → "
                f"{met['candidate_bad_pixel_times_over_0p2']:,}   ·   修复比例 {met['resolved_fraction']:.1%}")
        canvas = draw_label(canvas, 60, ytext, line, small, (226, 232, 240))
    canvas = draw_label(canvas, 60, 1060,
        "范围：覆盖率/深度竞争诊断候选；动态亮度未完全解决。Direct-HDR 是已保存状态参考，不能代表所有未采样中间时刻。",
        small, (148, 163, 184))
    path = output / "overview_before_after.png"
    cv2.imwrite(str(path), canvas)
    return path


def build_episode(ep, cache_root, output, diag, config):
    (episode_dir, source_path, evidence_path, src, times, direct, direct_rgb,
     archived_legacy, width, height, fov) = read_cache(ep, cache_root)
    legacy, candidate, legacy_replay_delta = make_candidate_frames(
        diag, src, src["timestamps_s"], times, archived_legacy, width, height, fov)
    if legacy.shape != direct.shape or candidate.shape != direct.shape:
        raise RuntimeError(f"episode {ep}: frame arrays are not aligned")

    left = int(np.flatnonzero(np.isclose(src["timestamps_s"], 0.5805, atol=1e-9, rtol=0))[0])
    I0 = src["radiance_luma"][left]
    dvs_values = config["dvs"]
    direct_events, direct_per = replay_events(direct, times, I0, dvs_values)
    legacy_events, legacy_per = replay_events(legacy, times, I0, dvs_values)
    cand_events, cand_per = replay_events(candidate, times, I0, dvs_values)

    def log_error(a, b):
        return np.abs(np.log(np.maximum(a, LOG_EPS)) - np.log(np.maximum(b, LOG_EPS)))

    err_legacy = log_error(legacy, direct)
    err_candidate = log_error(candidate, direct)
    legacy_bad = err_legacy > 0.2
    candidate_bad = err_candidate > 0.2
    metrics = {
        "episode": ep,
        "label": "center trajectory" if ep == "004" else "+55 mm lateral trajectory",
        "query_times_s": times.tolist(),
        "legacy_replay_max_abs_delta_vs_archived": legacy_replay_delta,
        "legacy_log_error_median": float(np.median(err_legacy)),
        "legacy_log_error_p95": float(np.percentile(err_legacy, 95)),
        "candidate_log_error_median": float(np.median(err_candidate)),
        "candidate_log_error_p95": float(np.percentile(err_candidate, 95)),
        "legacy_bad_pixel_times_over_0p2": int(legacy_bad.sum()),
        "candidate_bad_pixel_times_over_0p2": int(candidate_bad.sum()),
        "resolved_bad_pixel_times": int((legacy_bad & ~candidate_bad).sum()),
        "resolved_fraction": float((legacy_bad & ~candidate_bad).sum() / max(int(legacy_bad.sum()), 1)),
        "event_counts": {
            "direct": int(sum(direct_per)),
            "legacy": int(sum(legacy_per)),
            "candidate": int(sum(cand_per)),
        },
        "events_by_1ms_query": {"direct": direct_per, "legacy": legacy_per, "candidate": cand_per},
        "source_cache_sha256": sha256(source_path),
        "rendered_frame_evidence_sha256": sha256(evidence_path),
        "comparison_limit": "archived exact-state Direct-HDR frames; diagnostic reference, not hidden intermediate physical truth",
    }

    label = "center" if ep == "004" else "plus_55mm"
    video = save_video(ep, label, times, direct_rgb, legacy, candidate,
                       [direct_events, legacy_events, cand_events],
                       [direct_per, legacy_per, cand_per], metrics, output,
                       fps=int(config["visualization"]["fps"]),
                       frames_per_sample=int(config["visualization"]["video_frames_per_1ms_sample"]))
    return {
        "metrics": metrics,
        "label": metrics["label"],
        "direct_rgb": direct_rgb,
        "direct": direct,
        "legacy": legacy,
        "candidate": candidate,
        "video": video,
        "times": times,
        "source_path": source_path,
        "evidence_path": evidence_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "benchmark/robot_closed_loop/configs/r1_5_genesis_scene_demo_20261101a.json")
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite demo output: {output}")
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Chinese font not found: {FONT_PATH}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    diag = load_diagnostic()
    episodes = {ep: build_episode(ep, cache_root, output, diag, config) for ep in config["episodes"]}
    overview = save_overview(episodes, output)

    repo_head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    module_paths = {
        name: Path(path)
        for name, path in (
            ("physics_interpolator", ROOT / "genesis_event_plugin/physics_interpolator.py"),
            ("physics_warp", ROOT / "genesis_event_plugin/physics_warp.py"),
            ("dvs_emulator", ROOT / "genesis_event_plugin/dvs_emulator.py"),
            ("candidate_diagnostic", BASE_DIAGNOSTIC),
        )
    }
    result = {
        "purpose": "visual same-trajectory before/after demonstration for the Genesis robot interception scene",
        "execution": "offline replay of archived Genesis exact-state renders; no new renderer or physics rollout",
        "repository_commit_at_demo_run": repo_head,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "frozen_core_modified": False,
        "candidate_status": "diagnostic-only; not integrated into the frozen production core",
        "episodes": {ep: item["metrics"] for ep, item in episodes.items()},
        "runtime_modules": {name: {"path": str(path), "sha256": sha256(path)} for name, path in module_paths.items()},
        "artifacts": {"overview": str(overview.name),
                      "videos": {ep: str(item["video"].name) for ep, item in episodes.items()}},
        "interpretation": "The coverage/depth-competition candidate sharply reduces the old event burst and high-error pixel-times on these two cached clips. Residual error remains, especially from time-varying radiance; the video is not a claim of complete reconstruction or real-time behavior.",
    }
    (output / "demo_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = output / "per_query_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["episode", "time_s", "legacy_log_error_p95", "candidate_log_error_p95",
                  "direct_events_this_ms", "legacy_events_this_ms", "candidate_events_this_ms"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for ep, item in episodes.items():
            m = item["metrics"]
            for i, t in enumerate(m["query_times_s"]):
                legacy_e = np.abs(np.log(np.maximum(item["legacy"][i], LOG_EPS)) -
                                  np.log(np.maximum(item["direct"][i], LOG_EPS)))
                cand_e = np.abs(np.log(np.maximum(item["candidate"][i], LOG_EPS)) -
                                np.log(np.maximum(item["direct"][i], LOG_EPS)))
                writer.writerow({
                    "episode": ep, "time_s": f"{t:.6f}",
                    "legacy_log_error_p95": f"{np.percentile(legacy_e, 95):.8f}",
                    "candidate_log_error_p95": f"{np.percentile(cand_e, 95):.8f}",
                    "direct_events_this_ms": m["events_by_1ms_query"]["direct"][i],
                    "legacy_events_this_ms": m["events_by_1ms_query"]["legacy"][i],
                    "candidate_events_this_ms": m["events_by_1ms_query"]["candidate"][i],
                })
    print(json.dumps({"output": str(output), "commit": repo_head,
                      "episodes": {ep: episodes[ep]["metrics"] for ep in episodes},
                      "overview": overview.name}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
