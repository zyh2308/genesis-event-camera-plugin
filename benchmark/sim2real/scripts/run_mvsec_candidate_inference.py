#!/usr/bin/env python3
"""Run fixed-window, event-only STE-FlowNet inference on real MVSEC.

This is a Stage 3 inspection adapter, not a model-selection or ranking tool.
The selected APS start indices must be supplied explicitly before inference.
The official STE-FlowNet timestamp propagation helper is used to associate
MVSEC flow maps with each exact APS endpoint interval.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from verify_pinned_source import verify_pinned_source


PINNED_STEFLOW_COMMIT = "eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6"
NATIVE_HW = (260, 346)
SLICES_PER_APS_INTERVAL = 5
FLOW_UNIT = "px_per_interval"
SOURCE_BUNDLE_MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "downstream_source_bundle.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_text_attr(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def round_model_size(height: int, width: int, multiple: int = 16) -> int:
    if height <= 0 or width <= 0 or multiple <= 0:
        raise ValueError("height, width, and multiple must be positive")
    return int(math.ceil(max(height, width) / multiple) * multiple)


def center_pad_event_volume(volume: np.ndarray, target_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad [2,H,W,T] into [2,S,S,T], preserving sensor pixels and values."""
    volume = np.asarray(volume)
    if volume.ndim != 4 or volume.shape[0] != 2:
        raise ValueError(f"event volume must have shape [2,H,W,T], got {volume.shape}")
    _, height, width, time_bins = volume.shape
    if target_size < max(height, width) or target_size % 16:
        raise ValueError(f"target square size must be >= sensor bounds and divisible by 16; got {target_size}")
    y0 = (target_size - height) // 2
    x0 = (target_size - width) // 2
    padded = np.zeros((2, target_size, target_size, time_bins), dtype=volume.dtype)
    padded[:, y0:y0 + height, x0:x0 + width, :] = volume
    return padded, (y0, x0)


def crop_flow_to_sensor(flow: np.ndarray, sensor_hw: tuple[int, int], offset_yx: tuple[int, int]) -> np.ndarray:
    """Crop [2,S,S] prediction to the exact unpadded sensor raster."""
    flow = np.asarray(flow)
    if flow.ndim != 3 or flow.shape[0] != 2 or flow.shape[1] != flow.shape[2]:
        raise ValueError(f"padded flow must have shape [2,S,S], got {flow.shape}")
    height, width = sensor_hw
    y0, x0 = offset_yx
    cropped = flow[:, y0:y0 + height, x0:x0 + width]
    if cropped.shape != (2, height, width):
        raise ValueError(f"cropped flow has unexpected shape {cropped.shape}")
    return cropped


def strict_compute_epe(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
    *,
    prediction_unit: str,
    ground_truth_unit: str,
) -> tuple[float | None, int]:
    """Strict 2-vector EPE with a finite, binary [H,W] mask.

    Invalid GT values are permitted only outside the supplied mask. Prediction
    values must always be finite. An empty mask returns (None, 0) so an empty
    real interval is recorded without fabricating a zero error.
    """
    pred = np.asarray(prediction)
    gt = np.asarray(ground_truth)
    mask = np.asarray(valid_mask)
    if pred.ndim != 3 or pred.shape[0] != 2 or gt.shape != pred.shape:
        raise ValueError(f"prediction and GT must share [2,H,W], got {pred.shape} and {gt.shape}")
    if mask.shape != pred.shape[1:]:
        raise ValueError(f"mask must be [H,W]={pred.shape[1:]}, got {mask.shape}")
    if prediction_unit != FLOW_UNIT or ground_truth_unit != FLOW_UNIT:
        raise ValueError(f"both flow units must be {FLOW_UNIT!r}; got {prediction_unit!r}, {ground_truth_unit!r}")
    if not np.issubdtype(pred.dtype, np.number) or not np.isfinite(pred).all():
        raise ValueError("prediction must be finite numeric [u,v] flow")
    if not np.issubdtype(gt.dtype, np.number):
        raise ValueError("GT must have numeric dtype")
    if not (np.issubdtype(mask.dtype, np.bool_) or np.issubdtype(mask.dtype, np.number)):
        raise ValueError("mask must be boolean or numeric 0/1")
    if not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError("mask must contain only finite boolean/0/1 values")
    mask = mask.astype(bool, copy=False)
    count = int(mask.sum())
    if count == 0:
        return None, 0
    if not np.isfinite(gt[:, mask]).all():
        raise ValueError("GT contains non-finite values inside the valid mask")
    du = pred[0, mask].astype(np.float64) - gt[0, mask].astype(np.float64)
    dv = pred[1, mask].astype(np.float64) - gt[1, mask].astype(np.float64)
    value = float(np.hypot(du, dv).mean())
    if not np.isfinite(value):
        raise RuntimeError("computed EPE is non-finite")
    return value, count


def strict_model_output_to_hw(output: torch.Tensor) -> np.ndarray:
    if not torch.is_tensor(output) or output.ndim != 4 or output.shape[:2] != (1, 2):
        shape = tuple(output.shape) if torch.is_tensor(output) else type(output).__name__
        raise ValueError(f"model output must be [1,2,H,W], got {shape}")
    flow = output.detach().cpu().numpy()[0]
    if not np.issubdtype(flow.dtype, np.number) or not np.isfinite(flow).all():
        raise ValueError("model output must contain finite numeric u and v components")
    return flow


def normalize_polarity(polarity: np.ndarray) -> np.ndarray:
    polarity = np.asarray(polarity)
    if not np.issubdtype(polarity.dtype, np.number) or not np.isfinite(polarity).all():
        raise ValueError("event polarity must be finite numeric values")
    values = np.unique(polarity)
    if not np.isin(values, (-1, 0, 1)).all():
        raise ValueError(f"MVSEC polarity must be in {-1,0,1}, found {values[:8]}")
    # The upstream STE loader selects p==1 and p==-1. Some MVSEC HDF5
    # releases encode OFF as 0; map that representation explicitly.
    return np.where(polarity == 0, -1, polarity).astype(np.int8)


def gaussian_count_representation(events_xyts_p: np.ndarray, height: int, width: int) -> np.ndarray:
    """Pinned STE-style equal-event-count Gaussian representation.

    `events_xyts_p` uses [x,y,t_seconds,p] with p in {-1,+1}. Equal slices
    retain remainders; overflow fails closed rather than wrapping uint8.
    """
    events = np.asarray(events_xyts_p)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"events must be [N,4] x,y,t,p, got {events.shape}")
    if len(events) and (not np.isfinite(events).all()):
        raise ValueError("interval events contain non-finite values")
    result = np.zeros((2, height, width, SLICES_PER_APS_INTERVAL), dtype=np.uint8)
    for slice_idx, rows in enumerate(np.array_split(events, SLICES_PER_APS_INTERVAL)):
        if rows.size == 0:
            continue
        x = rows[:, 0].astype(np.int64)
        y = rows[:, 1].astype(np.int64)
        t = rows[:, 2].astype(np.float64)
        p = rows[:, 3].astype(np.int8)
        if (x.min() < 0 or x.max() >= width or y.min() < 0 or y.max() >= height):
            raise ValueError("event coordinates fall outside native MVSEC sensor raster")
        if not np.isin(p, (-1, 1)).all():
            raise ValueError("normalized polarity must be -1 or +1")
        span = float(t.max() - t.min())
        tn = (t - t.min()) / span if span > 0 else np.zeros_like(t)
        for polarity_index, polarity in enumerate((1, -1)):
            selected = p == polarity
            if not selected.any():
                continue
            tp = tn[selected]
            std = float(np.std(tp, ddof=1)) + 1e-4 if len(tp) > 1 else 1.0
            mean = float(tp.mean())
            gaussian = (1.0 / np.sqrt(2.0 * np.pi) / std) * np.exp(-((tp - mean) ** 2) / (2.0 * std**2))
            weights = np.ceil(gaussian * (int(selected.sum()) / float(gaussian.sum()))).astype(np.uint32)
            plane = np.zeros((height, width), dtype=np.uint32)
            np.add.at(plane, (y[selected], x[selected]), weights)
            maximum = int(plane.max(initial=0))
            if maximum > 255:
                raise OverflowError(f"official uint8 event-count plane would overflow: {maximum}")
            result[polarity_index, :, :, slice_idx] = plane.astype(np.uint8)
    return result


def interval_event_indices(event_timestamps_s: np.ndarray, start_s: float, end_s: float) -> tuple[int, int]:
    if not (np.isfinite(start_s) and np.isfinite(end_s) and end_s > start_s):
        raise ValueError(f"invalid event interval ({start_s}, {end_s}]")
    left = int(np.searchsorted(event_timestamps_s, start_s, side="right"))
    right = int(np.searchsorted(event_timestamps_s, end_s, side="right"))
    if right < left:
        raise RuntimeError("event timestamp search produced a reversed interval")
    return left, right


def collect_interval_events(
    events_xyts_p: np.ndarray,
    event_timestamps_s: np.ndarray,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray, dict[str, int]]:
    left, right = interval_event_indices(event_timestamps_s, start_s, end_s)
    selected = np.asarray(events_xyts_p[left:right]).copy()
    if len(selected):
        ts = selected[:, 2]
        if np.any(ts <= start_s) or np.any(ts > end_s):
            raise RuntimeError("selected events violate the registered (start,end] convention")
    return selected, {"left_exclusive_index": left, "right_inclusive_stop_index": right, "event_count": right - left}


def build_window_representation(
    events_xyts_p: np.ndarray,
    event_timestamps_s: np.ndarray,
    aps_timestamps_s: np.ndarray,
    index: int,
    dt: int,
    height: int,
    width: int,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], list[dict[str, Any]]]:
    if dt not in (1, 4):
        raise ValueError("dt must be 1 or 4 APS endpoint intervals")
    if index < 0 or index + dt >= len(aps_timestamps_s):
        raise IndexError(f"APS index {index} with dt={dt} is outside {len(aps_timestamps_s)} endpoints")
    representations = []
    interval_metadata = []
    for k in range(dt):
        interval_index = index + k
        start_s = float(aps_timestamps_s[interval_index])
        end_s = float(aps_timestamps_s[interval_index + 1])
        interval_events, info = collect_interval_events(
            events_xyts_p, event_timestamps_s, start_s, end_s
        )
        representations.append(gaussian_count_representation(interval_events, height, width))
        interval_metadata.append({
            "aps_interval_index": interval_index,
            "start_time_s": start_s,
            "end_time_s": end_s,
            "event_start_index": info["left_exclusive_index"],
            "event_stop_index_exclusive": info["right_inclusive_stop_index"],
            "event_count": info["event_count"],
        })
    native = np.concatenate(representations, axis=-1)
    padded, offset_yx = center_pad_event_volume(native, target_size)
    return native, padded, offset_yx, interval_metadata


def _propagate_flow(x_flow: np.ndarray, y_flow: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray,
                    x_mask: np.ndarray, y_mask: np.ndarray, scale: float = 1.0) -> None:
    import cv2
    flow_x = cv2.remap(x_flow, x_idx, y_idx, cv2.INTER_NEAREST)
    flow_y = cv2.remap(y_flow, x_idx, y_idx, cv2.INTER_NEAREST)
    x_mask[flow_x == 0] = False
    y_mask[flow_y == 0] = False
    x_idx += flow_x * scale
    y_idx += flow_y * scale


def official_flow_association_from_h5(
    flow_dataset: h5py.Dataset,
    gt_timestamps_s: np.ndarray,
    start_time_s: float,
    end_time_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match the pinned STE helper while loading only the needed GT maps.

    Official helper: `multiscaleloss.estimate_corresponding_gt_flow` at
    STE-FlowNet commit eb4498c. Each flow map at gt timestamp i represents
    displacement to timestamp i+1; the routine propagates with nearest-neighbor
    sampling and returns endpoint displacement in pixels.
    """
    ts = np.asarray(gt_timestamps_s, dtype=np.float64)
    if ts.ndim != 1 or len(ts) < 2 or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError("GT timestamps must be finite, strictly increasing seconds")
    if start_time_s < ts[0] or end_time_s > ts[-1] or end_time_s <= start_time_s:
        raise ValueError(
            f"requested APS endpoints [{start_time_s}, {end_time_s}] exceed GT timestamp coverage "
            f"[{ts[0]}, {ts[-1]}]"
        )
    first = int(np.searchsorted(ts, start_time_s, side="right") - 1)
    last = int(np.searchsorted(ts, end_time_s, side="left") - 1)
    end_timestamp_index = int(np.searchsorted(ts, end_time_s, side="left"))
    end_matches_timestamp = (
        end_timestamp_index < len(ts) and ts[end_timestamp_index] == end_time_s
    )
    if first < 0 or last < first or last >= int(flow_dataset.shape[0]):
        raise IndexError(f"GT flow index interval [{first},{last}] invalid for {flow_dataset.shape}")
    if int(flow_dataset.shape[1]) != 2:
        raise ValueError(f"official flow_dist must be [N,2,H,W], got {flow_dataset.shape}")
    # Read exactly the GT maps whose timestamp intervals intersect the APS
    # window. In particular, when end_time_s equals a GT timestamp, the last
    # contributing map is the preceding interval (last = endpoint_index - 1).
    # A later compatibility step mirrors the pinned helper's zero-duration
    # map read only on the control-flow path where that helper performs it.
    local_ts = ts[first:last + 2]
    local_flow = np.asarray(flow_dataset[first:last + 1], dtype=np.float64)
    if local_flow.shape[0] != len(local_ts) - 1:
        raise RuntimeError("selected flow maps do not match the selected GT timestamp intervals")
    height, width = int(local_flow.shape[2]), int(local_flow.shape[3])
    if (height, width) != NATIVE_HW:
        raise ValueError(f"MVSEC flow GT raster must be {NATIVE_HW}, got {(height, width)}")
    # Preserve any non-finite source sentinels. The pinned official evaluator
    # excludes non-finite GT pixels; candidate_valid_mask below records and
    # masks them explicitly. Do not replace them with plausible flow vectors.

    gt_dt = float(local_ts[1] - local_ts[0])
    dt = end_time_s - start_time_s
    if first == last and gt_dt > dt:
        # Keep the pinned helper's short-window approximation when the entire
        # APS interval is contained in one slower GT interval.
        result = np.stack((local_flow[0, 0] * dt / gt_dt,
                           local_flow[0, 1] * dt / gt_dt), axis=0)
        right_endpoint_noop_applied = False
        flow_indices_read = list(range(first, last + 1))
    else:
        x_idx, y_idx = np.meshgrid(np.arange(width), np.arange(height))
        x_idx = x_idx.astype(np.float32)
        y_idx = y_idx.astype(np.float32)
        orig_x, orig_y = x_idx.copy(), y_idx.copy()
        x_mask = np.ones((height, width), dtype=bool)
        y_mask = np.ones((height, width), dtype=bool)
        for local_index, flow_index in enumerate(range(first, last + 1)):
            interval_start = float(ts[flow_index])
            interval_end = float(ts[flow_index + 1])
            segment_start = max(start_time_s, interval_start)
            segment_end = min(end_time_s, interval_end)
            if segment_end <= segment_start:
                raise RuntimeError(
                    f"GT interval {flow_index} does not overlap requested window despite index selection"
                )
            scale = (segment_end - segment_start) / (interval_end - interval_start)
            _propagate_flow(local_flow[local_index, 0], local_flow[local_index, 1],
                            x_idx, y_idx, x_mask, y_mask, float(scale))
        # Match the pinned STE helper's zero-duration final propagation at an
        # exact interior GT timestamp. Although the displacement scale is 0,
        # the upstream helper still samples this next map and updates its
        # per-axis in-bounds masks. At the final available GT timestamp there
        # is no following interval; skip that undefined no-op and retain the
        # physically complete displacement through the requested endpoint.
        right_endpoint_noop_applied = bool(
            end_matches_timestamp
            and end_timestamp_index == first + 1
            and end_timestamp_index + 1 < len(ts)
            and end_timestamp_index < int(flow_dataset.shape[0])
        )
        flow_indices_read = list(range(first, last + 1))
        if right_endpoint_noop_applied:
            noop_flow = np.asarray(flow_dataset[end_timestamp_index], dtype=np.float64)
            if noop_flow.shape != (2, height, width):
                raise ValueError(f"right-endpoint GT flow map has unexpected shape {noop_flow.shape}")
            _propagate_flow(noop_flow[0], noop_flow[1], x_idx, y_idx,
                            x_mask, y_mask, 0.0)
            flow_indices_read.append(end_timestamp_index)
        x_shift, y_shift = x_idx - orig_x, y_idx - orig_y
        x_shift[~x_mask] = 0
        y_shift[~y_mask] = 0
        result = np.stack((x_shift, y_shift), axis=0)
    if result.shape != (2, height, width):
        raise RuntimeError(f"propagated GT has unexpected shape {result.shape}")
    timestamps_used = local_ts.tolist()
    if right_endpoint_noop_applied:
        timestamps_used.append(float(ts[end_timestamp_index + 1]))
    association = {
        "algorithm": "pinned STE-FlowNet estimate_corresponding_gt_flow semantics",
        "gt_map_indices_inclusive": [first, last],
        "gt_map_indices_read": flow_indices_read,
        "right_endpoint_zero_duration_map_read": right_endpoint_noop_applied,
        "gt_interval_start_index": int(np.searchsorted(ts, start_time_s, side="right") - 1),
        "gt_timestamps_used_s": timestamps_used,
        "start_time_s": float(start_time_s),
        "end_time_s": float(end_time_s),
        "flow_unit": FLOW_UNIT,
    }
    return result.astype(np.float32), association


@dataclass
class MVSECData:
    events_xyts_p: np.ndarray
    event_timestamps_s: np.ndarray
    aps_timestamps_s: np.ndarray
    event_index_boundaries: np.ndarray
    height: int
    width: int
    data_file_sha256: str
    source_full_file: str | None
    source_full_file_sha256: str | None
    registration_json_sha256: str | None


def load_mvsec_events_and_endpoints(data_h5: Path) -> MVSECData:
    with h5py.File(data_h5, "r") as source:
        base = "davis/left"
        required = ["events", "image_raw_ts", "image_raw_event_inds", "image_raw"]
        missing = [key for key in required if f"{base}/{key}" not in source]
        if missing:
            raise KeyError(f"MVSEC data HDF5 missing official fields: {missing}")
        raw = np.asarray(source[f"{base}/events"])
        aps_ts = np.asarray(source[f"{base}/image_raw_ts"], dtype=np.float64).reshape(-1)
        event_inds_raw = np.asarray(source[f"{base}/image_raw_event_inds"], dtype=np.float64).reshape(-1)
        image_shape = tuple(source[f"{base}/image_raw"].shape)
        source_full_file = source.attrs.get("source_full_file")
        source_full_file_sha256 = source.attrs.get("source_full_file_sha256")
        registration_json_sha256 = source.attrs.get("registration_json_sha256")
        source_full_file = optional_text_attr(source_full_file)
        source_full_file_sha256 = optional_text_attr(source_full_file_sha256)
        registration_json_sha256 = optional_text_attr(registration_json_sha256)
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"official STE input expects numeric events [N,4]=[x,y,t,p], got {raw.shape}/{raw.dtype}")
    if raw.shape[0] == 0:
        raise ValueError("MVSEC HDF5 contains no raw events")
    if not np.isfinite(raw).all():
        raise ValueError("raw MVSEC events contain NaN/Inf")
    if len(aps_ts) < 2 or not np.isfinite(aps_ts).all() or np.any(np.diff(aps_ts) <= 0):
        raise ValueError("APS endpoint timestamps must be finite, strictly increasing seconds")
    if image_shape != (len(aps_ts), *NATIVE_HW):
        raise ValueError(f"MVSEC APS image shape must be [N,{NATIVE_HW[0]},{NATIVE_HW[1]}], got {image_shape}")
    if event_inds_raw.shape != aps_ts.shape or not np.isfinite(event_inds_raw).all():
        raise ValueError("image_raw_event_inds must have one finite entry per APS endpoint")
    if np.any(np.diff(event_inds_raw) < 0) or np.any(event_inds_raw < -1) or np.any(event_inds_raw > len(raw)):
        raise ValueError("image_raw_event_inds is not a monotone in-range official index array")
    rounded_indices = np.rint(event_inds_raw).astype(np.int64)
    if not np.allclose(event_inds_raw, rounded_indices, rtol=0, atol=1e-6):
        raise ValueError("image_raw_event_inds contains non-integral event indices")

    x = raw[:, 0].astype(np.int64)
    y = raw[:, 1].astype(np.int64)
    event_ts = raw[:, 2].astype(np.float64)
    p = normalize_polarity(raw[:, 3])
    if not np.allclose(raw[:, 0], x, rtol=0, atol=1e-6) or not np.allclose(raw[:, 1], y, rtol=0, atol=1e-6):
        raise ValueError("MVSEC x/y event coordinates must be integer pixel locations")
    if x.min(initial=0) < 0 or x.max(initial=0) >= NATIVE_HW[1] or y.min(initial=0) < 0 or y.max(initial=0) >= NATIVE_HW[0]:
        raise ValueError("MVSEC event coordinates fall outside native 346x260 sensor")
    if np.any(np.diff(event_ts) < 0):
        raise ValueError("MVSEC event timestamp column is not sorted")
    if event_ts[0] > aps_ts[-1] or event_ts[-1] < aps_ts[0]:
        raise ValueError("MVSEC event and APS timestamps do not overlap in seconds")
    events = np.stack((x, y, event_ts, p), axis=1)
    return MVSECData(
        events_xyts_p=events,
        event_timestamps_s=event_ts,
        aps_timestamps_s=aps_ts,
        event_index_boundaries=rounded_indices,
        height=NATIVE_HW[0],
        width=NATIVE_HW[1],
        data_file_sha256=sha256_file(data_h5),
        source_full_file=source_full_file,
        source_full_file_sha256=source_full_file_sha256,
        registration_json_sha256=registration_json_sha256,
    )


def verify_event_index_boundary(data: MVSECData, endpoint_index: int) -> dict[str, int]:
    ts_index = int(np.searchsorted(data.event_timestamps_s, data.aps_timestamps_s[endpoint_index], side="right"))
    saved_index = int(data.event_index_boundaries[endpoint_index])
    # Some releases store the last event index instead of a half-open stop
    # boundary. Both are adjacent under the official convention; larger
    # disagreements indicate a unit/schema mismatch.
    difference = abs(ts_index - saved_index)
    if difference > 1:
        raise ValueError(
            f"timestamp-derived event endpoint ({ts_index}) and official image_raw_event_inds "
            f"({saved_index}) differ by {difference} at APS index {endpoint_index}"
        )
    return {"timestamp_right_search_index": ts_index, "official_image_raw_event_index": saved_index,
            "absolute_difference": difference}


def candidate_valid_mask(gt: np.ndarray, event_volume: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    if gt.ndim != 3 or gt.shape[0] != 2:
        raise ValueError(f"GT must have shape [2,H,W], got {gt.shape}")
    if event_volume.ndim != 4 or event_volume.shape[0] != 2 or event_volume.shape[1:3] != gt.shape[1:]:
        raise ValueError(f"event volume and GT raster mismatch: {event_volume.shape}, {gt.shape}")
    finite = np.isfinite(gt).all(axis=0)
    nonzero_gt = np.linalg.norm(np.where(np.isfinite(gt), gt, 0), axis=0) > 0
    event_support = np.asarray(event_volume, dtype=np.float64).sum(axis=(0, 3)) > 0
    valid = finite & nonzero_gt & event_support
    return valid, {
        "finite_gt_pixels": int(finite.sum()),
        "nonzero_gt_pixels": int((finite & nonzero_gt).sum()),
        "event_supported_pixels": int(event_support.sum()),
        "candidate_valid_pixels": int(valid.sum()),
    }


def _torch_corr_extension_compat(input1: torch.Tensor, input2: torch.Tensor, kernel_size=1,
                                 patch_size=9, stride=1, padding=0, dilation_patch=1) -> torch.Tensor:
    """Narrow inference fallback for the pinned 1x1/9x9 correlation call."""
    if input1.shape != input2.shape or input1.ndim != 4:
        raise ValueError("STE correlation requires matching [B,C,H,W] tensors")
    if (kernel_size, patch_size, stride, padding, dilation_patch) != (1, 9, 1, 0, 1):
        raise ValueError("only the pinned STE-FlowNet correlation arguments are supported")
    batch, _, height, width = input1.shape
    padded = F.pad(input2, (4, 4, 4, 4))
    maps = []
    for dy in range(9):
        for dx in range(9):
            shifted = padded[:, :, dy:dy + height, dx:dx + width]
            maps.append((input1 * shifted).sum(dim=1))
    return torch.stack(maps, dim=1).view(batch, 9, 9, height, width)


def load_pinned_model(model_root: Path, checkpoint_path: Path, dt: int, device: torch.device):
    source = model_root.resolve()
    if not (source / "models/STEFlow.py").is_file():
        raise FileNotFoundError(f"pinned STE-FlowNet source missing at {source}")
    source_verification = verify_pinned_source(
        source, "STE-FlowNet", PINNED_STEFLOW_COMMIT, SOURCE_BUNDLE_MANIFEST
    )
    sys.path.insert(0, str(source))
    corr_module = importlib.import_module("models.corr")
    corr_module.spatial_correlation_sample = _torch_corr_extension_compat
    models = importlib.import_module("models")
    model = models.steflow()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("candidate checkpoint must be a Stage 2 checkpoint with a 'state_dict' mapping")
    if "dt" in checkpoint and int(checkpoint["dt"]) != dt:
        raise ValueError(f"checkpoint dt={checkpoint['dt']} does not match requested dt={dt}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device).eval()
    return model, checkpoint, PINNED_STEFLOW_COMMIT, source_verification


def run_one_window(
    model: torch.nn.Module,
    data: MVSECData,
    gt_dataset: h5py.Dataset,
    gt_timestamps_s: np.ndarray,
    index: int,
    dt: int,
    target_size: int,
    device: torch.device,
) -> dict[str, Any]:
    native_volume, padded_volume, offset_yx, interval_meta = build_window_representation(
        data.events_xyts_p,
        data.event_timestamps_s,
        data.aps_timestamps_s,
        index,
        dt,
        data.height,
        data.width,
        target_size,
    )
    start_s = float(data.aps_timestamps_s[index])
    end_s = float(data.aps_timestamps_s[index + dt])
    if interval_meta[0]["start_time_s"] != start_s or interval_meta[-1]["end_time_s"] != end_s:
        raise RuntimeError("event window endpoints differ from the declared APS endpoints")
    gt, gt_association = official_flow_association_from_h5(gt_dataset, gt_timestamps_s, start_s, end_s)
    input_tensor = torch.from_numpy(padded_volume[None]).to(device=device, dtype=torch.float32)
    if input_tensor.shape != (1, 2, target_size, target_size, 5 * dt):
        raise ValueError(f"model input has unexpected shape {tuple(input_tensor.shape)}")
    if not torch.isfinite(input_tensor).all():
        raise ValueError("event-only model input contains NaN/Inf")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_tensor, target_size, use_dt4=(dt == 4), flow_seq=False)
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    padded_flow = strict_model_output_to_hw(output)
    if padded_flow.shape != (2, target_size, target_size):
        raise ValueError(f"model output raster does not match padded model input: {padded_flow.shape}")
    prediction = crop_flow_to_sensor(padded_flow, (data.height, data.width), offset_yx)
    if prediction.shape != gt.shape:
        raise ValueError(f"cropped prediction and MVSEC GT differ: {prediction.shape} vs {gt.shape}")
    valid, mask_counts = candidate_valid_mask(gt, native_volume)
    epe, valid_count = strict_compute_epe(
        prediction, gt, valid, prediction_unit=FLOW_UNIT, ground_truth_unit=FLOW_UNIT
    )
    endpoint_checks = [verify_event_index_boundary(data, index), verify_event_index_boundary(data, index + dt)]
    return {
        "sample_index": int(index),
        "rollout_id": "MVSEC_sequence",
        "dt": int(dt),
        "start_time_s": start_s,
        "label_time_s": end_s,
        "end_time_s": end_s,
        "event_window_convention": "(start_time_s, end_time_s]",
        "event_count": int(sum(item["event_count"] for item in interval_meta)),
        "intervals": interval_meta,
        "event_index_boundary_checks": endpoint_checks,
        "sensor_hw": [data.height, data.width],
        "model_input_shape": list(input_tensor.shape),
        "model_input_padding_yx": list(offset_yx),
        "model_input_padding_mode": "centered zero pad to square multiple of 16",
        "model_input_square_size": target_size,
        "output_crop": {"y_start": offset_yx[0], "x_start": offset_yx[1],
                        "height": data.height, "width": data.width},
        "prediction_shape": list(prediction.shape),
        "ground_truth_shape": list(gt.shape),
        "flow_unit": FLOW_UNIT,
        "gt_association": gt_association,
        "candidate_mask": {
            "rule": "finite GT vector AND nonzero GT magnitude AND any event support at that pixel",
            "paper_protocol_frozen": False,
            **mask_counts,
        },
        "candidate_epe_px_per_interval": epe,
        "candidate_epe_valid_pixels": valid_count,
        "inference_seconds": inference_seconds,
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
        "event_input_sha256": hashlib.sha256(np.ascontiguousarray(native_volume).view(np.uint8)).hexdigest(),
        "event_input_representation": "STE equal-event-count Gaussian weighted ON/OFF slices, 5 per APS interval",
        "inference_inputs": ["event representation only"],
    }, padded_volume, padded_flow, prediction, gt, valid, native_volume


def write_window(output_dir: Path, row: dict[str, Any], padded_events: np.ndarray, padded_prediction: np.ndarray,
                 prediction: np.ndarray, gt: np.ndarray, valid: np.ndarray, native_events: np.ndarray) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"dt{row['dt']}_aps{row['sample_index']:06d}"
    path = output_dir / f"{stem}.npz"
    np.savez_compressed(
        path,
        event_input_native_uint8=native_events,
        event_input_padded_uint8=padded_events,
        prediction_padded_px_per_interval=padded_prediction.astype(np.float32),
        prediction_px_per_interval=prediction.astype(np.float32),
        gt_px_per_interval=gt.astype(np.float32),
        candidate_valid_mask=valid.astype(np.uint8),
        sample_index=np.int64(row["sample_index"]),
        event_count=np.int64(row["event_count"]),
        start_time_s=np.float64(row["start_time_s"]),
        end_time_s=np.float64(row["end_time_s"]),
    )
    row["prediction_artifact"] = str(path)
    row["prediction_artifact_sha256"] = sha256_file(path)
    row["prediction_artifact_size_bytes"] = path.stat().st_size
    return row


def self_test() -> dict[str, Any]:
    gt = np.zeros((2, 2, 3), dtype=np.float32)
    gt[0] = 3.0
    gt[1] = 4.0
    mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.uint8)
    epe, n = strict_compute_epe(gt.copy(), gt, mask, prediction_unit=FLOW_UNIT, ground_truth_unit=FLOW_UNIT)
    assert epe == 0.0 and n == 4
    pred = gt.copy()
    pred[1] += 2.0
    epe_v, n_v = strict_compute_epe(pred, gt, mask, prediction_unit=FLOW_UNIT, ground_truth_unit=FLOW_UNIT)
    assert epe_v == 2.0 and n_v == 4
    pred[0] += 3.0
    epe_uv, _ = strict_compute_epe(pred, gt, mask, prediction_unit=FLOW_UNIT, ground_truth_unit=FLOW_UNIT)
    assert np.isclose(epe_uv, np.sqrt(13.0))
    empty, n_empty = strict_compute_epe(gt, gt, np.zeros((2, 3), dtype=bool),
                                       prediction_unit=FLOW_UNIT, ground_truth_unit=FLOW_UNIT)
    assert empty is None and n_empty == 0
    volume = np.zeros((2, 260, 346, 5), dtype=np.uint8)
    volume[0, 0, 0, 0] = 7
    padded, offset = center_pad_event_volume(volume, round_model_size(260, 346))
    restored = padded[:, offset[0]:offset[0] + 260, offset[1]:offset[1] + 346, :]
    assert np.array_equal(restored, volume)
    field = np.zeros((2, 352, 352), dtype=np.float32)
    field[0] = 11.0
    field[1] = -7.0
    cropped = crop_flow_to_sensor(field, (260, 346), offset)
    assert cropped.shape == (2, 260, 346) and np.all(cropped[0] == 11) and np.all(cropped[1] == -7)
    # Adjacent APS endpoints define dt1; dt4 uses exactly i..i+4.
    ts = np.array([0., .1, .2, .3, .4, .5])
    assert ts[2] - ts[1] == .1 and ts[4] - ts[0] == .4
    return {
        "status": "pass",
        "epe_perfect_uv_3_4": epe,
        "epe_vertical_only_plus_2": epe_v,
        "epe_u_plus_3_v_plus_2": epe_uv,
        "valid_pixels": n,
        "empty_mask": {"epe": empty, "valid_pixels": n_empty},
        "sensor_hw": [260, 346],
        "target_size": round_model_size(260, 346),
        "padding_yx": list(offset),
        "dt1_interval_seconds": float(ts[2] - ts[1]),
        "dt4_interval_seconds": float(ts[4] - ts[0]),
        "controlled_vector_channels": {"u": 11.0, "v": -7.0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run CPU-controlled vector/shape checks only")
    parser.add_argument("--data-h5", type=Path)
    parser.add_argument("--gt-h5", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dt", choices=(1, 4), type=int)
    parser.add_argument("--indices", type=int, nargs="+", help="pre-registered APS start indices; no auto-selection")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sequence", default="indoor_flying1")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    missing = [name for name in ("data_h5", "gt_h5", "model_root", "checkpoint", "dt", "indices", "output_dir")
               if getattr(args, name) is None]
    if missing:
        parser.error("real inference requires --" + " --".join(name.replace("_", "-") for name in missing))
    if args.device == "cpu" or not args.device.startswith("cuda"):
        raise ValueError("pinned STE-FlowNet forward hardcodes CUDA tensors; choose an explicit CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by pinned STE-FlowNet")
    if not args.data_h5.is_file() or not args.gt_h5.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("data HDF5, GT HDF5, and candidate checkpoint must all exist")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    model, checkpoint, model_commit, model_source_verification = load_pinned_model(
        args.model_root, args.checkpoint, args.dt, device
    )
    data = load_mvsec_events_and_endpoints(args.data_h5)
    with h5py.File(args.gt_h5, "r") as gt_file:
        gt_base = "davis/left"
        if f"{gt_base}/flow_dist" not in gt_file or f"{gt_base}/flow_dist_ts" not in gt_file:
            raise KeyError("MVSEC GT HDF5 missing official davis/left/flow_dist or flow_dist_ts")
        flow_dataset = gt_file[f"{gt_base}/flow_dist"]
        gt_timestamps = np.asarray(gt_file[f"{gt_base}/flow_dist_ts"], dtype=np.float64).reshape(-1)
        if flow_dataset.ndim != 4 or flow_dataset.shape[1] != 2 or flow_dataset.shape[2:] != NATIVE_HW:
            raise ValueError(f"official MVSEC flow_dist must be [N,2,260,346], got {flow_dataset.shape}")
        if not np.isfinite(gt_timestamps).all() or np.any(np.diff(gt_timestamps) <= 0):
            raise ValueError("flow_dist_ts must be finite, strictly increasing seconds")
        gt_hash = sha256_file(args.gt_h5)
        gt_source_full_file = optional_text_attr(gt_file.attrs.get("source_full_file"))
        gt_source_full_file_sha256 = optional_text_attr(gt_file.attrs.get("source_full_file_sha256"))
        gt_registration_json_sha256 = optional_text_attr(gt_file.attrs.get("registration_json_sha256"))
        target_size = round_model_size(data.height, data.width)
        if len(set(args.indices)) != len(args.indices):
            raise ValueError("registered APS indices must be unique")
        rows = []
        for index in args.indices:
            row, padded_events, padded_pred, pred, gt, valid, native_events = run_one_window(
                model, data, flow_dataset, gt_timestamps, index, args.dt, target_size, device
            )
            row["sequence"] = args.sequence
            rows.append(write_window(args.output_dir, row, padded_events, padded_pred, pred, gt, valid, native_events))
    manifest = {
        "status": "pass",
        "scope": "fixed-window event-only STE-FlowNet real-data inference and candidate per-window metric inspection",
        "real_scores_for_model_selection": False,
        "real_scores_used_for_checkpoint_or_generator_selection": False,
        "sequence": args.sequence,
        "registered_aps_start_indices": args.indices,
        "dt": args.dt,
        "dt_definition": "dt=1: APS endpoint i to i+1; dt=4: APS endpoint i to i+4",
        "flow_unit": FLOW_UNIT,
        "time_unit": "seconds",
        "event_window_convention": "(APS timestamp[i], APS timestamp[i+dt]]",
        "data_h5": str(args.data_h5.resolve()),
        "data_h5_size_bytes": args.data_h5.stat().st_size,
        "data_h5_sha256": data.data_file_sha256,
        "data_source_full_file": data.source_full_file,
        "data_source_full_file_sha256": data.source_full_file_sha256,
        "data_registration_json_sha256": data.registration_json_sha256,
        "gt_h5": str(args.gt_h5.resolve()),
        "gt_h5_size_bytes": args.gt_h5.stat().st_size,
        "gt_h5_sha256": gt_hash,
        "gt_source_full_file": gt_source_full_file,
        "gt_source_full_file_sha256": gt_source_full_file_sha256,
        "gt_registration_json_sha256": gt_registration_json_sha256,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_generator": checkpoint.get("generator"),
        "model_root": str(args.model_root.resolve()),
        "steflow_source_commit": model_commit,
        "steflow_source_verification": model_source_verification,
        "device": str(device),
        "deterministic_inference": {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "sensor_hw": [data.height, data.width],
        "model_size": target_size,
        "representation": "official STE equal-event-count Gaussian weighted ON/OFF event slices; 5 bins per APS interval",
        "inference_inputs": ["MVSEC events only"],
        "ground_truth_association": "pinned STE multiscaleloss.estimate_corresponding_gt_flow propagation semantics",
        "candidate_mask_protocol_frozen": False,
        "candidate_metric_mask": "finite nonzero GT vector and at least one event-supported pixel at the same location; full sensor raster",
        "windows": rows,
        "resource_summary": {
            "inference_seconds_sum": float(sum(row["inference_seconds"] for row in rows)),
            "peak_allocated_gpu_bytes_max": max(row["peak_allocated_gpu_bytes"] for row in rows),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
