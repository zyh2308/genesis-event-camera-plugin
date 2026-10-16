#!/usr/bin/env python3
"""Inspect and convert one registered HQF ROS1 bag without a ROS install."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp_seconds(stamp: Any) -> float:
    sec = getattr(stamp, "sec", getattr(stamp, "secs", None))
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", None))
    if sec is None or nsec is None:
        raise ValueError(f"ROS timestamp object lacks sec/nsec fields: {type(stamp)!r}")
    return float(sec) + float(nsec) * 1e-9


def inspect_bag(bag_path: Path) -> dict:
    from rosbags.rosbag1 import Reader

    with Reader(bag_path) as reader:
        topics = []
        for connection in reader.connections:
            msgdef = getattr(connection.msgdef, "data", "")
            topics.append({
                "topic": connection.topic,
                "message_type": connection.msgtype,
                "message_count": int(connection.msgcount),
                "connection_id": int(connection.id),
                "md5sum": getattr(connection, "digest", None),
                "message_definition": msgdef,
            })
    return {
        "bag_path": str(bag_path.resolve()),
        "bag_size_bytes": bag_path.stat().st_size,
        "bag_sha256": sha256(bag_path),
        "topics": topics,
    }


def image_to_mono8(message: Any) -> np.ndarray:
    height, width = int(message.height), int(message.width)
    encoding = str(message.encoding).lower()
    step = int(message.step)
    payload = np.frombuffer(message.data, dtype=np.uint8)
    if step <= 0 or payload.size < height * step:
        raise ValueError(f"sensor_msgs/Image has invalid step/data size: {step}/{payload.size}")
    rows = payload[:height * step].reshape(height, step)
    if encoding in ("mono8", "8uc1"):
        if step < width:
            raise ValueError(f"mono8 step {step} is shorter than width {width}")
        image = rows[:, :width].copy()
    elif encoding in ("rgb8", "bgr8"):
        if step < width * 3:
            raise ValueError(f"{encoding} step {step} is shorter than 3*width")
        color = rows[:, :width * 3].reshape(height, width, 3)
        code = cv2.COLOR_RGB2GRAY if encoding == "rgb8" else cv2.COLOR_BGR2GRAY
        image = cv2.cvtColor(color, code)
    else:
        raise ValueError(f"unsupported HQF reference image encoding {message.encoding!r}; expected mono8 or 8-bit RGB/BGR")
    if image.shape != (height, width) or image.dtype != np.uint8:
        raise RuntimeError(f"mono8 conversion returned {image.shape}/{image.dtype}")
    return image


def convert_bag(
    bag_path: Path,
    output_dir: Path,
    event_topic: str = "/dvs/events",
    image_topic: str = "/dvs/image_raw",
) -> dict:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing HQF sequence directory: {output_dir}")
    bag_info = inspect_bag(bag_path)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    event_blocks: list[np.ndarray] = []
    event_times: list[np.ndarray] = []
    polarity_blocks: list[np.ndarray] = []
    images: list[np.ndarray] = []
    image_times: list[float] = []
    image_encodings: set[str] = set()
    event_message_count = 0
    image_message_count = 0
    event_type = None
    image_type = None

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic in (event_topic, image_topic)]
        for topic in (event_topic, image_topic):
            if not any(c.topic == topic for c in connections):
                raise KeyError(f"registered HQF bag is missing expected topic {topic!r}")
        definitions = {}
        for connection in connections:
            definitions.update(get_types_from_msg(connection.msgdef.data, connection.msgtype))
            if connection.topic == event_topic:
                event_type = connection.msgtype
            elif connection.topic == image_topic:
                image_type = connection.msgtype
        typestore.register(definitions)

        for connection, _bag_timestamp, rawdata in reader.messages(connections=connections):
            message = typestore.deserialize_ros1(rawdata, connection.msgtype)
            if connection.topic == event_topic:
                event_message_count += 1
                events = message.events
                if not events:
                    continue
                xs = np.fromiter((int(event.x) for event in events), dtype=np.int32, count=len(events))
                ys = np.fromiter((int(event.y) for event in events), dtype=np.int32, count=len(events))
                times = np.fromiter((stamp_seconds(event.ts) for event in events), dtype=np.float64, count=len(events))
                polarity = np.fromiter((1 if bool(event.polarity) else 0 for event in events), dtype=np.uint8, count=len(events))
                event_blocks.append(np.column_stack((xs, ys)).astype(np.int32, copy=False))
                event_times.append(times)
                polarity_blocks.append(polarity)
            elif connection.topic == image_topic:
                image_message_count += 1
                image_encodings.add(str(message.encoding))
                images.append(image_to_mono8(message))
                image_times.append(stamp_seconds(message.header.stamp))

    if not event_blocks or not images:
        raise ValueError("registered HQF topics produced no decoded event records or images")
    events_xy = np.concatenate(event_blocks, axis=0)
    events_ts_absolute = np.concatenate(event_times)
    events_p = np.concatenate(polarity_blocks)
    images_array = np.stack(images, axis=0)
    images_ts_absolute = np.asarray(image_times, dtype=np.float64)
    if len(events_xy) != len(events_ts_absolute) or len(events_xy) != len(events_p):
        raise RuntimeError("decoded event x/y/t/p lengths differ")
    if events_xy.ndim != 2 or events_xy.shape[1] != 2:
        raise RuntimeError(f"decoded events_xy has invalid shape {events_xy.shape}")
    if images_array.ndim != 3:
        raise RuntimeError(f"decoded image stack has invalid shape {images_array.shape}")
    if not np.isfinite(events_ts_absolute).all() or not np.isfinite(images_ts_absolute).all():
        raise ValueError("decoded HQF timestamps contain NaN or Inf")
    if np.any(np.diff(events_ts_absolute) < 0):
        raise ValueError("decoded HQF event timestamps are not sorted in source message order")
    if np.any(np.diff(images_ts_absolute) <= 0):
        raise ValueError("decoded HQF image timestamps are not strictly increasing; refusing silent repair")
    height, width = map(int, images_array.shape[1:])
    if len(events_xy) and (
        events_xy[:, 0].min() < 0 or events_xy[:, 0].max() >= width
        or events_xy[:, 1].min() < 0 or events_xy[:, 1].max() >= height
    ):
        raise ValueError("decoded event coordinates fall outside HQF reference image resolution")

    origin = min(float(events_ts_absolute.min()), float(images_ts_absolute.min()))
    events_ts = events_ts_absolute - origin
    images_ts = images_ts_absolute - origin
    image_event_indices = np.searchsorted(events_ts, images_ts, side="right") - 1
    image_event_indices = np.clip(image_event_indices, 0, len(events_ts) - 1).astype(np.int64)
    if len(image_event_indices) != len(images_ts) or np.any(np.diff(image_event_indices) < 0):
        raise RuntimeError("decoded image-to-event endpoint association is inconsistent")
    if np.any(events_ts[image_event_indices] > images_ts + 1e-12):
        raise RuntimeError("image_event_indices points to an event after the image endpoint")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        np.save(temporary / "events_ts.npy", events_ts.astype(np.float64), allow_pickle=False)
        np.save(temporary / "events_xy.npy", events_xy.astype(np.int32), allow_pickle=False)
        np.save(temporary / "events_p.npy", events_p.astype(np.uint8), allow_pickle=False)
        np.save(temporary / "images.npy", images_array[..., None].astype(np.uint8), allow_pickle=False)
        np.save(temporary / "images_ts.npy", images_ts[:, None].astype(np.float64), allow_pickle=False)
        np.save(temporary / "image_event_indices.npy", image_event_indices[:, None], allow_pickle=False)
        metadata = {
            "sensor_resolution": [height, width],
            "source_bag": str(bag_path.resolve()),
            "source_bag_sha256": bag_info["bag_sha256"],
            "event_topic": event_topic,
            "image_topic": image_topic,
            "event_message_type": event_type,
            "image_message_type": image_type,
            "event_timestamp_unit": "seconds, ROS event ts",
            "image_timestamp_unit": "seconds, sensor_msgs/Image header.stamp",
            "time_origin_absolute_s": origin,
            "event_polarity_encoding": "0=OFF, 1=ON (same convention expected by pinned EVREAL loader)",
            "conversion": "rosbags 0.11.7; event x/y/t/p read from embedded ROS1 types; mono8 reference images; image_event_indices=clip(searchsorted(events_ts,images_ts,right)-1)",
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        files = {}
        for path in sorted(temporary.iterdir()):
            files[path.name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest = {
            "status": "pass",
            "scope": "registered public HQF bag conversion for candidate event-only reconstruction inference; reference images are not read by the model",
            "source_bag": bag_info,
            "topics": {"event_topic": event_topic, "event_message_type": event_type, "event_messages": event_message_count,
                       "image_topic": image_topic, "image_message_type": image_type, "image_messages": image_message_count},
            "event_count": int(len(events_ts)),
            "image_count": int(len(images_ts)),
            "sensor_resolution_hw": [height, width],
            "event_timestamp_absolute_s": [float(events_ts_absolute[0]), float(events_ts_absolute[-1])],
            "image_timestamp_absolute_s": [float(images_ts_absolute[0]), float(images_ts_absolute[-1])],
            "event_timestamp_relative_s": [float(events_ts[0]), float(events_ts[-1])],
            "image_timestamp_relative_s": [float(images_ts[0]), float(images_ts[-1])],
            "event_encoding": {"xy": "int32 pixel coordinates", "time": "float64 seconds from min source timestamp", "polarity": "uint8 0=OFF/1=ON"},
            "reference_image_encodings": sorted(image_encodings),
            "image_event_index_rule": "right-inclusive event endpoint; latest event timestamp <= image timestamp; clipped to [0,N-1] to match EVREAL bag_to_npy behavior",
            "reference_images_used_as_model_input": False,
            "files": files,
        }
        (temporary / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    manifest["output_dir"] = str(output_dir.resolve())
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--event-topic", default="/dvs/events")
    parser.add_argument("--image-topic", default="/dvs/image_raw")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if not args.bag.is_file():
        raise FileNotFoundError(args.bag)
    if args.inspect_only:
        print(json.dumps(inspect_bag(args.bag), indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --inspect-only is set")
    convert_bag(args.bag, args.output_dir, args.event_topic, args.image_topic)


if __name__ == "__main__":
    main()
