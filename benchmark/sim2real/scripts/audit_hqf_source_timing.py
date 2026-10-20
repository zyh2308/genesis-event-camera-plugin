#!/usr/bin/env python3
"""Audit HQF image header timestamps against ROS bag record order.

This is a read-only source audit. It does not sort, drop, or rewrite frames.
Integer nanoseconds are retained so float conversion cannot hide timestamp ties.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def stamp_to_ns(stamp) -> int:
    sec = getattr(stamp, "sec", getattr(stamp, "secs", None))
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsec", getattr(stamp, "nsecs", None)))
    if sec is None or nsec is None:
        raise ValueError(f"timestamp has no seconds/nanoseconds fields: {type(stamp)!r}")
    return int(sec) * 1_000_000_000 + int(nsec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--image-topic", default="/dvs/image_raw")
    parser.add_argument("--event-topic", default="/dvs/events")
    parser.add_argument("--include-event-timing", action="store_true",
                        help="also decode event batches and associate nearest batches by bag record time")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    rows = []
    event_batches = []
    store = get_typestore(Stores.ROS1_NOETIC)
    with Reader(args.bag) as reader:
        requested_topics = {args.image_topic}
        if args.include_event_timing:
            requested_topics.add(args.event_topic)
        connections = [c for c in reader.connections if c.topic in requested_topics]
        if not any(c.topic == args.image_topic for c in connections):
            raise KeyError(f"bag has no image topic {args.image_topic!r}")
        if args.include_event_timing and not any(c.topic == args.event_topic for c in connections):
            raise KeyError(f"bag has no event topic {args.event_topic!r}")
        for connection in connections:
            store.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
        for connection, bag_timestamp_ns, raw in reader.messages(connections=connections):
            message = store.deserialize_ros1(raw, connection.msgtype)
            if connection.topic == args.image_topic:
                stamp = message.header.stamp
                header_ns = stamp_to_ns(stamp)
                rows.append({
                    "bag_record_ns": int(bag_timestamp_ns),
                    "header_stamp_ns": header_ns,
                    "encoding": str(message.encoding),
                    "height": int(message.height),
                    "width": int(message.width),
                    "image_payload_sha256": hashlib.sha256(message.data).hexdigest(),
                })
            elif connection.topic == args.event_topic:
                events = message.events
                if events:
                    first, last = events[0].ts, events[-1].ts
                    event_batches.append({
                        "bag_record_ns": int(bag_timestamp_ns),
                        "first_event_ns": stamp_to_ns(first),
                        "last_event_ns": stamp_to_ns(last),
                        "event_count": len(events),
                    })

    if not rows:
        raise ValueError("image topic has no messages")
    header_ties = defaultdict(list)
    for index, row in enumerate(rows):
        header_ties[row["header_stamp_ns"]].append(index)
    adjacent_violations = [
        {"previous_index": index - 1, "index": index,
         "previous_header_stamp_ns": rows[index - 1]["header_stamp_ns"],
         "header_stamp_ns": rows[index]["header_stamp_ns"],
         "previous_bag_record_ns": rows[index - 1]["bag_record_ns"],
         "bag_record_ns": rows[index]["bag_record_ns"],
         "same_image_payload": rows[index - 1]["image_payload_sha256"] == rows[index]["image_payload_sha256"]}
        for index in range(1, len(rows))
        if rows[index]["header_stamp_ns"] <= rows[index - 1]["header_stamp_ns"]
    ]
    duplicate_groups = []
    for header_stamp_ns, indices in sorted(header_ties.items()):
        if len(indices) > 1:
            payload_hashes = sorted({rows[index]["image_payload_sha256"] for index in indices})
            duplicate_groups.append({
                "header_stamp_ns": header_stamp_ns,
                "indices_in_bag_order": indices,
                "distinct_image_payload_sha256": payload_hashes,
                "all_payloads_identical": len(payload_hashes) == 1,
            })
    for violation in adjacent_violations:
        if event_batches:
            record_ns = violation["bag_record_ns"]
            before = [batch for batch in event_batches if batch["bag_record_ns"] <= record_ns]
            after = [batch for batch in event_batches if batch["bag_record_ns"] >= record_ns]
            violation["nearest_event_batch_before"] = before[-1] if before else None
            violation["nearest_event_batch_after"] = after[0] if after else None
    event_order_violations = [
        {"previous_batch_index": index - 1, "batch_index": index,
         "previous_last_event_ns": event_batches[index - 1]["last_event_ns"],
         "first_event_ns": event_batches[index]["first_event_ns"]}
        for index in range(1, len(event_batches))
        if event_batches[index]["first_event_ns"] < event_batches[index - 1]["last_event_ns"]
    ]
    script_path = Path(__file__).resolve()
    script_sha = hashlib.sha256(script_path.read_bytes()).hexdigest()
    result = {
        "schema": "hqf_source_image_timing_audit_v1",
        "status": "audited_no_repair",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script": str(script_path),
        "audit_script_sha256": script_sha,
        "bag_path": str(args.bag.resolve()),
        "image_topic": args.image_topic,
        "image_message_count": len(rows),
        "timestamp_unit": "integer nanoseconds",
        "order": {
            "bag_record_nondecreasing": all(rows[i]["bag_record_ns"] >= rows[i - 1]["bag_record_ns"] for i in range(1, len(rows))),
            "header_stamp_strictly_increasing_in_bag_order": all(rows[i]["header_stamp_ns"] > rows[i - 1]["header_stamp_ns"] for i in range(1, len(rows))),
            "header_stamp_non_decreasing_in_bag_order": all(rows[i]["header_stamp_ns"] >= rows[i - 1]["header_stamp_ns"] for i in range(1, len(rows))),
        },
        "first_image": rows[0],
        "last_image": rows[-1],
        "adjacent_header_order_violations": adjacent_violations,
        "duplicate_header_stamp_groups": duplicate_groups,
        "event_timing": ({
            "event_topic": args.event_topic,
            "nonempty_event_message_count": len(event_batches),
            "event_count": sum(batch["event_count"] for batch in event_batches),
            "first_event_ns": event_batches[0]["first_event_ns"],
            "last_event_ns": event_batches[-1]["last_event_ns"],
            "bag_record_order_monotonic_for_event_times": not event_order_violations,
            "event_order_violations": event_order_violations,
        } if args.include_event_timing else None),
        "frames": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "image_message_count": len(rows),
        "order": result["order"],
        "adjacent_header_order_violations": len(adjacent_violations),
        "duplicate_header_stamp_groups": len(duplicate_groups),
        "event_timing": result["event_timing"],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
