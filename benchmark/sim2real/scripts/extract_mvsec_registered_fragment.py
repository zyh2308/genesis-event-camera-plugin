#!/usr/bin/env python3
"""Materialize only the preregistered MVSEC fragment, preserving original indices."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_dataset(source: h5py.Dataset, destination: h5py.Group, name: str, selection, **kwargs) -> None:
    values = np.asarray(source[selection])
    destination.create_dataset(name, data=values, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-h5", type=Path, required=True)
    parser.add_argument("--gt-h5", type=Path, required=True)
    parser.add_argument("--resolution-json", type=Path, required=True)
    parser.add_argument("--data-output", type=Path, required=True)
    parser.add_argument("--gt-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.data_output, args.gt_output, args.manifest_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    resolved = json.loads(args.resolution_json.read_text())
    if resolved.get("status") != "pass" or "dt1" not in resolved.get("windows", {}):
        raise ValueError("prerun timestamp resolution did not pass")
    registered = [row for dt in ("dt1", "dt4") for row in resolved["windows"][dt]]
    registered_source_hashes = resolved["inputs"]
    source_data_hash = registered_source_hashes[str(args.data_h5.resolve())]["sha256"]
    source_gt_hash = registered_source_hashes[str(args.gt_h5.resolve())]["sha256"]
    max_aps_index = max(row["aps_end_index"] for row in registered)
    max_event_stop = max(row["event_stop_exclusive_index"] for row in registered)

    with h5py.File(args.data_h5, "r") as source:
        src = source["davis/left"]
        event_ds = src["events"]
        if max_event_stop > len(event_ds) or max_aps_index >= len(src["image_raw_ts"]):
            raise IndexError("registered slice exceeds original public MVSEC data arrays")
        event_values = np.asarray(event_ds[:max_event_stop])
        if event_values.ndim != 2 or event_values.shape[1] != 4:
            raise ValueError(f"raw event shape is not [N,4]: {event_values.shape}")
        aps_stop = max_aps_index + 1
        aps_ts = np.asarray(src["image_raw_ts"][:aps_stop], dtype=np.float64).reshape(-1)
        aps_event_idx = np.asarray(src["image_raw_event_inds"][:aps_stop], dtype=np.int64).reshape(-1)
        aps_images = np.asarray(src["image_raw"][:aps_stop])
        # Independent per-window check against the original, unsliced events.
        original_ts = np.asarray(event_ds[:, 2], dtype=np.float64)
        event_checks = []
        for row in registered:
            lo = int(np.searchsorted(original_ts, row["start_time_s"], side="right"))
            hi = int(np.searchsorted(original_ts, row["end_time_s"], side="right"))
            if (lo, hi) != (row["event_left_exclusive_index"], row["event_stop_exclusive_index"]):
                raise RuntimeError(f"original event recount differs for APS {row['aps_start_index']}")
            if hi > len(event_values):
                raise RuntimeError("exported event slice truncates a registered interval")
            event_checks.append({
                "dt_window_start_index": row["aps_start_index"],
                "aps_end_index": row["aps_end_index"],
                "event_count_independent_original_recount": hi - lo,
                "event_stop_exclusive_index_original": hi,
            })

        args.data_output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.data_output, "w") as dest:
            dg = dest.create_group("davis/left")
            dg.create_dataset("events", data=event_values, compression="gzip", compression_opts=1, shuffle=True)
            dg.create_dataset("image_raw_ts", data=aps_ts)
            dg.create_dataset("image_raw_event_inds", data=aps_event_idx)
            dg.create_dataset("image_raw", data=aps_images, compression="gzip", compression_opts=1, shuffle=True)
            dest.attrs["source_full_file"] = str(args.data_h5.resolve())
            dest.attrs["source_full_file_sha256"] = source_data_hash
            dest.attrs["slice_definition"] = "public indoor_flying1 data; original APS rows [0,max_registered_endpoint] and raw event rows [0,event_stop_at_max_endpoint)"
            dest.attrs["original_max_aps_endpoint_index"] = max_aps_index
            dest.attrs["original_event_stop_exclusive"] = max_event_stop
            dest.attrs["registration_json_sha256"] = sha256(args.resolution_json)

    with h5py.File(args.gt_h5, "r") as source:
        src = source["davis/left"]
        flow = src["flow_dist"]
        gt_ts = np.asarray(src["flow_dist_ts"], dtype=np.float64).reshape(-1)
        if flow.ndim != 4 or flow.shape[1:] != (2, 260, 346):
            raise ValueError(f"unexpected original flow_dist shape {flow.shape}")
        if len(gt_ts) != flow.shape[0]:
            raise ValueError("original MVSEC GT contract unexpectedly lacks one timestamp per flow map")
        associations = []
        for row in registered:
            start, end = float(row["start_time_s"]), float(row["end_time_s"])
            first = int(np.searchsorted(gt_ts, start, side="right") - 1)
            last = int(np.searchsorted(gt_ts, end, side="left") - 1)
            if first < 0 or last < first or last >= flow.shape[0] - 1:
                raise IndexError(f"registered APS window has invalid source GT map range [{first},{last}]")
            associations.append((first, last))
        gt_first = min(pair[0] for pair in associations)
        gt_last = max(pair[1] for pair in associations)
        # Keep the next endpoint timestamp so the pinned STE association logic
        # has the same local [start,...,end] segment as it had in the full file.
        gt_ts_stop = gt_last + 2
        gt_maps_stop = gt_last + 1
        gt_ts_slice = gt_ts[gt_first:gt_ts_stop]
        gt_maps_slice = np.asarray(flow[gt_first:gt_maps_stop])
        if len(gt_ts_slice) != len(gt_maps_slice) + 1:
            raise RuntimeError("cropped GT map/timestamp endpoint contract is inconsistent")
        args.gt_output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.gt_output, "w") as dest:
            dg = dest.create_group("davis/left")
            dg.create_dataset("flow_dist", data=gt_maps_slice, compression="gzip", compression_opts=1, shuffle=True)
            dg.create_dataset("flow_dist_ts", data=gt_ts_slice)
            dest.attrs["source_full_file"] = str(args.gt_h5.resolve())
            dest.attrs["source_full_file_sha256"] = source_gt_hash
            dest.attrs["source_flow_map_indices_inclusive"] = [gt_first, gt_last]
            dest.attrs["source_timestamp_indices_inclusive"] = [gt_first, gt_last + 1]
            dest.attrs["slice_definition"] = "minimal contiguous source MVSEC flow maps needed by preregistered dt1/dt4 APS windows, with one trailing GT endpoint timestamp"
            dest.attrs["registration_json_sha256"] = sha256(args.resolution_json)

    manifest = {
        "status": "pass",
        "scope": "lossless fixed-window slice of public MVSEC indoor_flying1; no candidate model loaded",
        "preregistration": str(args.resolution_json.resolve()),
        "preregistration_sha256": sha256(args.resolution_json),
        "original_public_data": {
            str(args.data_h5.resolve()): {"size_bytes": args.data_h5.stat().st_size, "sha256": source_data_hash},
            str(args.gt_h5.resolve()): {"size_bytes": args.gt_h5.stat().st_size, "sha256": source_gt_hash},
        },
        "derived_slice": {
            "data_h5": {"path": str(args.data_output.resolve()), "size_bytes": args.data_output.stat().st_size, "sha256": sha256(args.data_output), "event_rows_original": [0, max_event_stop], "aps_rows_original": [0, max_aps_index + 1]},
            "gt_h5": {"path": str(args.gt_output.resolve()), "size_bytes": args.gt_output.stat().st_size, "sha256": sha256(args.gt_output), "flow_rows_original_inclusive": [gt_first, gt_last], "timestamp_rows_original_inclusive": [gt_first, gt_last + 1]},
            "aps_indices_for_inference": {"dt1": [row["aps_start_index"] for row in resolved["windows"]["dt1"]], "dt4": [row["aps_start_index"] for row in resolved["windows"]["dt4"]]},
            "window_event_recounts": event_checks,
        },
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
