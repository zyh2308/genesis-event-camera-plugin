"""Audit endpoint-window event counts against producer capture chunks."""
from __future__ import annotations

import numpy as np


def audit_capture_partition(
    event_timestamps: np.ndarray,
    endpoint_times: np.ndarray,
    timestamp_window_counts: np.ndarray,
    capture_window_counts: np.ndarray,
) -> dict:
    """Verify that count differences come only from exact endpoint ties.

    The shared event-window contract is ``(t_i, t_{i+1}]``. An event whose
    timestamp equals an endpoint is therefore placed in the preceding window
    by ``searchsorted(..., side="right")``. A producer may emit that event in
    the next capture call; this function allows and records that bookkeeping
    difference without changing the timestamp-based training window.
    """
    timestamps = np.asarray(event_timestamps, dtype=np.float64).reshape(-1)
    endpoints = np.asarray(endpoint_times, dtype=np.float64).reshape(-1)
    timestamp_counts = np.asarray(timestamp_window_counts, dtype=np.int64).reshape(-1)
    capture_counts = np.asarray(capture_window_counts, dtype=np.int64).reshape(-1)
    interval_count = len(endpoints) - 1
    if interval_count < 1 or timestamp_counts.shape != (interval_count,):
        raise ValueError("timestamp window count shape does not match endpoints")
    if capture_counts.shape != timestamp_counts.shape:
        raise ValueError("capture window count shape does not match endpoints")
    if int(timestamp_counts.sum()) != len(timestamps) or int(capture_counts.sum()) != len(timestamps):
        raise ValueError("capture/timestamp partitions do not cover the same event stream")

    # moved[k] is the number of exact-timestamp events at t_k that were emitted
    # by capture interval k instead of timestamp interval k-1.
    delta = timestamp_counts - capture_counts
    moved = np.zeros(interval_count + 1, dtype=np.int64)
    for interval in range(interval_count):
        moved[interval + 1] = moved[interval] + delta[interval]
    if moved[-1] != 0:
        raise ValueError(f"capture/timestamp count mismatch is not balanced: {delta.tolist()}")

    # Resolve all exact endpoint ties with two vectorized binary searches.
    # Counting across the entire event stream once per endpoint is O(N*T) and
    # made the 32-rollout reconstruction adapter effectively unusable.
    tied_at_endpoint = np.zeros(interval_count + 1, dtype=np.int64)
    internal_endpoints = endpoints[1:interval_count]
    if len(internal_endpoints):
        tie_left = np.searchsorted(timestamps, internal_endpoints, side="left")
        tie_right = np.searchsorted(timestamps, internal_endpoints, side="right")
        tied_at_endpoint[1:interval_count] = tie_right - tie_left
    for endpoint_index in range(1, interval_count):
        if moved[endpoint_index] < 0 or moved[endpoint_index] > tied_at_endpoint[endpoint_index]:
            raise ValueError(
                "capture/timestamp count mismatch is not explained by exact endpoint ties: "
                f"endpoint={endpoint_index}, moved={int(moved[endpoint_index])}, "
                f"ties={int(tied_at_endpoint[endpoint_index])}, delta={delta.tolist()}"
            )

    return {
        "status": "pass",
        "window_contract": "(t_i,t_i_plus_1] by event timestamp; tied events assigned to the right endpoint",
        "timestamp_window_counts": timestamp_counts.astype(int).tolist(),
        "producer_capture_window_counts": capture_counts.astype(int).tolist(),
        "timestamp_minus_capture_count_delta": delta.astype(int).tolist(),
        "exact_internal_endpoint_tie_count": int(tied_at_endpoint.sum()),
        "exact_tie_events_emitted_in_next_capture_call": int(moved.sum()),
        "internal_endpoint_ties": [
            {
                "endpoint_index": int(index),
                "endpoint_time_s": float(endpoints[index]),
                "tie_event_count": int(tied_at_endpoint[index]),
                "moved_to_next_capture_count": int(moved[index]),
            }
            for index in range(1, interval_count)
            if tied_at_endpoint[index]
        ],
    }
