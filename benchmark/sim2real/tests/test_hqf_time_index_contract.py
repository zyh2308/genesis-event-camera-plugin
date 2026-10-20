#!/usr/bin/env python3
"""Independent timestamp-boundary checks for the HQF EVREAL adapter."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from convert_hqf_bag_rosbags import image_endpoint_offsets, source_image_timestamp_keep_indices


class HQFTimeIndexContractTests(unittest.TestCase):
    def test_right_inclusive_endpoint_offsets_and_first_interval(self):
        # The first image is after one event; the next endpoint ties two events.
        # The last interval has no events. Offsets are exclusive slice ends.
        event_ts = np.asarray([0.01, 0.05, 0.05, 0.09], dtype=np.float64)
        image_ts = np.asarray([0.03, 0.05, 0.07, 0.20], dtype=np.float64)
        offsets = image_endpoint_offsets(event_ts, image_ts)
        self.assertEqual(offsets.tolist(), [1, 3, 3, 4])
        windows = [event_ts[:offsets[0]],
                   event_ts[offsets[0]:offsets[1]],
                   event_ts[offsets[1]:offsets[2]],
                   event_ts[offsets[2]:offsets[3]]]
        self.assertEqual([window.tolist() for window in windows], [[0.01], [0.05, 0.05], [], [0.09]])
        for index, window in enumerate(windows):
            self.assertFalse(np.any(window > image_ts[index]))
        self.assertEqual(windows[0][-1], 0.01)  # initial event interval is retained

    def test_legacy_evreal_index_would_drop_boundary_events(self):
        event_ts = np.asarray([0.01, 0.05, 0.05, 0.09], dtype=np.float64)
        image_ts = np.asarray([0.03, 0.05, 0.07, 0.20], dtype=np.float64)
        legacy = np.clip(np.searchsorted(event_ts, image_ts, side="right") - 1,
                         0, len(event_ts) - 1)
        corrected = image_endpoint_offsets(event_ts, image_ts)
        self.assertEqual(legacy.tolist(), [0, 2, 2, 3])
        self.assertEqual(corrected.tolist(), [1, 3, 3, 4])
        # The old value is subsequently consumed as an exclusive offset by
        # EVREAL's between_frames dataset, dropping the latest included event.
        self.assertLess(legacy[1], corrected[1])

    def test_invalid_timestamp_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sorted"):
            image_endpoint_offsets(np.asarray([0.2, 0.1]), np.asarray([0.3]))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            image_endpoint_offsets(np.asarray([0.1]), np.asarray([0.2, 0.2]))
        with self.assertRaisesRegex(ValueError, "finite"):
            image_endpoint_offsets(np.asarray([0.1]), np.asarray([np.nan]))

    def test_empty_event_stream_has_zero_offsets(self):
        offsets = image_endpoint_offsets(
            np.asarray([], dtype=np.float64),
            np.asarray([0.01, 0.02, 0.03], dtype=np.float64),
        )
        self.assertEqual(offsets.tolist(), [0, 0, 0])

    def test_bad_source_image_time_is_recorded_without_sorting_or_imputation(self):
        keep, excluded = source_image_timestamp_keep_indices(
            np.asarray([100, 200, 50, 300, 300, 400], dtype=np.int64)
        )
        self.assertEqual(keep, [0, 1, 3, 5])
        self.assertEqual([row["source_image_index"] for row in excluded], [2, 4])
        self.assertEqual([row["header_timestamp_ns"] for row in excluded], [50, 300])
        self.assertEqual(excluded[0]["previous_retained_source_image_index"], 1)
        self.assertEqual(excluded[1]["previous_retained_source_image_index"], 3)


if __name__ == "__main__":
    unittest.main()
