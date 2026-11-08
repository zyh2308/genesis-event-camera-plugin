"""Benchmark event-observation windows, extracted from the cached v2 diagnostic.

No target state, RGB/depth, renderer, physics or policy access. Camera poses are
exact available endpoints or explicitly approximate SE(3) interpolation at a
child-window end. A child retains its original parent's information availability.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: _serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_serial(v) for v in value]
    return value


def pose_between(t, t0, t1, T0, T1):
    assert t1 > t0 and t0-1e-8 <= t <= t1+1e-8
    if abs(t-t0) < 1e-10: return np.asarray(T0, dtype=float).copy()
    if abs(t-t1) < 1e-10: return np.asarray(T1, dtype=float).copy()
    alpha = (float(t)-t0)/(t1-t0)
    result = np.eye(4)
    result[:3, :3] = Slerp([0., 1.], Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]])))(alpha).as_matrix()
    result[:3, 3] = (1-alpha)*T0[:3, 3]+alpha*T1[:3, 3]
    return result


def history_camera_pose(t, times, poses, endpoint_available, parent_available):
    nearest = int(np.argmin(abs(times-t)))
    if abs(times[nearest]-t) < 1e-9:
        assert endpoint_available[nearest] <= parent_available+1e-9, 'Camera endpoint not yet available'
        return poses[nearest].copy(), [float(times[nearest])], [float(endpoint_available[nearest])]
    upper = int(np.searchsorted(times, t, side='right')); lower = upper-1
    assert 0 <= lower < upper < len(times), 'Camera history does not bracket event subwindow'
    assert endpoint_available[lower] <= parent_available+1e-9 and endpoint_available[upper] <= parent_available+1e-9, 'Camera interpolation would use an unavailable endpoint'
    T = pose_between(t, float(times[lower]), float(times[upper]), poses[lower], poses[upper])
    return T, [float(times[lower]), float(times[upper])], [float(endpoint_available[lower]), float(endpoint_available[upper])]


def split_packet(packet, times, poses, endpoint_available, window_s, parent_packet_ids):
    """Preserve actual event times, including carry-in, with unique input ranges."""
    end = float(packet['end_s']); available = float(packet['available_s'])
    k = int(np.argmin(abs(times-end)))
    assert abs(times[k]-end) < 1e-8
    if k == 0:
        assert abs(end) < 1e-8 and len(packet['events']) == 0, 'Only empty reset may lack a prior endpoint'
        assert endpoint_available[0] <= available+1e-9
        return []
    start = float(times[k-1]); parent_id = int(parent_packet_ids[k])
    assert parent_id >= 0 and start < end <= available+1e-8
    assert endpoint_available[k-1] <= available+1e-9 and endpoint_available[k] <= available+1e-9
    np.testing.assert_allclose(packet['T'], poses[k], rtol=0, atol=1e-10)
    events = np.asarray(packet['events'])
    if len(events):
        assert np.isfinite(events).all()
        assert float(events[:, 0].min()) >= float(times[0])-1e-8, 'Events precede camera history'
        assert float(events[:, 0].max()) <= end+1e-8, 'Events exceed parent endpoint; separate review required'
        assert np.all(np.diff(events[:, 0]) >= -1e-12)
    carry_windows = max(0, int(math.ceil((start-float(events[:, 0].min()))/window_s-1e-8))) if len(events) else 0
    back = [max(float(times[0]), start-i*window_s) for i in range(carry_windows, 0, -1)]
    n = max(1, int(math.ceil((end-start)/window_s-1e-8)))
    cuts = back+[start]+[min(end, start+i*window_s) for i in range(1, n)]+[end]
    assert all(hi > lo+1e-12 for lo, hi in zip(cuts[:-1], cuts[1:]))
    cursor = 0; parts = []
    for i, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
        stop = len(events) if i == len(cuts)-2 else int(np.searchsorted(events[:, 0], hi, side='left'))
        assert stop >= cursor
        first = cursor; part = events[cursor:stop]; cursor = stop
        T, pose_times, pose_available = history_camera_pose(hi, times, poses, endpoint_available, available)
        parts.append(dict(events=part, available_s=available, end_s=float(hi), T=T,
            parent_packet_id=parent_id, subwindow_index=i, input_event_index_range=[first, stop],
            start_s=float(lo), nominal_parent_start_s=start, carry_in_historical_window=bool(hi <= start+1e-10),
            pose_source=('exact_available_historical_camera_endpoint' if len(pose_times) == 1 else 'approx_available_camera_history_SE3_slerp_translation_linear_at_subwindow_end'),
            camera_endpoint_times_s=pose_times, camera_endpoint_available_s=pose_available,
            _observation_window_child=True))
    assert cursor == len(events) and sum(len(s['events']) for s in parts) == len(events)
    if len(events): np.testing.assert_array_equal(np.concatenate([s['events'] for s in parts]), events)
    return parts


class EventObservationWindows:
    """Only the fixed, cached-validated 10 ms benchmark candidate is enabled."""
    def __init__(self, window_ms):
        if float(window_ms) != 10.: raise ValueError('Observation windows support only the validated 10 ms candidate')
        self.window_s = .01
        self.endpoints = []
        self.parents = {}
        self.next_parent_id = 0
        self.consumed_parent_ids = set()
        self.last_raw_event_s = -np.inf
        self.seen_window_ends = {}
        self.processing_now_s = None
        self.audit = []

    def register_parent(self, packet):
        """push pre-registration and direct _detect auto-registration are idempotent."""
        key = (float(packet['end_s']), float(packet['available_s']))
        previous = self.parents.get(key)
        if previous is not None:
            if 'parent_packet_id' in packet: assert int(packet['parent_packet_id']) == previous['parent_packet_id']
            np.testing.assert_array_equal(packet['T'], previous['T'])
            assert len(packet['events']) == previous['event_count']
            packet['parent_packet_id'] = previous['parent_packet_id']
            return previous['parent_packet_id']
        if self.endpoints: assert key[0] > self.endpoints[-1]['time_s'], 'Camera endpoint registration must be chronological'
        parent_id = int(packet.get('parent_packet_id', self.next_parent_id))
        assert all(p['parent_packet_id'] != parent_id for p in self.endpoints), 'Parent packet ID reused'
        self.next_parent_id = max(self.next_parent_id, parent_id+1)
        entry = dict(time_s=key[0], available_s=key[1], T=np.asarray(packet['T'], dtype=float).copy(),
                     parent_packet_id=parent_id, event_count=len(packet['events']))
        self.endpoints.append(entry); self.parents[key] = entry
        packet['parent_packet_id'] = parent_id
        return parent_id

    def consume(self, tracker, packet, detect_child):
        parent_id = self.register_parent(packet)
        assert parent_id not in self.consumed_parent_ids, 'Arrival parent processed twice'
        if self.processing_now_s is not None:
            assert packet['available_s'] <= self.processing_now_s+1e-9, 'Parent not yet available at observe'
        raw = np.asarray(packet['events']); previous_raw_max = float(self.last_raw_event_s)
        if len(raw) and float(raw[:, 0].min()) < previous_raw_max-1e-12:
            self.audit.append(dict(status='STOP_OUT_OF_SEQUENCE_RAW', parent_packet_id=parent_id,
                minimum_event_s=float(raw[:, 0].min()), previous_consumed_raw_max_s=previous_raw_max))
            raise RuntimeError('Out-of-sequence raw events across arrival parents; filter was not rewound')
        times = np.asarray([p['time_s'] for p in self.endpoints])
        poses = np.asarray([p['T'] for p in self.endpoints])
        availability = np.asarray([p['available_s'] for p in self.endpoints])
        ids = np.asarray([p['parent_packet_id'] for p in self.endpoints])
        parts = split_packet(packet, times, poses, availability, self.window_s, ids)
        record = dict(status='PROCESSING', parent_packet_id=parent_id,
            original_end_s=float(packet['end_s']), original_available_s=float(packet['available_s']),
            processing_now_s=self.processing_now_s, processing_clock_measured=self.processing_now_s is not None,
            original_events=len(raw), previous_parent_raw_max_s=previous_raw_max if np.isfinite(previous_raw_max) else None,
            actual_event_min_s=float(raw[:, 0].min()) if len(raw) else None,
            actual_event_max_s=float(raw[:, 0].max()) if len(raw) else None,
            raw_events_equal_previous_parent_max=int(np.sum(raw[:, 0] == previous_raw_max)) if len(raw) else 0,
            windows=[])
        self.audit.append(record); before = len(tracker.samples)
        for child in parts:
            sample = detect_child(child)
            same_measurement_time = False
            if sample is not None:
                assert sample['available_s'] == packet['available_s']
                assert sample['t'] <= child['end_s']+1e-8
                assert sample['maximum_event_s'] <= child['end_s']+1e-8
                assert sample['maximum_event_s'] <= sample['available_s']+1e-8
                filter_time = getattr(tracker, 'filter_time', None)
                latest = tracker.samples[-1]['t'] if tracker.samples else None
                same_measurement_time = bool(latest is not None and abs(sample['t']-latest) <= 1e-12)
                if ((filter_time is not None and sample['t'] < filter_time-1e-12) or
                    (latest is not None and sample['t'] < latest-1e-12)):
                    record.update(status='STOP_OUT_OF_SEQUENCE_MEASUREMENT', measurement_t=float(sample['t']), filter_time_s=filter_time)
                    raise RuntimeError('Measurement older than processed filter/sample; filter was not rewound')
                tracker.samples.append(sample)
                tracker.last_event_time = max(tracker.last_event_time, sample['maximum_event_s'])
            # Apply each observed timestamp before the next child; query outputs
            # remain at the original observe polls and availability is unchanged.
            tracker._predict(child['end_s'])
            end_key = round(child['end_s'], 12)
            prior_parents = self.seen_window_ends.get(end_key, [])
            record['windows'].append(dict(start_s=child['start_s'], end_s=child['end_s'],
                parent_packet_id=parent_id, subwindow_index=child['subwindow_index'],
                input_event_index_range=child['input_event_index_range'], events=len(child['events']),
                carry_in_historical_window=child['carry_in_historical_window'],
                nominal_parent_start_s=child['nominal_parent_start_s'],
                same_window_end_previous_parent_ids=list(prior_parents),
                same_measurement_timestamp_as_previous=same_measurement_time,
                available_s=child['available_s'], camera_pose_time_s=child['end_s'],
                pose_source=child['pose_source'], camera_endpoint_times_s=child['camera_endpoint_times_s'],
                camera_endpoint_available_s=child['camera_endpoint_available_s'],
                camera_to_world=child['T'].tolist(), detected=_serial(sample)))
            self.seen_window_ends.setdefault(end_key, []).append(parent_id)
        if len(raw): self.last_raw_event_s = max(previous_raw_max, float(raw[:, 0].max()))
        self.consumed_parent_ids.add(parent_id)
        record.update(status='PASS', subwindows=len(parts), added_samples=len(tracker.samples)-before,
            carry_in_window_count=sum(w['carry_in_historical_window'] for w in record['windows']),
            carry_in_event_count=sum(w['events'] for w in record['windows'] if w['carry_in_historical_window']),
            event_partition_count=sum(w['events'] for w in record['windows']),
            event_partition_exact=True, raw_event_order_check='PASS')
        return None  # samples were appended once; prevent caller duplication
