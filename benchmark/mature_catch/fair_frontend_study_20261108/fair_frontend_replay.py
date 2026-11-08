"""Symmetric, CPU-only development-cache screening for the robot task frontend.

The estimator receives only archived events, original packet availability,
camera calibration/history, public gravity and the shared rendered RGB-D init.
Target labels are loaded after inference and used only by scoring/parity.
This is a fixed-trajectory flight estimator replay, not a closed-loop result.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '2')
import numpy as np


SCORE_RULE = dict(position_failure_cap_m=.5, velocity_failure_cap_m_s=2.,
    position_weight=.5, velocity_weight=.5,
    missing_or_nonmoving_query_loss=1.,
    false_moving_penalty='add pre-release actually published moving-query fraction',
    grouping='average variants within source/seed/cadence, then equal seed-cadence cells',
    ranking='total loss, greater moving coverage, smaller gap, prefer existing fit1, candidate ID',
    shortlist='one best per source/window family, union plus canonical window10-fit1-gap50')


@dataclass(frozen=True)
class Candidate:
    window_ms: int
    flight_fit: bool
    gap_ms: int

    @property
    def identifier(self):
        return f'w{self.window_ms}_fit{int(self.flight_fit)}_gap{self.gap_ms}'

    def receipt(self):
        return dict(candidate_id=self.identifier, frontend_id=self.identifier, **asdict(self),
            env=dict(CATCHIT_EVENT_OBSERVATION_WINDOW_MS=str(self.window_ms),
                     CATCHIT_EVENT_FLIGHT_FIT=str(int(self.flight_fit)),
                     CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS=str(self.gap_ms)))


CANDIDATES = tuple(Candidate(w, f, g) for w in (0, 10) for f in (False, True) for g in (50, 100))
CANONICAL = Candidate(10, True, 50).identifier


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(v) for v in value]
    return value


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(serial(value), indent=2, allow_nan=False), encoding='utf-8', newline='\n')
    temporary.replace(path)


@dataclass
class PublicCache:
    initialization: dict
    query_times: np.ndarray
    camera_times: np.ndarray
    camera_to_world: np.ndarray
    intrinsics: np.ndarray
    resolution_wh: tuple
    packets: list
    events: np.ndarray
    offsets: np.ndarray
    sphere_radius_m: float
    capture_schedule: dict
    receipt: dict


def load_public_cache(run, protocol):
    """Read an allowlisted public input bundle; no target poses/depth/seg labels."""
    import h5py
    sensor = run/'migration_audit/sensor_episode_000'
    initialization = read(sensor/'initialization.json')
    assert initialization['valid'], 'A failed common initialization cannot be silently discarded'
    assert not any('gt' in key.lower() or 'ground_truth' in key.lower() for key in initialization)
    packets = read(sensor/'packet_audit.json')
    with np.load(sensor/'raw_events.npz', allow_pickle=False) as archive:
        events = archive['events']
    assert events.ndim == 2 and events.shape[1] == 4
    offsets = np.cumsum([0]+[int(p['events']) for p in packets])
    assert len(events) == int(offsets[-1])
    # Only these arrays are accessed. Image/depth/segmentation/target-pose values
    # never enter this replay. The image dataset's shape is resolution metadata.
    hdf = sensor/'sparse_hdr_endpoints.h5'
    with h5py.File(hdf, 'r') as handle:
        times = handle['time_s'][:]
        poses = handle['camera_to_world_cv'][:]
        K = handle['intrinsics'][:]
        image_shape = handle['radiance'].shape
    assert len(times) == len(packets) and np.all(np.diff(times) > 0)
    assert poses.shape == (len(times), 4, 4)
    for timestamp, packet in zip(times, packets):
        assert abs(float(timestamp)-float(packet['time_s'])) < 1e-8
        assert packet['available_s'] >= packet['time_s']-1e-9
    # The trace supplies the original controller query schedule, not target data.
    query_times = np.asarray([row['t_before'] for row in read(run/'episode_000/trace.json')], dtype=float)
    assert len(query_times) and np.all(np.diff(query_times) > 0)
    paths = [sensor/'initialization.json', sensor/'packet_audit.json', sensor/'raw_events.npz',
             hdf, run/'protocol.json', run/'episode_000/trace.json']
    schedule_path = sensor/'capture_schedule.json'
    if schedule_path.exists(): paths.append(schedule_path)
    receipt = dict(files={str(p): sha(p) for p in paths},
        hdf_values_read=['time_s', 'camera_to_world_cv', 'intrinsics'],
        hdf_metadata_read=['radiance.shape'],
        trace_input_fields=['t_before'], target_GT_input=False,
        input_packets=len(packets), input_events=len(events))
    return PublicCache(initialization, query_times, times, poses, K,
        (int(image_shape[2]), int(image_shape[1])), packets, events, offsets,
        float(protocol['sphere_radius_m']), read(schedule_path) if schedule_path.exists() else {}, receipt)


def native_candidate(protocol):
    candidate = Candidate(int(float(protocol.get('event_observation_window_ms', 0))),
                          bool(protocol['event_flight_fit']),
                          int(float(protocol.get('event_confirmation_max_gap_ms', 50))))
    assert candidate in CANDIDATES, ('Archived configuration not in the registered grid', candidate)
    assert not protocol.get('reject_partial_contours') and not protocol.get('revocable_grasp')
    return candidate


def construct_tracker(factory, public, candidate):
    settings = dict(CATCHIT_EVENT_OBSERVATION_WINDOW_MS=str(candidate.window_ms),
        CATCHIT_EVENT_FLIGHT_FIT=str(int(candidate.flight_fit)),
        CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS=str(candidate.gap_ms),
        CATCHIT_REJECT_PARTIAL_CONTOURS='0', CATCHIT_REVOCABLE_GRASP='0')
    previous = {key: os.environ.get(key) for key in settings}
    try:
        os.environ.update(settings)
        tracker = factory(public.intrinsics, public.resolution_wh, public.sphere_radius_m)
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
    assert tracker.observation_window_ms == candidate.window_ms
    assert tracker.use_flight_fit == candidate.flight_fit
    assert abs(tracker.confirmation_max_gap_s-candidate.gap_ms/1000.) < 1e-15
    tracker.gravity = np.asarray(public.initialization['public_world_gravity_m_s2'], dtype=float)
    tracker.initialize_rgb(public.initialization, time_s=0., available_s=0.)
    return tracker


def native_visibility(predicted, tracker, now):
    if predicted is None: return None
    if not np.isfinite(np.r_[predicted['position'], predicted['velocity']]).all(): return None
    if tracker.motion_onset_s is None:
        return predicted if now-tracker.static_latest_s <= 1. else None
    if now-float(predicted['latest_s']) > .40 or float(predicted['fit_error_m']) > .12: return None
    return predicted


def infer(public, factory, candidate):
    """Pure flight replay; no result/target-label object is available to this function."""
    tracker = construct_tracker(factory, public, candidate)
    captured = consumed = 0; states = []
    for now_value in public.query_times:
        now = float(now_value)
        # Captures and availability are different clocks. A captured but not yet
        # available packet can be queued, but is never consumed early.
        while captured < len(public.packets) and public.packets[captured]['time_s'] <= now+1e-9:
            packet = public.packets[captured]
            tracker.push(public.events[public.offsets[captured]:public.offsets[captured+1]],
                available_s=packet['available_s'], packet_time_s=packet['time_s'],
                camera_to_world=public.camera_to_world[captured], parent_packet_id=captured)
            captured += 1
        windows = getattr(tracker, '_observation_windows', None)
        previous_now = windows.processing_now_s if windows is not None else None
        if windows is not None: windows.processing_now_s = now
        try:
            while tracker.pending and tracker.pending[0]['available_s'] <= now+1e-9:
                packet = tracker.pending.popleft()
                assert packet['available_s'] <= now+1e-9
                sample = tracker._detect(packet)
                if sample is not None:
                    assert sample['available_s'] <= now+1e-9
                    tracker.samples.append(sample)
                    tracker.last_event_time = max(tracker.last_event_time, sample['maximum_event_s'])
                consumed += 1
        finally:
            if windows is not None: windows.processing_now_s = previous_now
        predicted = tracker._predict(now)
        visible = native_visibility(predicted, tracker, now)
        state = dict(time_s=now, available=visible is not None,
            moving=bool(visible is not None and tracker.motion_onset_s is not None),
            motion_onset_s=None if tracker.motion_onset_s is None else float(tracker.motion_onset_s),
            position=None if visible is None else np.asarray(visible['position'], dtype=float).tolist(),
            velocity=None if visible is None else np.asarray(visible['velocity'], dtype=float).tolist(),
            latest_s=None if visible is None else float(visible['latest_s']),
            fit_error_m=None if visible is None else float(visible['fit_error_m']),
            sample_count=len(tracker.samples), consumed_packets=consumed,
            maximum_event_s=float(tracker.last_event_time) if np.isfinite(tracker.last_event_time) else None)
        assert state['maximum_event_s'] is None or state['maximum_event_s'] <= now+1e-8
        assert all(s['available_s'] <= now+1e-9 for s in tracker.samples)
        states.append(state)
    expected_consumed = sum(p['available_s'] <= public.query_times[-1]+1e-9 for p in public.packets)
    assert consumed == expected_consumed
    audit = getattr(tracker, 'observation_window_audit', [])
    assert all(p['status'] == 'PASS' and p['event_partition_exact'] for p in audit)
    return dict(status='PASS', states=states, samples=serial(tracker.samples),
        captured_packets=captured, consumed_packets=consumed,
        parent_availability_preserved=True, controller_queries_unchanged=True,
        child_windows=sum(p['subwindows'] for p in audit),
        pose_support='original camera endpoints; 10ms uses the registered available-history interpolation',
        assumed_grasp_propagation_replayed=False)


def verify_native_parity(run, inference):
    """All detector samples plus public-state prefix before assumed hand propagation."""
    archived = read(run/'migration_audit/sensor_episode_000/event_samples.json')
    samples = inference['samples']
    assert len(samples) == len(archived), ('Native archived sample count', len(samples), len(archived))
    maxima = dict(samples=0., position=0., velocity=0., latest_s=0.)
    for left, right in zip(samples, archived):
        for key in ('t', 'position', 'uv', 'radius_px', 'quality', 'available_s', 'maximum_event_s', 'camera_origin'):
            a, b = np.asarray(left[key], dtype=float), np.asarray(right[key], dtype=float)
            assert a.shape == b.shape
            difference = float(np.max(np.abs(a-b))) if a.size else 0.
            maxima['samples'] = max(maxima['samples'], difference)
            assert difference <= 1e-8, ('Native archived sample mismatch', key, difference)
    trace = read(run/'episode_000/trace.json')
    assert len(trace) == len(inference['states'])
    prefix_count = 0; first_attached = None
    for index, (actual, got) in enumerate(zip(trace, inference['states'])):
        expected = actual.get('event_state')
        if expected and str(expected.get('source', '')).startswith('event_acquisition_then_'):
            first_attached = index
            break
        assert (expected is not None) == got['available'], ('Native archived availability mismatch', index)
        if expected is not None:
            for key in ('position', 'velocity', 'latest_s'):
                difference = float(np.max(np.abs(np.asarray(expected[key])-np.asarray(got[key]))))
                maxima[key] = max(maxima[key], difference)
                assert difference <= 1e-8, ('Native archived state mismatch', index, key, difference)
            assert expected.get('motion_onset_s') == got['motion_onset_s']
        prefix_count += 1
    assert prefix_count > 0
    return dict(status='PASS', sample_count=len(samples), exact_public_state_prefix_queries=prefix_count,
        first_archived_assumed_grasp_query=first_attached,
        hand_propagation_tail_queries=0 if first_attached is None else len(trace)-first_attached,
        maximum_absolute_differences=maxima,
        note='No full observe/hand propagation parity is claimed after this fixed archived prefix')


def distribution(values):
    if not values: return dict(n=0, mean=None, median=None, p95=None)
    array = np.asarray(values, dtype=float)
    return dict(n=len(array), mean=float(np.mean(array)), median=float(np.median(array)), p95=float(np.percentile(array, 95)))


def grade(states, trace, release_s, indices):
    position = []; velocity = []; p_loss = []; v_loss = []; available = moving = 0
    for index in indices:
        state = states[index]; available += int(state['available']); moving += int(state['moving'])
        if state['moving']:
            assert index > 0, 'Post-release velocity label must come from the preceding physical state'
            p = float(np.linalg.norm(np.asarray(state['position'])-trace[index]['scoring_only_GT_position_at_control']))
            v = float(np.linalg.norm(np.asarray(state['velocity'])-np.asarray(trace[index-1]['qvel'][-6:-3])))
            position.append(p); velocity.append(v)
            p_loss.append(min(p/SCORE_RULE['position_failure_cap_m'], 1.))
            v_loss.append(min(v/SCORE_RULE['velocity_failure_cap_m_s'], 1.))
        else:
            p_loss.append(1.); v_loss.append(1.)
    n = len(indices)
    return dict(queries=n, available_states=available, moving_states=moving,
        available_coverage=None if n == 0 else available/n,
        moving_coverage=None if n == 0 else moving/n,
        position_error_m_conditional=distribution(position), velocity_error_m_s_conditional=distribution(velocity),
        penalized_position_error_m=distribution([x*SCORE_RULE['position_failure_cap_m'] for x in p_loss]),
        penalized_velocity_error_m_s=distribution([x*SCORE_RULE['velocity_failure_cap_m_s'] for x in v_loss]),
        normalized_equal_weight_loss=None if n == 0 else .5*float(np.mean(p_loss))+.5*float(np.mean(v_loss)))


def evaluate(run, inference, parity):
    # Target labels, release and contact determine grading only, after inference.
    trace = read(run/'episode_000/trace.json'); result = read(run/'episode_000/result.json')
    release = float(result['initial']['release_time_s'])
    first_contact = next((i for i, row in enumerate(trace) if row['hand_contact']), None)
    # A true first-contact flag refers to the following physical control step.
    # Its t_before query is still pre-contact and is included.
    end_index = len(trace)-1 if first_contact is None else first_contact
    indices = [i for i, row in enumerate(trace) if i <= end_index and row['t_before'] >= release-1e-9]
    assert indices, ('No post-release pre-contact evaluation queries; report this corpus problem', str(run))
    states = inference['states']
    assert len(states) == len(trace)
    pre_release = [i for i, row in enumerate(trace) if row['t_before'] < release-1e-9]
    false_count = sum(states[i]['moving'] for i in pre_release)
    false_fraction = false_count/len(pre_release) if pre_release else 0.
    full = grade(states, trace, release, indices)
    exact_indices = [i for i in indices if i < parity['exact_public_state_prefix_queries']]
    exact = grade(states, trace, release, exact_indices)
    full['selection_loss'] = full['normalized_equal_weight_loss']+false_fraction
    first = next((s['time_s'] for s in states if s['moving']), None)
    return dict(full_precontact=full, exact_archived_prefix_precontact=exact,
        pre_release_queries=len(pre_release), false_moving_pre_release_queries=false_count,
        false_moving_pre_release_fraction=false_fraction,
        first_control_moving_s=first,
        scoring_only_first_control_moving_delay_s=None if first is None else first-release,
        scoring_only_release_s=release, scoring_only_first_contact_step_index=first_contact,
        original_episode_outcome_not_used_for_candidate_selection=result['success'])


def aggregate_and_select(cache_results):
    sources = ('ours', 'v2e'); by_source = {source: [] for source in sources}
    corpus_cells = {source: set() for source in sources}
    ordered_caches = sorted(cache_results, key=lambda cache: cache['entry']['cache_id'])
    for cache in ordered_caches:
        entry = cache['entry']; corpus_cells[entry['method']].add((entry['seed'], entry['endpoint_hz'], entry['capture_variant']))
    assert corpus_cells['ours'] == corpus_cells['v2e'], 'Unequal cache screening budget/distribution'
    for source in sources:
        for candidate in CANDIDATES:
            selected = [cache for cache in ordered_caches if cache['entry']['method'] == source]
            cells = defaultdict(list); invalid = []
            for cache in selected:
                result = cache['candidates'][candidate.identifier]
                if result['status'] != 'PASS': invalid.append(cache['entry']['cache_id'])
                metric = result['metrics']['full_precontact']
                cells[(cache['entry']['seed'], cache['entry']['endpoint_hz'])].append(metric)
            assert cells
            loss = np.mean([np.mean([v['selection_loss'] for v in cells[key]]) for key in sorted(cells)])
            coverage = np.mean([np.mean([v['moving_coverage'] for v in cells[key]]) for key in sorted(cells)])
            by_source[source].append(dict(**candidate.receipt(), loss=float(loss), moving_coverage=float(coverage),
                valid=not invalid, invalid_cache_ids=invalid, independent_seed_cadence_cells=len(cells),
                cache_variants=len(selected)))
    def order(row):
        return (row['loss'], -row['moving_coverage'], row['gap_ms'], -int(row['flight_fit']), row['candidate_id'])
    best = {}
    for source in sources:
        best[source] = {}
        for window in (0, 10):
            allowed = [r for r in by_source[source] if r['valid'] and r['window_ms'] == window]
            assert allowed, ('No valid candidate remains for source/window family', source, window)
            best[source][str(window)] = sorted(allowed, key=order)[0]['candidate_id']
        by_source[source].sort(key=order)
    union_ids = sorted(set([CANONICAL]+[value for family in best.values() for value in family.values()]))
    assert len(union_ids) <= 5
    return dict(status='PASS', ranking_by_source=by_source, best_per_source_window_family=best,
        same_live_validation_candidate_union=[c.receipt() for c in CANDIDATES if c.identifier in union_ids],
        canonical_included=CANONICAL, source_specific_candidate_budget=8,
        live_validation_rule='Both generators must run every union candidate on the same paired development seeds/rates; pick a global configuration per source by equally weighted per-rate catch success; blind data untouched',
        interpretation='Fixed archived trajectories screen frontends only; scores are not a generator ranking or closed-loop success rate')


def source_receipt(source):
    names = ['event_sphere_tracker', 'event_sphere_tracker_rgbd_base', 'event_sphere_tracker_kalman',
             'event_sphere_tracker_contour_q80_occlusion', 'event_observation_windows', 'event_bearing_flight',
             'frontend_config']
    records = {}
    for name in names:
        module = importlib.import_module(name); path = Path(module.__file__).resolve()
        assert path.parent == source.resolve(), ('Wrong frontend module loaded', name, str(path))
        records[name+'.py'] = sha(path)
    return records


def finish(output, all_results, entries, identity, receipt, elapsed, shard, shards, result_paths):
    complete_corpus = len(all_results) == len(entries)
    assigned = sum(i % shards == shard for i in range(len(entries)))
    complete_shard = len(all_results) == assigned
    summary = dict(status='PASS' if complete_shard or complete_corpus else 'PARTIAL',
        complete=complete_shard or complete_corpus, complete_corpus=complete_corpus,
        shard=shard, shards=shards, completed_caches=len(all_results), planned_caches=len(entries),
        planned_shard_caches=assigned, candidates_per_source=8,
        native_default_parity_passed=sum(x['native_parity']['status'] == 'PASS' for x in all_results),
        source_files=receipt, corpus_sha256=identity['corpus_sha256'],
        replay_script_sha256=identity['replay_script_sha256'], scoring_rule=SCORE_RULE,
        physics_steps=0, renderer_calls=0, GPU_calls=0, target_GT_estimator_inputs=0,
        wall_s=elapsed, cache_result_paths=[str(p) for p in result_paths],
        cache_metrics=[dict(entry=x['entry'], native_parity=x['native_parity'],
            candidates={name: {k: v for k, v in result.items() if k in ('status', 'metrics', 'candidate_id', 'window_ms', 'flight_fit', 'gap_ms', 'error')}
                        for name, result in x['candidates'].items()}) for x in all_results])
    write(output/'SUMMARY.json', summary)
    if complete_corpus: write(output/'SHORTLIST.json', aggregate_and_select(all_results))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--cache-id', action='append', help='Smoke a specified cache; never produces a shortlist until the full corpus is complete')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--merge-shards', type=Path, nargs='+', help='Merge completed disjoint worker directories; no input cache is replayed')
    args = parser.parse_args(); started = time.monotonic()
    corpus = read(args.corpus); assert corpus['scoring_rule'] == SCORE_RULE
    assert corpus['candidate_grid'] == [c.receipt() for c in CANDIDATES]
    entries = corpus['caches']; ids = [entry['cache_id'] for entry in entries]
    assert len(ids) == len(set(ids)) and set(e['method'] for e in entries) == {'ours', 'v2e'}
    requested = set(args.cache_id) if args.cache_id else set(ids)
    assert requested <= set(ids)
    assert args.shards >= 1 and 0 <= args.shard < args.shards
    sys.path.insert(0, str(args.source.resolve()))
    factory = importlib.import_module('event_sphere_tracker').EventSphereTracker
    receipt = source_receipt(args.source)
    declared = importlib.import_module('frontend_config').candidate_grid()
    assert [(d['frontend_id'], d['env']) for d in declared] == [(c.identifier, c.receipt()['env']) for c in CANDIDATES]
    if args.source_manifest:
        manifest = read(args.source_manifest); expected = manifest.get('runtime_files', manifest)
        assert all(expected.get(name) == digest for name, digest in receipt.items()), 'Unsealed candidate frontend source'
    identity = dict(corpus_sha256=sha(args.corpus), replay_script_sha256=sha(__file__),
                    source_files=receipt, source_path=str(args.source),
                    scoring_rule=SCORE_RULE, candidate_grid=[c.receipt() for c in CANDIDATES])
    if args.output.exists():
        assert args.resume, 'Refuse to overwrite an existing cache-screen directory'
        assert read(args.output/'SCREEN_IDENTITY.json') == identity, 'Resume identity changed'
    else:
        args.output.mkdir(parents=True, exist_ok=False); write(args.output/'SCREEN_IDENTITY.json', identity)
    all_results = []; result_paths = []
    if args.merge_shards:
        assert args.shards == 1 and args.shard == 0 and not args.cache_id
        seen = set()
        for directory in args.merge_shards:
            assert read(directory/'SCREEN_IDENTITY.json') == identity
            summary = read(directory/'SUMMARY.json')
            assert summary['status'] == 'PASS' and summary['complete']
            for path in sorted((directory/'cache_results').glob('*.json')):
                cached = read(path); cache_id = cached['entry']['cache_id']
                assert cache_id in ids and cache_id not in seen
                assert cached['entry'] == entries[ids.index(cache_id)] and cached['native_parity']['status'] == 'PASS'
                assert set(cached['candidates']) == {c.identifier for c in CANDIDATES}
                seen.add(cache_id); all_results.append(cached); result_paths.append(path)
        assert seen == set(ids), ('Incomplete or duplicate shard merge', len(seen), len(ids))
        finish(args.output, all_results, entries, identity, receipt, time.monotonic()-started, 0, 1, result_paths)
        print(json.dumps(dict(status='PASS', complete_corpus=True, completed_caches=len(all_results), output=str(args.output))), flush=True)
        return
    for entry_index, entry in enumerate(entries):
        if entry_index % args.shards != args.shard: continue
        path = args.output/'cache_results'/(entry['cache_id']+'.json')
        if path.exists():
            cached = read(path); assert cached['entry'] == entry and cached['native_parity']['status'] == 'PASS'
            assert set(cached['candidates']) == {c.identifier for c in CANDIDATES}
            all_results.append(cached); result_paths.append(path); continue
        if entry['cache_id'] not in requested: continue
        run = Path(entry['run']); protocol = read(run/'protocol.json')
        assert protocol['event_generator'] == entry['method']
        assert protocol['seeds'] == [entry['seed']]
        assert abs(float(protocol['event_endpoint_rate_hz'])-entry['endpoint_hz']) < 1e-8
        public = load_public_cache(run, protocol)
        native = native_candidate(protocol)
        baseline = infer(public, factory, native)
        parity = verify_native_parity(run, baseline)
        results = {}
        for candidate in CANDIDATES:
            try:
                inference = baseline if candidate == native else infer(public, factory, candidate)
                metrics = evaluate(run, inference, parity)
                compact = {key: value for key, value in inference.items() if key != 'samples'}
                results[candidate.identifier] = dict(**candidate.receipt(), **compact, metrics=metrics,
                    detector_samples=len(inference['samples']), native_archive_configuration=candidate == native)
            except Exception as error:
                # Keep a spent candidate and all planned queries in the denominator.
                # Invalid candidates are not eligible for live selection.
                missing = [dict(time_s=float(t), available=False, moving=False, position=None, velocity=None)
                           for t in public.query_times]
                metrics = evaluate(run, dict(states=missing), parity)
                results[candidate.identifier] = dict(**candidate.receipt(), status='INVALID',
                    error=type(error).__name__+': '+str(error), states=missing, metrics=metrics)
        result = dict(status='PASS', entry=entry, input_receipt=public.receipt,
            native_candidate=native.receipt(), native_parity=parity,
            capture_schedule=public.capture_schedule, candidates=results,
            cache_censoring_note='Archived capture may stop after its own estimated grasp; the same available raw events and fixed trajectory are used for every candidate. No counterfactual new sensor data is invented.')
        write(path, result); all_results.append(result); result_paths.append(path)
        print(json.dumps(dict(cache=entry['cache_id'], native_parity='PASS', completed=len(all_results), total=len(entries))), flush=True)
    assert source_receipt(args.source) == receipt
    assert 'genesis' not in sys.modules and 'torch' not in sys.modules
    summary = finish(args.output, all_results, entries, identity, receipt, time.monotonic()-started,
                     args.shard, args.shards, result_paths)
    print(json.dumps(dict(status=summary['status'], complete=summary['complete'],
        complete_corpus=summary['complete_corpus'], completed_caches=len(all_results), output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
