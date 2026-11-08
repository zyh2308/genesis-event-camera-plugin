"""Independent CPU arithmetic/inventory check of the frozen GPU export."""
from pathlib import Path
import argparse, collections, hashlib, json, math
import numpy as np

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--completion-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    cp = a.root / 'completion.json'
    assert sha(cp) == a.completion_sha256
    completion = json.loads(cp.read_text())
    assert completion['status'] == 'complete' and completion['optimizer_updates'] == 0
    assert completion['all_control_native_peaks_exact'] is True
    assert completion['GT_forward_inputs'] is False and completion['real_data_used'] is False
    for r in completion['artifacts']:
        assert Path(r['path']).stat().st_size == r['size_bytes'] and sha(r['path']) == r['sha256']
    cases = [json.loads(s) for s in (a.root / 'paired_center_peaks.jsonl').open() if s.strip()]
    assert len(cases) == 1152
    assert len({(r['method'], r['seed'], r['condition'], r['sample_id']) for r in cases}) == 1152
    arrays = np.load(a.root / 'native_attention_maps.npz', allow_pickle=False)
    yy, xx = np.mgrid[:16, :16]
    grid = np.stack(((xx + .5) / 16, (yy + .5) / 16), -1)
    maxima = {}
    condition_counts = collections.defaultdict(collections.Counter)
    miss = 'no_target_support_in_endpoint_or_union_peak_cell'
    transitions = collections.defaultdict(collections.Counter)
    for role in ('control', 'candidate'):
        maps = arrays[role]
        assert maps.shape == (1152, 16, 16) and maps.dtype == np.float32
        assert np.isfinite(maps).all() and (maps >= 0).all()
        assert np.max(abs(maps.astype(np.float64).sum((1, 2)) - 1)) <= 1e-5
        error = 0.
        for i, r in enumerate(cases):
            assert r['map_index'] == i
            value = r[role]
            peak = int(np.argmax(maps[i]))
            row, col = divmod(peak, 16)
            assert (row, col, peak) == tuple(value['native_peak'][k] for k in ('row', 'column', 'flat_index'))
            np.testing.assert_array_equal(grid[row, col], value['native_peak']['uv'])
            uv = (maps[i].astype(np.float64)[..., None] * grid).sum((0, 1))
            error = max(error, float(np.max(abs(uv - value['pred_uv']))))
            assert r['control_native_peak_reference_exact'] is True
            endpoint = value['regions']['endpoint']
            union = value['regions']['endpoint_union_approx']
            cell = value['peak_cell_overlap']
            expected = ('region_unavailable_or_empty' if not endpoint['available'] or not union['available']
                        else 'native_peak_in_visible_target' if endpoint['peak_in_region']
                        else 'native_point_outside_endpoint_but_peak_cell_has_target_support' if cell['endpoint'] > 0
                        else 'no_endpoint_cell_support_but_endpoint_union_cell_support' if cell['endpoint_union_approx'] > 0
                        else miss)
            assert expected == value['peak_category']
            condition_counts[(r['method'], r['condition'], role)][expected] += 1
        assert error <= 2e-6
        maxima[role] = error
    for r in cases:
        if r['condition'] == 'eighth':
            old, new = r['control']['peak_category'] == miss, r['candidate']['peak_category'] == miss
            transitions[r['method']][('remaining' if old and new else 'resolved' if old else 'new' if new else 'neither')] += 1
    expected_control_misses = {('eventgenesis', 'normal'): 0, ('eventgenesis', 'half'): 0,
                               ('eventgenesis', 'eighth'): 28, ('v2e', 'normal'): 4,
                               ('v2e', 'half'): 4, ('v2e', 'eighth'): 0}
    for key, count in expected_control_misses.items():
        assert condition_counts[key + ('control',)][miss] == count
    exported = json.loads((a.root / 'three_seed_condition.json').read_text())
    assert len(exported) == 12
    for r in exported:
        key = (r['method'], r['condition'], r['role'])
        assert dict(condition_counts[key]) == r['peak_category_counts'] and r['prediction_rows'] == 192
    result = dict(status='passed',optimizer_updates=0,model_forwards=0,paired_cases=1152,
                  probability_maps_checked=2304,all_artifact_hashes_verified=True,
                  max_abs_probability_expectation_error=maxima,baseline_miss_counts_match_original=True,
                  dark_transitions={k:dict(v) for k,v in transitions.items()},
                  GT_regions_recomputed=False,GT_scope='Checks exported posthoc region fields; frozen producer verified original H5 lineage.',
                  input_completion_sha256=a.completion_sha256,audit_source_sha256=sha(__file__))
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
