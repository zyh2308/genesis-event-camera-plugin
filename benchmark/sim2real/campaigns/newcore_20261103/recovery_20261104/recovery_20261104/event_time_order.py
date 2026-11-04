import numpy as np

def chronological_events(raw, start_s, end_s):
    """Preserve all event values and order ties stably after bounded CUDA float rounding."""
    assert raw.ndim == 2 and raw.shape[1] == 4
    assert np.isfinite(raw).all()
    if not len(raw):
        return raw, {'backsteps_before': 0, 'stable_sort_applied': False}
    assert raw[:, 0].min() >= start_s - 1e-8
    assert raw[:, 0].max() <= end_s + 1e-8
    differences = np.diff(raw[:, 0])
    backward = differences < 0
    worst = float(-differences[backward].min()) if backward.any() else 0.0
    precision_bound = max(4e-9, 2 * float(np.spacing(np.float32(end_s * 1e9))) * 1e-9)
    assert worst <= precision_bound + 1e-15, (worst, precision_bound)
    metadata = {'backsteps_before': int(backward.sum()), 'max_backstep_s': worst,
                'float32_ns_precision_bound_s': precision_bound,
                'stable_sort_applied': bool(backward.any()), 'event_values_modified': False,
                'event_count': len(raw)}
    if backward.any():
        order = np.argsort(raw[:, 0], kind='stable')
        result = raw[order]
        assert np.array_equal(result[np.argsort(order)], raw)
        assert np.all(np.diff(result[:, 0]) >= 0)
        return result, metadata
    return raw, metadata
