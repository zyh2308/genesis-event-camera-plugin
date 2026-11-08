"""Fixed predicted-peak soft focus: probability algebra only, no GT inputs.

The only public transformation argument is the original 16x16 softmax map.
Coordinates follow the native row-major grid: (x,y)=((col+.5)/16,(row+.5)/16).
All algebra uses FP64 after a private copy. No clipping, epsilon, hard crop,
flat-map fallback, temperature, learned parameters, or parameter sweep exists.
"""
from __future__ import annotations
import numpy as np

GRID_SIZE = 16
SIGMA = 1.0 / GRID_SIZE
# Same probability-sum acceptance scale used by the frozen region diagnostics.
# This checks an FP32 softmax input; it does not tune scientific outcome gates.
PROBABILITY_SUM_ATOL = 1e-5


def cell_center_grid():
    """Return an independent FP64 (16,16,2) grid; x precedes y."""
    axis = (np.arange(GRID_SIZE, dtype=np.float64) + .5) / GRID_SIZE
    xx, yy = np.meshgrid(axis, axis, indexing='xy')
    return np.stack((xx, yy), axis=-1)


def softfocus_predicted_peak(attention):
    """Return the single prespecified soft-focus readout of a predicted map.

    Input is finite nonnegative float32/float64 softmax probabilities, exactly
    16x16, with sum within 1e-5 of one. Ties use first C-row-major argmax, matching
    the native flattened decoder. Uniform/incorrect peaks are retained: no GT,
    region mask, teacher, other exposure, initialization or fallback is allowed.

    ``original_fp64_uv`` intentionally preserves the original p@grid arithmetic
    without renormalizing p. Compare it to the native FP32 control in the caller;
    this module does not pretend NumPy FP32 reduction is the native Torch result.
    ``weights`` are normalized FP64 p_i*exp(-distance²/(2*(1/16)²)).
    """
    original = np.asarray(attention)
    if original.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError('Exactly one native 16x16 probability map is required')
    if original.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Native float32 or float64 probability map required')
    p = np.array(original, dtype=np.float64, copy=True, order='C')
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Softmax probabilities must be finite and nonnegative')
    map_sum = float(p.sum(dtype=np.float64))
    if abs(map_sum - 1.0) > PROBABILITY_SUM_ATOL:
        raise ValueError('Input probability sum differs from one beyond fixed tolerance')
    flat_index = int(np.argmax(p.ravel(order='C')))
    row, column = divmod(flat_index, GRID_SIZE)
    grid = cell_center_grid()
    peak_uv = grid[row, column].copy()
    distance_sq = ((grid - peak_uv) ** 2).sum(axis=-1)
    gaussian = np.exp(-distance_sq / (2.0 * SIGMA ** 2))
    product = p * gaussian
    weighted_mass = float(product.sum(dtype=np.float64))
    # g_peak=1 and p_peak>0 guarantee a positive denominator; never patch it.
    if not np.isfinite(weighted_mass) or weighted_mass <= 0:
        raise ValueError('Invalid positive soft-focus normalization mass')
    weights = product / weighted_mass
    uv = (weights[..., None] * grid).sum(axis=(0, 1), dtype=np.float64)
    original_uv = (p[..., None] * grid).sum(axis=(0, 1), dtype=np.float64)
    if not np.isfinite(uv).all() or not np.isfinite(weights).all():
        raise ValueError('Nonfinite FP64 soft-focus output')
    return {
        'map_sum': map_sum,
        'peak': {'flat_index': flat_index, 'row': row, 'column': column,
                 'uv': peak_uv.tolist(), 'probability': float(p[row, column])},
        'weights': weights,
        'uv': uv,
        'original_fp64_uv': original_uv,
        'numerical_diagnostics': {
            'input_dtype': str(original.dtype), 'algebra_dtype': 'float64',
            'sigma_normalized': SIGMA, 'grid_shape': [GRID_SIZE, GRID_SIZE],
            'probability_sum_atol': PROBABILITY_SUM_ATOL,
            'probability_sum_abs_error': abs(map_sum - 1.0),
            'map_min': float(p.min()), 'map_max': float(p.max()),
            'peak_tie_count': int(np.count_nonzero(p == p[row, column])),
            'argmax_order': 'first C-row-major flattened index',
            'gaussian_min': float(gaussian.min()), 'gaussian_peak': float(gaussian[row, column]),
            'weighted_mass_before_normalization': weighted_mass,
            'weights_sum': float(weights.sum(dtype=np.float64)),
            'normalization_abs_error': abs(float(weights.sum(dtype=np.float64)) - 1.0),
            'input_nonzero_count': int(np.count_nonzero(p)),
            'output_nonzero_count': int(np.count_nonzero(weights)),
            'hard_crop_applied': False, 'flat_fallback_applied': False,
            'GT_inputs_accepted': False,
        },
    }
