"""
PECS Chamfer Distance & Gaussian Distance 实现
================================================
参考: Han et al., "Physical-Based Event Camera Simulator", ECCV 2024

CD(R,Q) = 1/|R| Σᵣ min_q d(r,q) + 1/|Q| Σ_q min_r d(r,q)
GD(R,Q) = 1/|R| Σᵣ g(min_q d(r,q)) + 1/|Q| Σ_q g(min_r d(r,q))
where g(x) = 1 - exp(-||x||²₂/σ), σ = 0.4

归一化:
  L_β[i] = ((L'_β[i] - min L') / (max L' - min L' + ε)) * α
  α = 100  for (x, y, p)  — spatial + polarity
  α = 1000 for t           — temporal

加速: scipy.spatial.cKDTree → O(n log n)
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# PECS paper defaults
ALPHA_SPATIAL = 100.0   # α for x, y, p
ALPHA_TEMPORAL = 1000.0  # α for t
SIGMA = 0.4              # σ for Gaussian distance


def normalize_events(
    events: np.ndarray,
    alpha_xy: float = ALPHA_SPATIAL,
    alpha_p: float = ALPHA_SPATIAL,
    alpha_t: float = ALPHA_TEMPORAL,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    PECS 归一化: min-max normalize each dimension, then scale by α.
    
    Args:
        events: (N, 4) float32 [x, y, p, t]
        alpha_xy: spatial scaling (default 100)
        alpha_p: polarity scaling (default 100)
        alpha_t: temporal scaling (default 1000)
    
    Returns:
        normalized: (N, 4) float32, normalized and scaled
    """
    assert events.ndim == 2 and events.shape[1] == 4, \
        f"Expected (N,4), got {events.shape}"
    
    mins = events.min(axis=0)
    maxs = events.max(axis=0)
    ranges = maxs - mins + eps
    
    normalized = (events - mins) / ranges
    normalized[:, 0] *= alpha_xy   # x
    normalized[:, 1] *= alpha_xy   # y
    normalized[:, 2] *= alpha_p    # polarity
    normalized[:, 3] *= alpha_t    # timestamp
    
    return normalized.astype(np.float32)


def chamfer_distance(
    R: np.ndarray,
    Q: np.ndarray,
    normalize: bool = True,
    subsample: Optional[int] = None,
) -> float:
    """
    Chamfer Distance between two event point clouds.
    
    CD(R,Q) = mean(min d(r→q)) + mean(min d(q→r))
    
    Args:
        R: (N, 4) event cloud 1 (e.g., simulated)
        Q: (M, 4) event cloud 2 (e.g., real)
        normalize: apply PECS normalization
        subsample: if set, randomly subsample both to this many events
    
    Returns:
        CD value (lower = more similar)
    """
    R, Q = _prepare(R, Q, normalize, subsample)
    
    # R → Q: for each r, find nearest q
    tree_Q = cKDTree(Q)
    d_r2q, _ = tree_Q.query(R, k=1)
    
    # Q → R: for each q, find nearest r
    tree_R = cKDTree(R)
    d_q2r, _ = tree_R.query(Q, k=1)
    
    cd = float(d_r2q.mean() + d_q2r.mean())
    return cd


def gaussian_distance(
    R: np.ndarray,
    Q: np.ndarray,
    normalize: bool = True,
    subsample: Optional[int] = None,
    sigma: float = SIGMA,
) -> float:
    """
    Gaussian Distance between two event point clouds.
    
    GD(R,Q) = mean(g(min d(r→q))) + mean(g(min d(q→r)))
    g(x) = 1 - exp(-||x||²/σ)
    
    Args:
        R: (N, 4) event cloud 1
        Q: (M, 4) event cloud 2
        normalize: apply PECS normalization
        subsample: if set, randomly subsample
        sigma: Gaussian kernel width (default 0.4 from paper)
    
    Returns:
        GD value (lower = more similar)
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    R, Q = _prepare(R, Q, normalize, subsample)
    
    # R → Q
    tree_Q = cKDTree(Q)
    d_r2q, _ = tree_Q.query(R, k=1)
    g_r2q = 1.0 - np.exp(-np.square(d_r2q) / sigma)
    
    # Q → R
    tree_R = cKDTree(R)
    d_q2r, _ = tree_R.query(Q, k=1)
    g_q2r = 1.0 - np.exp(-np.square(d_q2r) / sigma)
    
    gd = float(g_r2q.mean() + g_q2r.mean())
    return gd


def compute_both(
    R: np.ndarray,
    Q: np.ndarray,
    normalize: bool = True,
    subsample: Optional[int] = None,
    sigma: float = SIGMA,
    label: str = "",
) -> dict:
    """
    Compute both CD and GD in one pass (reuses KD-trees).
    
    Returns:
        dict with keys: cd, gd, n_R, n_Q
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    R, Q = _prepare(R, Q, normalize, subsample)
    
    tree_Q = cKDTree(Q)
    d_r2q, _ = tree_Q.query(R, k=1)
    
    tree_R = cKDTree(R)
    d_q2r, _ = tree_R.query(Q, k=1)
    
    cd = float(d_r2q.mean() + d_q2r.mean())
    
    # scipy.spatial.cKDTree.query returns Euclidean distance, not squared
    # distance. PECS Eq. (15) applies the square inside the Gaussian.
    g_r2q = 1.0 - np.exp(-np.square(d_r2q) / sigma)
    g_q2r = 1.0 - np.exp(-np.square(d_q2r) / sigma)
    gd = float(g_r2q.mean() + g_q2r.mean())
    
    return {
        'cd': cd,
        'gd': gd,
        'n_R': len(R),
        'n_Q': len(Q),
        'label': label,
    }


def _prepare(R, Q, normalize, subsample):
    """Shared preprocessing."""
    R = np.asarray(R, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    assert R.shape[1] == 4, f"R shape {R.shape}, expected (N,4)"
    assert Q.shape[1] == 4, f"Q shape {Q.shape}, expected (M,4)"
    if len(R) == 0 or len(Q) == 0:
        raise ValueError("PECS distances require two non-empty event streams")
    
    if normalize:
        # Match the PECS reference implementation: each stream is min-max
        # normalized independently before nearest-neighbour matching. Do this
        # before optional subsampling so the coordinate transform is defined by
        # the complete evaluation window, as in the paper equation.
        R = normalize_events(R)
        Q = normalize_events(Q)

    if subsample and (len(R) > subsample or len(Q) > subsample):
        rng = np.random.RandomState(42)
        if len(R) > subsample:
            R = R[rng.choice(len(R), subsample, replace=False)]
        if len(Q) > subsample:
            Q = Q[rng.choice(len(Q), subsample, replace=False)]
    
    return R.astype(np.float32), Q.astype(np.float32)


# ════════════════════════════════════════════════════════
# PECS Paper Baselines (for reference)
# ════════════════════════════════════════════════════════

PECS_BASELINES = {
    # Format: scene_name: (GD, CD) for PECS simulator
    'R_360_H':  (0.960, 2.010),
    'R_360_L':  (1.631, 3.708),
    'R_60_H':   (1.504, 3.459),
    'T_1.0_H':  (1.667, 3.975),
    'T_0.6_H':  (1.798, 4.169),
    'T_1.0_L':  (1.710, 4.467),
    'average':  (1.545, 3.631),
}

V2E_BASELINES = {
    'R_360_H':  (1.671, 6.356),
    'R_360_L':  (1.831, 7.729),
    'R_60_H':   (1.816, 9.297),
    'T_1.0_H':  (1.827, 15.450),
    'T_0.6_H':  (1.839, 18.154),
    'T_1.0_L':  (1.700, 14.071),
    'average':  (1.781, 11.843),
}

ESIM_BASELINES = {
    'R_360_H':  (1.995, 45.512),
    'R_360_L':  (1.965, 35.082),
    'R_60_H':   (1.995, 39.425),
    'T_1.0_H':  (1.963, 62.206),
    'T_0.6_H':  (1.967, 63.939),
    'T_1.0_L':  (1.987, 68.115),
    'average':  (1.979, 52.380),
}


def compare_to_baselines(gd: float, cd: float, scene: str = 'average') -> dict:
    """Compare computed (GD, CD) against PECS paper baselines."""
    pecs_gd, pecs_cd = PECS_BASELINES.get(scene, PECS_BASELINES['average'])
    v2e_gd, v2e_cd = V2E_BASELINES.get(scene, V2E_BASELINES['average'])
    esim_gd, esim_cd = ESIM_BASELINES.get(scene, ESIM_BASELINES['average'])
    
    return {
        'scene': scene,
        'our_gd': gd, 'our_cd': cd,
        'pecs_gd': pecs_gd, 'pecs_cd': pecs_cd,
        'v2e_gd': v2e_gd, 'v2e_cd': v2e_cd,
        'esim_gd': esim_gd, 'esim_cd': esim_cd,
        'gd_vs_pecs': gd / pecs_gd if pecs_gd > 0 else float('inf'),
        'cd_vs_pecs': cd / pecs_cd if pecs_cd > 0 else float('inf'),
        'gd_vs_v2e': gd / v2e_gd if v2e_gd > 0 else float('inf'),
        'cd_vs_v2e': cd / v2e_cd if v2e_cd > 0 else float('inf'),
    }
