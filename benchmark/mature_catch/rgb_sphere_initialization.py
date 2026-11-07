"""Known-size sphere initialization from one RGB image and camera calibration.

No scene, object pose, depth or segmentation is accepted by this interface.
The launch-volume prior is the published Catch It! task distribution, padded
to accommodate pixel-level radius uncertainty, not an episode-specific crop.
"""
import cv2
import numpy as np

from event_sphere_tracker_contour_q80_occlusion import sphere_from_edge_pixels

cv2.setNumThreads(2)


def detect_sphere(image, intrinsics, camera_to_world, radius_m):
    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), .6)
    edges = cv2.Canny(smooth, 25, 75)
    ys, xs = np.nonzero(edges)
    points = np.column_stack((xs, ys)).astype(float) + .5
    circles = cv2.HoughCircles(smooth, cv2.HOUGH_GRADIENT, dp=1,
                              minDist=8, param1=75, param2=7,
                              minRadius=3, maxRadius=32)
    K = np.asarray(intrinsics, dtype=float)
    T = np.asarray(camera_to_world, dtype=float)
    candidates = []
    if circles is not None:
        for u, v, r in circles[0]:
            distances = np.linalg.norm(points - [u, v], axis=1)
            selected = points[np.abs(distances - r) < 1.7]
            if len(selected) < 12:
                continue
            fit = sphere_from_edge_pixels(selected, K, radius_m)
            if fit is None or fit['angle_bins'] < 10:
                continue
            world = (T @ np.r_[fit['camera_position'], 1.])[:3]
            if not (-.8 < world[0] < .8 and 1.8 < world[1] < 2.9
                    and .35 < world[2] < 1.95):
                continue
            yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
            dd = np.hypot(xx + .5 - fit['uv'][0], yy + .5 - fit['uv'][1])
            inside = gray[dd < .65 * fit['radius_px']]
            outside = gray[(dd > 1.4 * fit['radius_px']) &
                           (dd < 1.9 * fit['radius_px'])]
            if len(inside) == 0 or len(outside) == 0:
                continue
            contrast = abs(float(np.median(inside)) - float(np.median(outside)))
            if contrast < 12:
                continue
            score = contrast * fit['angle_bins'] / (.4 + fit['residual_px'])
            candidates.append(dict(position=world.tolist(), uv=fit['uv'].tolist(),
                                   radius_px=fit['radius_px'],
                                   residual_px=fit['residual_px'],
                                   contrast=contrast, score=score))
    if not candidates:
        return dict(valid=False, candidates=[], source='single actual RGB image')
    best = max(candidates, key=lambda c: c['score'])
    return dict(valid=True, **best, candidates=candidates,
                source='single actual RGB image; known-radius tangent-cone fit',
                camera_origin=T[:3, 3].tolist())
