"""One actual RGB-D observation, known sphere radius, no scene/GT/segmentation.

Genesis rasterizer depth is camera-z in linear metres. A visible surface point
is not the sphere centre: intersect its known-radius sphere with the RGB centre
bearing and take the farther (centre) solution. Only the interior RGB proposal
is read, avoiding antialiased silhouette/background depth mixtures.
"""
import numpy as np

from rgb_sphere_initialization import detect_sphere


def refine_with_depth(detection, depth_camera_z, intrinsics, camera_to_world, radius_m):
    if not detection.get('valid'):
        return dict(detection, initialization_mode='single_frame_RGBD')
    depth = np.asarray(depth_camera_z, dtype=float)
    K = np.asarray(intrinsics, dtype=float)
    T = np.asarray(camera_to_world, dtype=float)
    if depth.ndim != 2:
        raise ValueError('RGB-D initialization expects an HxW camera-z depth image')
    u, v = detection['uv']
    yy, xx = np.indices(depth.shape)
    keep = ((xx+.5-u)**2+(yy+.5-v)**2 < (.45*detection['radius_px'])**2)
    keep &= np.isfinite(depth) & (depth > radius_m) & (depth < 6.)
    ys, xs = np.nonzero(keep)
    if len(xs) < 6:
        return dict(valid=False, source='single actual RGB-D frame',
                    failure_reason='insufficient valid interior depth',
                    initialization_mode='single_frame_RGBD')
    pixels = np.column_stack((xs+.5, ys+.5, np.ones(len(xs))))
    points = (pixels @ np.linalg.inv(K).T) * depth[ys, xs, None]
    direction = T[:3, :3].T @ (np.asarray(detection['position'])-T[:3, 3])
    direction /= np.linalg.norm(direction)
    along = points @ direction
    transverse2 = np.maximum(0., np.sum(points*points, axis=1)-along*along)
    valid = transverse2 < radius_m**2
    if int(valid.sum()) < 6:
        return dict(valid=False, source='single actual RGB-D frame',
                    failure_reason='depth inconsistent with known sphere proposal',
                    initialization_mode='single_frame_RGBD')
    ranges = along[valid] + np.sqrt(radius_m**2-transverse2[valid])
    distance = float(np.median(ranges))
    mad = float(np.median(np.abs(ranges-distance)))
    if not np.isfinite(distance) or distance <= radius_m or mad > .015:
        return dict(valid=False, source='single actual RGB-D frame',
                    failure_reason='inconsistent sphere depth samples',
                    initialization_mode='single_frame_RGBD')
    centre = direction*distance
    world = (T @ np.r_[centre, 1.])[:3]
    return dict(detection, position=world.tolist(),
                RGB_only_position=list(detection['position']),
                source='single actual RGB-D frame; RGB proposal and interior camera-z depth',
                initialization_mode='single_frame_RGBD',
                depth_convention='linear camera-z metres',
                depth_interior_samples=int(valid.sum()),
                depth_range_mad_m=mad,
                # Declared uncertainty floor, not a calibrated RealSense noise model.
                range_std_m=max(.010, 1.4826*mad),
                depth_sensor_model='ideal Genesis depth; not real-device calibration')


def initialize_sphere_rgbd(image, depth_camera_z, intrinsics, camera_to_world, radius_m):
    detection = detect_sphere(image, intrinsics, camera_to_world, radius_m)
    return refine_with_depth(detection, depth_camera_z, intrinsics, camera_to_world, radius_m)
