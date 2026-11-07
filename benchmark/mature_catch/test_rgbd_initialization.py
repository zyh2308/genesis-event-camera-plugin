"""Camera-z/sphere-centre contract, including off-axis rays and world transforms."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from rgbd_sphere_initialization import refine_with_depth

K = np.array([[430., 0., 320.], [0., 430., 240.], [0., 0., 1.]])
T = np.eye(4)
T[:3, :3] = Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix()
T[:3, 3] = [.2, -.1, .8]
yy, xx = np.indices((480, 640))
rays = np.stack([(xx+.5-320)/430, (yy+.5-240)/430, np.ones(xx.shape)], axis=-1)
rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
results = []
for centre in [np.array([.2, -.3, 2.5]), np.array([-.4, .1, 1.5]), np.array([0., 0., 1.])]:
    R = .04
    dot = rays @ centre
    discriminant = dot*dot-(centre@centre-R*R)
    depth = np.zeros(xx.shape)
    valid = discriminant >= 0
    distance = dot[valid]-np.sqrt(discriminant[valid])
    depth[valid] = distance*rays[valid, 2]
    uv = (K@centre)[:2]/centre[2]
    biased = (T@np.r_[centre*1.12, 1.])[:3]
    proposal = dict(valid=True, position=biased.tolist(), uv=uv.tolist(),
                    radius_px=430*R/centre[2], camera_origin=T[:3, 3].tolist())
    result = refine_with_depth(proposal, depth, K, T, R)
    assert result['valid'], result
    expected = (T@np.r_[centre, 1.])[:3]
    error = float(np.linalg.norm(np.asarray(result['position'])-expected))
    assert error < 1e-9, error
    invalid = refine_with_depth(proposal, np.zeros_like(depth), K, T, R)
    assert not invalid['valid']
    results.append(dict(centre=centre.tolist(), position_error_m=error,
                        interior_samples=result['depth_interior_samples']))
print(json.dumps(dict(status='PASS', cases=results, blank_depth_rejected=True,
                      known_radius_surface_to_centre_correct=True)))
