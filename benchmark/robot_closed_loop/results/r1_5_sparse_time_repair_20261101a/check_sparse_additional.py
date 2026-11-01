"""Unknown-region fallback remains approximate and strict rejection survives."""
import json,numpy as np
from types import SimpleNamespace
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator,RenderRequiredError
from genesis_event_plugin.physics_warp import depth_aware_bidirectional_warp_projected_diagnostics as warp

interp=PhysicsInterpolator(np.eye(3),composite='coverage_time_log_linear',warp_backend='numpy')
interp._current_I0=np.ones((2,2),np.float32)
interp._current_I1=np.full((2,2),np.exp(.9),np.float32)
sampler=SimpleNamespace(p99_radiance_residual=0.)
values=[]
for alpha in [.49,.5,.51]:
    out,unknown=interp._handle_invalid(np.full((2,2),np.nan,np.float32),np.zeros((2,2),bool),
        alpha,0,.01,sampler,None)
    assert unknown.all() and np.isfinite(out).all()
    assert np.allclose(out,np.exp(.9*alpha))
    values.append(float(out[0,0]))
interp.strict_fidelity=True
try:
    interp._handle_invalid(np.ones((2,2),np.float32),np.zeros((2,2),bool),.5,0,.01,sampler,None)
except RenderRequiredError:pass
else:raise AssertionError('unknown support was accepted by strict mode')
yy,xx=np.mgrid[:2,:2];uv=np.stack([xx,yy]).astype(float);depth=np.ones((2,2));valid=np.ones((2,2),bool)
out,trust,diagnostic=warp(np.ones((2,2)),depth,uv,depth,valid,
    np.full((2,2),2.),depth,uv,depth,valid,composite='coverage_time_log_linear',alpha=.5,
    surface_id0=np.zeros((2,2),int),surface_id1=np.ones((2,2),int))
assert not trust.any() and diagnostic['intermediate_unknown'].all() and np.isfinite(out).all()
print(json.dumps(dict(status='PASS',controls=['continuous unknown fallback across midpoint',
    'unknown mask not relaxed','strict rejects approximate reconstruction','different surfaces remain unknown'],
    midpoint_values=values)))
