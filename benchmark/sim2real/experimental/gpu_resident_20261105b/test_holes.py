"""Identical projected radiance/masks isolate fallback precision from splat RNG."""
import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
root=Path(__file__).resolve().parent;P=Path('/home/zhaoyuhan/Eventbased_WAM')
sys.path.insert(0,json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text())['core_path'])
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator,RenderRequiredError
from resident import finish_geometry
torch.set_num_threads(2)
out=root/'hole_tests';out.mkdir(exist_ok=False);rng=np.random.default_rng(7901)
report=dict(cases=[],exact_mode_passed=True,gpu_math_bitwise_passed=True,real_data_used=False)
for case in ['all_supported','conditional_unknown','mixed_holes','all_holes','hdr_holes','strict_rejection']:
    T,H,W=19,24,32;alphas=np.linspace(1/T,1,T,dtype=np.float64)
    first=rng.uniform(.00001,15 if case=='hdr_holes' else 1,(H,W)).astype(np.float32)
    last=rng.uniform(.00001,10 if case=='hdr_holes' else 1,(H,W)).astype(np.float32)
    raw=rng.uniform(.001,1,(T,H,W)).astype(np.float32);valid=rng.random((T,H,W))>.3
    if case=='all_supported':valid[:]=True
    if case in ['mixed_holes','hdr_holes','strict_rejection']:raw[(rng.random(raw.shape)>.3)&~valid]=np.nan
    if case=='all_holes':raw[:]=np.nan;valid[:]=False
    sampling=SimpleNamespace(p99_radiance_residual=.7)
    def state():return SimpleNamespace(_current_I0=first,_current_I1=last,disocclusion_policy='hold',composite='coverage_time_log_linear',strict_fidelity=case=='strict_rejection',last_render_requests=[],last_visibility={})
    reference=state();expected=[];reference_error=False
    for index,alpha in enumerate(alphas):
        try:
            frame,_=PhysicsInterpolator._handle_invalid(reference,raw[index].copy(),valid[index],float(alpha),1.,1.01,sampling,None)
            expected.append(frame)
        except RenderRequiredError:reference_error=True;break
    expected=np.stack(expected) if expected else None
    for mode in ['exact','gpu']:
        candidate=state();candidate_error=False
        try:
            frames,times=finish_geometry(candidate,torch.tensor(raw,device='cuda:0')[None],torch.tensor(valid,device='cuda:0')[None],alphas,1.,1.01,sampling,None,False,False,mode)
            actual=torch.stack(frames).cpu().numpy()
        except RenderRequiredError:candidate_error=True;actual=None
        assert candidate_error==reference_error,(case,mode,'strict error differs')
        equal=reference_error or bool(np.array_equal(expected,actual,equal_nan=True))
        error=0.0 if reference_error else float(np.max(np.abs(expected-actual)))
        requests_equal=len(reference.last_render_requests)==len(candidate.last_render_requests) and all(a.as_dict()==b.as_dict() and np.array_equal(a.mask,b.mask) for a,b in zip(reference.last_render_requests,candidate.last_render_requests))
        assert requests_equal,(case,mode,'render request semantics differ')
        if mode=='exact':assert equal,(case,error)
        else:report['gpu_math_bitwise_passed'] &= equal
        report['cases'].append(dict(case=case,mode=mode,frame_bitwise_equal=equal,max_abs=error,strict_rejection=reference_error,render_requests_equal=requests_equal))
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
