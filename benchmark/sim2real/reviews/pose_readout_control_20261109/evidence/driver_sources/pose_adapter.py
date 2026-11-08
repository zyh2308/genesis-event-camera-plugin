"""Thin adapter: original model/rotation/crop/ADD-S; only center readout differs."""
import numpy as np
import torch


def need(ok,message):
    if not ok:raise ValueError(message)


@torch.inference_mode()
def center_readouts(model,images,math_module):
    grid=model.grid.detach().cpu().numpy()
    need(np.array_equal(grid,math_module.cell_center_grid().reshape(256,2)),'Native grid differs from registered cell centers')
    native=[];focused=[];maps=[];weights=[];diagnostics=[]
    for offset in range(0,len(images),32):
        logits,uv,_,_=model(images[offset:offset+32]);p=logits.softmax(1)
        need(p.shape==(len(uv),256) and torch.isfinite(p).all(),'Invalid native attention')
        for original,prob in zip(uv.detach().cpu().numpy(),p.detach().cpu().numpy()):
            result=math_module.softfocus_predicted_peak(prob.reshape(16,16))
            error=float(np.max(np.abs(original-result['original_fp64_uv'])));need(error<=2e-6,'FP64 original expectation differs from native control')
            native.append(original);focused.append(result['uv']);maps.append(prob.reshape(16,16));weights.append(result['weights'])
            diagnostics.append(dict(peak=result['peak'],original_fp64_uv=result['original_fp64_uv'],native_fp64_max_abs=error,numerical=result['numerical_diagnostics']))
    return dict(control_uv=np.stack(native),candidate_uv=np.stack(focused),attention=np.stack(maps),weights=np.stack(weights),diagnostics=diagnostics)


def equivalent_score(current,reference):
    names=('depth_mae_m','depth_bias_m','log_size_mae','log_size_bias','center_px','success','auc')
    need(set(current)==set(reference)=={'summary','by_rollout'},'Native score schema differs')
    need(current['by_rollout'].keys()==reference['by_rollout'].keys(),'Native score rollout cohort differs')
    for a,b in [(current['summary'],reference['summary'])]+[(current['by_rollout'][k],reference['by_rollout'][k]) for k in current['by_rollout']]:
        need(set(a)==set(b)==set(names),'Native metric names differ')
        need(all(np.isfinite(a[k]) and np.isclose(a[k],b[k],rtol=1e-6,atol=1e-8) for k in names),'Full-pose control differs from frozen native score')


def pose_pair(ctx,rotation_head,images,rows,uv0,uv1,z,size,gt_log_size,tree,reference_score):
    """Candidate rotation is actually inferred again; never reuse control R."""
    before_z=z.copy();before_size=size.copy();K=[r['K'] for r in rows];out={}
    for arm,uv in (('control',uv0),('candidate',uv1)):
        need(np.isfinite(uv).all() and np.isfinite(z).all() and np.isfinite(size).all() and (z>0).all() and (size>0).all(),'Invalid full-pose inputs')
        R=ctx.shared.frozen_rotation(ctx.rotation,rotation_head,images,uv,size,K)
        need(R.shape==(len(rows),3,3) and np.isfinite(R).all(),'Invalid inferred rotation')
        score=ctx.shared.score(ctx.rotation,rows,uv,z,size,gt_log_size,R,ctx.vertices,tree,ctx.threshold)
        if arm=='control':equivalent_score(score,reference_score)
        out[arm]=dict(R=R,score=score)
    need(np.array_equal(z,before_z) and np.array_equal(size,before_size),'Frozen depth or size changed')
    return out
