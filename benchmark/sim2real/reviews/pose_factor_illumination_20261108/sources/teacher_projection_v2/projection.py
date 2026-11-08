"""One-sided normal-teacher gradient projection in the full24-tensor space.

Caller supplies P=the original complete supervised-loss gradient and T=the
already weighted2*normal-teacher gradient. This module neither differentiates a
loss nor assigns .grad, applies an optimizer, changes a weight or chooses a loss
coefficient. One common projection scalar is computed across the entire model.
"""
from __future__ import annotations
from typing import Optional,Sequence
import torch

PARAMETER_TENSORS=24

def _need(ok,message):
    if not ok:raise ValueError(message)

def project_teacher_gradients(parameters:Sequence[torch.Tensor],
                              physical_grads:Sequence[Optional[torch.Tensor]],
                              weighted_teacher_grads:Sequence[Optional[torch.Tensor]],
                              parameter_names:Optional[Sequence[str]]=None):
    """Return {combined_grads,projected_teacher_grads,diagnostics} without mutation.

    Each list has exactly24 entries in the same complete named-parameter order.
    None represents an unused/frozen zero gradient; it never removes that slot.
    Dot products, norms, coefficient and negative-case projection arithmetic use
    FP64; outputs retain each parameter's original FP32/FP64 dtype/device. A
    nonnegative dot or a zero physical norm preserves T bit-for-bit. No norm
    threshold, epsilon, clipping or iterative orthogonality correction is used.
    FP32 casts can leave a tiny residual projected dot, which is reported.
    """
    _need(isinstance(parameters,(list,tuple)) and isinstance(physical_grads,(list,tuple)) and isinstance(weighted_teacher_grads,(list,tuple)),
          'Use explicit full parameter and gradient lists/tuples')
    params=list(parameters);P=list(physical_grads);T=list(weighted_teacher_grads)
    _need(len(params)==len(P)==len(T)==PARAMETER_TENSORS,'Exactly24 full parameter slots are required; do not filter unused tensors')
    _need(all(isinstance(p,torch.Tensor) and p.dtype in (torch.float32,torch.float64) for p in params),'Parameters must be floating FP32/FP64 tensors')
    dtype=params[0].dtype;device=params[0].device
    _need(all(p.dtype==dtype and p.device==device for p in params),'The pinned model must use one floating dtype/device')
    names=list(parameter_names) if parameter_names is not None else [str(i) for i in range(PARAMETER_TENSORS)]
    _need(len(names)==PARAMETER_TENSORS and all(isinstance(x,str) for x in names) and len(set(names))==PARAMETER_TENSORS,'Provide24 unique ordered parameter names')
    for label,grads in (('physical',P),('weighted_teacher',T)):
        for i,(p,g) in enumerate(zip(params,grads)):
            if g is not None:
                _need(isinstance(g,torch.Tensor) and g.shape==p.shape and g.device==p.device and g.dtype==p.dtype,
                      f'{label} gradient shape/dtype/device mismatch at slot{i}')
                _need(bool(torch.isfinite(g).all()),f'Nonfinite {label} gradient at slot{i}')
    physical_none=[i for i,g in enumerate(P) if g is None];teacher_none=[i for i,g in enumerate(T) if g is None]
    with torch.no_grad():
        p64=[torch.zeros_like(p,dtype=torch.float64) if g is None else g.detach().double() for p,g in zip(params,P)]
        t64=[torch.zeros_like(p,dtype=torch.float64) if g is None else g.detach().double() for p,g in zip(params,T)]
        p2=sum(x.square().sum() for x in p64);t2=sum(x.square().sum() for x in t64)
        dot=sum((x*y).sum() for x,y in zip(p64,t64))
        _need(bool(torch.isfinite(p2)&torch.isfinite(t2)&torch.isfinite(dot)),'Overflow/nonfinite FP64 full-space norm or dot')
        projected=bool(p2>0) and bool(dot<0)
        coefficient=dot/p2 if projected else torch.zeros((),device=device,dtype=torch.float64)
        _need(bool(torch.isfinite(coefficient)),'Nonfinite full-space projection coefficient')
        teacher_out=[];combined=[]
        for p,pp,tt,g in zip(params,p64,t64,T):
            if projected:out=(tt-coefficient*pp).to(dtype=p.dtype)
            else:out=torch.zeros_like(p) if g is None else g.detach().clone()
            base=torch.zeros_like(p) if P[len(combined)] is None else P[len(combined)].detach()
            total=base+out
            _need(bool(torch.isfinite(out).all()&torch.isfinite(total).all()),'Projection output overflow; never silently clip gradients')
            teacher_out.append(out);combined.append(total)
        out64=[g.double() for g in teacher_out];combined64=[g.double() for g in combined]
        projected_dot=sum((x*y).sum() for x,y in zip(p64,out64))
        combined_dot=sum((x*y).sum() for x,y in zip(p64,combined64))
        out2=sum(g.square().sum() for g in out64);combined2=sum(g.square().sum() for g in combined64)
        pnorm=float(p2.sqrt());tnorm=float(t2.sqrt());outnorm=float(out2.sqrt());totalnorm=float(combined2.sqrt())
        diagnostics=dict(schema='pose_depth_teacher_one_sided_projection_v1',parameter_tensors=PARAMETER_TENSORS,
            parameter_names=names,projection_space='one complete24-tensor vector; unused/frozen slots contribute zero',
            physical_unused_slots=physical_none,teacher_unused_slots=teacher_none,
            physical_gradient_l2=pnorm,weighted_teacher_gradient_l2=tnorm,projected_teacher_gradient_l2=outnorm,
            combined_gradient_l2=totalnorm,physical_teacher_dot=float(dot),physical_projected_teacher_dot=float(projected_dot),
            physical_combined_dot=float(combined_dot),physical_gradient_norm_squared=float(p2),coefficient=float(coefficient),
            projection_applied=projected,reason='negative_teacher_physical_dot' if projected else 'zero_physical_norm' if pnorm==0 else 'nonnegative_teacher_physical_dot',
            physical_teacher_cosine=float(dot)/(pnorm*tnorm) if pnorm and tnorm else None,
            physical_combined_cosine=float(combined_dot)/(pnorm*totalnorm) if pnorm and totalnorm else None,
            arithmetic='FP64 dots/norms/coefficient/negative-case projection; returned gradients cast to originaldtype',
            dtype=str(dtype),device=str(device),teacher_input_already_weighted=True,
            formula='Tprime=T-min(dot(T,P)/dot(P,P),0)*P; G=P+Tprime; Pnorm0 leaves T unchanged',
            input_gradient_fields_mutated=False,optimizer_step_applied=False,
            limitations='Local Euclidean gradient operation, not an AdamW direction or guarantee of validation/pose improvement; reverse gradients at a trained endpoint can reflect regularized equilibrium.')
    return dict(combined_grads=combined,projected_teacher_grads=teacher_out,diagnostics=diagnostics)
