"""Diagnostic-only native STE photometric maps and fixed directional oracles."""
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

EPSILONS=(.01,.05)
TEXTURE_THRESHOLD=.01

def need(x,message):
    if not x:raise ValueError(message)

def gray_at_scale(gray,hw):
    # Original compute_photometric_loss uses cv2(width=height,height=width).
    # All original256/352 model pyramid levels are square, so preserve exactly.
    h,w=hw;need(h==w,'Pinned native loss only used square pyramid levels')
    images=gray.detach().cpu().numpy()
    return torch.from_numpy(np.stack([cv2.resize(im[0],(h,w),interpolation=cv2.INTER_LINEAR) for im in images])[:,None]).to(gray.device)

def target_at_scale(flow,valid,common,sensor,hw):
    h,w=hw;H,W=flow.shape[-2:]
    target=F.interpolate(flow,size=(h,w),mode='bilinear',align_corners=False)
    target=target*target.new_tensor([w/W,h/H]).view(1,2,1,1)
    # Oracle excludes bilinear target pixels contaminated by invalid zero labels.
    strict=F.interpolate(valid.float(),size=(h,w),mode='bilinear',align_corners=False)[:,0]>=.9999
    nearest=F.interpolate(valid.float(),size=(h,w),mode='nearest')[:,0]>.5
    sensor_scaled=F.interpolate(sensor.float(),size=(h,w),mode='nearest')[:,0]>.5
    active=F.adaptive_max_pool2d(common.float(),(h,w))[:,0]>0
    return target,strict,nearest,active,sensor_scaled

def warp_inbounds(field):
    _,_,h,w=field.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=field.device),torch.arange(w,device=field.device),indexing='ij')
    return (xx+field[:,0]>=0)&(xx+field[:,0]<=w-1)&(yy+field[:,1]>=0)&(yy+field[:,1]<=h-1)

def texture(prev):
    dx=F.pad((prev[:,:,:,2:]-prev[:,:,:,:-2])*.5,(1,1,0,0),mode='replicate')
    dy=F.pad((prev[:,:,2:,:]-prev[:,:,:-2,:])*.5,(0,0,1,1),mode='replicate')
    return (dx.square()+dy.square()).sqrt()[:,0]

def native_map(prev,nxt,field,warp):
    residual=warp(nxt,field)-prev
    # Preserve upstream float32 epsilon construction, not Python1e-6 rounding.
    return torch.pow(torch.mul(residual,residual)+torch.mul(1e-3,1e-3),.45)[:,0]

def analyze_scale(prev,nxt,pred,target,valid,nearest,common,sensor,warp,scale_count=4,extra_masks=None,input_hw=None):
    need(pred.shape==target.shape and pred.shape[0]==1,'One fixed example at a time')
    field=pred.detach().clone().requires_grad_(True)
    rho=native_map(prev,nxt,field,warp)
    native_loss=rho.sum()/scale_count
    grad=torch.autograd.grad(native_loss,field,retain_graph=True)[0]
    # Isolates the often suggested mask-only change. With the original hard
    # warp mask, removing the invalid residual changes constants, not gradients.
    hard_inbounds=(warp(torch.ones_like(nxt),field).detach()[:,0]>=.9999)
    masked_gradient=torch.autograd.grad((rho*hard_inbounds).sum()/scale_count,field)[0]
    need(torch.isfinite(grad).all() and torch.isfinite(masked_gradient).all(),'Nonfinite detached-flow gradients')
    difference=target.detach()-field.detach();error=difference.square().sum(dim=1).sqrt()
    direction=difference/error[:,None].clamp_min(1e-12);direction=direction*valid[:,None]
    directed=(grad.detach()*direction).sum(dim=1)
    energy=grad.detach().square().sum(dim=1)
    original_energy=float(energy.sum())
    maps={'prediction':rho.detach()}
    with torch.no_grad():
        maps['zero']=native_map(prev,nxt,torch.zeros_like(field),warp)
        maps['GT_oracle_valid_only']=native_map(prev,nxt,target,warp)
        # MinusGT is a direction sanity diagnostic, never an alternative label.
        maps['minus_GT_diagnostic']=native_map(prev,nxt,-target,warp)
        fd={eps:(native_map(prev,nxt,field.detach()+eps*direction,warp),native_map(prev,nxt,field.detach()-eps*direction,warp)) for eps in EPSILONS}
    tex=texture(prev)
    masks=dict(all_strict_valid=valid,nearest_valid=nearest,common_active_strict_valid=valid&common,
        inactive_strict_valid=valid&~common,weak_texture_valid=valid&(tex<TEXTURE_THRESHOLD),strong_texture_valid=valid&(tex>=TEXTURE_THRESHOLD),
        weak_texture_active_valid=valid&common&(tex<TEXTURE_THRESHOLD),strong_texture_active_valid=valid&common&(tex>=TEXTURE_THRESHOLD),
        invalid_geometry_or_visibility=sensor&~nearest,invalid_GT_interpolation=sensor&nearest&~valid,padding_or_augmentation_outside=~sensor,
        predicted_out_of_bounds=sensor&~hard_inbounds,inbounds_strict_valid=valid&hard_inbounds,
        inbounds_common_active_valid=valid&common&hard_inbounds,inbounds_invalid_geometry=sensor&~nearest&hard_inbounds,
        out_of_bounds_invalid_geometry=sensor&~nearest&~hard_inbounds)
    H,W=input_hw or pred.shape[-2:];h,w=pred.shape[-2:]
    native_target=target*target.new_tensor([W/w,H/h]).view(1,2,1,1)
    mag=native_target.square().sum(dim=1).sqrt()
    for lo,hi in zip((0,.1,.5,1,2,5),(.1,.5,1,2,5,math.inf)):
        key=f'motion_{lo:g}_{hi:g}px_in_input_raster';masks[key]=valid&(mag>=lo)&(mag<hi)
        masks['active_'+key]=masks[key]&common
    if extra_masks:masks.update(extra_masks)
    need(all(torch.isfinite(v).all() for v in maps.values()),'Nonfinite native residual oracle')
    result={};total_loss=float(rho.detach().sum())
    for name,mask in masks.items():
        count=int(mask.sum());known=mask&valid;oracle_count=int(known.sum());eligible=known&(error>1e-6)
        n=int(eligible.sum());entry=dict(pixels=count,fraction_of_raster=count/mask.numel(),oracle_valid_pixels=oracle_count,alignment_eligible_pixels=n,
            photo_mean_prediction=float(maps['prediction'][mask].mean()) if count else None,
            photo_mean_zero=float(maps['zero'][mask].mean()) if count else None,
            photo_mean_GT_valid_only=float(maps['GT_oracle_valid_only'][known].mean()) if oracle_count else None,
            photo_mean_minus_GT_valid_only=float(maps['minus_GT_diagnostic'][known].mean()) if oracle_count else None,
            photo_energy_fraction=float(rho.detach()[mask].sum())/total_loss if total_loss else None,
            photo_gradient_energy_fraction=float(energy[mask].sum())/original_energy if original_energy else None,
            mean_epe_px_at_scale=float(error[known].mean()) if oracle_count else None,
            gradient_toward_GT_cosine=None,fraction_photo_descent_toward_GT=None,directional_derivative_native_sum=None,finite_differences=[])
        if n:
            g=grad.detach().permute(0,2,3,1)[eligible].double();d=direction.permute(0,2,3,1)[eligible].double()
            denom=float(g.norm()*d.norm());entry['gradient_toward_GT_cosine']=float((-g*d).sum())/denom if denom else None
            nonzero=g.norm(dim=1)>1e-12
            entry['gradient_nonzero_pixels']=int(nonzero.sum())
            entry['fraction_photo_descent_toward_GT']=float(((-g*d).sum(dim=1)[nonzero]>0).double().mean()) if bool(nonzero.any()) else None
            entry['directional_derivative_native_sum']=float(directed[eligible].sum())
            for eps,(plus,minus) in fd.items():
                dp=float((plus-rho.detach())[eligible].sum())/(eps*scale_count)
                dm=float((rho.detach()-minus)[eligible].sum())/(eps*scale_count)
                maskplus=warp_inbounds(field.detach()+eps*direction);maskminus=warp_inbounds(field.detach()-eps*direction)
                entry['finite_differences'].append(dict(epsilon_px_at_scale=eps,central_native_sum=(dp+dm)/2,forward_native_sum=dp,backward_native_sum=dm,
                    native_photo_decreases_toward_GT=float((plus-rho.detach())[eligible].sum())<0,
                    analytical_geometric_boundary_crossing_pixels=int(((maskplus!=maskminus)&eligible).sum())))
        result[name]=entry
    delta=(grad.detach()-masked_gradient.detach()).double()
    return dict(hw=list(pred.shape[-2:]),flow_unit='pixels_in_this_scale_per_30ms',scale_weight=1/scale_count,
        native_photo_sum=float(native_loss.detach()),regions=result,finite=True,
        mask_only_diagnostic=dict(max_abs_gradient_difference=float(delta.abs().max()),relative_L2_gradient_difference=float(delta.norm()/grad.detach().double().norm().clamp_min(1e-30)),
            native_sum=float(native_loss.detach()),inbounds_only_sum=float((rho.detach()*hard_inbounds).sum()/scale_count),
            note='This diagnostic removes hard-mask out-of-bounds constants only; equal gradients provide no learning-signal improvement.'),
        definitions=dict(positive_cosine='negative photo gradient aligned with GT-minus-prediction',negative_directional_derivative='photo decreases toward GT',
            GT_invalid='no valid oracle; invalid zero sentinel is never treated as truth',texture_threshold=TEXTURE_THRESHOLD,texture='central gray gradient magnitude after exact native cv2 resize',
            motion_bins='displacement converted back to input-raster pixel units; same bins across pyramid scales'))

def controls(warp,device):
    size=64;yy,xx=torch.meshgrid(torch.arange(size,device=device,dtype=torch.float32),torch.arange(size,device=device,dtype=torch.float32),indexing='ij')
    def image(x,y):return .5+.15*torch.sin(.23*x+.17*y)+.12*torch.cos(.11*x-.21*y)+.05*torch.sin(.37*x)
    prev=image(xx,yy)[None,None];clean=image(xx-2,yy+1)[None,None]
    gt=torch.zeros((1,2,size,size),device=device);gt[:,0]=2;gt[:,1]=-1;pred=gt*.25
    interior=((xx>=5)&(xx<size-5)&(yy>=5)&(yy<size-5))[None]
    sensor=torch.ones((1,size,size),device=device,dtype=torch.bool);common=sensor
    occlusion=(xx>=24)&(xx<40)&(yy>=24)&(yy<40)
    projected_occlusion=((xx+2>=24)&(xx+2<40)&(yy-1>=24)&(yy-1<40))[None]
    result=[]
    for name,brightness,occlude in [('clean_translation',False,False),('brightness_gain_offset',True,False),('occlusion',False,True),('brightness_and_occlusion',True,True)]:
        nxt=clean.clone()
        if brightness:nxt=nxt*1.1+.05
        if occlude:nxt[0,0,occlusion]=.9
        valid=interior&~projected_occlusion if occlude else interior
        diag=analyze_scale(prev,nxt,pred,gt,valid,valid,common,sensor,warp,scale_count=1,
            extra_masks={'known_occluded_correspondence':interior&projected_occlusion} if occlude else None)
        good=diag['regions']['all_strict_valid'];result.append(dict(case=name,diagnostic=diag))
        if name=='clean_translation':
            need(good['photo_mean_GT_valid_only']<good['photo_mean_zero'],'Ideal warp sign/unit control failed')
            need(good['photo_mean_GT_valid_only']<good['photo_mean_minus_GT_valid_only'],'MinusGT sign control failed')
            need(good['gradient_toward_GT_cosine']>0 and all(x['native_photo_decreases_toward_GT'] for x in good['finite_differences']),'Ideal directional gradient control failed')
    return dict(status='passed',cases=result,clean_required_controls_passed=True,intervention='deterministic analytic2px right/1px up; gain1.1+offset.05; known next-image16x16 occluder',
        brightness_or_occlusion_improvement_required=False,warp_equation='warped_next(x,y)=next(x+u,y+v); compare against prev(x,y)',optimizer_updates=0)
