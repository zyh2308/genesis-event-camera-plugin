"""CPU-only cached-image diagnosis. No Genesis init/render, training or repo writes."""
from pathlib import Path
import sys,json,hashlib,time,contextlib
import numpy as np,h5py
from PIL import Image,ImageDraw

matches=list(Path('/home').glob('*/Eventbased_WAM/repositories/genesis-event-camera-plugin'))
if len(matches)!=1:raise RuntimeError(f'ambiguous local research checkout: {matches}')
ROOT=matches[0]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059'
out={'scope':'two historical development clips; two-endpoint surface candidate, not full multi-reference rollout; no new renderer data'}
ART=Path('/mnt/c/Users/zhaoyuhan/Documents/ChatGPT/New project/docs/robot_structural_diagnosis_2026-10-24')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def boundary(seg,radius=2):
    b=np.zeros(seg.shape,bool)
    b[1:]|=seg[1:]!=seg[:-1];b[:-1]|=seg[:-1]!=seg[1:]
    b[:,1:]|=seg[:,1:]!=seg[:,:-1];b[:,:-1]|=seg[:,:-1]!=seg[:,1:]
    for _ in range(radius):
        p=np.pad(b,1);b=np.logical_or.reduce([p[y:y+seg.shape[0],x:x+seg.shape[1]] for y in range(3) for x in range(3)])
    return b
def metrics(diff,mask):
    d=diff[mask]
    if not len(d):return {'pixels':0}
    return {'pixels':len(d),'mean_abs_log_error':float(d.mean()),'p95_abs_log_error':float(np.quantile(d,.95)),'p99_abs_log_error':float(np.quantile(d,.99)),'over_0_05':int((d>.05).sum()),'over_0_2':int((d>.2).sum())}
with contextlib.redirect_stdout(sys.stderr):
    import torch
    torch.set_num_threads(1)
    from genesis_event_plugin.physics_warp import project_endpoint_to_alpha,interpolate_se3,depth_aware_bidirectional_warp_projected_diagnostics
    from benchmark.robot_closed_loop.diagnostics.r1_5_surface_raster_candidate import surface_raster_warp,compose_surface_warps
    rows=[]
    for ep in ['004','005']:
        src=BASE/f'episode_{ep}/same_state_source_cache.h5';evidence=BASE/f'episode_{ep}/rendered_frame_evidence.h5'
        with h5py.File(src) as f:
            times=f['timestamps_s'][:];I=f['radiance_luma'][:];dep=f['depth'][:];seg=f['segmentation'][:];cams=f['camera_to_world_cv'][:];ids=f['segmentation_pose_ids'][:];poses=f['object_poses'][:]
            focal=240/(2*np.tan(np.deg2rad(float(f.attrs['vertical_fov_deg']))/2))
        with h5py.File(evidence) as f:
            dt=f['dense_times_s'][:];direct=f['dense_direct_hdr_luminance'][:];dseg=f['dense_direct_segmentation'][:]
        out.setdefault('inputs',{})[ep]={'source_sha256':sha(src),'direct_sha256':sha(evidence),'source_times':times.tolist(),'dense_times':dt.tolist()}
        chosen=sorted(set(int(np.argmin(abs(dt-t))) for t in [.5755,.5855,.5905]))
        for di in chosen:
            t=float(dt[di]);j=int(np.searchsorted(times,t,side='right')-1);j=max(0,min(j,len(times)-2));a=float((t-times[j])/(times[j+1]-times[j]))
            # Timestamps are float; this only recognises observed endpoints.
            at_end=np.flatnonzero(abs(times-t)<1e-8)
            region_edge=boundary(dseg[di]);ref=direct[di]
            if len(at_end):
                diff=np.abs(np.log(np.maximum(I[int(at_end[0])],1e-6))-np.log(np.maximum(ref,1e-6)))
                rows.append({'episode':ep,'time_s':t,'alpha':None,'K_center_offset_px':0.,'candidate':'observed_endpoint_identity','all_supported':metrics(diff,np.ones(ref.shape,bool))})
                continue
            o0={int(k):poses[j,n] for n,k in enumerate(ids)};o1={int(k):poses[j+1,n] for n,k in enumerate(ids)}
            for offset in [0.,-.5]:
                K=np.array([[focal,0,160+offset],[0,focal,120+offset],[0,0,1.]])
                T=interpolate_se3(cams[j],cams[j+1],a)
                projections=[project_endpoint_to_alpha(dep[j+k],seg[j+k],K,cams[j+k],T,a,k,o0,o1) for k in [0,1]]
                frozen,valid,_=depth_aware_bidirectional_warp_projected_diagnostics(I[j],dep[j],*projections[0],I[j+1],dep[j+1],*projections[1])
                warped=[surface_raster_warp(I[j+k],dep[j+k],seg[j+k],*projections[k],K) for k in [0,1]]
                c,has,request,diag=compose_surface_warps(*warped[0][:4],*warped[1][:4],a)
                shared=valid&has&~request&np.isfinite(c)&(diag['target_labels']==dseg[di])
                candidates=[('frozen_point_next_primary',frozen,valid,np.zeros_like(valid),None),('continuous_surface_log_blend',c,has,request,diag['target_labels'])]
                if len(at_end):candidates.append(('observed_endpoint_identity',I[int(at_end[0])],np.ones(ref.shape,bool),np.zeros(ref.shape,bool),seg[int(at_end[0])]))
                for name,im,v,req,labels in candidates:
                    diff=np.abs(np.log(np.maximum(im,1e-6))-np.log(np.maximum(ref,1e-6)))
                    support=v&~req&np.isfinite(im)
                    # Same-label support excludes known geometric mislabeling,
                    # but is NOT proof of matched physical surface identity.
                    same=support if labels is None else support&(labels==dseg[di])
                    floor_fixed=shared&(dseg[di]==1)&(seg[j]==1)&(seg[j+1]==1)
                    source_change=np.abs(np.log(np.maximum(I[j],1e-6))-np.log(np.maximum(I[j+1],1e-6)))
                    rows.append({'episode':ep,'time_s':t,'alpha':a,'K_center_offset_px':offset,'candidate':name,'supported_fraction':float(support.mean()),'same_label_fraction_of_supported':None if labels is None or not support.any() else float((labels[support]==dseg[di][support]).mean()),'all_supported':metrics(diff,same),'contour_band_2px_supported':metrics(diff,same&region_edge),'interior_supported':metrics(diff,same&~region_edge),'common_mask_all':metrics(diff,shared),'common_mask_contour':metrics(diff,shared&region_edge),'common_mask_interior':metrics(diff,shared&~region_edge),'static_floor':{'camera_pose_same':bool(np.allclose(cams[j],cams[j+1],atol=1e-12,rtol=0)),'floor_has_no_dynamic_pose':1 not in o0 and 1 not in o1,'errors':metrics(diff,floor_fixed),'bad_pixels_with_source_log_change_over_0_05':int(np.sum(floor_fixed&(diff>.2)&(source_change>.05)))},'per_label':{str(int(k)):metrics(diff,same&(dseg[di]==k)) for k in np.unique(dseg[di])}})
                if offset==0 and abs(a-.5)<1e-6:
                    lo,hi=np.percentile(np.log(np.maximum(ref,1e-6)),[1,99.5])
                    canvas=Image.new('RGB',(960,270),(20,20,20));draw=ImageDraw.Draw(canvas)
                    for ix,(title,pic) in enumerate([('Direct cached HDR',ref),('Supported surface candidate',np.where(shared,c,ref))]):
                        vis=np.uint8(255*np.clip((np.log(np.maximum(pic,1e-6))-lo)/(hi-lo),0,1));canvas.paste(Image.fromarray(vis).convert('RGB'),(ix*320,28));draw.text((ix*320+4,8),title,fill='white')
                    err=np.abs(np.log(np.maximum(c,1e-6))-np.log(np.maximum(ref,1e-6)))
                    rgb=np.zeros((240,320,3),np.uint8);rgb[...,0]=np.uint8(255*np.clip(np.where(shared,err,0)/.2,0,1));rgb[...,2]=np.where(shared,0,80).astype(np.uint8)
                    canvas.paste(Image.fromarray(rgb),(640,28));draw.text((644,8),'Error: red; unsupported: blue',fill='white')
                    canvas.save(ART/f'episode_{ep}_midpoint_residual.png')
    out['rows']=rows
    out['source_hashes']={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'genesis_event_plugin/physics_warp.py',ROOT/'benchmark/robot_closed_loop/diagnostics/r1_5_surface_raster_candidate.py',ROOT/'benchmark/robot_closed_loop/diagnostics/r1_5_gl_surface_raster.py']}
print(json.dumps(out,indent=2))
