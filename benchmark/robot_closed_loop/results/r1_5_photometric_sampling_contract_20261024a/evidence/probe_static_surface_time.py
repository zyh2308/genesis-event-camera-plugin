"""Isolate temporal radiance on a permanently visible, stationary floor."""
from pathlib import Path
import sys,contextlib,json,hashlib
import numpy as np,h5py
ROOT=next(Path('/home').glob('*/Eventbased_WAM/repositories/genesis-event-camera-plugin'))
sys.path.insert(0,str(ROOT))
ART=Path('/mnt/c/Users/zhaoyuhan/Documents/ChatGPT/New project/docs/robot_structural_diagnosis_2026-10-24')
BASE=ROOT/'benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059'
with contextlib.redirect_stdout(sys.stderr):
    import torch
    torch.set_num_threads(1)
    from genesis_event_plugin.dvs_emulator import DvsEmulator
    from genesis_event_plugin.noise_model import DvsNoiseConfig
    cfg=DvsNoiseConfig(pos_thres=.2,neg_thres=.2,sigma_thres=0.,cutoff_hz=0.,leak_rate_hz=0.,leak_jitter_fraction=0.,shot_noise_rate_hz=0.,noise_rate_cov_decades=0.,refractory_period_s=0.,radiance_input=True,response_mode='physical_log',readout_model='ideal_readout')
    result={}
    for ep in ['004','005']:
        src=BASE/f'episode_{ep}/same_state_source_cache.h5';den=BASE/f'episode_{ep}/rendered_frame_evidence.h5'
        with h5py.File(src) as f:
            st=f['timestamps_s'][:];im=f['radiance_luma'][:];sg=f['segmentation'][:];depth=f['depth'][:];cam=f['camera_to_world_cv'][:];ids=f['segmentation_pose_ids'][:];poses=f['object_poses'][:]
        with h5py.File(den) as f:
            ts=f['dense_times_s'][:];di=f['dense_direct_hdr_luminance'][:];ds=f['dense_direct_segmentation'][:]
        lo=int(np.searchsorted(st,ts[0],side='right')-1);hi=int(np.searchsorted(st,ts[-1],side='left'))
        assert abs(st[hi]-ts[-1])<1e-8 and hi==lo+1
        # Fixed mask: the same floor pixels are visible in BOTH endpoints and
        # EVERY exact direct row, away from geometry-label boundaries.
        mask=(sg[lo]==1)&(sg[hi]==1)&np.all(ds==1,axis=0)
        for _ in range(2):
            p=np.pad(mask,1,constant_values=False)
            mask=np.logical_and.reduce([p[y:y+mask.shape[0],x:x+mask.shape[1]] for y in range(3) for x in range(3)])
        mask &= np.isfinite(depth[lo])&np.isfinite(depth[hi])&(abs(depth[lo]-depth[hi])<1e-7)
        # The fixed plane is absent from the moving-body pose map and is
        # treated as static world geometry by the plugin, matching the scene.
        assert 1 not in ids
        assert np.allclose(cam[lo],cam[hi],atol=1e-12,rtol=0)
        init=np.where(mask,im[lo],1.)
        log0=np.log(np.maximum(im[lo],1e-6));log1=np.log(np.maximum(im[hi],1e-6))
        streams={};frame_counts={}
        for name in ['direct_cached','endpoint_log_linear']:
            em=DvsEmulator(res=mask.shape,config=cfg,device='cpu');em.initialize(init,float(st[lo]));chunks=[];ns=[]
            for i,t in enumerate(ts):
                alpha=float((t-st[lo])/(st[hi]-st[lo]))
                image=di[i] if name=='direct_cached' else np.exp((1-alpha)*log0+alpha*log1)
                ev=em.generate_events(np.where(mask,image,1.),float(t));chunks.append(ev);ns.append(len(ev))
            streams[name]=np.concatenate(chunks);frame_counts[name]=ns
        def buckets(events):
            d={}
            for t,x,y,p in events:d.setdefault((int(x),int(y),int(p)),[]).append(float(t))
            return d
        a,b=(buckets(streams[k]) for k in ['direct_cached','endpoint_log_linear'])
        dif=[]
        for key in a.keys()&b.keys():dif.extend(abs(x-y) for x,y in zip(sorted(a[key]),sorted(b[key])))
        endpoint_subthreshold=mask&(abs(log1-log0)<.2)
        sub_ev=streams['direct_cached']
        selected=endpoint_subthreshold[sub_ev[:,2].astype(int),sub_ev[:,1].astype(int)]
        actual_subthreshold_events=int(selected.sum())
        active_coords={(int(x),int(y)) for _,x,y,_ in sub_ev[selected]}
        eventfile=ART/f'episode_{ep}_static_floor_temporal_events.npz'
        np.savez_compressed(eventfile,mask=mask,times=ts,direct=streams['direct_cached'],predicted=streams['endpoint_log_linear'])
        result[ep]={'source_bracket_s':[float(st[lo]),float(st[hi])],'direct_times_s':ts.tolist(),'fixed_stationary_floor_pixels':int(mask.sum()),'camera_and_floor_pose_unchanged':True,'max_floor_depth_difference_m':float(np.max(abs(depth[lo][mask]-depth[hi][mask]))),'events':{k:len(v) for k,v in streams.items()},'endpoint_subthreshold_pixels':int(endpoint_subthreshold.sum()),'direct_events_on_endpoint_subthreshold_pixels':actual_subthreshold_events,'active_endpoint_subthreshold_pixel_count':len(active_coords),'frame_event_counts':frame_counts,'same_pixel_polarity_pairs_by_temporal_order':len(dif),'paired_time_error_ms_median':float(1000*np.median(dif)) if dif else None,'paired_time_error_ms_p95':float(1000*np.quantile(dif,.95)) if dif else None,'pairs_within_0_1ms':int(np.sum(np.array(dif)<=.0001)),'pairs_within_1ms':int(np.sum(np.array(dif)<=.001)),'event_npz_sha256':hashlib.sha256(eventfile.read_bytes()).hexdigest()}
print(json.dumps(result,indent=2))
