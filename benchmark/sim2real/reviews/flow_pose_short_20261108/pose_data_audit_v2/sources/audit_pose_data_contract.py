"""CPU read-only TRAIN coverage and native-window representation audit; no scores."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import cv2
import h5py
import numpy as np

METHODS=('eventgenesis','v2e')
LEVELS=('normal','half','eighth')
V2E_FLAGS=('input_frame_rate','auto_timestamp_resolution','timestamp_resolution','pos_thres','neg_thres',
 'sigma_thres','cutoff_hz','leak_rate_hz','leak_jitter_fraction','noise_rate_cov_decades',
 'shot_noise_rate_hz','refractory_period','dvs_emulator_seed')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def summarize(values,bins):
    a=np.asarray(values,dtype=float)
    if not len(a):return dict(count=0)
    assert np.isfinite(a).all()
    count,_=np.histogram(a,bins=bins)
    return dict(count=len(a),min=float(a.min()),q10=float(np.quantile(a,.1)),median=float(np.median(a)),
                q90=float(np.quantile(a,.9)),max=float(a.max()),histogram=count.tolist(),
                bins=[str(v) if not np.isfinite(v) else float(v) for v in bins])


def row_id(row):return row.get('rollout_id',row.get('rollout'))


def projection(group,k,vertices):
    K=np.asarray(group['camera_intrinsics_K_px']);keys=np.asarray(group['cad_keypoints'])
    T=np.linalg.inv(np.asarray(group['camera_to_world_cv'][k]))@np.asarray(group['target_pose'][k])
    xyz=keys@T[:3,:3].T+T[:3,3];q=xyz@K.T;uv=q[:,:2]/q[:,2:]
    v=vertices@T[:3,:3].T+T[:3,3];v=v@K.T;vu=v[:,:2]/v[:,2:]
    scale=float(np.exp(np.log(np.ptp(vu/[346,260],axis=0)).mean()))
    left,_,right=np.linalg.svd(T[:3,:3]);R=left@right
    if np.linalg.det(R)<0:left[:,-1]*=-1;R=left@right
    return dict(K=K,keys=keys,T=T,uv=uv,z=float(xyz[0,2]),center3=xyz[0],scale=scale,R=R)


def geometry(group,rid,vertices):
    ts=np.asarray(group['frame_timestamps_s']);rows={};skipped=[]
    assert len(ts)==21 and np.all(np.diff(ts)>0)
    for k in range(4,20):
        p=projection(group,k+1,vertices)
        if not ((p['uv'][0]>0).all() and (p['uv'][0]<[346,260]).all()):skipped.append(f'{rid}/{k:03d}');continue
        prior=projection(group,k,vertices);dt=ts[k+1]-ts[k]
        angle=float(np.arccos(np.clip((np.trace(p['R'])-1)/2,-1,1)))
        angular=float(np.arccos(np.clip((np.trace(p['R']@prior['R'].T)-1)/2,-1,1))/dt)
        p.update(id=f'{rid}/{k:03d}',rid=rid,k=k,start_s=float(ts[k]),end_s=float(ts[k+1]),
                 translational_speed_m_s=float(np.linalg.norm(p['center3']-prior['center3'])/dt),
                 screen_speed_px_s=float(np.linalg.norm(p['uv'][0]-prior['uv'][0])/dt),
                 orientation_angle_rad=angle,angular_speed_rad_s=angular)
        rows[p['id']]=p
    return rows,skipped


def distributions(rows):
    values=list(rows.values())
    return dict(z_m=summarize([r['z'] for r in values],[0,.38,.5,.65,.8,1,1.15,1.5,np.inf]),
      center_u=summarize([r['uv'][0,0]/346 for r in values],[0,.1,.25,.5,.75,.9,1]),
      center_v=summarize([r['uv'][0,1]/260 for r in values],[0,.1,.25,.5,.75,.9,1]),
      normalized_projected_scale=summarize([r['scale'] for r in values],[0,.05,.1,.2,.3,.5,1,np.inf]),
      orientation_rad=summarize([r['orientation_angle_rad'] for r in values],np.linspace(0,np.pi,7)),
      translation_speed_m_s=summarize([r['translational_speed_m_s'] for r in values],[0,.05,.1,.25,.5,1,2,5,np.inf]),
      angular_speed_rad_s=summarize([r['angular_speed_rad_s'] for r in values],[0,.1,.5,1,2,4,8,np.inf]),
      screen_speed_px_s=summarize([r['screen_speed_px_s'] for r in values],[0,10,50,100,250,500,1000,2000,np.inf]))


def encoder(path):
    tree=ast.parse(Path(path).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='encode_event_window')
    size=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NETWORK_SIZE' for t in n.targets))
    scope=dict(np=np,cv2=cv2,NETWORK_SIZE=size)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['encode_event_window'],size


def v2e_settings(row):
    command=row.get('command',[]);out={}
    for name in V2E_FLAGS:
        flag='--'+name
        if flag in command:out[name]=command[command.index(flag)+1]
    out['weight_sha256']=row.get('weight_sha256')
    return out


def histogram_quantiles(values,probabilities=(.5,.9,.99)):
    a=np.asarray(values,dtype=np.int64).reshape(-1)
    if not len(a):return [None]*len(probabilities)
    counts=np.bincount(a);cumulative=counts.cumsum();n=len(a)
    out=[]
    for probability in probabilities:
        position=(n-1)*probability;lo=int(np.floor(position));hi=int(np.ceil(position))
        left=np.searchsorted(cumulative,lo,side='right');right=np.searchsorted(cumulative,hi,side='right')
        out.append(float(left+(position-lo)*(right-left)))
    return out


def count_metrics(values):
    values=np.asarray(values).reshape(-1)
    if not len(values):return None
    q=histogram_quantiles(values);active=values[values>0];aq=histogram_quantiles(active)
    return dict(nonzero_rate=float(np.mean(values>0)),greater_than8_rate=float(np.mean(values>8)),
        at_least8_rate=float(np.mean(values>=8)),max_count=int(values.max()),
        count_p50=q[0],count_p90=q[1],count_p99=q[2],
        active_count_p50=aq[0],active_count_p90=aq[1],active_count_p99=aq[2])


def information_audit(banks,geometries,encode,vertices):
    """Raw sensor counts and actual whole-frame encoder; GT box is statistics only."""
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2)
    buckets={};evidence=[]
    for split,level_banks in banks.items():
        for level,bank in level_banks.items():
            geometry_rows=geometries[split][level]
            grouped={}
            for row in geometry_rows.values():grouped.setdefault(row['rid'],[]).append(row)
            for method in METHODS:
                manifest=json.loads((bank/'events'/method/'manifest.json').read_text())
                items={row_id(r):r for r in manifest['rollouts']}
                assert manifest['physical_manifest_sha256']==sha(bank/'bank_manifest.json') and not manifest['real_data_used']
                for rid,rows in sorted(grouped.items()):
                    raw_path=bank/'events'/method/(rid+'.npy');assert sha(raw_path)==items[rid]['events_sha256']
                    raw=np.load(raw_path,allow_pickle=False)
                    assert np.isfinite(raw).all() and (len(raw)==0 or (np.diff(raw[:,0])>=0).all())
                    t,x,y,p=raw[:,0],raw[:,1].astype(int),raw[:,2].astype(int),(raw[:,3]>0).astype(np.uint8)
                    event_counts=[]
                    for row in rows:
                        first=np.searchsorted(t,row['start_s'],side='right');last=np.searchsorted(t,row['end_s'],side='right')
                        xs,ys,ps=x[first:last],y[first:last],p[first:last]
                        assert len(xs)==0 or (xs.min()>=0 and xs.max()<346 and ys.min()>=0 and ys.max()<260)
                        pixel=ys*346+xs
                        on=np.bincount(pixel[ps==1],minlength=260*346).reshape(260,346)
                        off=np.bincount(pixel[ps==0],minlength=260*346).reshape(260,346)
                        counts=np.stack([on,off,on+off])
                        frame=encode(t,x,y,p,row['start_s'],row['end_s'],346,260,8.)
                        resized=F.interpolate(torch.from_numpy(frame[None]),(128,128),mode='area')[0].numpy()
                        camera=vertices@row['T'][:3,:3].T+row['T'][:3,3];q=camera@row['K'].T;uv=q[:,:2]/q[:,2:]
                        lower=np.floor(uv.min(0)).astype(int);upper=np.ceil(uv.max(0)).astype(int)
                        lower=np.clip(lower,[0,0],[346,260]);upper=np.clip(upper,[0,0],[346,260])
                        mask=np.zeros((260,346),bool);mask[lower[1]:upper[1],lower[0]:upper[0]]=True
                        zbin=('z_lt0.5' if row['z']<.5 else 'z_0.5_0.8' if row['z']<.8 else 'z_0.8_1.15' if row['z']<1.15 else 'z_ge1.15')
                        for channel,name in enumerate(('ON','OFF','TOTAL')):
                            for region,selection in [('full',np.ones_like(mask)),('CAD_bbox',mask),('outside_bbox',~mask)]:
                                metrics=count_metrics(counts[channel][selection])
                                if metrics is None:continue
                                if region=='full':
                                    metrics['encoded_network_exact_one_rate']=float(np.mean(frame[channel]==1))
                                    metrics['actual_model128_exact_one_rate']=float(np.mean(resized[channel]==1))
                                for depth_group in ('all_z',zbin):
                                    key=(split,method,level,depth_group,region,name)
                                    bucket=buckets.setdefault(key,{})
                                    for metric,value in metrics.items():
                                        if value is not None:bucket.setdefault(metric,[]).append(value)
                        event_counts.append(last-first)
                    evidence.append(dict(split=split,method=method,level=level,rollout=rid,windows=len(rows),
                        raw_events_sha256=items[rid]['events_sha256'],window_event_count_min_median_max=[min(event_counts),float(np.median(event_counts)),max(event_counts)]))
                print('INPUT_INFORMATION',split,method,level,sum(len(r) for r in grouped.values()),flush=True)
    summary=[]
    for key,bucket in sorted(buckets.items()):
        summary.append(dict(zip(('split','method','level','depth_bin','region','channel'),key),
            windows=max(len(v) for v in bucket.values()),
            metrics={metric:dict(window_equal_mean=float(np.mean(v)),window_median=float(np.median(v)),
                                window_q10=float(np.quantile(v,.1)),window_q90=float(np.quantile(v,.9))) for metric,v in bucket.items()}))
    return dict(summary=summary,raw_event_evidence=evidence,
        interpretation='Raw count quantiles are computed within each sensor region/window, then summarized across windows. Count>=8 already encodes to1; count>8 is reported separately. Bboxes are GT diagnostic regions only; all actual model inputs stay full frame.',
        model_input='unchanged native log1p(count)/log1p(8),clip1,bilinear NETWORK_SIZE,then native area128',
        training_or_inference_crops=False,GPU_used=False,model_inference=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--radiance-train',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--representations',action='store_true')
    p.add_argument('--input-information',action='store_true');p.add_argument('--radiance-validation',type=Path)
    args=p.parse_args();assert args.out.is_absolute() and not args.out.exists() and args.out.parent.is_dir()
    cv2.setNumThreads(1);start=time.monotonic();R=args.campaign;normal_bank=R/'pose_bank'
    normal_manifest=json.loads((normal_bank/'bank_manifest.json').read_text())
    train_ids={row_id(r) for r in normal_manifest['rollouts'] if r['split']=='train'}
    split_counts={s:sum(r['split']==s for r in normal_manifest['rollouts']) for s in sorted({r['split'] for r in normal_manifest['rollouts']})}
    prepared=json.loads((args.radiance_train/'completion.json').read_text());selected=set(prepared['selected'])
    assert prepared['status']=='complete' and not prepared['real_data_used']
    assets=json.loads((R/'recovery_20261105_mainline/pose_prepared_full_v5/protocol.json').read_text())['assets']
    assert sha(assets['vertices']['path'])==assets['vertices']['sha256'];vertices=np.load(assets['vertices']['path'],allow_pickle=False)
    encode,network_size=encoder(R/'sources/benchmark/sim2real/scripts/run_pose_stage2_pilot.py')
    banks=dict(normal=normal_bank,half=args.radiance_train/'half/pose_bank',eighth=args.radiance_train/'eighth/pose_bank')
    geom={};skipped={};geometry_equal={};file_rows={}
    for level,bank in banks.items():
        rows={};skip=[];paths=[]
        for rid in sorted(selected):
            path=bank/'physical'/(rid+'.h5');paths.append(str(path))
            with h5py.File(path,'r') as stream:
                g=stream['episodes'][rid];assert g.attrs['split']=='train'
                batch,missing=geometry(g,rid,vertices);rows.update(batch);skip+=missing
        geom[level]=rows;skipped[level]=skip;file_rows[level]=paths
    for level in ('half','eighth'):
        a,b=geom['normal'],geom[level];same=set(a)==set(b)
        maxima={k:max(float(np.max(np.abs(a[s][k]-b[s][k]))) for s in a.keys()&b.keys()) for k in ('T','K','keys','uv')}
        time_same=all(a[s]['start_s']==b[s]['start_s'] and a[s]['end_s']==b[s]['end_s'] for s in a.keys()&b.keys())
        geometry_equal[level]=dict(same_ids=same,timewindows_equal=time_same,max_abs=maxima)
    manifests={};contracts={};adapted={};warnings=[];repr_records={};normal_ids_per_method={}
    for method in METHODS:
        settings={};total_normal={};normal_files=sorted((R/'adapters'/method/'pose'/method/'train').glob('*.npz'))
        assert normal_files
        for level,bank in banks.items():
            mp=bank/'events'/method/'manifest.json';m=json.loads(mp.read_text())
            assert not m['real_data_used'] and m['physical_manifest_sha256']==sha(bank/'bank_manifest.json')
            items={row_id(r):r for r in m['rollouts']};assert selected<=set(items)
            manifests[(method,level)]=(m,items)
            settings[level]=dict(schema=m.get('schema'),core_release=m.get('core_release'),
                sensor_profile_sha256=m.get('sensor_profile_sha256'),source_commit=m.get('source_commit'),
                per_rollout_profile_sha256=sorted({r['profile_sha256'] for r in items.values() if r.get('profile_sha256')}),
                per_rollout_v2e_parameters=[dict(v) for v in {json.dumps(v2e_settings(items[rid]),sort_keys=True):v2e_settings(items[rid]) for rid in selected}.values()] if method=='v2e' else None,
                manifest_sha256=sha(mp),source_cadence_hz=sorted({r['source_cadence_hz'] for r in items.values() if 'source_cadence_hz' in r}))
        invariant_keys=('core_release','sensor_profile_sha256','source_commit','per_rollout_profile_sha256','per_rollout_v2e_parameters','source_cadence_hz')
        contracts[method]=dict(settings=settings,invariant_settings_equal=all(all(settings['normal'][k]==settings[l][k] for k in invariant_keys) for l in ('half','eighth')))
        maxima=dict(K=0.,keys=0.,T=0.,uv=0.);repr_max=0.;repr_mismatch=[];label_time_issues=[]
        for path in normal_files:
            with np.load(path,allow_pickle=False) as archive:
                # NpzFile does not cache decompression. Read each array once per rollout.
                z={name:archive[name] for name in archive.files}
                ids=[str(s) for s in z['sample_ids']];assert len(ids)==len(z['images'])
                for i,sid in enumerate(ids):
                    assert sid not in total_normal;g=geom['normal'][sid]
                    for key,zk in [('K','K'),('keys','cad_keypoints')]:maxima[key]=max(maxima[key],float(np.max(np.abs(z[zk]-g[key]))))
                    maxima['T']=max(maxima['T'],float(np.max(np.abs(z['target_to_camera'][i]-g['T']))))
                    maxima['uv']=max(maxima['uv'],float(np.max(np.abs(z['uv_normalized'][i]-g['uv']/[346,260]))))
                    total_normal[sid]=g
                if 'label_times_s' in z:
                    times=z['label_times_s'];aligned=len(times)==len(ids) and all(abs(float(t)-geom['normal'][sid]['end_s'])<1e-9 for t,sid in zip(times,ids))
                    if not aligned:label_time_issues.append(dict(rollout=path.stem,images=len(ids),label_times=len(times)))
                if args.representations:
                    rid=path.stem;raw_path=normal_bank/'events'/method/(rid+'.npy');item=manifests[(method,'normal')][1][rid]
                    assert sha(raw_path)==item['events_sha256'];raw=np.load(raw_path,allow_pickle=False)
                    assert np.isfinite(raw).all() and (len(raw)==0 or (np.diff(raw[:,0])>=0).all())
                    for i,sid in enumerate(ids):
                        g=geom['normal'][sid];fresh=encode(raw[:,0],raw[:,1].astype(int),raw[:,2].astype(int),(raw[:,3]>0).astype(np.uint8),g['start_s'],g['end_s'],346,260,8.)
                        delta=float(np.max(np.abs(z['images'][i]-fresh)));repr_max=max(repr_max,delta)
                        if delta:repr_mismatch.append(dict(sample_id=sid,max_abs=delta))
        adapted[method]=dict(files=len(normal_files),rollouts=len({r['rid'] for r in total_normal.values()}),windows=len(total_normal),
             same_ids_as_fresh_base=set(total_normal)==set(geom['normal']),max_abs_geometry=maxima,
             unused_label_times_metadata_issues=label_time_issues,distribution=distributions(total_normal))
        normal_ids_per_method[method]=set(total_normal)
        repr_records[method]=dict(executed=args.representations,normal_npz_vs_fresh_base_max_abs=repr_max if args.representations else None,
             mismatch_windows=len(repr_mismatch) if args.representations else None,first_mismatches=repr_mismatch[:8],
             equivalence='exact arrays before the common deterministic area resize to128' if args.representations and not repr_mismatch else 'not established')
        print('COVERAGE_METHOD',method,adapted[method]['rollouts'],adapted[method]['windows'],'REPR_MAX',repr_max if args.representations else None,flush=True)
    n=len(geom['normal']);half=len(geom['half']);dark=len(geom['eighth']);draws=6*n
    expected=dict(normal_draws_total=draws,half_draws_total=draws*.75,eighth_draws_total=draws*.25,
                  normal_visits_per_window=6,half_expected_visits_per_window=draws*.75/half,
                  eighth_expected_visits_per_window=draws*.25/dark)
    information=None
    if args.input_information:
        assert args.radiance_validation is not None
        validation_root=R/'recovery_20261105_mainline/jobs/119_pose_fresh_confirm_paired_20261106/results/pose_bank'
        val_protocol=json.loads((args.radiance_validation/'protocol.json').read_text())
        val_ids=val_protocol['selected'];assert len(val_ids)==4 and not (set(val_ids)&selected)
        val_banks=dict(normal=validation_root,half=args.radiance_validation/'half/pose_bank',eighth=args.radiance_validation/'eighth/pose_bank')
        val_geom={}
        for level,bank in val_banks.items():
            value={}
            for rid in val_ids:
                with h5py.File(bank/'physical'/(rid+'.h5'),'r') as stream:
                    group=stream['episodes'][rid];assert group.attrs['split']=='validation'
                    batch,_=geometry(group,rid,vertices);value.update(batch)
            val_geom[level]=value
        information=information_audit(dict(train=banks,validation=val_banks),dict(train=geom,validation=val_geom),encode,vertices)
    passed=(train_ids==selected and normal_ids_per_method['eventgenesis']==normal_ids_per_method['v2e']
        and all(x['same_ids'] and x['timewindows_equal'] and max(x['max_abs'].values())<1e-7 for x in geometry_equal.values())
        and all(x['invariant_settings_equal'] for x in contracts.values())
        and all(x['same_ids_as_fresh_base'] and max(x['max_abs_geometry'].values())<1e-6 for x in adapted.values())
        and (not args.representations or all(x['mismatch_windows']==0 for x in repr_records.values())))
    result=dict(schema='pose_train_data_contract_audit_v1',status='passed' if passed else 'mismatch',
       real_data_used=False,model_inference=False,optimizer_updates=0,GPU_used=False,
       representations_executed=args.representations,split_counts=split_counts,selected_rollouts=len(selected),
       all_train_selected=train_ids==selected,missing_train=sorted(train_ids-selected),foreign_selected=sorted(selected-train_ids),
       fresh_geometry={l:dict(windows=len(geom[l]),skipped_ids=skipped[l],distribution=distributions(geom[l])) for l in LEVELS},
       normal_vs_relit_geometry=geometry_equal,normal_adapter=adapted,representation=repr_records,
       event_generator_contracts=contracts,expected_six_epoch_exposure=expected,
       input_information=information,
       network_size=network_size,encode_source_sha256=sha(R/'sources/benchmark/sim2real/scripts/run_pose_stage2_pilot.py'),
       source_script_sha256=sha(Path(__file__)),elapsed_seconds=time.monotonic()-start,
       limitation='TRAIN coverage/encoding contract only; this cannot establish a cause for validation failures. No new render/replay/train/evaluation launched.')
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],all_train_selected=result['all_train_selected'],split_counts=split_counts,windows=n,
          representation=repr_records,expected_six_epoch_exposure=expected,elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
