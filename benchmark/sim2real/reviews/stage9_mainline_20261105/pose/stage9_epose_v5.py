"""Stage9 E-POSE full wood-block preparation and explicitly authorized evaluation.

Preparation and fixture commands never load a model. Scoring requires immutable
preparation, all15 formal checkpoint receipts, and a separately approved latch.
No simulator, training source, environment or existing result is modified.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np

METHODS=('eventgenesis','v2e','vid2e','film_esim','rife_esim')
SEEDS=(0,1,2)
WIDTH,HEIGHT,NETWORK=346,260,416
DIST=np.array([-.362,.115,.0009576,.00000455,.130],dtype=np.float64)
SEEN={'r_01_gl','r_05_gl'}
DEFAULT_BASE=Path('/home/zhaoyuhan/Eventbased_WAM')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def model_source_manifest(root):
    root=Path(root).resolve()
    paths=sorted(set(root.rglob('*.py'))|{root/'cfg/yolo-pose.cfg'})
    return {str(p.relative_to(root)):sha(p) for p in paths}

def validate_gt_linear(A,policy):
    """Accept strict rotations or the independently audited fixed native factor."""
    gram=A.T@A;orth=float(np.max(np.abs(gram-np.eye(3))));det=float(np.linalg.det(A))
    sv=np.linalg.svd(A,compute_uv=False)
    residual=float(np.max(np.abs(gram-np.asarray(policy['reference_right_gram']))))
    assert np.isfinite(A).all() and det>0,'Nonfinite/reflected GT linear transform'
    assert np.max(np.abs(sv-1))<=policy['max_singular_value_relative_strain'],'GT exceeds registered1% strain hard ceiling'
    rigid=orth<=policy['strict_rigid_max_orthogonality'] and abs(det-1)<=policy['strict_rigid_det_abs_tolerance']
    native=residual<=policy['native_gram_max_abs_residual_tolerance'] and abs(det-policy['native_det_reference'])<=policy['native_det_abs_tolerance']
    assert rigid or native,'GT linear transform is neither strict rigid nor the audited fixed native structure'
    return dict(pose_linear_orthogonality_max_abs=orth,pose_linear_determinant=det,
        pose_linear_singular_values=sv.tolist(),pose_linear_reference_gram_max_abs_residual=residual,
        pose_linear_class='strict_rigid' if rigid else 'audited_native_fixed_nonrigid',
        published_pose_preserved=True)

def metrics(errors,diameter,operator='lt'):
    """All-frame success and exact integral of the empirical success CDF."""
    a=np.array([float(x) if x is not None else np.inf for x in errors],dtype=np.float64)
    assert a.size and diameter>0 and np.all(a>=0)
    limit=.1*diameter
    valid=np.isfinite(a)
    success=(a<limit) if operator=='lt' else (a<=limit)
    auc=np.maximum(1-a/limit,0)
    auc_m=np.maximum(1-a/.1,0)
    return dict(samples=len(a),valid_errors=int(valid.sum()),success_count=int(success.sum()),
        adds_success_0_1D=float(success.mean()),adds_auc_0_to_0_1m_normalized=float(auc_m.mean()),
        adds_auc_0_to_0_1D_normalized=float(auc.mean()),
        auc_conventional_upper_limit_m=.1,auc_diameter_relative_upper_limit_m=limit,
        adds_mean_valid_m=float(a[valid].mean()) if valid.any() else None,
        adds_median_valid_m=float(np.median(a[valid])) if valid.any() else None)

def rigid_metrics(vertices,tree,Rp,tp,T):
    """ADI GT-to-estimated; isometry uses predicted Rp only, never GT inverse."""
    pred=vertices@Rp.T+tp
    truth=vertices@T[:3,:3].T+T[:3,3]
    gt_in_pred=(truth-tp)@Rp
    add=float(np.linalg.norm(pred-truth,axis=1).mean())
    adds=float(tree.query(gt_in_pred,workers=1)[0].mean())
    # Published GT is not necessarily SO3: its transpose is not a rigid inverse.
    from scipy.spatial import cKDTree
    legacy=float(cKDTree(truth).query(pred,workers=1)[0].mean())
    A=T[:3,:3];strict=np.max(np.abs(A.T@A-np.eye(3)))<=1e-4 and abs(np.linalg.det(A)-1)<=1e-4
    angle=lambda R:float(np.degrees(np.arccos(np.clip((np.trace(Rp@R.T)-1)/2,-1,1))))
    U,_,Vt=np.linalg.svd(A);polar=U@Vt
    assert np.linalg.det(polar)>0
    return dict(add_m=add,adds_m=adds,adds_legacy_est_to_gt_m=legacy,translation_error_m=float(np.linalg.norm(tp-T[:3,3])),
        rotation_raw_deg=angle(A) if strict else None,rotation_polar_SO3_aux_deg=angle(polar),
        rotation_raw_defined=bool(strict),GT_projected_for_primary_geometry=False)

def fixture(out):
    from scipy.spatial import cKDTree
    v=np.array([[x,y,z] for x in [-.05,.05] for y in [-.04,.04] for z in [-.03,.03]],dtype=np.float64)
    tree=cKDTree(v);T=np.eye(4);T[2,3]=.7
    a=rigid_metrics(v,tree,np.eye(3),T[:3,3].copy(),T)
    assert abs(a['add_m'])<1e-14 and abs(a['adds_m'])<1e-14
    shifted=T[:3,3]+[.001,0,0];b=rigid_metrics(v,tree,np.eye(3),shifted,T)
    assert abs(b['add_m']-.001)<1e-14 and abs(b['adds_m']-.001)<1e-14
    # Independent transformed-cloud tree agrees for a nontrivial rigid pose.
    ang=.31;R=np.array([[math.cos(ang),-math.sin(ang),0],[math.sin(ang),math.cos(ang),0],[0,0,1]])
    z=rigid_metrics(v,tree,R,shifted,T)
    oracle=float(cKDTree(v@R.T+shifted).query(v@T[:3,:3].T+T[:3,3])[0].mean())
    assert abs(z['adds_m']-oracle)<1e-14
    # Nonuniform nonsymmetric finite cloud exposes the two directed distances.
    asym=np.array([[0,0,0],[.001,0,0],[.003,.002,0],[.09,.03,.02],[.07,-.01,.06]],np.float64)
    angle=1.1;asym_R=np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
    asym_t=shifted+[.04,-.01,0]
    asym_result=rigid_metrics(asym,cKDTree(asym),asym_R,asym_t,T)
    asym_pred=asym@asym_R.T+asym_t;asym_gt=asym@T[:3,:3].T+T[:3,3]
    primary=float(cKDTree(asym_pred).query(asym_gt)[0].mean())
    reverse=float(cKDTree(asym_gt).query(asym_pred)[0].mean())
    assert abs(asym_result['adds_m']-primary)<1e-14
    assert abs(asym_result['adds_legacy_est_to_gt_m']-reverse)<1e-14
    assert abs(primary-reverse)>1e-7
    # Official native nonrigid GT: preserve original points in both directions.
    diag_path=Path(__file__).with_name('pose_rotation_native_diagnostic.json')
    native=np.asarray(json.loads(diag_path.read_text())['samples'][0]['pose'])
    native_before=native.copy();native_result=rigid_metrics(asym,cKDTree(asym),asym_R,asym_t,native)
    native_truth=asym@native[:3,:3].T+native[:3,3]
    native_primary=float(cKDTree(asym_pred).query(native_truth)[0].mean())
    native_reverse=float(cKDTree(native_truth).query(asym_pred)[0].mean())
    assert abs(native_result['adds_m']-native_primary)<1e-14 and abs(native_result['adds_legacy_est_to_gt_m']-native_reverse)<1e-14
    assert native_result['rotation_raw_deg'] is None and native_result['rotation_polar_SO3_aux_deg'] is not None
    assert np.array_equal(native,native_before) and not native_result['GT_projected_for_primary_geometry']
    policy=json.loads(Path(__file__).with_name('native_pose_audit_policy.json').read_text())
    assert sha(diag_path)==policy['audit_source_sha256'],'Native audit fixture differs from registered source'
    audited=validate_gt_linear(native[:3,:3],policy);assert audited['pose_linear_class']=='audited_native_fixed_nonrigid'
    rejected=[]
    for bad in [np.diag([1.02,1,1]),np.diag([-1,1,1]),np.diag([1.001,1,1])]:
        try:validate_gt_linear(bad,policy)
        except AssertionError:rejected.append(True)
    assert len(rejected)==3,'Excess strain/reflection/unregistered small nonrigidity must reject'
    # A MAT native packet can split an equal-timestamp run by row identity.
    canonical=np.array([[11,21,0.,1],[12,22,0.,-1],[13,23,.01,1],[14,24,.02,-1],[15,25,.02,1]],np.float64)
    raw=np.stack((canonical[:,0],canonical[:,1],(canonical[:,3]>0).astype(float),canonical[:,2]))
    start,stop,prefix=match_native_packet(raw,canonical[:,2],canonical[1:])
    assert (start,stop)==(1,5) and prefix['prefix_boundary_tie_trim_count']==1
    start,stop,suffix=match_native_packet(raw,canonical[:,2],canonical[:4])
    assert (start,stop)==(0,4) and suffix['suffix_boundary_tie_trim_count']==1
    start,stop,both=match_native_packet(raw,canonical[:,2],canonical[1:4])
    assert (start,stop)==(1,4) and both['prefix_boundary_tie_trim_count']==both['suffix_boundary_tie_trim_count']==1
    broken=canonical[1:4].copy();broken[1,0]+=1
    internal_deleted=canonical[[0,1,3,4]]
    duplicate=np.repeat(canonical[:1],3,axis=0)
    duplicate_raw=np.stack((duplicate[:,0],duplicate[:,1],(duplicate[:,3]>0).astype(float),duplicate[:,2]))
    packet_rejections=0
    for ds,ts,target in [(raw,canonical[:,2],broken),(raw,canonical[:,2],internal_deleted),(duplicate_raw,duplicate[:,2],duplicate[:1])]:
        try:match_native_packet(ds,ts,target)
        except AssertionError:packet_rejections+=1
    assert packet_rejections==3,'Internal mismatch/deletion and ambiguous raw rows must reject'
    # At0.1D strict success is false; AUC is dimensionless on[0,0.1D].
    boundary=.1*.2;m=metrics([0,boundary/2,boundary,None],.2,'lt')
    assert m['success_count']==2 and abs(m['adds_auc_0_to_0_1D_normalized']-.375)<1e-14
    assert metrics([.02],.2,'le')['success_count']==1
    assert metrics([None],.2)['adds_success_0_1D']==0
    assert abs(metrics([0,.05,.1,None],.2)['adds_auc_0_to_0_1m_normalized']-.375)<1e-14
    # Threshold boundary uses exact shared float expression, not decimal alias.
    q=.1*.23737
    assert metrics([q],.23737,'lt')['success_count']==0
    assert metrics([q],.23737,'le')['success_count']==1
    result=dict(passed=True,identity=a,translation=b,independent_cloud_tree_adds=oracle,
        exact_CDF_integral_fixture=m,asymmetric_directed_distance_fixture=asym_result,
        nonrigid_GT_direct_cloud_oracle=dict(result=native_result,independent_primary=native_primary,independent_reverse=native_reverse,
            original_GT_unchanged=True,registered_native_accepted=True,unregistered_reflected_or_excess_strain_rejected=3),
        native_packet_tie_oracles=dict(prefix=prefix,suffix=suffix,both=both,internal_mismatch_deletion_ambiguity_rejected=packet_rejections),
        success_operator='lt',auc_ranges=['0 to0.1 metre conventional','0 to0.1 CAD diameter supplement'],
        official_symmetry_rotation_claimed=False,model_loaded=False,real_scores_computed=False)
    dump(out,result);print('METRIC_FIXTURE_PASSED',out)

def cad_assets(vertices_path,keys_path,physical):
    import h5py
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist
    v=np.load(vertices_path,allow_pickle=False).astype(np.float64)
    keys=np.load(keys_path,allow_pickle=False).astype(np.float64)
    assert v.ndim==2 and v.shape[1]==3 and keys.shape==(9,3)
    assert np.isfinite(v).all() and np.isfinite(keys).all()
    diameter=float(pdist(v[ConvexHull(v).vertices]).max());assert .2<diameter<.3
    with h5py.File(physical,'r') as h:
        ep=next(iter(h['episodes'].values()));actual=np.asarray(ep['cad_keypoints'])
        assert actual.shape==(9,3) and np.array_equal(actual,keys),'Stage9 keypoint ordering differs'
    return v,keys,dict(vertices=dict(path=str(vertices_path.resolve()),sha256=sha(vertices_path)),
        keypoints=dict(path=str(keys_path.resolve()),sha256=sha(keys_path)),
        synthetic_physical_sample=dict(path=str(physical.resolve()),sha256=sha(physical)),
        diameter_m=diameter,keypoint_order_matches_stage9=True,points=len(v))

def mat_sample(path,policy):
    from scipy.io import loadmat
    m=loadmat(path,variable_names=['time','intrinsicMatrix','pose','events'])
    K=np.asarray(m['intrinsicMatrix'],dtype=np.float64);T=np.asarray(m['pose'],dtype=np.float64)
    label=float(np.asarray(m['time']).squeeze());ev=np.asarray(m['events'],dtype=np.float64).reshape(-1,4)
    assert K.shape==(3,3) and T.shape==(4,4) and np.isfinite(K).all() and np.isfinite(T).all()
    assert K[0,0]>0 and K[1,1]>0 and np.allclose(T[3],[0,0,0,1],atol=1e-12,rtol=0)
    linear_audit=validate_gt_linear(T[:3,:3],policy)
    assert math.isfinite(label) and np.isfinite(ev).all()
    if len(ev):
        assert np.all(np.diff(ev[:,2])>=0) and ev[-1,2]<=label+1e-7,'Future/nonmonotone MAT packet'
        assert np.all(ev[:,:2]==np.rint(ev[:,:2]))
    return K,T,label,ev,linear_audit

def encode_rectified(x,y,p):
    import cv2
    on=np.zeros((HEIGHT,WIDTH),np.float32);off=np.zeros_like(on);total=np.zeros_like(on)
    if len(x):
        np.add.at(on,(y[p],x[p]),1.);np.add.at(off,(y[~p],x[~p]),1.);np.add.at(total,(y,x),1.)
    scale=np.float32(np.log1p(8.))
    return np.ascontiguousarray(np.stack([cv2.resize(np.minimum(np.log1p(c)/scale,1.),(NETWORK,NETWORK),interpolation=cv2.INTER_LINEAR)
                                          for c in [on,off,total]]),dtype=np.float32)

def rectification_lut(K,dist):
    import cv2
    xx,yy=np.meshgrid(np.arange(WIDTH),np.arange(HEIGHT));points=np.stack((xx,yy),axis=-1).astype(np.float64)
    coords=cv2.undistortPoints(points.reshape(-1,1,2),K,dist,P=K).reshape(HEIGHT,WIDTH,2)
    return np.rint(coords).astype(np.int64)

def frame_from_raw(ds,row,lut):
    raw=np.asarray(ds[:,row['raw_first_index']:row['raw_stop_index']],dtype=np.float64).T
    assert len(raw)==row['raw_event_count']
    xy=np.rint(raw[:,:2]).astype(np.int64)
    assert np.array_equal(xy,raw[:,:2])
    assert not len(xy) or ((xy[:,0]>=0)&(xy[:,0]<WIDTH)&(xy[:,1]>=0)&(xy[:,1]<HEIGHT)).all()
    mapped=lut[xy[:,1],xy[:,0]] if len(xy) else np.empty((0,2),np.int64)
    inside=(mapped[:,0]>=0)&(mapped[:,0]<WIDTH)&(mapped[:,1]>=0)&(mapped[:,1]<HEIGHT)
    chosen=mapped[inside];pol=raw[inside,2]>0
    return encode_rectified(chosen[:,0],chosen[:,1],pol),int(inside.sum())

def match_native_packet(ds,relative,ev):
    """Unique complete-row match; only equal-time packet boundary ties trim."""
    assert len(ev)>0
    first,last=float(ev[0,2]),float(ev[-1,2])
    outer_first=int(np.searchsorted(relative,first,'left'));outer_stop=int(np.searchsorted(relative,last,'right'))
    raw=np.asarray(ds[:,outer_first:outer_stop],dtype=np.float64).T
    canonical=np.c_[raw[:,:2],relative[outer_first:outer_stop],np.where(raw[:,2]>0,1.,-1.)]
    n=len(ev);assert len(canonical)>=n,'Native packet longer than its raw boundary envelope'
    matches=[];tested=0
    for offset in range(len(canonical)-n+1):
        stop=offset+n
        if offset and not np.all(canonical[:offset,2]==first):continue
        if stop<len(canonical) and not np.all(canonical[stop:,2]==last):continue
        # Row-content pruning is event-only and does not refer to GT geometry.
        if not np.array_equal(canonical[offset],ev[0]) or not np.array_equal(canonical[stop-1],ev[-1]):continue
        tested+=1
        if np.array_equal(canonical[offset:stop],ev):matches.append(offset)
    assert len(matches)==1,('Native packet must have one unique contiguous raw match; exact candidates',len(matches))
    offset=matches[0];left=outer_first+offset;right=left+n
    selected=canonical[offset:offset+n]
    return left,right,dict(candidate_envelope_first_index=outer_first,candidate_envelope_stop_index=outer_stop,
        candidate_envelope_events=len(canonical),prefix_boundary_tie_trim_count=offset,
        suffix_boundary_tie_trim_count=len(canonical)-offset-n,edge_content_candidates_tested=tested,
        exact_contiguous_match_count=1,boundary_timestamps_only_trimmed=True,
        native_xy_t_p_f64_sha256=hashlib.sha256(np.ascontiguousarray(ev,dtype=np.float64).tobytes()).hexdigest(),
        selected_raw_xy_t_p_f64_sha256=hashlib.sha256(np.ascontiguousarray(selected,dtype=np.float64).tobytes()).hexdigest(),
        all_native_rows_exact=True)

def prepare(args):
    import h5py,cv2
    cv2.setNumThreads(1)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    policy_path=getattr(args,'native_pose_policy',Path(__file__).with_name('native_pose_audit_policy.json'))
    policy=json.loads(policy_path.read_text())
    v,keys,assets=cad_assets(args.cad_vertices,args.cad_keypoints,args.physical_sample)
    mats={p.name:p for p in args.mat_root.iterdir() if p.is_dir() and list(p.glob('*_data.mat'))}
    raws={p.parent.name:p for p in args.raw_root.glob('*/data.h5')}
    assert mats and raws
    missing=sorted(set(mats)-set(raws));assert not missing,('Annotated sequences missing raw data',missing)
    seqs=[];preflight=[]
    for name in sorted(mats):
        files=sorted(mats[name].glob('*_data.mat'),key=lambda p:int(p.name.split('_')[0]))
        if args.interface_only:files=[files[i] for i in sorted(set([0,len(files)//2,len(files)-1]))]
        sample_file=out/(name+'_samples.jsonl');packet_count=0;event_count=0;prefix_trims=0;suffix_trims=0;trimmed_packets=0
        with h5py.File(raws[name],'r') as h,sample_file.open('x',encoding='utf-8') as output:
            ds=h['events_parallel'];assert ds.ndim==2 and ds.shape[0]==4
            times=np.asarray(ds[3,:],dtype=np.float64);assert len(times) and np.isfinite(times).all() and np.all(np.diff(times)>=0)
            origin=float(times[0]);relative=times-origin
            first_K=None;lut=None
            for f in files:
                K,T,label,ev,linear_audit=mat_sample(f,policy)
                if first_K is None:first_K=K.copy();lut=rectification_lut(K,DIST)
                assert np.array_equal(K,first_K),'Per-packet K changes; requires explicit registration'
                if len(ev):
                    first=float(ev[0,2]);last=float(ev[-1,2]);left,right,alignment=match_native_packet(ds,relative,ev)
                    assert relative[right-1]<=label+1e-7
                else:
                    first=label-.010;last=label;left=int(np.searchsorted(relative,first,'right'));right=int(np.searchsorted(relative,last,'right'))
                    assert right==left,'Empty exported packet contains raw events'
                    alignment=dict(candidate_envelope_first_index=left,candidate_envelope_stop_index=right,
                        candidate_envelope_events=0,prefix_boundary_tie_trim_count=0,suffix_boundary_tie_trim_count=0,
                        exact_contiguous_match_count=1,boundary_timestamps_only_trimmed=True,all_native_rows_exact=True)
                xyz=(T@np.c_[keys,np.ones(9)].T).T[:,:3]
                assert np.isfinite(xyz).all() and (xyz[:,2]>0).all(),'Invalid GT camera-frame CAD depth'
                q=(K@xyz.T).T;uv=q[:,:2]/q[:,2:]
                row=dict(sequence='block/'+name,index=int(f.name.split('_')[0]),mat_path=str(f.resolve()),mat_sha256=sha(f),
                    label_time_s=label,window_first_relative_s=first,window_last_relative_s=last,
                    raw_first_index=left,raw_stop_index=right,raw_event_count=right-left,
                    K=K.tolist(),target_to_camera=T.tolist(),gt_uv_ideal=uv.tolist(),
                    supplied_keypointprojections_read=False,supplied_cameraPose_read=False)
                row.update(linear_audit)
                row['native_packet_alignment']=alignment
                prefix_trims+=alignment['prefix_boundary_tie_trim_count'];suffix_trims+=alignment['suffix_boundary_tie_trim_count']
                trimmed_packets+=int(alignment['prefix_boundary_tie_trim_count']+alignment['suffix_boundary_tie_trim_count']>0)
                output.write(json.dumps(row,allow_nan=False)+'\n');packet_count+=1;event_count+=right-left
                if packet_count==1 or args.interface_only:
                    image,count=frame_from_raw(ds,row,lut)
                    preflight.append(dict(sequence='block/'+name,index=row['index'],events_raw=len(ev),events_rectified=count,
                        tensor_shape=list(image.shape),tensor_sha256=hashlib.sha256(image.tobytes()).hexdigest(),finite=bool(np.isfinite(image).all()),
                        shape_qualified=image.shape==(3,416,416),raw_mat_packet_exact=True,
                        native_packet_alignment=alignment,**linear_audit))
        # Both the JSONL output stream and input HDF5 are closed before hashing.
        row=dict(sequence='block/'+name,seen=name in SEEN,group='seen_exploratory' if name in SEEN else 'untouched_primary',
            raw_h5=str(raws[name].resolve()),raw_h5_sha256=sha(raws[name]),raw_dataset='events_parallel',
            raw_event_count=len(times),raw_absolute_origin_s=origin,MAT_directory=str(mats[name].resolve()),
            sample_manifest=str(sample_file),sample_manifest_sha256=sha(sample_file),sample_count=packet_count,
            total_overlapping_packet_events=event_count,all_mat_packets=not args.interface_only,
            boundary_tie_alignment_summary=dict(packets_with_boundary_tie_trims=trimmed_packets,prefix_trim_events=prefix_trims,suffix_trim_events=suffix_trims))
        seqs.append(row);print('PREPARED',name,packet_count,flush=True)
    assert all(x['finite'] and x['shape_qualified'] for x in preflight)
    protocol=dict(schema='stage9_epose_full_woodblock_protocol_v5_closed_manifest_hash',created_utc=utc(),
        preparation_only=True,real_candidate_predictions_computed=False,interface_only=args.interface_only,
        score_launch_enabled=False,scope='all annotated complete YCB036woodblock sequences; not allEPOSE objects',
        mat_root=str(args.mat_root.resolve()),raw_root=str(args.raw_root.resolve()),raw_unannotated_sequences=sorted(set(raws)-set(mats)),
        sequences=seqs,assets=assets,distortion_k1_k2_p1_p2_k3=DIST.tolist(),width=WIDTH,height=HEIGHT,
        native_pose_policy=dict(path=str(policy_path.resolve()),sha256=sha(policy_path),policy=policy),
        network_hw=[NETWORK,NETWORK],input_encoding='full frame ON/OFF/total; min(log1p(count)/log(9),1); independent channel linear resize416',
        event_selection='Each complete native MAT packet is matched to exactly one continuous rawH5 row interval inside its inclusive timestamp envelope; only same-first/last-time boundary rows may trim; all x/y/t/p rows exact; ambiguity or internal mismatch rejected; no GT mask/crop/box/cell; empty packet retained',
        pose_reference='Published MAT object-to-camera4x4 in metres, public CAD/K; original possibly nonorthogonal linear block preserved in primary geometry; faulty keypointprojections and cameraPose unused',
        primary='ADD-S success at0.1CADdiameter',success_operator=args.success_operator,
        adds_direction='GT model points to nearest estimated model points, PoseCNN pose_error.py adi convention',
        adds_legacy_direction='estimated to GT reported as legacy auxiliary only; oldpilot is not directly mixed with Stage9',
        AUC=dict(conventional=dict(name='ADD-S AUC0to0.1m normalized',upper_limit_m=.1,normalization_m=.1,
                 source='PoseCNN CVPR2018 Sec.V-F, arXiv1711.00199; same published range, exact empirical CDF integral'),
                 supplement=dict(name='ADD-S AUC0to0.1D normalized',upper_limit_m=.1*assets['diameter_m'],normalization_m=.1*assets['diameter_m']),
                 formula='mean(max(1-error/upper_limit,0)); invalidPnP contributes0; normalized fractions, multiply100 for percentage',
                 official_epose_metric_claimed=False),
        rotation='raw angle only when GT qualifies as strict rigid; rotation_polar_SO3_aux_deg explicitly uses nearest proper polar rotation for auxiliary only; no official/symmetry-aware rotation claim',
        reprojection='mean ideal-pinhole distance over9 CAD keypoints after estimated PnP, validPnP only; auxiliary, no symmetry claim',
        invalid_prediction='included as failure in every success/AUC denominator; valid-only pose means labeled',
        aggregation='per-sequence per-seed; group aggregate equal-sequence means; finalseed/sampleSD and pairedsequencebootstrap outside runner',
        source_code_sha256=sha(__file__))
    dump(out/'interface_preflight.json',dict(passed=True,rows=preflight,model_loaded=False,scoring_performed=False))
    fixture(out/'metric_fixture.json')
    dump(out/'protocol.json',protocol)
    dump(out/'preparation_receipt.json',dict(status='interface_only' if args.interface_only else 'complete',protocol_sha256=sha(out/'protocol.json'),
        sequences=len(seqs),samples=sum(x['sample_count'] for x in seqs),seen_sequences=[x['sequence'] for x in seqs if x['seen']],
        untouched_sequences=[x['sequence'] for x in seqs if not x['seen']],model_loaded=False,real_scores_computed=False,
        interface_preflight_sha256=sha(out/'interface_preflight.json'),metric_fixture_sha256=sha(out/'metric_fixture.json')))
    print('EPOSE_PREPARATION_COMPLETE',out,flush=True)

def decode(raw):
    import torch
    B,channels,H,W=raw.shape;assert channels==20
    idx=raw[:,18].flatten(1).argmax(1);yy=idx//W;xx=idx%W;b=torch.arange(B,device=raw.device)
    selected=raw[:,:18].reshape(B,9,2,H,W).permute(0,3,4,1,2)[b,yy,xx]
    uv=torch.cat([selected[:,:1].sigmoid(),selected[:,1:]],dim=1)
    uv=(uv+torch.stack([xx,yy],1)[:,None])/torch.tensor([W,H],device=raw.device)
    return uv.detach().cpu().numpy(),idx.detach().cpu().numpy()

def score(args):
    import h5py,cv2,torch
    from scipy.spatial import cKDTree
    prep=args.prepared.resolve();protocol=json.loads((prep/'protocol.json').read_text());receipt=json.loads((prep/'preparation_receipt.json').read_text())
    assert receipt['status']=='complete' and not protocol['interface_only']
    assert sha(prep/'protocol.json')==receipt['protocol_sha256']
    frozen=json.loads(args.frozen_checkpoints.read_text());auth=json.loads(args.authorization.read_text())
    assert auth['score_launch_enabled'] is True and auth['task']=='pose'
    assert auth['protocol_sha256']==sha(prep/'protocol.json') and auth['frozen_checkpoints_sha256']==sha(args.frozen_checkpoints)
    assert auth['evaluator_source_sha256']==sha(__file__),'Evaluator source not approved'
    assert auth['model_source_files']==model_source_manifest(args.model_root),'SingleShotPose source/cfg not frozen by authorization'
    assert frozen['task']=='pose' and len(frozen['runs'])==15
    assert {(x['method'],x['seed']) for x in frozen['runs']}=={(m,s) for m in METHODS for s in SEEDS}
    assert all(x['completed_updates']==30000 and x['real_data_used_in_training'] is False for x in frozen['runs'])
    for seed in SEEDS:assert len({x['initial_state_sha256'] for x in frozen['runs'] if x['seed']==seed})==1
    for x in frozen['runs']:
        assert sha(x['selected_checkpoint'])==x['selected_checkpoint_sha256']
        assert Path(x['selected_checkpoint']).name=='synthetic_best.pt'
        assert x['recipe_sha256']==frozen['recipe_sha256']
    assets=protocol['assets']
    assert sha(protocol['native_pose_policy']['path'])==protocol['native_pose_policy']['sha256']
    for k in ['vertices','keypoints','synthetic_physical_sample']:assert sha(assets[k]['path'])==assets[k]['sha256']
    # Immutable real inputs are hashed once before loading any candidate.
    for seq in protocol['sequences']:
        assert sha(seq['sample_manifest'])==seq['sample_manifest_sha256']
        assert sha(seq['raw_h5'])==seq['raw_h5_sha256']
    v=np.load(assets['vertices']['path'],allow_pickle=False).astype(np.float64);keys=np.load(assets['keypoints']['path'],allow_pickle=False).astype(np.float64)
    tree=cKDTree(v);limit=.1*assets['diameter_m']
    sys.path.insert(0,str(args.model_root.resolve()));from darknet import Darknet
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device(args.device)
    if device.type=='cuda':
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        if args.expected_gpu_uuid:assert os.environ.get('CUDA_VISIBLE_DEVICES')==args.expected_gpu_uuid
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    dump(out/'evaluation_registration.json',dict(protocol_sha256=sha(prep/'protocol.json'),frozen_checkpoints_sha256=sha(args.frozen_checkpoints),
        authorization_sha256=sha(args.authorization),source_sha256=sha(__file__),device=str(device),batch_size=args.batch_size,
        started_utc=utc(),no_GT_crop_mask_box_cell=True))
    sequence_summaries=[];started=time.monotonic()
    for run in frozen['runs']:
        model=Darknet(str(args.model_root/'cfg/yolo-pose.cfg')).to(device)
        checkpoint=torch.load(run['selected_checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model'],strict=True);model.eval();del checkpoint
        tag=run['method']+'_seed'+str(run['seed']);predictions=out/(tag+'_predictions.jsonl')
        with predictions.open('x',encoding='utf-8') as output:
            for seq in protocol['sequences']:
                assert sha(seq['sample_manifest'])==seq['sample_manifest_sha256']
                rows=[json.loads(line) for line in Path(seq['sample_manifest']).read_text().splitlines()]
                assert len(rows)==seq['sample_count'];metrics_rows=[]
                with h5py.File(seq['raw_h5'],'r') as h:
                    ds=h[seq['raw_dataset']];lut=rectification_lut(np.asarray(rows[0]['K']),np.asarray(protocol['distortion_k1_k2_p1_p2_k3']))
                    for start in range(0,len(rows),args.batch_size):
                        block=rows[start:start+args.batch_size];images=[];rectified=[]
                        for item in block:
                            image,count=frame_from_raw(ds,item,lut);images.append(image);rectified.append(count)
                        with torch.inference_mode():uv,cells=decode(model(torch.from_numpy(np.stack(images)).to(device)))
                        for item,norm,cell,count in zip(block,uv,cells,rectified):
                            pred=norm.astype(np.float64)*[WIDTH,HEIGHT];K=np.asarray(item['K']);T=np.asarray(item['target_to_camera']);gtuv=np.asarray(item['gt_uv_ideal'])
                            row=dict(method=run['method'],seed=run['seed'],sequence=seq['sequence'],group=seq['group'],index=item['index'],
                                raw_event_count=item['raw_event_count'],rectified_event_count=count,confidence_cell=int(cell),pnp_valid=False,
                                adds_m=None,adds_legacy_est_to_gt_m=None,add_m=None,translation_error_m=None,rotation_raw_deg=None,centroid_epe_px=None,corner_epe_px=None,
                                reprojection_keypoints_px=None,rotation_polar_SO3_aux_deg=None,
                                predicted_uv=None,adds_success_0_1D=False,invalid_reason=None)
                            row.update({k:value for k,value in item.items() if k.startswith('pose_linear_')})
                            if np.isfinite(pred).all():
                                row.update(predicted_uv=pred.tolist(),centroid_epe_px=float(np.linalg.norm(pred[0]-gtuv[0])),corner_epe_px=float(np.linalg.norm(pred[1:]-gtuv[1:],axis=1).mean()))
                                try:
                                    ok,rv,tv=cv2.solvePnP(keys[1:],pred[1:],K,None,flags=cv2.SOLVEPNP_EPNP)
                                    if ok:
                                        Rp=cv2.Rodrigues(rv)[0];tp=tv[:,0];z=(keys@Rp.T+tp)[:,2]
                                        if np.isfinite(Rp).all() and np.isfinite(tp).all() and (z>0).all():
                                            row.update(rigid_metrics(v,tree,Rp,tp,T));row['pnp_valid']=True
                                            q=(K@(keys@Rp.T+tp).T).T;projected=q[:,:2]/q[:,2:]
                                            row['reprojection_keypoints_px']=float(np.linalg.norm(projected-gtuv,axis=1).mean())
                                            row['estimated_R']=Rp.tolist();row['estimated_t_m']=tp.tolist()
                                        else:row['invalid_reason']='nonfinite_or_nonpositive_camera_depth'
                                    else:row['invalid_reason']='solvePnP_false'
                                except cv2.error as e:row['invalid_reason']='solvePnP_error:'+str(e)[:150]
                            else:row['invalid_reason']='nonfinite_model_keypoints'
                            if row['pnp_valid']:row['adds_success_0_1D']=row['adds_m']<limit if protocol['success_operator']=='lt' else row['adds_m']<=limit
                            output.write(json.dumps(row,allow_nan=False)+'\n');metrics_rows.append(row)
                s=metrics([x['adds_m'] for x in metrics_rows],assets['diameter_m'],protocol['success_operator'])
                s.update(method=run['method'],seed=run['seed'],sequence=seq['sequence'],group=seq['group'],valid_pnp_fraction=sum(x['pnp_valid'] for x in metrics_rows)/len(metrics_rows),
                    centroid_epe_mean_valid_uv_px=float(np.mean([x['centroid_epe_px'] for x in metrics_rows if x['centroid_epe_px'] is not None])) if any(x['centroid_epe_px'] is not None for x in metrics_rows) else None,
                    checkpoint_sha256=run['selected_checkpoint_sha256'])
                a=[x['corner_epe_px'] for x in metrics_rows if x['corner_epe_px'] is not None]
                s['corner_epe_mean_valid_uv_px']=float(np.mean(a)) if a else None
                for column in ['add_m','adds_legacy_est_to_gt_m','translation_error_m','rotation_raw_deg','rotation_polar_SO3_aux_deg','reprojection_keypoints_px']:
                    a=[x[column] for x in metrics_rows if x[column] is not None];s[column+'_mean_valid_pnp']=float(np.mean(a)) if a else None
                sequence_summaries.append(s);print('SCORED',tag,seq['sequence'],s['samples'],s['adds_success_0_1D'],flush=True)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    dump(out/'per_sequence_summary.json',sequence_summaries)
    with (out/'per_sequence_scores.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['task','metric','method','seed','sequence','value','group','denominator','valid_pnp'])
        for x in sequence_summaries:
            for name in ['adds_success_0_1D','adds_auc_0_to_0_1m_normalized','adds_auc_0_to_0_1D_normalized','valid_pnp_fraction',
                         'centroid_epe_mean_valid_uv_px','corner_epe_mean_valid_uv_px','reprojection_keypoints_px_mean_valid_pnp',
                         'translation_error_m_mean_valid_pnp','rotation_raw_deg_mean_valid_pnp','rotation_polar_SO3_aux_deg_mean_valid_pnp']:
                writer.writerow(['pose',name,x['method'],x['seed'],x['sequence'],x[name],x['group'],x['samples'],x['valid_errors']])
    dump(out/'completion.json',dict(status='complete',evaluated_runs=15,sequence_results=len(sequence_summaries),
        protocol_sha256=sha(prep/'protocol.json'),frozen_checkpoints_sha256=sha(args.frozen_checkpoints),elapsed_s=time.monotonic()-started,
        primary_group='untouched_primary',seen_group_separate=True,real_feedback_used_for_training_or_selection=False))
    print('STAGE9_EPOSE_COMPLETE',out,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fixture');f.add_argument('--out',type=Path,required=True)
    z=sub.add_parser('model-source');z.add_argument('--model-root',type=Path,default=DEFAULT_BASE/'external/sources/downstream/SingleShotPose')
    z.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('prepare');a.add_argument('--mat-root',type=Path,default=DEFAULT_BASE/'data/real/epose/figshare_28106717_v1/extracted/block/block')
    a.add_argument('--raw-root',type=Path,default=DEFAULT_BASE/'data/real/epose/figshare_28106717_v1/extracted/original_data/original_data/block')
    a.add_argument('--cad-vertices',type=Path,default=DEFAULT_BASE/'runs/sim2real/20261101-epose-wood-short-research-v1/cad_vertices.npy')
    a.add_argument('--cad-keypoints',type=Path,default=DEFAULT_BASE/'runs/sim2real/20261101-epose-wood-short-research-v1/cad_keypoints.npy')
    a.add_argument('--physical-sample',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--native-pose-policy',type=Path,default=Path(__file__).with_name('native_pose_audit_policy.json'))
    a.add_argument('--interface-only',action='store_true');a.add_argument('--success-operator',choices=['lt','le'],default='lt')
    s=sub.add_parser('score');s.add_argument('--prepared',type=Path,required=True);s.add_argument('--frozen-checkpoints',type=Path,required=True)
    s.add_argument('--authorization',type=Path,required=True);s.add_argument('--model-root',type=Path,default=DEFAULT_BASE/'external/sources/downstream/SingleShotPose')
    s.add_argument('--out',type=Path,required=True);s.add_argument('--device',default='cuda:0');s.add_argument('--batch-size',type=int,default=8);s.add_argument('--expected-gpu-uuid')
    args=p.parse_args()
    if args.command=='fixture':fixture(args.out)
    elif args.command=='model-source':dump(args.out,model_source_manifest(args.model_root))
    elif args.command=='prepare':prepare(args)
    else:score(args)

if __name__=='__main__':main()
