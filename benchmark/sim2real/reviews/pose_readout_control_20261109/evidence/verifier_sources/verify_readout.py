"""Independent CPU probability and CAD-geometry audit; no model/helper imports."""
import argparse,hashlib,itertools,json,math,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

EXPORT_SHA='fea5095c363b2d4a02602c4fe2dec6144058564be877c55caf0ef1dc24631093'
METHODS=('eventgenesis','v2e');LIGHTS=('normal','half','eighth');RIDS=('validation_000','validation_031','validation_063','validation_095')
METRICS=('depth_mae_m','depth_bias_m','log_size_mae','log_size_bias','center_px','success','auc')
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def checked(r):
    p=Path(r['path']);need(p.is_file() and p.stat().st_size==r['size_bytes'] and sha(p)==r['sha256'],'Pinned artifact changed: '+str(p));return p


def probability(p):
    """Independent integer-cell form of sigma=1/16; peak is prediction only."""
    p=np.array(p,dtype=np.float64,copy=True)
    need(p.shape==(16,16) and np.isfinite(p).all() and (p>=0).all() and abs(float(p.sum())-1)<=1e-5,'Invalid original probability map')
    k=int(np.argmax(p.ravel(order='C')));row,col=divmod(k,16);yy,xx=np.mgrid[:16,:16]
    gaussian=np.exp(-.5*((xx-col)**2+(yy-row)**2));q=p*gaussian
    need(q.sum()>0 and np.isfinite(q).all(),'Invalid focus normalization');q/=q.sum()
    grid=np.stack(((xx+.5)/16,(yy+.5)/16),-1).reshape(256,2)
    return q,q.ravel()@grid,p.ravel()@grid,grid[k]


def measure(case,arm,vertices,keys,tree,diameter):
    # Match native control subtraction in FP32; candidate remains FP64.
    uv=np.asarray(case[arm+'_uv'],dtype=np.float32 if arm=='control' else np.float64)
    gtuv=np.asarray(case['gt_uv'],dtype=np.float32)
    K=np.asarray(case['K'],float);T=np.asarray(case['T_gt'],float);R=np.asarray(case['R_'+arm],float);anchor=np.asarray(case['anchor'],float)
    z=np.float32(case['z_m']);size=np.float32(case['size'])
    need(K.shape==(3,3) and T.shape==(4,4) and R.shape==(3,3) and anchor.shape==(3,) and uv.shape==(2,),'Geometry shape changed')
    need(np.isfinite(np.r_[K.ravel(),T.ravel(),R.ravel(),anchor,uv,z,size]).all() and z>0 and size>0,'Invalid finite pose/depth/size')
    need(np.array_equal(anchor,keys[0]),'Anchor differs from CAD keypoints[0]')
    ray=np.linalg.solve(K,np.r_[uv*[346,260],1.]);need(abs(ray[2]-1)<1e-12,'Expected unnormalized ray with z=1')
    translation=ray*z-R@anchor
    valid=bool(((keys@R.T+translation)[:,2]>0).all())
    truth=vertices@T[:3,:3].T+T[:3,3]
    adds=float(tree.query((truth-translation)@R,workers=1)[0].mean()) if valid else None
    gt_z=(T[:3,:3]@anchor+T[:3,3])[2]
    projected=truth@K.T;uv_vertices=projected[:,:2]/projected[:,2:]/[346,260]
    gt_log_size=np.float32(np.log(np.ptp(uv_vertices,axis=0)).mean())
    log_size_error=float(np.log(size)-gt_log_size)
    return dict(depth_mae_m=float(abs(z-gt_z)),depth_bias_m=float(z-gt_z),log_size_mae=abs(log_size_error),log_size_bias=log_size_error,
                center_px=float(np.linalg.norm((uv-gtuv)*[346,260])),success=float(adds is not None and adds<.1*diameter),
                auc=max(1-adds/.1,0) if adds is not None else 0.),dict(valid=valid,adds_m=adds)


def audit(root,completion_sha):
    cp=root/'completion.json';need(sha(cp)==completion_sha,'Externally pinned full completion differs');c=read(cp)
    need(c['status']=='complete' and c['mode']=='full' and c['paired_cases']==1152 and c['score_rows']==36 and c['optimizer_updates']==0 and
         c['real_data_used'] is False and c['all_reference_checks_passed'] is True and c['all_models_unchanged'] is True,'Wrong/incomplete full diagnostic receipt')
    models=c['models'];need(len(models)==6 and {(r['method'],r['seed']) for r in models}==set(itertools.product(METHODS,range(3))) and
        all(r['unchanged'] and all(len(r[k])==64 for k in ('center_state','depth_state','rotation_state')) for r in models),'Incomplete frozen-model state receipts')
    artifacts={}
    for r in c['artifacts']:
        p=checked(r);need(p.parent.resolve()==root.resolve(),'Full output artifact outside full root');need(p.name not in artifacts,'Duplicate artifact name');artifacts[p.name]=p
    need({'paired_cases.jsonl','readout_maps.npz','readout_map_index.json','scores.json','vertices.npy','keypoints.npy'}<=artifacts.keys(),'Missing pinned audit inputs')
    export_records={r['path']:r for r in c['source_records'] if r['sha256']==EXPORT_SHA};need(len(export_records)==1,'Original export receipt pin missing/ambiguous')
    export=read(checked(next(iter(export_records.values()))));assets=export['assets'];diameter=float(assets['diameter_m']);need(diameter>0,'Invalid CAD diameter')
    vertices=np.load(artifacts['vertices.npy'],allow_pickle=False);keys=np.load(artifacts['keypoints.npy'],allow_pickle=False)
    for name,array in (('vertices',vertices),('keypoints',keys)):
        original=np.load(checked(assets[name]),allow_pickle=False);need(np.array_equal(array,original),'CAD asset values differ from pinned original export')
    need(vertices.ndim==2 and vertices.shape[1]==3 and keys.shape==(9,3) and np.isfinite(vertices).all() and np.isfinite(keys).all(),'Invalid CAD geometry')
    # Rehash model files, without loading framework state or doing inference.
    model_files={r['path']:r for r in c['source_records'] if Path(r['path']).suffix in ('.pt','.pth')}
    for r in model_files.values():checked(r)
    rows=[json.loads(line) for line in artifacts['paired_cases.jsonl'].read_text().splitlines()];index=read(artifacts['readout_map_index.json'])
    maps=np.load(artifacts['readout_maps.npz'],allow_pickle=False);p=maps['original_attention'];q=maps['focused_weights']
    need(len(rows)==len(index)==1152 and p.shape==q.shape==(1152,16,16) and p.dtype==np.float32 and q.dtype==np.float64,'Wrong fixed map inventory/dtypes')
    need(np.isfinite(q).all() and (q>=0).all(),'Invalid saved focus probabilities')
    expected=set(itertools.product(METHODS,range(3),LIGHTS,RIDS,range(4,20)));seen=set();groups=defaultdict(lambda:defaultdict(list));tree=cKDTree(vertices)
    maxerrors=dict(weights=0.,candidate_uv=0.,native_control_shadow=0.);invalid=dict(control=0,candidate=0)
    for i,(row,idx) in enumerate(zip(rows,index)):
        need(row['map_index']==idx['map_index']==i,'Map order index differs')
        need(all(row[k]==idx[k] for k in ('method','seed','condition','rid','sample_id')),'Map and case physical identity differ')
        key=(row['method'],row['seed'],row['condition'],row['rid'],int(row['sample_id'].rsplit('/',1)[1]));need(key in expected and key not in seen and row['sample_id']==f'{row["rid"]}/{key[-1]:03d}','Duplicate/unregistered physical case');seen.add(key)
        weights,uv,original,peak=probability(p[i]);errs=dict(weights=float(np.max(abs(weights-q[i]))),candidate_uv=float(np.max(abs(uv-row['candidate_uv']))),native_control_shadow=float(np.max(abs(original-row['control_uv']))))
        need(errs['weights']<=1e-12 and errs['candidate_uv']<=1e-12 and errs['native_control_shadow']<=2e-6,'Independent readout probability parity failed')
        need(np.array_equal(peak,row['peak_uv']),'Peak does not match original prediction argmax')
        for k,v in errs.items():maxerrors[k]=max(maxerrors[k],v)
        need(row['finite_prediction_control'] is True and row['finite_prediction_candidate'] is True,'Producer declared nonfinite prediction')
        for arm in ('control','candidate'):
            values,extra=measure(row,arm,vertices,keys,tree,diameter);invalid[arm]+=int(not extra['valid'])
            groups[(row['method'],row['seed'],row['condition'],arm)][row['rid']].append(values)
    need(seen==expected,'Incomplete fixed physical cohort')
    scores=read(artifacts['scores.json']);scoremap={(r['method'],r['seed'],r['condition'],r['arm']):r['score'] for r in scores}
    need(len(scores)==len(scoremap)==36 and set(scoremap)==set(groups),'Score cell inventory differs')
    recomputed=[];checks=0;metric_error={k:0. for k in METRICS}
    for key,rollouts in sorted(groups.items()):
        need(set(rollouts)==set(RIDS) and all(len(v)==16 for v in rollouts.values()),'Per-rollout inventory differs')
        by={rid:dict(zip(METRICS,np.asarray([[r[m] for m in METRICS] for r in values],float).mean(0))) for rid,values in sorted(rollouts.items())}
        summary={m:float(np.mean([v[m] for v in by.values()])) for m in METRICS};reference=scoremap[key]
        need(set(reference['by_rollout'])==set(by),'Reference rollout inventory differs')
        for scope in ('summary',*sorted(by)):
            a=summary if scope=='summary' else by[scope];b=reference['summary'] if scope=='summary' else reference['by_rollout'][scope]
            for m in METRICS:
                error=abs(a[m]-b[m]);metric_error[m]=max(metric_error[m],float(error));checks+=1
                need(a[m]==b[m] if m=='success' else np.isclose(a[m],b[m],atol=1e-8,rtol=1e-6),'Independent native metric parity failed: '+str((key,scope,m,a[m],b[m])))
        recomputed.append(dict(zip(('method','seed','condition','arm'),key),score=dict(summary=summary,by_rollout=by)))
    return dict(status='passed',schema='pose_readout_independent_cpu_audit_v1',completion_sha256=completion_sha,artifact_files_verified=len(artifacts),
        probability_cases_recomputed=1152,geometry_predictions_recomputed=2304,score_cells_recomputed=36,scalar_metric_checks=checks,success_exact=True,
        readout_max_abs_errors=maxerrors,metric_max_abs_errors=metric_error,invalid_pose_counts=invalid,CAD_diameter_m=diameter,ADD_S_threshold_m=.1*diameter,
        frozen_model_receipts_checked=6,model_source_files_sha256_verified=len(model_files),optimizer_updates=0,model_forward_calls=0,real_data_used=False,
        control_center_dtype='FP32 UV-GT subtraction, then native [346,260] scaling',candidate_center_dtype='FP64 UV minus original FP32 GT',
        state_scope='Model bytes and declared beginning/end state SHA receipts checked; framework states not reloaded by this CPU-only audit'),recomputed


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--full-root',type=Path,required=True);parser.add_argument('--completion-sha256',required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    need(not a.out.exists(),'Fresh audit output required');a.out.mkdir(parents=True);start=time.monotonic()
    try:
        result,scores=audit(a.full_root,a.completion_sha256);result.update(verifier_sha256=sha(Path(__file__)),elapsed_seconds=time.monotonic()-start)
        write(a.out/'AUDIT.json',result);write(a.out/'scores_recomputed.json',scores);print(json.dumps(result))
    except BaseException:
        write(a.out/'failure.json',dict(status='failed',error=traceback.format_exc(),model_forward_calls=0));raise
