"""Read-only CPU audit against the exact server data/source used by Stage 4.

This is an independent diagnostic, not a production launcher or repair.
Exit status is managed by the SSH wrapper after JSON has been preserved.
"""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib.util,json,math,sys,time,warnings
import cv2,h5py,numpy as np,torch
torch.set_num_threads(2)
R=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real')
B=R/'benchmark/sim2real/scripts'
P=Path('/home/zhaoyuhan/Eventbased_WAM')
M=P/'external/sources/downstream/STE-FlowNet'
G=P/'runtime/sim2real/genesis_hdr_v2/genesis'
BANK=P/'data/sim2real/stage4/training_bank_v1_torch_cuda_32x2s'
DATA=P/'data/sim2real/stage4/20261016-stage4-flowdt1-seed0-v4/steflow_30ms'
sys.path[:0]=[str(R),str(B)]
from genesis_event_plugin.physics_warp import project_endpoint_to_alpha
import train_steflow_stage4 as training
import run_steflow_stage2_pilot as pilot
from prepare_steflow_stage0 import gaussian_count_representation
from stage4_v2e_artifacts import locate_shard_artifacts
t0=time.monotonic()
out={'checked_at_UTC':datetime.now(timezone.utc).isoformat(),'torch_version':torch.__version__,'audit_device':'CPU','source_hashes':{},'checks':{}}
sources=[B/n for n in ['generate_stage4_flow_bank.py','prepare_stage4_steflow_bank.py','prepare_steflow_stage0.py','train_steflow_stage4.py','run_steflow_stage2_pilot.py']]+[R/'genesis_event_plugin'/n for n in ['physics_plugin.py','physics_interpolator.py','torch_physics.py','physics_warp.py']]+[M/'multiscaleloss.py',M/'models/STEFlow.py',M/'models/corr.py',G/'vis/rasterizer_context.py',G/'vis/camera.py']
for p in sources:out['source_hashes'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()

# Execute the actual renderer's small ID-mapping class, without initializing a renderer.
tree=ast.parse((G/'vis/rasterizer_context.py').read_text())
node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SegmentationColorMap')
ns={'np':np,'torch':torch}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-renderer-mapping>','exec'),ns)
cmap=ns['SegmentationColorMap']()
nonconsecutive={k:cmap.seg_key_to_idxc(k) for k in [17,5,42]}
out['checks']['renderer_mapping']={'background_link':cmap.idxc_map[0],'physical_to_render_nonconsecutive':nonconsecutive,'render_to_physical':cmap.idxc_map,'constant_minus_one_is_general_solution':False}

# Pure analytic geometry: independent closed-form expectations, asymmetric u/v.
def analytic_check(camera_dx,object_dx,object_dy):
    H,W=7,9;K=np.array([[80.,0,4],[0,70.,3],[0,0,1]])
    depth=np.full((H,W),2.0);seg=np.full((H,W),7,np.int32)
    c0=np.eye(4);c1=np.eye(4);c1[0,3]=camera_dx
    o0=np.eye(4);o1=np.eye(4);o1[:2,3]=[object_dx,object_dy]
    uv,_,valid=project_endpoint_to_alpha(depth,seg,K,c0,c1,1.,source_index=0,object_world0={7:o0},object_world1={7:o1})
    yy,xx=np.mgrid[:H,:W];got=np.stack([uv[0]-xx,uv[1]-yy]);expected=np.array([80*(object_dx-camera_dx)/2,70*object_dy/2])[:,None,None]
    return {'expected_uv':expected[:,0,0].tolist(),'max_error':float(np.abs(got-expected).max()),'all_geometry_valid':bool(valid.all())}
out['checks']['analytic_projection']=[analytic_check(0,0,0),analytic_check(.025,0,0),analytic_check(0,.05,-.02)]

# Same source functions on CPU. Only remove cuda() calls; retain numerical expressions.
class CPU(ast.NodeTransformer):
    def visit_Call(self,n):
        n=self.generic_visit(n)
        return n.func.value if isinstance(n.func,ast.Attribute) and n.func.attr=='cuda' else n
class Consistent(ast.NodeTransformer):
    def visit_Call(self,n):
        n=self.generic_visit(n)
        if isinstance(n.func,ast.Attribute) and n.func.attr=='grid_sample':n.keywords.append(ast.keyword(arg='align_corners',value=ast.Constant(True)))
        return n
def loss_fn(consistent):
    f=next(n for n in ast.parse((M/'multiscaleloss.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='warp')
    f=CPU().visit(copy.deepcopy(f));f=Consistent().visit(f) if consistent else f
    ns={'torch':torch,'nn':torch.nn}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),'<actual-loss-warp>','exec'),ns)
    return ns['warp']
def model_fn(consistent):
    c=next(n for n in ast.parse((M/'models/STEFlow.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='FlowNetS_spike')
    c.body=[x for x in c.body if isinstance(x,ast.FunctionDef) and x.name in ['flow_warp','mesh_grid','norm_grid']]
    c.bases=[];c=Consistent().visit(c) if consistent else c
    ns={'torch':torch,'nn':torch.nn}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[c],type_ignores=[])),'<actual-model-warp>','exec'),ns)
    return ns['FlowNetS_spike']().flow_warp
torch.manual_seed(314)
x=torch.rand(1,1,32,40);zero=torch.zeros(1,2,32,40)
out['checks']['warp']=[]
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for kind,factory in [('loss',loss_fn),('model',model_fn)]:
        for consistent in [False,True]:
            fn=factory(consistent);row={'kind':kind,'consistent_control':consistent,'tests':[]}
            for dx,dy in [(0,0),(1,0),(0,-2),(2,1)]:
                flow=zero.clone();flow[:,0]=dx;flow[:,1]=dy;flow.requires_grad_()
                target=torch.roll(x,shifts=(dy,dx),dims=(-2,-1));pred=fn(target,flow);err=(pred-x)[...,4:-4,4:-4]
                err.abs().mean().backward()
                row['tests'].append({'u':dx,'v':dy,'interior_MAE':float(err.abs().mean()),'gradient_finite':bool(torch.isfinite(flow.grad).all())})
            out['checks']['warp'].append(row)

# Metric must preserve both u and v and known units, without model inference.
truth=np.zeros((2,5,7),np.float32);truth[0]=3;truth[1]=4;valid=np.ones((5,7),np.uint8)
out['checks']['metric_controls']={
 'identical_prediction_epe':pilot.compute_epe(truth.copy(),truth,valid,prediction_unit='px_per_interval',gt_unit='px_per_interval')[0],
 'zero_prediction_epe_expected_5':pilot.compute_epe(np.zeros_like(truth),truth,valid,prediction_unit='px_per_interval',gt_unit='px_per_interval')[0],
 'v_only_error_expected_4':pilot.compute_epe(np.stack([truth[0],np.zeros_like(truth[1])]),truth,valid,prediction_unit='px_per_interval',gt_unit='px_per_interval')[0]}
test=torch.arange(2*352*352,dtype=torch.float32).reshape(1,2,352,352)
out['checks']['output_conversion_identity']=bool(np.array_equal(pilot.model_output_to_flow_hw(test),test.numpy()[0]))

# Check the real augmentation transform using coordinate-valued images/vectors.
yy,xx=np.mgrid[:260,:346].astype(np.float32)
sample={'events':np.zeros((1,2,260,346,5),np.float32),'gray_prev':xx[None]/346,'gray_next':yy[None]/260,'flow_gt':np.stack([np.full_like(xx,3),np.full_like(xx,-2)])[None],'flow_valid':np.ones((1,260,346),np.uint8)}
augment=[]
for seed in range(12):
    grid,inside,A=training.transform_matrix_and_grid(260,346,np.random.default_rng(seed))
    event,gray,flow,mask=training.apply_joint_transform(sample,np.random.default_rng(seed),torch.device('cpu'))
    source_x=(grid[0,:,:,0]+1)*346/2-.5;source_y=(grid[0,:,:,1]+1)*260/2-.5
    m=inside[0,0]&(source_x>1)&(source_x<344)&(source_y>1)&(source_y<258)
    expected=A@torch.tensor([3.,-2.]);e=(flow[0,:,m]-expected[:,None]).abs()
    augment.append({'seed':seed,'valid_pixels':int(m.sum()),'coordinate_image_max_error':float(torch.max(torch.stack([(gray[0,0,m]-source_x[m]/346).abs(),(gray[0,1,m]-source_y[m]/260).abs()]))),'vector_max_error':float(e.max())})
out['checks']['augmentation_coordinate_controls']=augment

# Dataset-wide semantics on 3 endpoints per rollout, all strata/train/validation.
mapping_checks=[];label_adapter=[];initialization=[];selected_rep=[]
generator_src=(B/'generate_stage4_flow_bank.py').read_text();tree=ast.parse(generator_src)
trajectory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='trajectory_pose');ns={'np':np,'math':math}
exec(compile(ast.Module(body=[trajectory],type_ignores=[]),'<actual-trajectory>','exec'),ns)
registered=json.loads((BANK/'registered_split.json').read_text());episodes={e['id']:e for e in registered}
manifest=json.loads((DATA/'training_bank_manifest.json').read_text())
v2e_root=Path(manifest['raw_v2e_manifest']).parent
with h5py.File(BANK/'shared_trajectory.h5','r') as h:
    for eid in sorted(h['episodes']):
        e=h[f'episodes/{eid}'];ids=e['object_segmentation_link_ids'][:];K=e['camera_intrinsics_K_px'][:];cams=e['camera_to_world_cv'][:];poses=e['object_to_world'][:];times=e['frame_timestamps_s'][:]
        raw0=e['segmentation_link_idx'][0];seen=[]
        for i in [0,99,195]:
            raw=e['segmentation_link_idx'][i];depth=e['depth_m'][i];y,x=np.mgrid[:raw.shape[0],:raw.shape[1]]
            cam_points=np.stack([(x-K[0,2])*depth/K[0,0],(y-K[1,2])*depth/K[1,1],depth,np.ones_like(depth)])
            world=(cams[i]@cam_points.reshape(4,-1)).reshape(4,*raw.shape)
            ground=(raw==1)&np.isfinite(depth)&(depth>0)
            uv,_,good=project_endpoint_to_alpha(depth,raw,K,cams[i],cams[i+3],1.,source_index=0,object_world0={int(k):poses[i,j] for j,k in enumerate(ids)},object_world1={int(k):poses[i+3,j] for j,k in enumerate(ids)})
            old=e['flow_forward_30ms_px'][i//3];valid_old=e['flow_valid_30ms'][i//3].astype(bool);computed=np.stack([uv[0]-x,uv[1]-y])
            corrected=np.where(raw>0,raw-1,-1)
            uv_c,_,_=project_endpoint_to_alpha(depth,corrected,K,cams[i],cams[i+3],1.,source_index=0,object_world0={int(k):poses[i,j] for j,k in enumerate(ids)},object_world1={int(k):poses[i+3,j] for j,k in enumerate(ids)})
            gc=(ground&valid_old);motion=np.hypot(old[0],old[1]);cmotion=np.hypot(uv_c[0]-x,uv_c[1]-y)
            seen.append({'endpoint':i,'camera_delta':float(abs(cams[i+3]-cams[i]).max()),'plane_pose_delta':float(abs(poses[i+3,0]-poses[i,0]).max()),'raw_seg_ids':np.unique(raw).tolist(),'plane_world_height_abs_p99_m':float(np.quantile(abs(world[2,ground]),.99)),'old_GT_reproduced_max_error_valid':float(abs(old[:,valid_old]-computed[:,valid_old]).max()),'old_plane_mean_magnitude':float(motion[gc].mean()) if gc.any() else None,'mapped_plane_mean_magnitude':float(cmotion[gc].mean()) if gc.any() else None})
        mapping_checks.append({'rollout':eid,'split':str(e.attrs['split']),'pose_link_ids':ids.tolist(),'checks':seen})
        ep=episodes[eid]
        for oi,link in enumerate(ids[1:]):
            base=poses[0,oi+1,:3,3];expected=ns['trajectory_pose'](0.,ep,oi,base)
            initialization.append({'rollout':eid,'object_link':int(link),'first_path_offset_m':float(np.linalg.norm(expected-base)),'observed_first_10ms_delta_m':float(np.linalg.norm(poses[1,oi+1,:3,3]-base)),'observed_next_10ms_delta_m':float(np.linalg.norm(poses[2,oi+1,:3,3]-poses[1,oi+1,:3,3]))})
        # All 64 adapter files: no representation decompression except selected cases.
        for gen in ['eventgenesis','v2e']:
            path=DATA/gen/str(e.attrs['split'])/(eid+'.npz')
            with np.load(path,allow_pickle=False) as z:
                flow=z['flow_gt'];mask=z['flow_valid'];left=z['event_start_s'];right=z['event_end_s'];g0=z['gray_prev'];g1=z['gray_next'];idx=np.arange(0,198,3)
                label_adapter.append({'rollout':eid,'generator':gen,'label_equal_shared':bool(np.array_equal(flow,e['flow_forward_30ms_px'][:])),'mask_equal_shared':bool(np.array_equal(mask,e['flow_valid_30ms'][:])),'windows_match_shared':bool(np.array_equal(left,times[idx]) and np.array_equal(right,times[idx+3])),'gray_endpoints_equal_shared':bool(np.array_equal(g0,e['v2e_display_luma_u8'][:][idx].astype(np.float32)/255) and np.array_equal(g1,e['v2e_display_luma_u8'][:][idx+3].astype(np.float32)/255)),'counts_nonnegative':bool((z['event_counts']>=0).all())})
                if eid in ['s0_00','s1_06','s2_00','s3_06']:
                    if gen=='eventgenesis':raw=e['eventgenesis_events_txyp'][:]
                    else:
                        p,_=locate_shard_artifacts(v2e_root,eid)
                        with h5py.File(p,'r') as vh:r=vh['events'][:]
                        raw=np.stack([r[:,0].astype(float)*1e-6,r[:,1],r[:,2],np.where(r[:,3]>0,1,-1)],axis=1)
                    rep=z['events'];counts=z['event_counts'];rows=[]
                    for n in [0,33,65]:
                        a,b=np.searchsorted(raw[:,0],[left[n],right[n]],side='right');events=raw[a:b].copy();events[:,0]-=left[n]
                        expected=gaussian_count_representation(events,260,346)
                        rows.append({'sample_index':n,'timestamp_count_matches':int(b-a)==int(counts[n]),'representation_exact_rebuild':bool(np.array_equal(expected,rep[n])),'no_future_events':bool(not len(events) or events[:,0].max()<=right[n]-left[n]+1e-12)})
                    selected_rep.append({'rollout':eid,'generator':gen,'checks':rows})
out['checks']['dataset_semantics']=mapping_checks
out['checks']['all_adapter_labels_timestamps_images']=label_adapter
out['checks']['trajectory_initialization']=initialization
out['checks']['selected_representation_rebuild']=selected_rep
out['runtime_seconds']=time.monotonic()-t0
print(json.dumps(out,ensure_ascii=False))
