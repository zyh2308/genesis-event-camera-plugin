from pathlib import Path
import hashlib,importlib.util,json,sys
import h5py,numpy as np,torch,cv2
torch.set_num_threads(2)
R=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real');B=R/'benchmark/sim2real';E=B/'evidence/S3/20261016-stage3';S=B/'scripts';sys.path.insert(0,str(S))
x=json.loads((E/'external_artifacts_stage3.json').read_text());out={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
pose=load('review_pose',S/'run_pose_stage2_pilot.py');data=Path('/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage2/20261016-s2-pilot-v1');size=json.loads((B/'configs/stage2_pilot.json').read_text())['simulation']['target_box']['size_m'];presult=json.loads(Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage3-pose-paired-output/result.json').read_text());checks=[]
for gen in ['eventgenesis','v2e']:
    result=next(r for r in presult['results'] if r['generator']==gen)
    for split in ['train','validation']:
        rows=pose.collect_samples(gen,data/'shared/shared_trajectory.h5',data/'v2e_events',size,split,2,8.0)
        with h5py.File(data/'shared/shared_trajectory.h5','r') as bank:
            for row in rows:
                ep=bank[f"episodes/{row['episode_id']}"];i=row['interval_index'];t=ep['frame_timestamps_s'][:];ts,*_=pose.read_event_table(gen,row['episode_id'],bank,data/'v2e_events')
                expected=int(((ts>t[i])&(ts<=t[i+1])).sum());logs=result['training_detail'] if split=='train' else result['validation_detail'];rec=next(z for z in logs if z['episode_id']==row['episode_id'] and z['interval_index']==i)
                checks.append({'generator':gen,'sample_id':row['sample_id'],'window_is_10ms':bool(np.isclose(t[i+1]-t[i],0.01)),'end_equals_label_and_correct_endpoint':row['event_end_s']==row['label_time_s']==float(t[i+1]),'count_matches_independent_raw_and_log':row['event_count']==expected==rec['event_count']})
out['pose_actual_sample_checks']=checks
out['pose_BN_buffers_unchanged_by_diagnostics']={r['generator']:r['batchnorm_after_training_state']==r['batchnorm_after_validation_diagnostics_state'] for r in presult['results']}
out['pose_training_record_counts']={r['generator']:len(r['training_detail']) for r in presult['results']}

# Saved real predictions: independent full-vector EPE, crop, pairing and raw support.
realchecks=[];pair={}
mv=load('review_mv',S/'run_mvsec_candidate_inference.py')
src=Path('/home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/STE-FlowNet');sys.path.insert(0,str(src));from multiscaleloss import estimate_corresponding_gt_flow
frag=x['derived_public_data_on_g1']['mvsec_fixed_fragment'];dh=Path(frag['files'][0]['server_path']);gh=Path(frag['files'][1]['server_path'])
with h5py.File(dh,'r') as f,h5py.File(gh,'r') as g:
    raw=f['davis/left/events'][:];aps=f['davis/left/image_raw_ts'][:].reshape(-1);gt=g['davis/left/flow_dist'][:];gt_ts=g['davis/left/flow_dist_ts'][:].reshape(-1)
    for run in x['mvsec_event_only_inference_outputs']:
        manifest=json.loads((Path(run['server_output_dir'])/'manifest.json').read_text());bad=[];maxerr=0.0;rawdiff=[];inputequal=True;countmatches=True;gtmatches=True;cropmatches=True
        for row in manifest['windows']:
            p=Path(row['prediction_artifact']);dt=run['dt'];i=row['sample_index'];key=(dt,i)
            with np.load(p) as z:
                pred=z['prediction_px_per_interval'];target=z['gt_px_per_interval'];mask=z['candidate_valid_mask'].astype(bool);native=z['event_input_native_uint8'];padded=z['prediction_padded_px_per_interval']
                wanted=np.stack(estimate_corresponding_gt_flow(gt[:,0],gt[:,1],gt_ts,float(aps[i]),float(aps[i+dt])))
                gtmatches &= bool(np.allclose(target,wanted,rtol=1e-6,atol=1e-6,equal_nan=True))
                cropmatches &= bool(np.array_equal(pred,padded[:,46:46+260,3:3+346]))
                # Recompute metric, but only expose agreement, not new real scores/rankings.
                actual=float(np.hypot((pred[0]-target[0]).astype(np.float64),(pred[1]-target[1]).astype(np.float64))[mask].mean()) if mask.any() else None
                expected=row['candidate_epe_px_per_interval'];delta=abs(actual-expected) if actual is not None and expected is not None else 0.0;maxerr=max(maxerr,delta)
                rawwindow=raw[(raw[:,2]>aps[i])&(raw[:,2]<=aps[i+dt])];support=np.zeros((260,346),bool);support[rawwindow[:,1].astype(int),rawwindow[:,0].astype(int)]=True
                encoded=native.sum(axis=(0,3))>0;rawdiff.append(int((support&~encoded).sum()));countmatches &= len(rawwindow)==row['event_count']
                current=[hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest() for k in ['event_input_native_uint8','gt_px_per_interval','candidate_valid_mask']]
                if key in pair:inputequal &= current==pair[key]
                else:pair[key]=current
        realchecks.append({'generator':run['generator'],'dt':run['dt'],'windows':len(manifest['windows']),'independent_EPE_max_roundoff':maxerr,'GT_matches_pinned_official_helper':gtmatches,'crop_matches':cropmatches,'raw_count_matches':bool(countmatches),'paired_input_GT_mask_hashes_equal':bool(inputequal),'raw_support_pixels_lost_by_encoder_minmax':[min(rawdiff),max(rawdiff)]})
out['MVSEC_saved_prediction_checks']=realchecks
out['HQF_real_inference']=[]
for run in x['hqf_event_only_inference']:
    d=json.loads((Path(run['server_output_dir'])/'inference_manifest.json').read_text());seq=d['sequence'];samples=seq['samples'];arr=Path(seq['path']);ts=np.load(arr/'events_ts.npy',mmap_mode='r');its=np.load(arr/'images_ts.npy').reshape(-1);idx=np.load(arr/'image_event_indices.npy').reshape(-1)
    index_correct=bool(np.array_equal(idx,np.maximum(np.searchsorted(ts,its,side='right')-1,0)))
    pngs=d['registered_inference']['saved_images'];pngpass=all(sha(p['path'])==p['sha256'] and cv2.imread(p['path'],cv2.IMREAD_GRAYSCALE).shape==(180,240) for p in pngs)
    out['HQF_real_inference'].append({'generator':run['generator'],'samples':len(samples),'timestamps_match_all_images':all(s['target_timestamp_s']==its[i] for i,s in enumerate(samples)),'all_event_endpoints_not_after_label':all(s['event_timestamp_last_s'] is None or s['event_timestamp_last_s']<=s['target_timestamp_s'] for s in samples),'recurrent_resets':d['model']['recurrent_reset_calls'],'saved_png_count':len(pngs),'all_saved_png_hash_and_shape_pass':pngpass,'event_index_table_matches_conversion_rule':index_correct,'GT_scoring':d['model']['real_GT_scoring_performed']})
bundle=json.loads((B/'manifests/downstream_source_bundle.json').read_text());out['source_hashes']=[{'repo':p['repo'],'files':len(p['files']),'mismatches':[f['path'] for f in p['files'] if sha(Path(p['server_path'])/f['path'])!=f['sha256']]} for p in bundle['repositories']]
print(json.dumps(out,ensure_ascii=False))
