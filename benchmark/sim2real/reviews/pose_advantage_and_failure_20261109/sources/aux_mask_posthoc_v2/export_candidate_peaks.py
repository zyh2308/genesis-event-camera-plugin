"""Zero-update RAW C1/candidate native attention export; GT regions posthoc only."""
from __future__ import annotations
import argparse,collections,hashlib,importlib.util,json,math,sys,time,traceback
from pathlib import Path
import summarize_actual_results as A


def category(endpoint,union,endpoint_cell,union_cell):
    if endpoint['available'] is not True or union['available'] is not True:return 'region_unavailable_or_empty'
    if endpoint['peak_in_region'] is True:return 'native_peak_in_visible_target'
    if endpoint_cell>0:return 'native_point_outside_endpoint_but_peak_cell_has_target_support'
    if union_cell>0:return 'no_endpoint_cell_support_but_endpoint_union_cell_support'
    return 'no_target_support_in_endpoint_or_union_peak_cell'


def decode_original_probability(probability,original_uv):
    import numpy as np
    p=np.asarray(probability);A.need(p.shape==(16,16) and p.dtype==np.float32 and np.isfinite(p).all() and (p>=0).all() and abs(float(p.sum(dtype=np.float64))-1)<=1e-5,'Invalid native FP32 probability map')
    y,x=np.mgrid[:16,:16];grid=np.stack(((x+.5)/16,(y+.5)/16),-1);error=float(np.max(abs((p.astype(np.float64)[...,None]*grid).sum((0,1))-original_uv)))
    A.need(math.isfinite(error) and error<=2e-6,'Native original UV expectation sanity failed');i=int(p.argmax());r,c=divmod(i,16)
    return dict(row=r,column=c,flat_index=i,uv=grid[r,c].tolist(),probability=float(p[r,c]),original_fp64_expectation_max_abs=error)


def verify_control_reference(uv,peak,reference):
    """Exact native grid peak and continuous UV are separate admission checks."""
    import numpy as np
    error=float(np.max(abs(np.asarray(uv)-reference['pred_uv'])));A.need(math.isfinite(error) and error<=2e-6,'Original C1 UV reference parity failed')
    np.testing.assert_array_equal(reference['center_argmax_uv'],reference['alt_uv_peak'],err_msg='Frozen reference peak aliases disagree')
    np.testing.assert_array_equal(peak['uv'],reference['center_argmax_uv'],err_msg='Original C1 native peak differs despite UV parity')
    return dict(max_abs_uv=error,native_peak_exact=True)


def checked_lineage(records):
    """Actually hash current source bytes; retain the pinned cache's records."""
    A.need(isinstance(records,list) and len(records)>0,'Prepared RAW cache lineage missing');seen={}
    for r in records:
        p=Path(r['path']);A.need(p.is_file() and p.stat().st_size==r['size_bytes'] and A.digest(p)==r['sha256'],'Prepared RAW cache input bytes changed: '+str(p))
        if r['path'] in seen:A.need(seen[r['path']]==r,'Conflicting cache lineage record')
        seen[r['path']]=r
    return list(seen.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','contract','resource-contract','v2-sources','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('study','contract','resource-contract','analysis-source-manifest'):p.add_argument('--'+name+'-sha256',required=True)
    a=p.parse_args();A.need(not a.out.exists() and a.out.parent.is_dir(),'Fresh output directory required');a.out.mkdir();started=time.monotonic()
    try:
        study=A.pinned(a.study,a.study_sha256);A.verify_common(study,'study');A.need(a.contract_sha256==A.SCIENCE and a.resource_contract_sha256==A.RESOURCE,'Unregistered contracts')
        own=Path(__file__).parent;A.need(A.digest(own/'source_manifest.json')==a.analysis_source_manifest_sha256,'Exporter source manifest differs');sources=[]
        for name,rec in json.loads((own/'source_manifest.json').read_text()).items():
            expected=rec['sha256'] if isinstance(rec,dict) else rec;A.need(Path(name).name==name and A.digest(own/name)==expected,'Exporter source changed');sources.append(A.record(own/name))
        sources.append(A.record(own/'source_manifest.json'));sys.path.insert(0,str(a.v2_sources));import run_aux_mask_center as W
        import numpy as np,torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        W.cv2.setNumThreads(1);W.P.verify_sources(a.v2_sources,A.SOURCES,sources);c=A.pinned(a.contract,a.contract_sha256);W.validate_contract(c)
        start_resource=W.G.check_resource(a.resource_contract,a.resource_contract_sha256,a.contract_sha256);D=Path(c['diagnostic_source_dir']);W.P.verify_sources(D,W.DIAGNOSTIC_SHA,sources);E=W.P.load('candidate_peak_original_context',D/'export_pose_factors.py')
        base=A.pinned(c['base_contract_path'],c['base_contract_sha256']);ctx=E.load_context(base,check_gpu=False);ctx.device=torch.device('cuda:0');W.G.check_resource(a.resource_contract,a.resource_contract_sha256,a.contract_sha256)
        for path,sha in [(a.study,a.study_sha256),(a.contract,a.contract_sha256),(a.resource_contract,a.resource_contract_sha256),(c['base_contract_path'],c['base_contract_sha256'])]:E.remember(ctx.source_records,path,sha)
        forward=A.pinned(c['forward_completion'],c['forward_completion_sha256']);E.remember(ctx.source_records,c['forward_completion'],c['forward_completion_sha256']);ref={}
        rec=next(r for r in forward['artifacts'] if Path(r['path']).name=='predictions.jsonl');E.remember(ctx.source_records,rec['path'],rec['sha256'])
        with Path(rec['path']).open() as f:
            for line in f:
                r=json.loads(line)
                if r['arm']=='C1':
                    key=(r['method'],r['seed'],r['condition'],r['sample_id']);A.need(key not in ref,'Duplicate frozen control reference');ref[key]=r
        A.need(len(ref)==1152,'Frozen reference inventory incomplete');cases=[];maps={r:[] for r in ('control','candidate')};models=[];regions={};pyramids={};expected_ids=None
        for method in A.METHODS:
            cache=next(r for r in study['input_caches'] if Path(r['path']).name==f'{method}_RAW_inputs_masks128.pt');E.remember(ctx.source_records,cache['path'],cache['sha256']);cached=torch.load(cache['path'],map_location='cpu',weights_only=False)
            A.need(cached['contract_sha256']==A.SCIENCE and cached['source_manifest_sha256']==A.SOURCES,'RAW cache provenance differs');data=cached['data']
            W.P.verify_input_lineage(data['input_lineage'])
            for original in checked_lineage(data['input_lineage']):
                prior=[r for r in ctx.source_records if r['path']==original['path']]
                A.need(not prior or prior[0]==original,'Cache/source context lineage conflicts')
                if not prior:ctx.source_records.append(original)
            for seed in A.SEEDS:
                old,_,_=E.load_branch(ctx,method,seed,'physical_supervised','center');run=next(r for r in study['records'] if (r['method'],r['seed'])==(method,seed));cp=run['final_model'];E.remember(ctx.source_records,cp['path'],cp['sha256'])
                state=torch.load(cp['path'],map_location='cpu',weights_only=True);A.need(E.state_sha(state)==run['final_state_sha256'],'Candidate model state differs from completed study');new=ctx.scale.CenterScale().to(ctx.device);new.load_state_dict(state,strict=True);new.eval();before={role:E.state_sha(model.state_dict()) for role,model in [('control',old),('candidate',new)]}
                for view,condition in (('matched_normal','normal'),('half','half'),('eighth','eighth')):
                    v=data['views'][view];rows=v['rows'];images=v['raw'].to(ctx.device);A.need(images.shape==(64,3,128,128) and images.dtype==torch.float32 and torch.isfinite(images).all(),'Wrong RAW input inventory/type')
                    ids=[r['id'] for r in rows];A.need(len(set(ids))==64 and {r['rid'] for r in rows}=={'validation_000','validation_031','validation_063','validation_095'},'Wrong fixed cohort')
                    if expected_ids is None:expected_ids=ids
                    A.need(ids==expected_ids,'Physical window order differs across methods/seeds/conditions')
                    if condition not in regions:
                        bank=ctx.D/'pose_bank' if condition=='normal' else Path(c['radiance_validation'])/condition/'pose_bank';regions[condition]=W.S.regions_for_rows(bank,rows,strict=True)
                        if condition!='normal':W.S.verify_matched_regions(regions['normal'],regions[condition])
                        pyramids[condition]={sid:{k:W.S.region_pyramid(r[k]) for k in ('endpoint','endpoint_union_approx')} for sid,r in regions[condition].items()}
                        for r in regions[condition].values():E.remember(ctx.source_records,r['source_record']['path'],r['source_record']['sha256'])
                    outputs={}
                    for role,model in [('control',old),('candidate',new)]:
                        uv=[];ps=[]
                        with torch.inference_mode():
                            for k in range(0,64,32):
                                pred=model(images[k:k+32]);uv.extend(pred[1].cpu().numpy());ps.extend(pred[0].softmax(1).reshape(-1,16,16).cpu().numpy())
                        outputs[role]=(np.asarray(uv),np.asarray(ps))
                    for i,row in enumerate(rows):
                        reference=ref[(method,seed,condition,row['id'])];control_peak=decode_original_probability(outputs['control'][1][i],outputs['control'][0][i]);control_parity=verify_control_reference(outputs['control'][0][i],control_peak,reference);error=control_parity['max_abs_uv']
                        r=dict(map_index=len(cases),method=method,seed=seed,condition=condition,rid=row['rid'],sample_id=row['id'],gt_uv=np.asarray(row['uv'][0]).tolist(),control_reference_max_abs_uv=error)
                        r['control_native_peak_reference_exact']=control_parity['native_peak_exact']
                        for role in ('control','candidate'):
                            uv=outputs[role][0][i];prob=outputs[role][1][i];peak=decode_original_probability(prob,uv);rr=regions[condition][row['id']];py=pyramids[condition][row['id']]
                            summaries={k:W.S.summarize_attention(rr[k],prob,uv,peak['uv'],pyramid=py[k]) for k in py};cell={k:float(py[k]['attention16'][peak['row'],peak['column']]) for k in py}
                            r[role]=dict(pred_uv=uv.tolist(),center_error_px=float(np.linalg.norm((uv-row['uv'][0])*[346,260])),native_peak=peak,regions=summaries,peak_cell_overlap=cell,
                                peak_category=category(summaries['endpoint'],summaries['endpoint_union_approx'],cell['endpoint'],cell['endpoint_union_approx']))
                            maps[role].append(prob)
                        cases.append(r)
                    del images
                A.need(all(E.state_sha(model.state_dict())==before[role] for role,model in [('control',old),('candidate',new)]),'Center model state changed');models.append(dict(method=method,seed=seed,control_state_sha256=before['control'],candidate_state_sha256=before['candidate'],unchanged=True));del old,new,state;torch.cuda.empty_cache()
            del data,cached
        A.need(len(cases)==1152 and len(models)==6,'Incomplete candidate/native-peak export');all_sources=sources+ctx.source_records;A.need(all(W.sha(r['path'])==r['sha256'] for r in all_sources),'Source/input/model bytes changed')
        with (a.out/'paired_center_peaks.jsonl').open('x') as f:
            for r in cases:f.write(json.dumps(r,allow_nan=False)+'\n')
        np.savez_compressed(a.out/'native_attention_maps.npz',control=np.stack(maps['control']),candidate=np.stack(maps['candidate']));W.write(a.out/'map_index.json',[{k:r[k] for k in ('map_index','method','seed','condition','rid','sample_id')} for r in cases])
        groups=collections.defaultdict(list)
        for row in cases:
            for role in ('control','candidate'):groups[(row['method'],row['seed'],row['condition'],row['rid'],role)].append(row[role])
        detail=[dict(zip(('method','seed','condition','rid','role'),key),windows=len(rows),mean_center_error_px=sum(r['center_error_px'] for r in rows)/len(rows),peak_category_counts=dict(collections.Counter(r['peak_category'] for r in rows))) for key,rows in sorted(groups.items())]
        A.need(len(detail)==144 and all(r['windows']==16 for r in detail),'Per-rid/seed inventory differs');W.write(a.out/'per_seed_rid.json',detail)
        compact=[]
        for method in A.METHODS:
            for condition in ('normal','half','eighth'):
                for role in ('control','candidate'):
                    rows=[r[role] for r in cases if (r['method'],r['condition'])==(method,condition)];compact.append(dict(method=method,condition=condition,role=role,prediction_rows=len(rows),physical_windows=64,physical_rollouts=4,model_seeds=3,mean_center_error_px=sum(r['center_error_px'] for r in rows)/len(rows),peak_category_counts=dict(collections.Counter(r['peak_category'] for r in rows))))
        W.write(a.out/'three_seed_condition.json',compact);end_resource=W.G.check_resource(a.resource_contract,a.resource_contract_sha256,a.contract_sha256)
        W.write(a.out/'completion.json',dict(schema='pose_aux_mask_candidate_peak_export_v1',status='complete',optimizer_updates=0,real_data_used=False,GT_forward_inputs=False,GT_regions_used_only_posthoc=True,
            paired_prediction_cases=1152,model_forward_prediction_rows=2304,physical_windows=64,physical_rollouts=4,methods=list(A.METHODS),seeds=list(A.SEEDS),source_records=all_sources,models=models,
            resource_start=start_resource,resource_end=end_resource,scientific_contract_sha256=A.SCIENCE,training_source_manifest_sha256=A.SOURCES,resource_contract_sha256=A.RESOURCE,actual_study_sha256=a.study_sha256,
            no_decode_or_model_change=True,all_control_native_peaks_exact=True,cache_input_lineage_rehashed_at_admission_and_completion=True,formal30_and8='not_retested_not_passed_by_this_probe',artifacts=[A.record(q) for q in a.out.iterdir() if q.is_file()],elapsed_seconds=time.monotonic()-started,
            limitations=['Endpoint mask is exact at its frame, not per-event origin. Endpoint union remains an approximation.','Peak cell overlap is a fractional spatial proxy, not feature provenance.','A reduced wrong-peak count does not by itself establish full-pose or real-world superiority.']))
    except BaseException:
        (a.out/'failure.json').write_text(json.dumps(dict(status='failed',optimizer_updates=0,error=traceback.format_exc()),indent=2));raise


if __name__=='__main__':main()
