"""Read-only CPU joint export of frozen support/attention/scale diagnostics."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,itertools,json,math,os,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(_name,'2')
import numpy as np
METRICS_SHA='cc71eb2585cb3a2cdd2162cd646854a488bde22a19d9e9ce7b38561504b9237f'
REGIONS=('endpoint','endpoint_union_approx')
PAIR_KEYS=('method','seed','arm','branch','condition','rid','sample_id')
SHARED=('motion','target_identity','geometric_sanity','nominal_GT_heat_reference','visible_geometry')
SCALE_FIELDS={'log_size':'signed_log_size_error','log_shape':'signed_log_shape_error','log_depth':'signed_log_depth_error','depth_m':'signed_depth_error_m'}

def need(x,message):
    if not x:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def record(p):
    p=Path(p);return dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size)
def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,allow_nan=False,separators=(',',':'))
def jsonl(p):
    with Path(p).open() as f:return [json.loads(line) for line in f]
def dump_lines(p,rows):
    with Path(p).open('x') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n')
def csv_rows(p,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def mean(v):
    a=[float(x) for x in v if x is not None];need(all(math.isfinite(x) for x in a),'Nonfinite descriptive metric');return float(np.mean(a)) if a else None
def moments(v):
    a=np.asarray(v,float);need(len(a)>0 and np.isfinite(a).all(),'Finite fixed-window moments required')
    return dict(bias=float(a.mean()),mae=float(np.abs(a).mean()),scatter=float(a.std(ddof=0)),rmse=float(np.sqrt((a*a).mean())))
def compensation(size,shape,logz):
    s,h,z=map(lambda a:np.asarray(a,float),(size,shape,logz));res=float(np.mean(z*z)-np.mean(s*s+h*h-2*s*h))
    need(np.isfinite([*s,*h,*z,res]).all() and abs(res)<1e-10,'Scale-component identity broken')
    return dict(size_shape_product_mean=float(np.mean(s*h)),centered_covariance=float(np.mean((s-s.mean())*(h-h.mean()))),
        correlation=float(np.corrcoef(s,h)[0,1]) if s.std()>0 and h.std()>0 else None,logz_square_mean=float(np.mean(z*z)),square_identity_residual=res,
        interpretation='descriptive algebraic cancellation; not causal attribution')
def null_counts(rows):
    counts=Counter()
    def visit(v,key):
        if v is None:counts[key]+=1
        elif isinstance(v,dict):
            for k,x in v.items():visit(x,key+'/'+k)
        elif isinstance(v,list):
            for x in v:visit(x,key+'[]')
    for r in rows:visit(r,'')
    return dict(sorted(counts.items()))
def scalar(row,region):
    a=row['attention'][region];s=row['scale'];out=dict(center_epe_px=a['center_error_px'],fg_mass=a['region_attention_mass'],
        enrichment=a['region_mass_enrichment'],peak_hit=a['peak_in_region'],uv_hit=a['prediction_in_region'])
    for name,field in [('log_size_error','signed_log_size_error'),('log_shape_error','signed_log_shape_error'),('log_depth_error','signed_log_depth_error')]:out[name]=None if s is None else s[field]
    out.update(depth_error_cm=None if s is None else 100*s['signed_depth_error_m'],abs_log_size_error=None if s is None else abs(s['signed_log_size_error']),abs_depth_error_cm=None if s is None else 100*abs(s['signed_depth_error_m']))
    return out

def matched_pairs(rows,support):
    need(len(rows)==9216 and len(support)==384,'Full frozen matrix required');supports={};pairs=defaultdict(dict);shared={};cases=defaultdict(set)
    for s in support:
        key=(s['method'],s['condition'],s['sample_id']);need(key not in supports and s['status']=='complete','Duplicate/failed unique support');supports[key]=s
    for r in rows:
        key=tuple(r[k] for k in PAIR_KEYS);need(r['input_view'] in ('raw','baf') and r['input_view'] not in pairs[key],'Duplicate/unknown view');pairs[key][r['input_view']]=r
        sk=(r['method'],r['condition'],r['sample_id']);source=supports[sk];need(r['rid']==source['rid'] and r['sample_id'].split('/')[0]==r['rid'],'Physical ID/rid mismatch');need(r['support']=={k:v for k,v in source.items() if k not in ('method','condition')},'Embedded support differs from unique table')
        common={k:r[k] for k in SHARED}
        if sk in shared:need(shared[sk]==common,'Repeated physical geometry/reference differs across model/view')
        else:shared[sk]=common
        cases[(r['method'],r['seed'],r['arm'],r['branch'],r['condition'],r['input_view'])].add((r['rid'],r['sample_id']))
    expected=set(itertools.product(('eventgenesis','v2e'),(0,1,2),('C0','C1'),('center','depth'),('normal','half','eighth'),('raw','baf')))
    need(set(cases)==expected and len(pairs)==4608 and all(set(v)=={'raw','baf'} for v in pairs.values()),'Missing paired model cell')
    ids=next(iter(cases.values()));need(len(ids)==64 and len({k[0] for k in ids})==4 and all(v==ids for v in cases.values()),'Physical cohort changed')
    need(all({int(s.rsplit('/',1)[1]) for rr,s in ids if rr==rid}==set(range(4,20)) for rid in {r for r,s in ids}),'Original16 windows per rollout required')
    groups=defaultdict(list)
    for key,views in pairs.items():
        for region in REGIONS:groups[key[:-1]+(region,)].append((scalar(views['raw'],region),scalar(views['baf'],region)))
    out=[]
    for key,items in sorted(groups.items()):
        stats={};raw={};baf={};valid={};mismatch={}
        for metric in items[0][0]:
            values=[(a[metric],b[metric]) for a,b in items if a[metric] is not None and b[metric] is not None]
            raw[metric]=mean(a for a,b in values);baf[metric]=mean(b for a,b in values);stats[metric]=mean(float(b)-float(a) for a,b in values);valid[metric]=len(values)
            mismatch[metric]=sum((a[metric] is None)!=(b[metric] is None) for a,b in items)
        out.append(dict(zip(PAIR_KEYS[:-1]+('region',),key),windows=len(items),raw=raw,baf=baf,delta=stats,paired_available=valid,availability_mismatch=mismatch))
    need(len(out)==576 and all(r['windows']==16 for r in out),'Matched per-seed/rid inventory changed');return out,supports,shared

def support_groups(supports,shared):
    groups=defaultdict(list)
    for key,s in supports.items():
        for region in REGIONS:groups[(s['method'],s['condition'],s['rid'],region)].append((s['regions'][region],shared[key]['motion']))
    out={}
    for key,items in sorted(groups.items()):
        row=dict(zip(('method','condition','rid','region'),key),unique_support_windows=len(items))
        for part in ('foreground','background_region'):
            values=[s[part] for s,m in items];n=sum(v['raw_events'] for v in values);k=sum(v['baf_events'] for v in values)
            row[part]=dict(raw_events=n,baf_events=k,event_retention_ratio=k/n if n else None,
                region_nonempty_windows=sum(v['region_pixels']>0 for v in values),raw_nonempty_windows=sum(v['raw_events']>0 for v in values),occupancy_available_windows=sum(v['raw_occupancy_ratio'] is not None for v in values),
                **{name:mean(v[name] for v in values) for name in ('raw_occupancy_ratio','baf_occupancy_ratio','raw_same_window_spatially_isolated_event_fraction','raw_prior_neighbour8_within2ms_fraction')})
        row['corner_motion_px_mean']=mean(m['max_projected_corner_displacement_px'] for s,m in items);row['corner_motion_px_max']=max(m['max_projected_corner_displacement_px'] for s,m in items)
        row['anchor_motion_px_mean']=mean(m['projected_anchor_displacement_px'] for s,m in items);out[key]=row
    need(len(out)==48 and all(r['unique_support_windows']==16 for r in out.values()),'Support grouping changed');return out

def scale_tables(rows,hybrids):
    within=defaultdict(list);cross=defaultdict(dict);keys=('method','seed','arm','input_view','condition','rid')
    for r in rows:
        if r['branch']=='depth':within[tuple(r[k] for k in keys)].append(r);cross[(r['method'],r['seed'],r['arm'],r['input_view'],r['rid'],r['sample_id'])][r['condition']]=r
    table=[]
    for key,items in sorted(within.items()):
        values={n:[r['scale'][field] for r in items] for n,field in SCALE_FIELDS.items()}
        table.append(dict(zip(keys,key),windows=len(items),moments={n:moments(v) for n,v in values.items()},compensation=compensation(values['log_size'],values['log_shape'],values['log_depth'])))
    drift=defaultdict(list)
    for key,lights in cross.items():
        need(set(lights)=={'normal','half','eighth'},'Incomplete matched illumination')
        for light in ('half','eighth'):
            a,b=lights['normal'],lights[light];need(a['labels']==b['labels'],'GT scale geometry changes with illumination')
            drift[key[:-1]+(light,)].append({n:b['scale'][field]-a['scale'][field] for n,field in SCALE_FIELDS.items()})
    drifts=[]
    for key,items in sorted(drift.items()):
        v={n:[r[n] for r in items] for n in items[0]};drifts.append(dict(zip(('method','seed','arm','input_view','rid','paired_light'),key),windows=len(items),reference='normal',moments={n:moments(x) for n,x in v.items()},compensation=compensation(v['log_size'],v['log_shape'],v['log_depth'])))
    hg=defaultdict(list)
    seen=set();physical={(r['rid'],r['sample_id']) for r in rows}
    for r in hybrids:
        need(r['paired_light'] in ('half','eighth') and r['input_view']=='raw' and r['diagnostic_only'] is True and r['deployable'] is False and r['context_source']=='same_as_features','Hybrid semantics changed')
        key=tuple(r[k] for k in ('method','seed','arm','paired_light','feature_source','attention_source','rid'));identity=key+(r['sample_id'],)
        need(identity not in seen and (r['rid'],r['sample_id']) in physical and r['feature_source'] in ('normal',r['paired_light']) and r['attention_source'] in ('normal',r['paired_light']),'Hybrid duplicate/cohort/cell mismatch');seen.add(identity);hg[key].append(r)
    ht=[]
    for key,items in sorted(hg.items()):
        need(len(items)==16,'Hybrid cell must retain all16 windows')
        s=[r['log_size']-r['gt_log_size'] for r in items];h=[r['log_shape']-r['gt_log_shape'] for r in items];z=[b-a for a,b in zip(s,h)];d=[r['signed_depth_error_m'] for r in items]
        row=dict(zip(('method','seed','arm','paired_light','feature_source','attention_source','rid'),key),windows=len(items),diagnostic_only=True,deployable=False)
        row.update({prefix+'_'+n:v for prefix,x in (('log_size',s),('log_shape',h),('log_depth',z),('depth_m',d)) for n,v in moments(x).items()});row.update(compensation(s,h,z));row.pop('interpretation');ht.append(row)
    need((len(table),len(drifts),len(ht))==(288,192,384),'Scale/hybrid group inventory changed');return table,drifts,ht

def seed_ranges(paired,support):
    groups=defaultdict(list)
    for r in paired:groups[tuple(r[k] for k in ('method','arm','branch','condition','rid','region'))].append(r)
    rows=[]
    for key,cells in sorted(groups.items()):
        need(sorted(r['seed'] for r in cells)==[0,1,2],'Three seeds required');m,arm,branch,light,rid,region=key;s=support[(m,light,rid,region)]
        row=dict(zip(('method','arm','branch','condition','rid','region'),key),windows_per_seed=16,model_seeds='0|1|2',support_unique_key='|'.join((m,light,rid,region)))
        for prefix,names in (('raw',('center_epe_px','fg_mass','peak_hit','uv_hit')),('baf',('center_epe_px','fg_mass','peak_hit','uv_hit')),('delta',('center_epe_px','fg_mass','enrichment','abs_log_size_error','abs_depth_error_cm'))):
            for name in names:
                values=[r[prefix][name] for r in cells if r[prefix][name] is not None];label=prefix+'_'+name;row[label+'_seed_mean']=mean(values)
                if prefix=='delta':row[label+'_seed_min']=min(values) if values else None;row[label+'_seed_max']=max(values) if values else None;row[label+'_available_seeds']=len(values)
        for part,short in (('foreground','FG'),('background_region','BG')):
            row.update({short+'_'+name:s[part][name] for name in ('raw_events','baf_events','event_retention_ratio','raw_occupancy_ratio','baf_occupancy_ratio')})
        row['max_corner_motion_px']=s['corner_motion_px_max'];rows.append(row)
    need(len(rows)==192,'Compact CSV matrix changed');return rows

def close_json(a,b):
    if isinstance(a,dict):need(a.keys()==b.keys(),'Producer summary keys differ');[close_json(a[k],b[k]) for k in a]
    elif isinstance(a,list):need(len(a)==len(b),'Producer summary length differs');[close_json(x,y) for x,y in zip(a,b)]
    elif isinstance(a,(int,float)) and not isinstance(a,bool):need(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),'Producer summary value differs')
    else:need(a==b,'Producer summary identity differs')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--completion-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    need(a.out.is_absolute() and a.out.parent.is_dir() and not a.out.exists(),'Unique absolute output required');a.out.mkdir();start=time.monotonic()
    try:
        cp=a.input_root/'completion.json';need(sha(cp)==a.completion_sha256,'Completion digest differs');c=read(cp)
        need(c['schema']=='pose_support_attention_scale_frozen_v1' and c['status']=='complete' and c['mode']=='full','Full frozen diagnostic incomplete')
        need(c['optimizer_updates']==0 and c['real_data_used'] is False and c['GT_modified_forward'] is False and c['raw_parity_passed'] and c['raw_parity_records']==4608,'Isolation/parity contract failed')
        need(c['GT_used_only_posthoc'] is True and c['filters_adopted'] is False and c['hybrids_deployable'] is False and c['attention_RF_interpretation'] is False and c['clean384_mixed'] is False,'Diagnostic-only semantics changed')
        need((c['diagnostic_rows'],c['raw_hybrid_rows'],c['support_physical_method_light_rows'])==(9216,6144,384) and len(c['model_records'])==24 and all(r['unchanged'] for r in c['model_records']),'Producer inventory/state changed')
        inputs=[record(cp)];paths={}
        for r in c['artifacts']:
            q=Path(r['path']);need(q.parent.resolve()==a.input_root.resolve() and sha(q)==r['sha256'] and q.stat().st_size==r['size_bytes'],'Raw artifact digest/scope differs');inputs.append(r);paths[q.name]=q
        ms=next(r for r in c['source_records'] if Path(r['path']).name=='attention_scale_metrics.py' and r['sha256']==METRICS_SHA);need(sha(ms['path'])==METRICS_SHA,'Original metric source changed')
        spec=importlib.util.spec_from_file_location('joint_original_metrics',ms['path']);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
        rows=jsonl(paths['per_window.jsonl']);support=jsonl(paths['event_support.jsonl']);hybrids=jsonl(paths['raw_feature_attention_hybrids.jsonl'])
        need(len(hybrids)==6144,'Hybrid raw inventory changed');paired,supports,shared=matched_pairs(rows,support);sg=support_groups(supports,shared);within,drift,ht=scale_tables(rows,hybrids)
        original=read(paths['raw_hybrid_summaries.json']);recomputed=M.hybrid_summaries(hybrids);close_json(recomputed,original);need(len(original['three_seed_means'])==32,'All32 original hybrid means required')
        csv_rows(a.out/'per_rid_three_seed_ranges.csv',seed_ranges(paired,sg));csv_rows(a.out/'all_hybrid_cells_by_rid_seed.csv',ht)
        sentinels=[];shared_sentinels={}
        for r in rows:
            if int(r['sample_id'].rsplit('/',1)[1])==12:
                sk=(r['method'],r['condition'],r['sample_id']);name='|'.join(sk);sentinels.append({k:v for k,v in r.items() if k not in (*SHARED,'support')}|dict(shared_key=name));shared_sentinels[name]=dict(shared[sk],support=supports[sk])
        need(len(sentinels)==576 and len(shared_sentinels)==24,'Sentinel inventory changed');dump_lines(a.out/'sentinel_k12_models.jsonl',sentinels);write(a.out/'sentinel_k12_shared.json',shared_sentinels)
        dump_lines(a.out/'sentinel_k12_hybrids.jsonl',[r for r in hybrids if int(r['sample_id'].rsplit('/',1)[1])==12])
        summary=dict(schema='pose_support_attention_scale_joint_v1',scope='all4 fixed synthetic rollouts; descriptive, no independent-window/seed sampling claim',input_completion_sha256=a.completion_sha256,producer_source_manifest_sha256=c['source_manifest_sha256'],metrics_sha256=METRICS_SHA,
            counts=dict(raw_model_rows=9216,paired_model_windows=4608,unique_support_rows=384,physical_windows=64,physical_rollouts=4,model_seeds=[0,1,2]),
            metric_semantics=dict(delta='BAF minus RAW within same model/window',scatter='population within16 adjacent windows, not standard error',support_retention='ratio of event-count sums within one rid; support never multiplied by model seeds',region='endpoint exact at frame; endpoint_union_approx not continuous visibility or physical event origin',branch='center is localizer; depth UV is internal attention, not final pose center'),
            matched_per_seed_rid=paired,support_unique_groups=list(sg.values()),within_light_scale=within,cross_light_scale=drift,producer_hybrid32_means=original['three_seed_means'],producer_mean_recomputation_check=dict(passed=True,absolute_relative_tolerance=1e-10,independent_GPU_validation=False),
            null_counts=dict(model_payload=null_counts([{k:v for k,v in r.items() if k not in (*SHARED,'support')} for r in rows]),support_unique=null_counts(support),shared_geometry_unique=null_counts(list(shared.values())),hybrid=null_counts(hybrids)),
            expected_null_note='scale=None for4608 center-branch records is not a failed depth measurement',p_values_computed=False,hybrid_selected=False,training_decision=False)
        write(a.out/'joint_summary.json',summary);write(a.out/'raw_server_artifact_index.json',dict(input_completion=record(cp),all_raw_files_retained_on_server=inputs,producer_source=ms,producer_recorded_source_records=c['source_records'],verification_scope='postprocessor rehashes raw artifacts and metric source; frozen models/input assets verified in producer receipt, not loaded here'))
        need(all(sha(r['path'])==r['sha256'] for r in inputs) and sha(ms['path'])==METRICS_SHA,'Input changed during postprocess')
        write(a.out/'completion.json',dict(schema='pose_joint_readonly_export_v1',status='complete',optimizer_updates=0,model_forward_passes=0,GPU_used=False,real_data_used=False,input_completion_sha256=a.completion_sha256,
            source=record(Path(__file__)),artifacts=[record(q) for q in a.out.iterdir() if q.is_file()],elapsed_seconds=time.monotonic()-start))
    except BaseException as e:
        write(a.out/'failure.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),optimizer_updates=0));raise

if __name__=='__main__':main()
