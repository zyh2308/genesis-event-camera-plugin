"""CPU-only full-cohort frozen C1 wrong-peak diagnosis; no candidate readout.

GT regions/grid references are posthoc diagnostics, never inference inputs.
The prior failed soft-focus run supplies ONLY its immutable original attention
and original control predictions. All candidate/focused predictions are ignored.
"""
import argparse,csv,hashlib,importlib.util,itertools,json,math,os,sys,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

SUPPORT_SHA='6102a77a44c6604cc5349b9e58d2787fb8080c779561b7dc0b2bc1a143c0f91b'
CONTRACT_SHA='d2368c9f246a61efc4382238741f1e7a62cf891aa8d6d653136832ed64ea3b62'
METHODS=('eventgenesis','v2e');LIGHTS=('normal','half','eighth');RIDS=('validation_000','validation_031','validation_063','validation_095')
PIXELS=np.array([346.,260.])

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def checked(r):
    p=Path(r['path']);need(p.is_file() and p.stat().st_size==r['size_bytes'] and sha(p)==r['sha256'],'Pinned input differs: '+str(p));return p
def identity(row):return tuple(row[k] for k in ('method','seed','condition','rid','sample_id'))
def grid(n):
    y,x=np.mgrid[:n,:n];return np.stack(((x+.5)/n,(y+.5)/n),-1).reshape(-1,2)
def norm_px(a,b):return float(np.linalg.norm((np.asarray(a,float)-np.asarray(b,float))*PIXELS))
def cosine(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float);d=float(np.linalg.norm(a)*np.linalg.norm(b));return float(a@b/d) if d>0 else None


def peak_diagnostics(case,p,sup,regions,pyramids):
    """Pure posthoc map/region statistics; no scoring-based output selection."""
    p=np.asarray(p,float);need(p.shape==(16,16) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<=1e-5,'Invalid original map')
    gt=np.asarray(case['gt_uv'],np.float32);uv=np.asarray(case['control_uv'],np.float32)
    g16=grid(16);i=int(p.ravel().argmax());peak=g16[i];row,col=divmod(i,16)
    need(np.array_equal(peak,case['peak_uv']) and np.array_equal(peak,sup['peak_uv']),'Original predicted peak differs')
    need(np.max(abs(uv-np.asarray(sup['pred_uv'])))<=2e-6 and np.max(abs(p.ravel()@g16-uv))<=2e-6,'Original RAW control/map parity failed')
    d16=np.linalg.norm((g16-gt)*PIXELS,axis=1);d128=np.linalg.norm((grid(128)-gt)*PIXELS,axis=1)
    endpoint=sup['attention']['endpoint'];union=sup['attention']['endpoint_union_approx']
    px,py=np.floor(peak*PIXELS).astype(int)
    need(bool(regions['endpoint'].mask[py,px])==endpoint['peak_in_region'] and
         bool(regions['endpoint_union_approx'].mask[py,px])==union['peak_in_region'],'Old native peak membership differs from pinned physical masks')
    ep=pyramids['endpoint'];un=pyramids['endpoint_union_approx']
    ep_overlap=None if not ep['available'] else float(ep['attention16'][row,col]);un_overlap=None if not un['available'] else float(un['attention16'][row,col])
    if endpoint['peak_in_region'] is True:category='native_peak_in_visible_target'
    elif ep_overlap is not None and ep_overlap>0:category='native_point_outside_endpoint_but_peak_cell_has_target_support'
    elif un_overlap is not None and un_overlap>0:category='no_endpoint_cell_support_but_endpoint_union_cell_support'
    elif ep_overlap is not None and un_overlap is not None:category='no_target_support_in_endpoint_or_union_peak_cell'
    else:category='region_unavailable'
    center_error=float(np.linalg.norm((uv-gt)*[346,260]));peak_error=norm_px(peak,gt)
    geom=sup['geometric_sanity'];visible=sup['visible_geometry']['endpoint'];support=sup['support']['regions']['endpoint']
    need(np.max(abs(gt*PIXELS-np.asarray(geom['GT_anchor_uv_px'])))<=3e-5,'Physical GT anchor differs from original float32 label')
    fg,bg=support['foreground'],support['background_region'];total=fg['raw_events']+bg['raw_events'];bbox=visible['bbox_xyxy']
    out=dict(center_error_px=center_error,peak_error_px=peak_error,hard_peak_grid16_floor_px=float(d16.min()),hard_peak_grid128_floor_px=float(d128.min()),
        peak_error_above_hard16_floor_px=peak_error-float(d16.min()),peak_to_center_distance_px=norm_px(peak,uv),center_error_minus_peak_error_px=center_error-peak_error,
        endpoint_peak_point_hit=endpoint['peak_in_region'],endpoint_peak_cell_overlap=ep_overlap,endpoint_peak_cell_has_support=None if ep_overlap is None else ep_overlap>0,
        union_peak_point_hit=union['peak_in_region'],union_peak_cell_overlap=un_overlap,union_peak_cell_has_support=None if un_overlap is None else un_overlap>0,
        endpoint_uv_point_hit=endpoint['prediction_in_region'],foreground_attention_mass=endpoint['region_attention_mass'],foreground_attention_enrichment=endpoint['region_mass_enrichment'],
        nominal_GT_target_foreground_mass=sup['nominal_GT_heat_reference']['endpoint']['region_attention_mass'],
        GT_anchor_visible=geom['GT_anchor_visible_membership']['inside'],GT_anchor_visible_centroid_distance_px=geom['GT_anchor_to_visible_centroid_distance_px'],
        GT_anchor_outside_model_grid_range=sup['nominal_GT_heat_reference']['GT_anchor_outside_grid_range'],nominal_GT_heat_decoded_bias_px=sup['nominal_GT_heat_reference']['target_decoded_center_bias_px'],
        max_corner_motion_px=sup['motion']['max_projected_corner_displacement_px'],anchor_motion_px=sup['motion']['projected_anchor_displacement_px'],
        visible_area_px=visible['area_px'],visible_mask_border_contact=visible['visible_mask_touches_border'],
        visible_bbox_min_side_in_attention_cells=min((bbox[2]-bbox[0])/346*16,(bbox[3]-bbox[1])/260*16) if bbox is not None else None,
        foreground_raw_events=fg['raw_events'],background_region_raw_events=bg['raw_events'],foreground_event_share=fg['raw_events']/total if total else None,
        foreground_raw_occupancy=fg['raw_occupancy_ratio'],background_raw_occupancy=bg['raw_occupancy_ratio'],
        foreground_BAF_retention=fg['event_retention_ratio'],background_BAF_retention=bg['event_retention_ratio'],
        local_feature_norm=sup['features']['local_l2'],context_feature_norm=sup['features']['context_l2'],
        peak_probability=float(p.max()),top1_minus_top2_probability=float(np.sort(p.ravel())[-1]-np.sort(p.ravel())[-2]),
        attention_entropy=float(-np.sum(p[p>0]*np.log(p[p>0]))),peak_category=category)
    return dict(**{k:case[k] for k in ('method','seed','condition','rid','sample_id')},metrics=out)


def summaries(rows,keys):
    grouped=defaultdict(list)
    for row in rows:grouped[tuple(row[k] for k in keys)].append(row)
    result=[]
    for key,values in sorted(grouped.items()):
        numeric={k for k,v in values[0]['metrics'].items() if not isinstance(v,str)}
        mean={};available={}
        for name in sorted(numeric):
            vals=[r['metrics'][name] for r in values if r['metrics'][name] is not None]
            need(all(math.isfinite(float(v)) for v in vals),'Nonfinite diagnostic scalar')
            mean[name]=float(np.mean(vals)) if vals else None;available[name]=len(vals)
        result.append(dict(zip(keys,key),prediction_rows=len(values),metric_mean=mean,available=available,
                           category_counts=dict(Counter(r['metrics']['peak_category'] for r in values))))
    return result


def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    for n in ('readout-root','support-root','out'):a.add_argument('--'+n,type=Path,required=True)
    a.add_argument('--readout-sha',required=True);a.add_argument('--support-sha',required=True);args=a.parse_args()
    need(not args.out.exists(),'Fresh output required');args.out.mkdir(parents=True);start=time.monotonic();inputs={}
    def pin(r):
        p=checked(r);inputs[str(p)]=r;return p
    def artifact(c,name):
        recs=[r for r in c['artifacts'] if Path(r['path']).name==name];need(len(recs)==1,'Missing/ambiguous pinned artifact '+name);return pin(recs[0])
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        for root,expected in ((args.readout_root,args.readout_sha),(args.support_root,args.support_sha)):
            cp=root/'completion.json';need(sha(cp)==expected,'Externally pinned completion differs');inputs[str(cp)]=dict(path=str(cp),size_bytes=cp.stat().st_size,sha256=expected)
        rc,supc=read(args.readout_root/'completion.json'),read(args.support_root/'completion.json')
        need(rc['status']==supc['status']=='complete' and rc['mode']=='full' and rc['paired_cases']==1152 and rc['optimizer_updates']==supc['optimizer_updates']==0 and
             rc['real_data_used'] is False and supc['real_data_used'] is False,'Wrong producer scope')
        cases=[json.loads(line) for line in artifact(rc,'paired_cases.jsonl').read_text().splitlines()];maps=np.load(artifact(rc,'readout_maps.npz'),allow_pickle=False)['original_attention'];index=read(artifact(rc,'readout_map_index.json'))
        support={}
        with artifact(supc,'per_window.jsonl').open() as f:
            for line in f:
                r=json.loads(line)
                if (r['arm'],r['branch'],r['input_view'])==('C1','center','raw'):
                    key=identity(r);need(key not in support,'Duplicate SUP row');support[key]=r
        need(len(cases)==len(index)==len(support)==1152 and maps.shape==(1152,16,16),'Incomplete fixed1152 inputs')
        rcmodels={(r['method'],r['seed']):r for r in rc['models']};supmodels={(r['method'],r['seed'],r['arm'],r['branch']):r for r in supc['model_records']}
        need(len(rcmodels)==6 and all(r['unchanged'] is True for r in rcmodels.values()),'Frozen C1 model receipts incomplete')
        for key,model in rcmodels.items():
            for branch in ('center','depth'):need(model[branch+'_state']==supmodels[(*key,'C1',branch)]['state_sha256'],'Frozen model state differs across producers')
        sources={r['path']:r for r in supc['source_records']};sr=[r for r in sources.values() if Path(r['path']).name=='event_support.py' and r['sha256']==SUPPORT_SHA];need(len(sr)==1,'Frozen region helper missing')
        sp=pin(sr[0]);spec=importlib.util.spec_from_file_location('peak_diagnostic_frozen_regions',sp);S=importlib.util.module_from_spec(spec);sys.modules[spec.name]=S;spec.loader.exec_module(S)
        contracts={r['path']:r for r in rc['source_records'] if r['sha256']==CONTRACT_SHA};need(len(contracts)==1,'Original physical contract pin missing');base=read(pin(next(iter(contracts.values()))))
        physical_rc={r['path']:r for r in rc['source_records'] if '/physical/validation_' in r['path']};physical_sup={r['path']:r for r in supc['source_records'] if '/physical/validation_' in r['path']}
        need(len(physical_rc)==12 and physical_rc==physical_sup,'Physical K/T/mask/time source receipt differs')
        for r in physical_rc.values():pin(r)
        keys=np.load(artifact(rc,'keypoints.npy'),allow_pickle=False);regions={};pyramids={}
        by_id={c['sample_id']:c for c in cases if c['method']=='eventgenesis' and c['seed']==0 and c['condition']=='normal'};need(len(by_id)==64,'Full physical64 reference missing')
        relit_dirs={Path(base['radiance_validation'])/light/'pose_bank/physical' for light in ('half','eighth')}
        normals=[Path(p) for p in physical_rc if Path(p).parent not in relit_dirs]
        need(len(normals)==4 and len({p.parent for p in normals})==1,'Normal physical bank path ambiguous');normal_physical=normals[0]
        for condition in LIGHTS:
            # Normal bank path comes from the original original-export source;
            # relit bank roots are fixed in the pinned base contract.
            bank=normal_physical.parent.parent if condition=='normal' else Path(base['radiance_validation'])/condition/'pose_bank'
            values=[dict(id=c['sample_id'],rid=c['rid'],K=np.asarray(c['K']),T=np.asarray(c['T_gt']),keys=keys,
                         uv=np.asarray([c['gt_uv']],np.float32)) for c in sorted(by_id.values(),key=lambda c:c['sample_id'])]
            regions[condition]=S.regions_for_rows(bank,values,strict=True)
            if condition!='normal':S.verify_matched_regions(regions['normal'],regions[condition])
            pyramids[condition]={sid:{name:S.region_pyramid(rr[name]) for name in ('endpoint','endpoint_union_approx')} for sid,rr in regions[condition].items()}
        expected=set(itertools.product(METHODS,range(3),LIGHTS,RIDS,range(4,20)));seen=set();output=[];event_unique={}
        for i,case in enumerate(cases):
            need(case['map_index']==index[i]['map_index']==i and all(case[k]==index[i][k] for k in ('method','seed','condition','rid','sample_id')),'Readout map order differs')
            key=identity(case);r=support[key];sid=case['sample_id'];rid=case['rid'];k=int(sid.rsplit('/',1)[1]);cell=(*key[:3],rid,k)
            need(cell in expected and cell not in seen and sid==f'{rid}/{k:03d}','Unregistered/duplicate physical prediction');seen.add(cell)
            for field in ('K','T_gt','anchor','gt_uv'):need(np.array_equal(case[field],by_id[sid][field]),'RC physical labels differ across seed/method/illumination: '+field)
            gt_z=float((np.asarray(case['T_gt'])[:3,:3]@case['anchor']+np.asarray(case['T_gt'])[:3,3])[2]);need(abs(gt_z-r['labels']['gt_z_m'])<1e-12,'GT depth label pin differs')
            region=regions[case['condition']][sid]
            need(r['support']['start_s']==region['start_s'] and r['support']['end_s']==region['end_s'],'Event support time window differs')
            diag=peak_diagnostics(case,maps[i],r,regions[case['condition']][sid],pyramids[case['condition']][sid]);output.append(diag)
            physical=(case['method'],case['condition'],sid)
            if physical in event_unique:need(event_unique[physical]==r['support'],'Seed duplicated support differs')
            event_unique[physical]=r['support']
            normal=support[(case['method'],case['seed'],'normal',rid,sid)]['features']
            diag['metrics']['local_feature_cos_to_normal']=cosine(r['features']['local_feature'],normal['local_feature'])
            diag['metrics']['context_feature_cos_to_normal']=cosine(r['features']['context_feature'],normal['context_feature'])
            need(time.monotonic()-start<600,'Fixed CPU600s runtime exceeded')
        need(seen==expected and len(event_unique)==384,'Incomplete all-window cohort/support deduplication')
        for name,groups in (('per_seed_rid',('method','seed','condition','rid')),('three_seed_per_rid',('method','condition','rid')),('three_seed_condition',('method','condition'))):write(args.out/(name+'.json'),summaries(output,groups))
        with (args.out/'per_prediction.jsonl').open('x') as f:
            for r in output:f.write(json.dumps(r,allow_nan=False)+'\n')
        with (args.out/'unique_event_support.jsonl').open('x') as f:
            for (m,c,sid),r in sorted(event_unique.items()):f.write(json.dumps(dict(method=m,condition=c,sample_id=sid,support=r),allow_nan=False)+'\n')
        unique_groups=defaultdict(list)
        for (m,c,sid),r in event_unique.items():unique_groups[(m,c)].append(r)
        unique_summary=[]
        for (m,c),values in sorted(unique_groups.items()):
            need(len(values)==64,'Unique event support windows incomplete');totals={}
            for region_name in ('endpoint','endpoint_union_approx'):
                totals[region_name]={}
                for part in ('foreground','background_region'):
                    raw=sum(v['regions'][region_name][part]['raw_events'] for v in values);kept=sum(v['regions'][region_name][part]['baf_events'] for v in values)
                    totals[region_name][part]=dict(raw_events=raw,baf_events=kept,event_count_weighted_retention=kept/raw if raw else None)
            unique_summary.append(dict(method=m,condition=c,physical_windows=64,model_seed_multiplier_applied=False,regions=totals))
        write(args.out/'unique_event_support_summary.json',unique_summary)
        need(all(sha(p)==r['sha256'] for p,r in inputs.items()),'Input changed during CPU diagnosis')
        write(args.out/'completion.json',dict(schema='pose_wrong_peak_full_cohort_cpu_v1',status='complete',prediction_cases=1152,physical_windows=64,physical_rollouts=4,
            unique_event_support_rows=384,input_records=list(inputs.values()),script_sha256=sha(Path(__file__)),GT_used_only_posthoc=True,model_forward_calls=0,optimizer_updates=0,GPU_used=False,real_data_used=False,
            stopped_softfocus_candidate_outputs_used=False,training_peak_data_available=False,p_values_computed=False,
            limitations=['Hard16/128 nearest-grid references apply to discrete peaks, not a continuous softargmax accuracy lower bound.',
                         'Fractional cell overlap is a spatial proxy, not convolution feature provenance. Endpoint union is not exact swept visibility.',
                         'No-target-support classification is not proof of a physical noise source or a unique failure cause.',
                         'Descriptive adjacent-window/model-seed associations do not establish population significance.'],elapsed_seconds=time.monotonic()-start))
        print('PEAK_DIAGNOSIS_COMPLETE',len(output),len(event_unique),flush=True)
    except BaseException:
        write(args.out/'failure.json',dict(status='failed',error=traceback.format_exc(),model_forward_calls=0));raise


if __name__=='__main__':main()
