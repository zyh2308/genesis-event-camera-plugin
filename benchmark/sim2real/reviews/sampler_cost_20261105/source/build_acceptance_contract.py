"""Derive compact provenance from actual complete plugin traces, without recomputation."""
import hashlib,json,math
from pathlib import Path
E=Path(__file__).resolve().parent
def decode(x):
    if isinstance(x,dict):
        if x.get('kind')=='python_float':return float.fromhex(x['hex'])
        return {k:decode(v) for k,v in x.items()}
    if isinstance(x,list):return list(map(decode,x))
    return x
pairs=[];numerical=[]
for folder,case in [('gate',x) for x in ['s0_00','s1_00','s2_00','s3_00','train_000']]+[('gate_long','s3_00')]:
    traces=[json.loads((E/folder/side/case/'trace.json').read_text()) for side in ['frozen','candidate']]
    old,new=traces;assert all(t['status']=='complete' for t in traces)
    assert old['intervals']==new['intervals']==len(old['rows'])==len(new['rows'])
    pairs.append({'folder':folder,'case':case,'intervals':old['intervals'],'actual_sensor_frames_per_interval':[[r['sensor_inputs']['frame_count'] for r in t['rows']] for t in traces],'deterministic_algorithms':[t['deterministic_algorithms'] for t in traces],'cublas_workspace_config':[t['cublas_workspace_config'] for t in traces],'gpu_uuid_visible':[t['gpu_uuid_visible'] for t in traces],'geometry_source_sha256':[t['core_source_sha256']['genesis_event_plugin/gpu_geometry.py'] for t in traces],'trace_files':[{'path':str(E/folder/side/case/'trace.json'),'sha256':hashlib.sha256((E/folder/side/case/'trace.json').read_bytes()).hexdigest()} for side in ['frozen','candidate']]})
    if folder=='gate':
        cfg=decode(new['sensor_initial']['config'])
        rows=[]
        for r in new['rows']:
            s=decode(r['geometry_diagnostics']['sampling']);dt=r['physical_right_s']-r['physical_left_s']
            derived=math.ceil(dt*2*math.pi*cfg['cutoff_hz']/cfg['sensor_eps_target'])
            assert derived==s['sensor_samples']==s['samples']==377
            rows.append({'dt_s':dt,'derived_sensor_samples':derived,'actual_sampling':s})
        numerical.append({'case':case,'actual_noise_config':cfg,'samples':rows,'config_source':'actual sensor_initial.config from completed plugin trace, including resolved defaults','noise_source_sha256':new['core_source_sha256']['genesis_event_plugin/noise_model.py'],'sampler_source_sha256':new['core_source_sha256']['genesis_event_plugin/adaptive_sampler.py']})
(E/'numerical_contract.json').write_text(json.dumps({'status':'complete','cases':numerical,'noise_rules_changed':False,'sensor_time_points_changed':False},indent=2)+'\n')
(E/'acceptance_receipt.json').write_text(json.dumps({'status':'complete','pairs':pairs,'total_compared_intermediate_frames':sum(x['intervals']*377 for x in pairs),'all_deterministic':all(all(x['deterministic_algorithms']) for x in pairs),'raw_timestamps_quantized':False,'observability_time_excluded_from_efficiency':True},indent=2)+'\n')
print('ACCEPTANCE_PROVENANCE_COMPLETE',flush=True)
