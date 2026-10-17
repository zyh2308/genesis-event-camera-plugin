"""Assess captured independent evidence; exit 2 means BLOCKED.

This assesses this audit bundle. Production launch enforcement is engineering work.
"""
from pathlib import Path
import hashlib,json,sys
O=Path(__file__).resolve().parent
d=json.loads((O/'remote_preflight.json').read_text());c=d['checks']
a=json.loads((O/'additional_contracts.json').read_text())
r=json.loads((O/'render_minimal.json').read_text())
rows=[]
def add(name,passed,evidence):rows.append({'check':name,'status':'PASS' if passed else 'FAIL','evidence':evidence})
static=[f for e in c['dataset_semantics'] if e['rollout'].startswith('s1_') for f in e['checks'] if f['old_plane_mean_magnitude'] is not None]
add('static_ground_physical_labels',bool(static) and max(f['old_plane_mean_magnitude'] for f in static)<1e-5,{'cases_with_old_valid_ground':len(static),'max_ground_displacement_px':max(f['old_plane_mean_magnitude'] for f in static),'tolerance_px':1e-5})
for kind in ['loss','model']:
 tests=next(x['tests'] for x in c['warp'] if x['kind']==kind and not x['consistent_control'])
 add(kind+'_warp_identity_and_integer_translation',all(x['interior_MAE']<1e-6 and x['gradient_finite'] for x in tests),{'max_interior_MAE':max(x['interior_MAE'] for x in tests),'tolerance_MAE':1e-6})
render=next(x for x in r['rows'] if x['render_path']=='radiance_plus_depth' and x['offset']==0.)
add('actual_frame_provider_depth_K_pixel_contract',render['plane_z_abs_quantiles_eroded_m'][2]<.001,{'plane_height_abs_p99_m':render['plane_z_abs_quantiles_eroded_m'][2],'tolerance_m':.001})
add('analytic_projection',max(x['max_error'] for x in c['analytic_projection'])<1e-5,c['analytic_projection'])
ad=c['all_adapter_labels_timestamps_images']
add('adapter_transport_not_physical_GT_validity',len(ad)==64 and all(all(x[k] for k in ['label_equal_shared','mask_equal_shared','windows_match_shared','gray_endpoints_equal_shared','counts_nonnegative']) for x in ad),{'adapter_files':len(ad)})
rep=[x for e in c['selected_representation_rebuild'] for x in e['checks']]
add('selected_raw_events_to_representation',len(rep)==24 and all(all(x[k] for k in ['timestamp_count_matches','representation_exact_rebuild','no_future_events']) for x in rep),{'windows_rebuilt':len(rep)})
add('augmentation',all(x['coordinate_image_max_error']<1e-5 and x['vector_max_error']<1e-5 for x in c['augmentation_coordinate_controls']),{'cases':len(c['augmentation_coordinate_controls'])})
add('EPE_and_padding',c['metric_controls']=={'identical_prediction_epe':0.,'zero_prediction_epe_expected_5':5.,'v_only_error_expected_4':4.} and c['output_conversion_identity'] and all(x['rejected'] for x in a['metric_negative_controls']) and all(a['validation_padding_roundtrip'][k] for k in ['event_exact','gray0_exact','gray1_exact']),{'metric_controls':c['metric_controls'],'negative_controls':a['metric_negative_controls'],'padding':a['validation_padding_roundtrip']})
add('correlation_forward_and_backward',a['correlation_forward_max_error']<1e-10 and max(a['correlation_gradient_max_errors'])<1e-10 and a['correlation_gradcheck_fast'],{k:a[k] for k in ['correlation_forward_max_error','correlation_gradient_max_errors','correlation_gradcheck_fast']})
out={'scope':'Independent assessment of captured legacy Stage-4 evidence; not an installed production launcher gate.','status':'PASS' if all(x['status']=='PASS' for x in rows) else 'BLOCKED','checks':rows,'source_and_input_fingerprints':{'sources':{**d['source_hashes'],**a['extra_source_hashes']},'inputs':a['input_artifact_hashes']},'evidence_file_sha256':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['remote_preflight.json','additional_contracts.json','render_minimal.json']},'additional_requirements_for_new_full_run':['Exact new CUDA training entrypoint paired short run','Actual interrupted/resumed next optimizer steps, including RNG and epoch boundary','Launcher rejects missing/failed/stale receipt without spawning training']}
(O/'gate_verdict.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'checks':[{'check':x['check'],'status':x['status']} for x in rows]},ensure_ascii=False))
sys.exit(0 if out['status']=='PASS' else 2)
