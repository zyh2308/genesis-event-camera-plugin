from pathlib import Path
import hashlib,json
O=Path(__file__).resolve().parent
d=json.loads((O/'remote_preflight.json').read_text());c=d['checks']
root=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real')
source=[]
for remote,h in d['source_hashes'].items():
    if '/repositories/gec-sim2real/' in remote:p=root/remote.split('/repositories/gec-sim2real/',1)[1]
    elif '/external/sources/downstream/STE-FlowNet/' in remote:p=Path('/home/科研/Eventbased_WAM/external_sources/STE-FlowNet')/remote.split('/STE-FlowNet/',1)[1]
    else:continue
    source.append({'file':str(p),'matches_g1':p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==h})
static=[{'rollout':e['rollout'],**f} for e in c['dataset_semantics'] for f in e['checks'] if e['rollout'].startswith('s1_')]
static_valid=[r for r in static if r['old_plane_mean_magnitude'] is not None]
ad=c['all_adapter_labels_timestamps_images'];rep=[f for e in c['selected_representation_rebuild'] for f in e['checks']]
init=[r for r in c['trajectory_initialization'] if r['first_path_offset_m']>1e-8]
summary={
 'runtime_seconds':d['runtime_seconds'],'source_matches':source,
 'rollouts_checked':len(c['dataset_semantics']),'geometry_endpoint_checks':sum(len(e['checks']) for e in c['dataset_semantics']),
 'all_sampled_old_GT_reproduced_max_error':max(f['old_GT_reproduced_max_error_valid'] for e in c['dataset_semantics'] for f in e['checks']),
 'ground_unprojection_max_p99_height_m':max(f['plane_world_height_abs_p99_m'] for e in c['dataset_semantics'] for f in e['checks']),
 'static_camera_plane_cases':len(static),'static_camera_plane_cases_with_old_valid_ground':len(static_valid),'static_camera_plane_old_nonzero_cases':sum(r['old_plane_mean_magnitude']>1e-6 for r in static_valid),
 'static_camera_pose_max_delta':max(r['camera_delta'] for r in static),'static_plane_pose_max_delta':max(r['plane_pose_delta'] for r in static),
 'static_plane_old_mean_magnitude_range':[min(r['old_plane_mean_magnitude'] for r in static_valid),max(r['old_plane_mean_magnitude'] for r in static_valid)],
 'static_plane_mapped_max_mean_magnitude':max(r['mapped_plane_mean_magnitude'] for r in static_valid),
 'analytic_projection':c['analytic_projection'],'warp':c['warp'],'metric_controls':c['metric_controls'],
 'output_conversion_identity':c['output_conversion_identity'],
 'augmentation_cases':len(c['augmentation_coordinate_controls']),
 'augmentation_coordinate_max_error':max(r['coordinate_image_max_error'] for r in c['augmentation_coordinate_controls']),
 'augmentation_vector_max_error':max(r['vector_max_error'] for r in c['augmentation_coordinate_controls']),
 'adapter_files_checked':len(ad),'adapter_all_pass':all(all(r[k] for k in ['label_equal_shared','mask_equal_shared','windows_match_shared','gray_endpoints_equal_shared','counts_nonnegative']) for r in ad),
 'raw_representation_windows_rebuilt':len(rep),'raw_representation_all_pass':all(all(r[k] for k in ['timestamp_count_matches','representation_exact_rebuild','no_future_events']) for r in rep),
 'trajectory_init_moving_object_cases':len(init),'trajectory_init_offset_m_range':[min(r['first_path_offset_m'] for r in init),max(r['first_path_offset_m'] for r in init)],
 'legacy_data_preflight_acceptance':'BLOCKED: semantic ID mapping and both loss/model warp controls fail.',
 'no_new_training_started':True,'no_real_data_scored':True,'production_code_changed':False}
(O/'preflight_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
