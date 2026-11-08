"""Explicit synthetic schema fixtures, never published as actual run results."""
import copy,unittest
import summarize_actual_results as S


def common(phase):
    count=1 if phase=='smoke' else 288;seeds=[0] if phase=='smoke' else S.SEEDS
    records=[dict(method=m,seed=s,arm='aux_visible_mask_center',branch='center',updates=count,new_optimizer_updates=count,teacher_unchanged=True,size_unchanged=True,independent_model_optimizer_rebuild=True,optimizer_state_equal=True,
        first_native_loss_oracle=dict(passed=True,absolute_errors=dict(loss=0.)),source_state_sha256='a'*64,final_state_sha256='b'*64,auxiliary_state_sha256='c'*64,labels_sha256='d'*64,checkpoint_sha256='e'*64,
        exposure=dict(normal=32 if phase=='smoke' else 9174,half=24 if phase=='smoke' else 6880,eighth=8 if phase=='smoke' else 2294)) for m in S.METHODS for s in seeds]
    bchecks=[dict(method=m,seed=s,condition=k,windows=64,passed=True,native_summary_and_by_rollout_parity=True,max_abs={k:0. for k in ('center_uv','depth_uv','log_size','log_shape','z','size')},rotation_max_abs=0.) for m in S.METHODS for s in S.SEEDS for k in ('normal','half','eighth')]
    resource=dict(passed=True,gpu_uuid=S.GPU3,gpu_index=3,cuda_device_count=1,cuda_visible_devices=S.GPU3,queue_parent_pid=335161,queue_parent_start_ticks=53273268,worker_pid=1234,active_gpu_pids=[335161,1234],resource_contract=dict(sha256=S.RESOURCE))
    return dict(schema='pose_aux_visible_mask_center_v1',status='complete_synthetic_only',phase=phase,canonical_candidate='aux_visible_mask_center',contract_sha256=S.SCIENCE,source_manifest_sha256=S.SOURCES,runtime_resource_contract_sha256=S.RESOURCE,
        execution_gpu_uuid=S.GPU3,original_frozen_source_gpu_uuid=S.GPU4,real_scores_computed=False,GT_forward_inputs=False,pose_prediction_uses_auxiliary_head=False,resume_supported=False,checkpoint_verification_scope='model_auxiliary_optimizer_readback_only_no_resume_claim',
        original30_gate=dict(status='not_retested_not_passed_by_this_probe',thresholds_unchanged=True),original8_confirmation=dict(status='not_retested_not_passed_by_this_probe',thresholds_unchanged=True),new_optimizer_updates=count*len(records),records=records,
        full288_admissions=[dict(method=m,seed=s,full288_replay_exact=True,labels_sha256='f'*64,trajectory_sha256='a'*64) for m in S.METHODS for s in S.SEEDS],mask_labels=[dict(method=m,normal_sha256='a'*64,auxiliary_sha256='b'*64) for m in S.METHODS],
        input_caches=[dict(path=f'/fixture/{m}_RAW_inputs_masks128.pt',sha256='e'*64,size_bytes=1) for m in S.METHODS],runtime_reference_baseline=dict(status='passed',optimizer_updates=0,real_data_used=False,all_models_unchanged=True,prediction_windows=1152,checks=bchecks),
        resource_start=resource,resource_end=copy.deepcopy(resource),source_records=[dict(path='/fixture/smoke/completion.json',sha256='9'*64,size_bytes=1)],results=[],stage_gates=None)


def fixtures():
    smoke=common('smoke');study=common('study')
    for m in S.METHODS:
        for seed in S.SEEDS:
            arms={}
            for arm in S.ARMS:
                views={}
                for condition in S.CONDITIONS:
                    n,r=(384,96) if condition=='clean' else (64,4);rids=[f'validation_{i:03d}' for i in range(r)];ks=(4,9,14,19) if condition=='clean' else range(4,20)
                    center=1. if arm=='aux_visible_mask_center' and condition=='eighth' else 2.
                    metric=dict(depth_mae_m=.03,depth_bias_m=.001,log_size_mae=.04,log_size_bias=.002,center_px=center,success=.5,auc=.6)
                    views[condition]=dict(windows=n,rollouts=r,window_ids=[f'{rid}/{k:03d}' for rid in rids for k in ks],center_epe_px=center,pose=dict(summary=metric,by_rollout={rid:metric.copy() for rid in rids}),auxiliary_mask_diagnostic=None)
                arms[arm]=views
            study['results'].append(dict(method=m,seed=seed,arms=arms))
    checks=S.independent_checks(study['results']);study['stage_gates']={name:dict(checks=v,passed=all(r['passed'] for r in v)) for name,v in checks.items()}
    return smoke,study


class StrictActualSummaryTests(unittest.TestCase):
    def test_full_matrix_has48rows36plus12_and_no_wrong_peak_claim(self):
        smoke,study=fixtures();r,rows,rollouts=S.summarize(smoke,study,'9'*64)
        self.assertEqual(len(rows),48);self.assertEqual(len(rollouts),1296);self.assertTrue(r['both_gate_groups_passed']);self.assertEqual(r['stage_gates']['center_stage36']['total_checks'],36);self.assertEqual(r['stage_gates']['matchednormal12']['total_checks'],12)
        self.assertFalse(r['post_training_wrong_peak_measurement_available']);self.assertEqual(len(r['paired_generator_comparisons']),8)

    def test_smoke_only_does_not_invent_metrics_or_completion(self):
        smoke,_=fixtures();r,rows,_=S.summarize(smoke);self.assertEqual(rows,[]);self.assertEqual(r['status'],'smoke_complete_study_not_supplied')

    def test_one_condition_regression_cannot_hide_behind_average(self):
        smoke,study=fixtures();study['results'][0]['arms']['aux_visible_mask_center']['half']['pose']['summary']['success']=.4
        with self.assertRaises(ValueError):S.summarize(smoke,study,'9'*64)

    def test_missing_seed_nonfinite_wrong_source_or_smoke_rejected(self):
        smoke,study=fixtures();bad=copy.deepcopy(study);bad['results'].pop()
        with self.assertRaises(ValueError):S.summarize(smoke,bad,'9'*64)
        bad=copy.deepcopy(study);bad['results'][0]['arms']['raw_C1']['half']['pose']['summary']['center_px']=float('nan')
        with self.assertRaises(ValueError):S.summarize(smoke,bad,'9'*64)
        bad=copy.deepcopy(study);bad['source_manifest_sha256']='0'*64
        with self.assertRaises(ValueError):S.summarize(smoke,bad,'9'*64)
        with self.assertRaises(ValueError):S.summarize(smoke,study,'0'*64)

    def test_frozen_depth_change_rejected_even_with_complete_gates(self):
        smoke,study=fixtures();study['results'][0]['arms']['aux_visible_mask_center']['eighth']['pose']['summary']['depth_mae_m']+=.01
        with self.assertRaises(ValueError):S.summarize(smoke,study,'9'*64)


if __name__=='__main__':unittest.main(verbosity=2)
