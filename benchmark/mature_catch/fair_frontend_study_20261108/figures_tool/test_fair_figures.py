"""Statistical and evidence-contract checks; fixtures never become study results."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
import make_fair_figures as f


def actor(seed=1001, rate=25., method='ours', frontend='w10_fit1_gap50', success=False):
    return dict(physical_case_id=f'case_{seed}', seed=seed, endpoint_hz=rate,
        method=method, frontend_id=frontend if method in f.METHODS else '',
        verified_catch=success, raw_success=success, terminated=False,
        hand_contact_steps=int(success), final_contact=success,
        initialization_valid=None if method=='state_reference' else True,
        sim_time_s=1.5, wall_s=999., output=f'test_fixture/{seed}/{method}/{rate}/{frontend}')


def crossover():
    return {'rows': [f.episode(actor(seed, rate, method, frontend))
        for seed in range(41,45) for rate in f.VALIDATION_RATES
        for method in f.METHODS for frontend in f.DEFAULT_FRONTENDS]}


def blind_stage():
    rows=[]
    for seed in range(1001,1033):
        for rate in f.BLIND_RATES:
            for method in f.METHODS:
                # Keep each block's cross-source result identical at every rate.
                rows.append(f.episode(actor(seed,rate,method,success=(seed%2==0 if method=='ours' else False))))
        for method in f.REFERENCES:
            rows.append(f.episode(actor(seed,None,method,success=method=='state_reference')))
    return dict(rows=rows)


def cost_rows():
    return [dict(method=method, source_hz=rate, frontend_id='w10_fit1_gap50', repeat_id=repeat,
        simulated_s=.8, measured_wall_s=.8*(repeat+1), measurement_start_s=.32,
        measurement_end_s=1.12, cost_scope=f.COST_SCOPE, complete_window='true',
        native_only='true', audit_included='false', primary_validated='true', instrumented='false',
        reference_trace_sha256='a'*64, physics_reference_sha256='d'*64,physics_reference_kind='public_recorded_fixed_archived_actions',
        source_snapshot_sha256='b'*64, policy_sha256='c'*64,
        host='fixture_host',gpu_uuid='GPU-fixture',equivalence_kind='raw_exact',functional_query_exact='true',
        raw_event_exact='true',actor_query_payload_sha256='f'*64,actor_query_count=28,equivalence_receipt_sha256='1'*64)
        for method in f.METHODS for rate in f.BLIND_RATES for repeat in range(3)]


class FigureContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='contract_', dir=Path(__file__).parent)
        self.root=Path(self.temp.name)
        self.selection=self.root/'FRONTEND_SELECTION.json'
        self.selection.write_text(json.dumps({'frontend_by_method':dict.fromkeys(f.METHODS,'w10_fit1_gap50')}),encoding='utf-8')

    def tearDown(self): self.temp.cleanup()

    def write_csv(self,path,rows,fields=None):
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(fields or rows[0]))
            writer.writeheader();writer.writerows(rows)

    def load_blind(self,stage=None):
        return f.validate_blind(stage or blind_stage(),self.selection,32,f.BLIND_RATES)

    def test_wilson_zero_and_all_and_mcnemar(self):
        lo,hi=f.wilson(0,32);self.assertAlmostEqual(lo,0);self.assertAlmostEqual(hi,.1071791982550706)
        lo2,hi2=f.wilson(32,32);self.assertAlmostEqual(hi2,1);self.assertAlmostEqual(lo2,1-hi)
        self.assertEqual(f.exact_mcnemar(0,0),1)
        self.assertEqual(f.exact_mcnemar(5,0),.0625)
        self.assertEqual(f.exact_mcnemar(6,0),.03125)
        self.assertEqual(f.exact_mcnemar(4,2),f.exact_mcnemar(2,4))
        self.assertEqual(f.holm([.01,.04,.03,.1]),[.04,.09,.09,.1])

    def test_empty_schema_has_no_observations(self):
        path=self.root/'empty.csv';self.write_csv(path,[],f.EPISODE_FIELDS)
        with self.assertRaisesRegex(f.DataError,'no completed'):f.csv_rows(path,f.EPISODE_FIELDS)
        output=self.root/'must_not_exist'
        result=subprocess.run([sys.executable,str(Path(f.__file__)),'--validation',str(self.root),'--output',str(output)],capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('NOT_READY',result.stderr)
        self.assertFalse(output.exists())

    def test_complete_symmetric_crossover_and_missing_duplicate(self):
        stage=crossover();summary=f.validate_crossover(stage,f.DEFAULT_FRONTENDS,4)
        self.assertEqual(len(summary['summaries']),12)
        self.assertEqual(len(stage['rows']),48)
        with self.assertRaises(f.DataError):f.validate_crossover(dict(rows=stage['rows'][:-1]),f.DEFAULT_FRONTENDS,4)
        with self.assertRaises(f.DataError):f.validate_crossover(dict(rows=stage['rows']+[stage['rows'][0]]),f.DEFAULT_FRONTENDS,4)

    def test_full_blind_keeps_32_pairs_not_128(self):
        result=self.load_blind()
        self.assertEqual(len(result['blocks']),32)
        self.assertEqual(len(result['summaries']),8)
        self.assertEqual([r['n'] for r in result['references']],[32,32])
        for row in result['paired']:
            self.assertEqual(row['paired_blocks'],32)
            self.assertEqual(row['ours_only'],16)
            self.assertEqual(row['v2e_only'],0)

    def test_blind_missing_cell_or_per_rate_frontend_tuning_rejected(self):
        stage=blind_stage();stage['rows'].pop()
        with self.assertRaises(f.DataError):self.load_blind(stage)
        stage=blind_stage();stage['rows'][0]['frontend_id']='w0_fit1_gap100'
        with self.assertRaisesRegex(f.DataError,'Blind frontend changes'):self.load_blind(stage)
        with self.assertRaisesRegex(f.DataError,'References must appear once'):
            f.episode(actor(1,25.,'init_only'))

    def test_block_bootstrap_retains_cross_rate_identity(self):
        result=f.block_bootstrap(self.load_blind(),1000,20261109)
        self.assertEqual(result['independent_blocks'],32)
        self.assertTrue(result['never_pool_rates_as_independent'])
        intervals={(r['percentile95_low'],r['percentile95_high']) for r in result['per_rate']}
        self.assertEqual(len(intervals),1)

    def test_cost_exact_fixed_window_and_independent_wall(self):
        path=self.root/'cost.csv';self.write_csv(path,cost_rows())
        result=f.validate_cost(path,self.load_blind())
        self.assertEqual(len(result['accepted']),24)
        self.assertTrue(all(r['repeats']==3 for r in result['summaries']))
        self.assertTrue(all(abs(r['mean_wall_s_per_sim_s']-2)<1e-12 for r in result['summaries']))
        self.assertFalse(result['episode_wall_time_used'])

    def test_cost_partial_window_mixed_hardware_and_repeats_rejected(self):
        path=self.root/'cost.csv'
        for field,value in [('complete_window','false'),('measurement_end_s',1.0),('gpu_uuid','GPU-other'),
            ('physics_reference_sha256','e'*64),('physics_reference_kind','legacy_utility_rollout')]:
            rows=cost_rows();rows[0][field]=value;self.write_csv(path,rows)
            with self.assertRaises(f.DataError,msg=field):f.validate_cost(path,self.load_blind())
        rows=cost_rows()[:-1];self.write_csv(path,rows)
        with self.assertRaisesRegex(f.DataError,'cost repeats'):f.validate_cost(path,self.load_blind())

    def test_audited_cost_cannot_replace_native_complete_grid(self):
        rows=cost_rows();rows[0]['audit_included']='true'
        path=self.root/'cost.csv';self.write_csv(path,rows)
        with self.assertRaises(f.DataError):f.validate_cost(path,self.load_blind())
        rows=cost_rows();rows.append(dict(rows[0],audit_included='true'))
        self.write_csv(path,rows);result=f.validate_cost(path,self.load_blind())
        self.assertEqual(len(result['excluded']),1)
        self.assertEqual(result['excluded'][0]['reason'],'audit_or_instrumentation_workload')

    def test_cost_v4_functional_contract_and_repeat_fingerprints(self):
        path=self.root/'cost_v4.csv';rows=cost_rows()
        for row in rows:row.update(equivalence_kind='task_functional',raw_event_exact='false')
        self.write_csv(path,rows);result=f.validate_cost(path,self.load_blind())
        self.assertEqual(result['equivalence_kind'],'task_functional')
        self.assertTrue(all(not r['all_repeats_raw_event_exact'] for r in result['summaries']))
        rows[0]['functional_query_exact']='false';self.write_csv(path,rows)
        with self.assertRaises(f.DataError):f.validate_cost(path,self.load_blind())
        rows[0]['functional_query_exact']='true';rows[0]['actor_query_payload_sha256']='2'*64;self.write_csv(path,rows)
        with self.assertRaisesRegex(f.DataError,'actor-query fingerprint'):f.validate_cost(path,self.load_blind())
        rows[0]['actor_query_payload_sha256']='f'*64;rows[0].update(equivalence_kind='raw_exact',raw_event_exact='true');self.write_csv(path,rows)
        with self.assertRaisesRegex(f.DataError,'cannot mix'):f.validate_cost(path,self.load_blind())

    def test_completion_receipt_and_csv_json_integrity(self):
        rows=crossover()['rows'];data=dict(metadata=dict(stage='validation',actual_planned_actors=48),rows=rows)
        result_path=self.root/'FAIR_STAGE_RESULTS.json';result_path.write_text(json.dumps(data),encoding='utf-8')
        self.write_csv(self.root/'FAIR_EPISODES.csv',rows)
        marker=dict(status='PASS',all_actual_jobs_normal_exit=True,all_task_failures_retained=True,
            actual_episodes=48,results_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
        (self.root/'FAIR_COLLECTION_COMPLETE.json').write_text(json.dumps(marker),encoding='utf-8')
        self.assertEqual(len(f.load_stage(self.root,'validation')['rows']),48)
        rows[0]['verified_catch']=True;self.write_csv(self.root/'FAIR_EPISODES.csv',rows)
        with self.assertRaises(f.DataError):f.load_stage(self.root,'validation')

    def test_rendering_functions_only_in_disposable_contract_directory(self):
        # Software QA only. TemporaryDirectory deletes every fixture/plot below.
        stage=crossover();stage['inputs']=[self.selection]
        validation=f.validate_crossover(stage,f.DEFAULT_FRONTENDS,4)
        blind=self.load_blind();blind_stage_input=dict(inputs=[self.selection])
        path=self.root/'cost.csv';self.write_csv(path,cost_rows())
        cost=f.validate_cost(path,blind)
        f.figure_crossover(stage,validation,self.root)
        f.figure_blind(blind_stage_input,blind,self.root,None)
        f.figure_cost(blind_stage_input,blind,cost,path,self.root)
        for prefix in ['frontend_crossover','blind_success_vs_rate','blind_success_vs_cost']:
            self.assertTrue((self.root/(prefix+'.png')).read_bytes().startswith(b'\x89PNG'))
            self.assertTrue((self.root/(prefix+'.pdf')).read_bytes().startswith(b'%PDF'))
            manifest=json.loads((self.root/(prefix+'_MANIFEST.json')).read_text(encoding='utf-8'))
            self.assertEqual(manifest['code_sha256'],f.sha(f.__file__))

    def test_retained_oom_history_only_passes_with_closed_raw_audited_recovery(self):
        rows=crossover()['rows'];data=dict(metadata=dict(stage='validation',actual_planned_actors=48),rows=rows)
        f.write_csv(self.root/'FAIR_EPISODES.csv',rows)
        (self.root/'FAIR_STAGE_RESULTS.json').write_text(json.dumps(data),encoding='utf-8')
        protocol=dict(cases=[dict(shard=0),dict(shard=1)])
        (self.root/'EVALUATION_PROTOCOL.json').write_text(json.dumps(protocol),encoding='utf-8')
        failed=[dict(worker_id='old',archived_failed_claims=[dict(shard=0)])]
        recovery=dict(schema='fair_workqueue_infra_recovery_v1',reason='cuda_out_of_memory',
            no_algorithm_scene_seed_configuration_changed=True,registered_utc='2026-11-08T00:00:00+00:00',
            protocol_sha256=f.sha(self.root/'EVALUATION_PROTOCOL.json'),historical_failed_workers=failed,replacement_worker_ids=['new'])
        (self.root/'INFRASTRUCTURE_RECOVERY.json').write_text(json.dumps(recovery),encoding='utf-8')
        declared=dict(recovery,sha256=f.sha(self.root/'INFRASTRUCTURE_RECOVERY.json'))
        scheduling=dict(mode='finite_registered_workqueue',infrastructure_recovery=declared,
            registered_workers=['old','new'],worker_job_receipts=[dict(worker_id='old',result='old.result',result_sha256='a'*64),
                dict(worker_id='new',result='new.result',result_sha256='b'*64)],
            completed_claim_receipts=[dict(shard=0,worker_id='new'),dict(shard=1,worker_id='new')])
        receipt=dict(status='PASS',valid_completed_unique_episodes=48,actual_planned_actors=48,
            frozen_source_policy_profile_hashes_verified=True,all_task_failures_retained=True,
            selected_normal_exit_job_receipts=[dict(path='new.result',sha256='b'*64)],all_shards_checked=[0,1],scheduling_audit=scheduling)
        audit=dict(status='PASS',episodes=48,raw_event_contract_and_packet_counts_pass=True,all_defined_successes_and_failures_retained=True)
        marker=dict(status='PASS',actual_episodes=48,all_actual_jobs_normal_exit=False,all_final_actor_attempts_normal_exit=True,
            historical_infrastructure_failures_retained=True,all_task_failures_retained=True,
            results_sha256=f.sha(self.root/'FAIR_STAGE_RESULTS.json'))
        def reseal():
            (self.root/'DATA_COMPLETION_RECEIPT.json').write_text(json.dumps(receipt),encoding='utf-8')
            (self.root/'FINAL_AUDIT.json').write_text(json.dumps(audit),encoding='utf-8')
            names=['DATA_COMPLETION_RECEIPT.json','FINAL_AUDIT.json','EVALUATION_PROTOCOL.json',
                'INFRASTRUCTURE_RECOVERY.json','FAIR_STAGE_RESULTS.json','FAIR_EPISODES.csv']
            (self.root/'SHA256SUMS').write_text(''.join(f'{f.sha(self.root/name)}  {name}\n' for name in names),encoding='utf-8')
            marker.update(raw_manifest_sha256=f.sha(self.root/'SHA256SUMS'),final_audit_sha256=f.sha(self.root/'FINAL_AUDIT.json'))
            (self.root/'FAIR_COLLECTION_COMPLETE.json').write_text(json.dumps(marker),encoding='utf-8')
        reseal();loaded=f.load_stage(self.root,'validation')
        self.assertEqual(loaded['completion']['mode'],'closed_registered_oom_recovery')
        self.assertEqual(loaded['completion']['closed_shards'],2)
        # Merely changing "final normal" is insufficient if a claim is missing.
        scheduling['completed_claim_receipts'].pop();reseal()
        with self.assertRaisesRegex(f.DataError,'close every registered shard'):f.load_stage(self.root,'validation')
        scheduling['completed_claim_receipts'].append(dict(shard=1,worker_id='new'))
        audit['raw_event_contract_and_packet_counts_pass']=False;reseal()
        with self.assertRaisesRegex(f.DataError,'complete raw audit'):f.load_stage(self.root,'validation')
        audit['raw_event_contract_and_packet_counts_pass']=True
        marker['all_final_actor_attempts_normal_exit']=False;reseal()
        with self.assertRaisesRegex(f.DataError,'unresolved infrastructure'):f.load_stage(self.root,'validation')


if __name__=='__main__': unittest.main()
