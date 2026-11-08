"""Temporary schema fixture only; no actual study outcomes are read or run."""
from pathlib import Path
import csv, json, tempfile, unittest
import write_blind_summary as s


def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))


class SummaryContract(unittest.TestCase):
 def test_complete320_summary_keeps_references_once_and_rejects_missing_actor(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);stage=root/'stage';figures=root/'figures';stage.mkdir();figures.mkdir();rows=[];cases=[]
   env={'CATCHIT_EVENT_OBSERVATION_WINDOW_MS':'10','CATCHIT_EVENT_FLIGHT_FIT':'1','CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS':'50'}
   lock=dict(status='LOCKED',locked_utc='2026-11-09T00:00:00+00:00',no_blind_outcome_used=True,
    single_global_configuration_per_source=True,actual_primary_episodes=48,physical_seed_blocks=4,frontend_by_method={'ours':env,'v2e':env})
   write(stage/'FRONTEND_SELECTION.json',lock)
   for seed in range(1001,1033):
    for rate,method in [(r,m) for r in s.RATES for m in s.f.METHODS]+[(None,m) for m in s.f.REFERENCES]:
     success=method=='state_reference';physical='fixture_'+str(seed)
     row=dict(case_id=physical+'_'+str(rate)+'_'+method,physical_case_id=physical,seed=seed,endpoint_hz=rate,method=method,
      frontend_id='w10_fit1_gap50' if method in s.f.METHODS else 'not_applicable_reference',verified_catch=success,raw_success=success,
      terminated=False,hand_contact_steps=int(success),final_contact=success,initialization_valid=None if method=='state_reference' else True,
      sim_time_s=1.5,wall_s=1.,output='temporary_fixture_not_experiment',failure_class='none' if success else 'no_contact')
     rows.append(row);cases.append(dict(physical_case_id=physical,seed=seed,endpoint_hz=rate or 25,method_order=[method]))
   write(stage/'EVALUATION_PROTOCOL.json',dict(cases=cases,frontend_selection_file='/nonexistent/server/lock.json',frontend_selection_sha256=s.f.sha(stage/'FRONTEND_SELECTION.json')))
   write(stage/'FAIR_STAGE_RESULTS.json',dict(status='PASS',complete=True,metadata={'stage':'blind','actual_planned_actors':320},rows=rows))
   s.f.write_csv(stage/'FAIR_EPISODES.csv',rows)
   write(stage/'FAIR_COLLECTION_COMPLETE.json',dict(status='PASS',actual_episodes=320,all_task_failures_retained=True,all_actual_jobs_normal_exit=True,results_sha256=s.f.sha(stage/'FAIR_STAGE_RESULTS.json')))
   actual=s.f.validate_blind(s.f.load_stage(stage,'blind'),stage/'FRONTEND_SELECTION.json',32,s.f.BLIND_RATES)
   s.f.write_csv(figures/'blind_paired_by_rate.csv',actual['paired'])
   write(figures/'blind_success_vs_rate_MANIFEST.json',dict(inputs=[dict(path=str(stage/n),sha256=s.f.sha(stage/n)) for n in ['FAIR_STAGE_RESULTS.json','FAIR_COLLECTION_COMPLETE.json']],
    outputs=[dict(path=str(figures/'blind_paired_by_rate.csv'),sha256=s.f.sha(figures/'blind_paired_by_rate.csv'))]))
   output=root/'summary';result=s.write_summary(stage,figures,output)
   self.assertEqual(result['actual_actors'],320);self.assertEqual(len(result['per_rate']),4)
   self.assertEqual([r['n'] for r in result['references']],[32,32])
   self.assertEqual(result['cost']['status'],'NOT_AVAILABLE')
   text=(output/'README.md').read_text(encoding='utf-8');self.assertIn('尚无有效成本曲线',text);self.assertIn('没有确立',text)
   with (output/'逐条结果.csv').open(encoding='utf-8-sig',newline='') as stream:self.assertEqual(len(list(csv.DictReader(stream))),320)
   s.f.write_csv(stage/'FAIR_EPISODES.csv',rows[:-1])
   with self.assertRaises(s.f.DataError):s.write_summary(stage,figures,root/'missing_actor')
   self.assertFalse((root/'missing_actor').exists())
   (stage/'FAIR_COLLECTION_COMPLETE.json').unlink()
   with self.assertRaises(s.f.DataError):s.write_summary(stage,figures,root/'incomplete')
   self.assertFalse((root/'incomplete').exists())


if __name__=='__main__':unittest.main(verbosity=2)
