"""Plan/evidence tests only. No synthetic scene demo is saved or delivered."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
sys.dont_write_bytecode=True
import make_blind_demo as media


def row(seed,source,success=True,initialized=True):
    return dict(verified_catch=success,initialization_valid=initialized,method=source,seed=seed)


def mini_blind():
    blocks=[('block_a',900),('block_b',100),('block_c',800),('block_d',200),('block_e',700),('block_f',300)]
    indices={b:i+1 for i,b in enumerate(blocks)};indexed={}
    outcomes=[(True,False),(False,True),(True,True),(False,False),(False,True),(False,False)]
    for rate in media.fair.BLIND_RATES:
        for i,(b,(a,v)) in enumerate(zip(blocks,outcomes)):
            indexed[b,rate,'ours']=row(b[1],'ours',a,initialized=i!=4)
            indexed[b,rate,'v2e']=row(b[1],'v2e',v)
    return dict(blocks=blocks,rates=media.fair.BLIND_RATES,indexed=indexed),indices


class BlindMediaContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='contract_',dir=Path(__file__).parent)
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_fixed_minimum_registered_block_not_min_seed(self):
        blind,indices=mini_blind();pairs,decisions=media.representative_pairs(blind,indices)
        self.assertEqual(len(pairs),20)
        for decision in decisions:
            expected={'ours_only':1,'v2e_only':2,'both_success':3,'both_fail':4,'initialization_failure':5}[decision['category']]
            self.assertEqual(decision['selected_block_index'],expected)
        self.assertEqual(pairs[0]['seed'],900) # Minimum index, not minimum seed100.
        self.assertEqual(sum('v2e_only' in p['selection_categories'] for p in pairs),4)

    def test_overlapping_categories_deduplicated_and_absent_explicit(self):
        blind,indices=mini_blind()
        for rate in blind['rates']:
            for b in blind['blocks']:
                blind['indexed'][b,rate,'ours']['verified_catch']=False
                blind['indexed'][b,rate,'v2e']['verified_catch']=False
            blind['indexed'][blind['blocks'][0],rate,'ours']['initialization_valid']=False
        pairs,decisions=media.representative_pairs(blind,indices)
        self.assertEqual(len(pairs),4)
        self.assertTrue(all(p['selection_categories']==['both_fail','initialization_failure'] for p in pairs))
        absent=[d for d in decisions if d['category']=='v2e_only']
        self.assertTrue(all(d['selected_block_index'] is None and d['eligible_blocks']==0 for d in absent))

    def test_safe_paths_only_and_portable_local_rebase(self):
        self.assertEqual(media.remote_child('/actual/actor','episode_000/genesis.mp4'),'/actual/actor/episode_000/genesis.mp4')
        for root,child in [('relative/actor','video.mp4'),('/actual/actor','../../video.mp4'),('/actual/../actor','video.mp4')]:
            with self.assertRaises(media.fair.DataError):media.remote_child(root,child)
        with self.assertRaises(media.fair.DataError):media.local_child(self.root,'../outside.mp4')

    def test_partial_collector_creates_no_plan_or_demo(self):
        output=self.root/'must_not_exist'
        with self.assertRaises(media.fair.DataError):media.prepare_plan(self.root,self.root/'selection.json',self.root/'protocol.json',output)
        self.assertFalse(output.exists())

    def test_seal_and_transfer_receipt_checks_actual_bytes(self):
        path=self.root/'DEMO_DOWNLOAD_PLAN.json';artifact=dict(artifact_id='one',remote='/actual/a',local_relative='assets/a',kind='transfer_contract')
        plan=dict(schema=media.SCHEMA,selected_pair_count=0,pairs=[],artifacts=[artifact])
        media.write_json(path,plan);media.write_json(self.root/'PLAN_SHA256.json',dict(plan_sha256=media.fair.sha(path)))
        self.assertEqual(media.read_plan(path)['artifacts'],[artifact])
        local=self.root/'assets/a';local.parent.mkdir();local.write_bytes(b'transfer hash contract, not a scene')
        receipts=[dict(remote='/actual/a',local=str(local),size=local.stat().st_size,sha256=media.fair.sha(local))]
        receipt_path=self.root/'receipt.json';media.write_json(receipt_path,receipts)
        self.assertEqual(len(media.validate_receipts(plan,self.root,receipt_path)),1)
        local.write_bytes(b'changed bytes')
        with self.assertRaises(media.fair.DataError):media.validate_receipts(plan,self.root,receipt_path)
        plan['title']='changed plan';media.write_json(path,plan)
        with self.assertRaises(media.fair.DataError):media.read_plan(path)

    def test_actual_frame_times_require_full_control_cadence(self):
        trace=[dict(t_before=1.2,t=1.24),dict(t_before=1.24,t=1.28)]
        times,start=media.elapsed_times(trace)
        self.assertAlmostEqual(start,1.2);self.assertAlmostEqual(times[-1],.08)
        with self.assertRaises(media.fair.DataError):media.elapsed_times([])
        with self.assertRaises(media.fair.DataError):media.elapsed_times([dict(t_before=0.,t=.08)])

    def test_complete_320_row_contract_yields_plan_but_no_scene_media(self):
        # Tabular software fixture only; no source frames or demo are invented.
        selection=self.root/'FRONTEND_SELECTION.json'
        media.write_json(selection,dict(frontend_by_method=dict.fromkeys(media.fair.METHODS,'w10_fit1_gap50')))
        rows=[];cases=[]
        for index in range(1,33):
            physical=f'blind_b{index:02d}';seed=1000+index
            for rate in media.fair.BLIND_RATES:
                for method in media.fair.METHODS:
                    success=(index%4 in (0,1)) if method=='ours' else (index%4 in (0,2))
                    initialized=not(index==3 and method=='ours')
                    rows.append(dict(physical_case_id=physical,seed=seed,endpoint_hz=rate,method=method,
                        frontend_id='w10_fit1_gap50',verified_catch=success,raw_success=success,
                        terminated=False,hand_contact_steps=int(success),final_contact=success,
                        initialization_valid=initialized,sim_time_s=1.5 if initialized else 0.,wall_s=3.,
                        output=f'/actual/actor/{physical}/{rate}/{method}'))
                    cases.append(dict(physical_case_id=physical,seed=seed,block_index=index))
            for method in media.fair.REFERENCES:
                rows.append(dict(physical_case_id=physical,seed=seed,endpoint_hz=None,method=method,
                    frontend_id='',verified_catch=False,raw_success=False,terminated=False,
                    hand_contact_steps=0,final_contact=False,initialization_valid=None if method=='state_reference' else True,
                    sim_time_s=1.5,wall_s=3.,output=f'/actual/actor/{physical}/{method}'))
                cases.append(dict(physical_case_id=physical,seed=seed,block_index=index))
        result_path=self.root/'FAIR_STAGE_RESULTS.json'
        media.write_json(result_path,dict(metadata=dict(stage='blind',actual_planned_actors=320),rows=rows))
        with (self.root/'FAIR_EPISODES.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        media.write_json(self.root/'FAIR_COLLECTION_COMPLETE.json',dict(status='PASS',actual_episodes=320,
            all_actual_jobs_normal_exit=True,all_task_failures_retained=True,results_sha256=media.fair.sha(result_path)))
        protocol=self.root/'EVALUATION_PROTOCOL.json'
        media.write_json(protocol,dict(cases=cases,frontend_selection_sha256=media.fair.sha(selection),checkpoint_sha256='a'*64))
        out=self.root/'planned_only';plan=media.prepare_plan(self.root,selection,protocol,out)
        self.assertEqual(plan['total_completed_actors'],320)
        self.assertLessEqual(plan['selected_pair_count'],20)
        self.assertTrue(any(d['category']=='v2e_only' and d['eligible_blocks']>0 for d in plan['decisions']))
        self.assertTrue(all(p['panels'][0]['method']=='ours' and p['panels'][1]['method']=='v2e' for p in plan['pairs']))
        self.assertTrue(all(p.suffix=='.json' for p in out.iterdir()))
        exported=json.loads((out/'DOWNLOAD_FETCH_PLAN.json').read_text(encoding='utf-8'))
        self.assertTrue(all(set(row)=={'remote','local'} for row in exported))


if __name__=='__main__':unittest.main()
