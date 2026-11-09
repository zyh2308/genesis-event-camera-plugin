import copy,hashlib,json,tempfile,unittest,signal,subprocess,sys
from unittest.mock import Mock,patch,call
from pathlib import Path
import continue_after_events as L

def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
def producer():
    argv=[str(L.PYTHON),'-B','-u',str(L.PREP),'events','--registration',str(L.REG),'--registration-sha256',L.REG_SHA,'--physical-receipt-sha256',L.PHYSICAL_SHA]
    spec=dict(job_id=L.JOB,argv=argv,output_dir=str(L.R/'recovery_20261105_mainline/jobs'/L.JOB),phase='pose_advantage_events')
    return dict(spec,status='complete',exit_code=0),spec

def full_inventory(root):
    rids=[f'validation_{i:03d}' for i in range(32)];physical_files=[];banks={}
    for condition in L.CONDITIONS:
        bank=root/condition/'pose_bank';banks[condition]=str(bank)
        for rid in rids:
            p=bank/'physical'/(rid+'.h5');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((condition+rid).encode())
            if condition=='normal':physical_files.append(L.record(p))
        dump(bank/'bank_manifest.json',dict(rollouts=rids,real_data_used=False))
        for method in L.METHODS:
            items=[]
            for rid in rids:
                p=bank/'events'/method/(rid+'.npy');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((condition+method+rid).encode());items.append(dict(rollout_id=rid,events_sha256=L.sha(p)))
            dump(bank/'events'/method/'manifest.json',dict(rollouts=items,real_data_used=False,physical_manifest_sha256=L.sha(bank/'bank_manifest.json')))
    physical=dict(schema='pose_confirm32_physical_v1',status='complete',optimizer_updates=0,real_data_used=False,count=32,physical_seed=202611090137,cuda_initialized=False,
        registration_sha256=L.REG_SHA,physical_files=physical_files,bank_manifest=L.record(root/'normal/pose_bank/bank_manifest.json'))
    dump(root/'physical_completion.json',physical)
    artifacts=[L.record(p) for c in L.CONDITIONS for p in (root/c/'pose_bank').rglob('*') if p.is_file()]
    event=dict(schema='pose_confirm32_events_v1',status='complete',optimizer_updates=0,real_data_used=False,registration_sha256=L.REG_SHA,
        physical_receipt=L.record(root/'physical_completion.json'),banks=banks,artifacts=artifacts)
    return event,physical

class ContinuationContracts(unittest.TestCase):
    def test_actual_archived_contract_physical_and_job_spec_bindings(self):
        archive=Path(__file__).resolve().parent.parent
        lane=L.R/'gpu4'
        def observed(relative,remote):
            local=archive/relative
            return local if local.is_file() else remote
        reg=observed('actual_evidence/PREPARATION_CONTRACT.json',L.REG)
        physical=observed('actual_evidence/physical_completion.json',L.PHYSICAL)
        spec_path=observed('actual_evidence_later/queue/'+L.JOB+'.running_spec.json',lane/(L.JOB+'.running_spec.json'))
        for p,digest in ((reg,L.REG_SHA),(physical,L.PHYSICAL_SHA),(spec_path,L.SPEC_SHA)):
            self.assertEqual(L.verify(dict(path=str(p),sha256=digest)),p)
        c=L.read(reg);receipt=L.read(physical);spec=L.read(spec_path)
        self.assertEqual((c['root'],c['campaign'],receipt['registration_sha256']),(str(L.ROOT),str(L.R),L.REG_SHA))
        self.assertEqual((c['primary_arm'],c['primary_conditions'],c['per_condition_ci_level']),('C1_RAW',['normal','half'],.975))
        self.assertEqual((receipt['count'],len(receipt['physical_files'])),(32,32))
        self.assertEqual(spec['source_sha256'][str(L.PREP)],L.PREP_SHA)
        self.assertEqual(spec['env']['CUDA_VISIBLE_DEVICES'],L.GPU4)
        if reg==L.REG:
            done_path=lane/(L.JOB+'.done.json')
            L.verify(dict(path=str(done_path),sha256=L.DONE_SHA))
            L.verify(dict(path=str(L.EVENT),sha256=L.EVENT_SHA))
            done=L.read(done_path);event=L.read(L.EVENT)
            self.assertEqual(event['physical_receipt'],L.record(L.PHYSICAL))
            self.assertEqual(event['registration_sha256'],L.REG_SHA)
        else:
            done=dict(spec,status='complete',exit_code=0)
        self.assertEqual(L.verify_job(done,spec),L.PREP)

    def test_frozen_evaluation_manifest_and_leaf_hashes(self):
        archive=Path(__file__).resolve().parent.parent
        sources=archive if (archive/'evaluation_source_manifest.json').is_file() else L.EDIR
        manifest=sources/'evaluation_source_manifest.json'
        self.assertEqual(L.sha(manifest),L.EMANIFEST_SHA)
        for name,digest in L.read(manifest).items():L.verify(dict(path=str(sources/name),sha256=digest))
        for digest in (L.REG_SHA,L.PHYSICAL_SHA,L.EVENT_SHA,L.DONE_SHA,L.SPEC_SHA,L.EMANIFEST_SHA,L.PREP_SHA):
            self.assertEqual(len(digest),64)
            self.assertTrue(all(ch in '0123456789abcdef' for ch in digest))

    def test_malformed_sha_reports_pin_error_before_file_check(self):
        with self.assertRaisesRegex(ValueError,'Invalid SHA256'):
            L.verify(dict(path='/missing/fixture',sha256='a'*65))

    def test_parent_start_time_must_match_pinned_instance(self):
        for start_ticks in (L.PARENT_START_TICKS,L.PARENT_START_TICKS+1):
            fields=['S']+['0']*18+[str(start_ticks)]
            stat=f'{L.PARENT} (python (queue)) '+' '.join(fields)
            with patch.object(L.os,'getppid',return_value=L.PARENT),patch.object(Path,'read_text',return_value=stat),patch.object(L,'PYTHON',Path(sys.executable)),patch.dict(L.os.environ,{'CUDA_VISIBLE_DEVICES':L.GPU4}):
                if start_ticks==L.PARENT_START_TICKS:L.verify_queue_parent()
                else:
                    with self.assertRaisesRegex(ValueError,'process identity changed'):L.verify_queue_parent()

    def test_exact_done_zero_and_argv(self):
        d,s=producer();self.assertEqual(L.verify_job(d,s),L.PREP)
        d['argv']=d['argv'][0:1]+d['argv'][3:];s['argv']=d['argv'];self.assertEqual(L.verify_job(d,s),L.PREP)
    def test_failed_incomplete_wrong_id_or_spec_refused(self):
        for key,value in [('exit_code',1),('status','running'),('job_id','171_other_job')]:
            d,s=producer();d[key]=value
            with self.assertRaises(ValueError):L.verify_job(d,s)
        d,s=producer();s=copy.deepcopy(s);s['argv'].append('--real-on-pass')
        with self.assertRaises(ValueError):L.verify_job(d,s)
    def test_changed_contract_or_physical_or_source_refused(self):
        for old,new in [(L.REG_SHA,'0'*64),(L.PHYSICAL_SHA,'1'*64),(str(L.PREP),'/tmp/prepare_confirmation.py')]:
            d,s=producer();d['argv']=[new if x==old else x for x in d['argv']];s['argv']=d['argv']
            with self.assertRaises(ValueError):L.verify_job(d,s)
    def test_complete_all32_six_banks_rehashed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);e,p=full_inventory(root);result=L.verify_event_inventory(e,p,root)
            self.assertEqual(result['required_event_streams'],192);self.assertEqual(result['physical_rollouts'],32)
    def test_missing_one_event_or_byte_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);e,p=full_inventory(root);bad=copy.deepcopy(e);bad['artifacts'].pop()
            with self.assertRaises(ValueError):L.verify_event_inventory(bad,p,root)
            fp=root/'half/pose_bank/events/v2e/validation_031.npy';fp.write_bytes(b'mutated')
            with self.assertRaises(ValueError):L.verify_event_inventory(e,p,root)
    def test_bad_physical_binding_or_real_flag_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);e,p=full_inventory(root)
            for key,value in [('real_data_used',True),('registration_sha256','0'*64),('physical_receipt',dict(path='wrong',sha256='0'*64))]:
                bad=copy.deepcopy(e);bad[key]=value
                with self.assertRaises(ValueError):L.verify_event_inventory(bad,p,root)
    def test_eval_command_is_fixed_c1_source_and_actual_receipt(self):
        digest='a'*64;argv=L.build_argv(digest)
        self.assertEqual(argv,[str(L.PYTHON),'-B','-u',str(L.EDIR/'evaluate_confirmation.py'),'--registration',str(L.REG),'--registration-sha256',L.REG_SHA,'--event-receipt-sha256',digest,'--out',str(L.ROOT/'evaluation')])
        self.assertNotIn('--real-on-pass',argv)
        with self.assertRaises(ValueError):L.build_argv('not-a-hash')
    def test_cleanup_refuses_parent_or_foreign_process_group(self):
        child=Mock(pid=500001);child.poll.return_value=None
        with patch.object(L.os,'getpgid',return_value=L.PARENT),patch.object(L.os,'killpg') as kill:
            with self.assertRaises(ValueError):L.cleanup_child(child)
            kill.assert_not_called()
    def test_timeout_cleanup_only_terminates_own_child_group(self):
        child=Mock(pid=500001);child.poll.return_value=None;child.wait.side_effect=[subprocess.TimeoutExpired(['fixture'],10),0]
        with patch.object(L.os,'getpgid',return_value=child.pid),patch.object(L.os,'killpg') as kill:
            L.cleanup_child(child);self.assertEqual(kill.call_args_list,[call(child.pid,signal.SIGTERM),call(child.pid,signal.SIGKILL)])
    def test_finished_child_never_receives_a_signal(self):
        child=Mock(pid=500001);child.poll.return_value=0
        with patch.object(L.os,'killpg') as kill:L.cleanup_child(child);kill.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
