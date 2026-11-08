"""One CPU handoff simulation; no actual processes, SSH or GPU are started."""
from pathlib import Path
from unittest import mock
import json, tempfile, types, unittest
import checkpoint_blind_for_cost_v4 as s


def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))


class Handoff(unittest.TestCase):
 def test_complete_block_drain_cost_failure_and_resume_preserve_protocol_and_claims(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);blind=root/'blind32';queue=blind/'workqueue';res=root/'reservations';res.mkdir()
   write(root/'lock.json',{'fixture':True})
   spec=dict(code_commit=s.COMMIT,source=str(root/'source_cae1ab9e7f48'),frontend_selection_file=str(root/'lock.json'),
    frontend_selection_sha256=s.sha(root/'lock.json'),cases=[dict(shard=i//10,method_order=['ours']) for i in range(320)])
   write(blind/'EVALUATION_PROTOCOL.json',spec);digest=s.sha(blind/'EVALUATION_PROTOCOL.json')
   write(queue/'PLAN.json',dict(ordered_shards=list(range(32)),protocol_sha256=digest))
   oldhashes={}
   for gpu in [0,1]:
    name='catchit_fair_blind32_g1_gpu'+str(gpu)+'_blind_20261108b';worker='g1_gpu'+str(gpu)+'_blind'
    write(blind/(name+'.job.json'),dict(name=name,gpu_candidates=[gpu],env={'FIXTURE':'unchanged'},command=['python','fair_workqueue_worker.py','--protocol',str(blind/'EVALUATION_PROTOCOL.json'),'--worker-id',worker]))
    write(blind/(name+'.result.json'),dict(status='completed',exit_code=0))
    allocation=dict(gpu_index=gpu,pid=100+gpu,process_start_ticks='fixture',gpu_uuid='fixture'+str(gpu))
    write(blind/(name+'.allocation.json'),allocation)
    write(res/(name+'_gpu'+str(gpu)+'.json'),dict(allocation,owner=s.OWNER,run_id=name,output_path=str(blind),gpu_reserved=False,status='released_after_fixture'))
    claim=dict(shard=gpu,worker_id=worker,status='completed',exit_code=0,protocol_sha256=digest)
    write(queue/'claims'/('shard%03d.json'%gpu),claim);write(queue/(worker+'.progress.json'),[claim]);write(queue/(worker+'.finished.json'),dict(status='PASS',shards=1))
    for p in [blind/(name+'.job.json'),blind/(name+'.result.json'),queue/(worker+'.progress.json'),queue/'claims'/('shard%03d.json'%gpu)]:oldhashes[p]=s.sha(p)
   affinity=[];events=[]
   def process(command,**kw):
    self.assertEqual(affinity[-1],list(range(128)))
    jobfile=Path(command[-1]);job=s.read(jobfile);events.append(job['name'])
    self.assertEqual(job['scheduling_checkpoint_sha256'],s.sha(blind/'SCHEDULING_COST_V4_CHECKPOINT.json'))
    if 'costv4_resume1' in job['name']:
     self.assertTrue((blind/'catchit_fair_blind32_g1_gpu0_blind_costv4_resume1_20261108b.job.json').exists())
     self.assertEqual(len(list((queue/'claims').glob('shard*.json'))),2)
    if 'gpu0_blind_costv4' in job['name']:
     self.assertEqual(s.read(root/'BLIND_COST_V4_CHECKPOINT_COST_RECEIPT.json')['resource_release']['status'],'RELEASE_CONFIRMED')
    def wait():
     self.assertIn('cost_v4_gpu0',job['name']);self.assertFalse((root/'cost_v4_matched').exists())
     write(jobfile.parent/(job['name']+'.result.json'),dict(status='failed',exit_code=1))
     allocation=dict(gpu_index=0,pid=777,process_start_ticks='fixture',gpu_uuid='costfixture')
     write(jobfile.parent/(job['name']+'.allocation.json'),allocation)
     write(res/(job['name']+'_gpu0.json'),dict(allocation,owner=s.OWNER,run_id=job['name'],output_path=str(jobfile.parent),gpu_reserved=False,status='released_after_fixture'))
     return 0
    return types.SimpleNamespace(pid=1000+len(events),wait=wait)
   with (mock.patch.multiple(s,ROOT=root,RESERVATIONS=res),mock.patch.object(s,'start_ticks',return_value='fixture'),
    mock.patch.object(s.os,'cpu_count',return_value=128),mock.patch.object(s.os,'sched_getaffinity',return_value=set(range(128)),create=True),
    mock.patch.object(s.os,'sched_setaffinity',side_effect=lambda pid,v:affinity.append(list(v)),create=True),
    mock.patch.object(s.subprocess,'Popen',side_effect=process)):
    s.main()
   self.assertEqual(s.sha(blind/'EVALUATION_PROTOCOL.json'),digest)
   for p,oldsha in oldhashes.items():self.assertEqual(s.sha(p),oldsha)
   self.assertEqual(len(list((blind/'scheduling_checkpoints/cost_v4/held_claims').glob('*.json'))),30)
   self.assertIn('gpu1_blind_costv4',events[0]);self.assertIn('cost_v4_gpu0',events[1]);self.assertIn('gpu0_blind_costv4',events[2])
   self.assertEqual(s.read(root/'BLIND_COST_V4_CHECKPOINT_RECEIPT.json')['status'],'HANDED_BACK')
   self.assertEqual(s.read(root/'BLIND_COST_V4_CHECKPOINT_RECEIPT.json')['cost_status'],'FAILED')


if __name__=='__main__':unittest.main(verbosity=2)
