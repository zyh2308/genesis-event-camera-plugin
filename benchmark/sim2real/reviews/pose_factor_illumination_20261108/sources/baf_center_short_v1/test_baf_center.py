"""CPU fixtures only: actual bounded micro-updates, replay, restore, and gates."""
import contextlib,copy,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import baf
import run_baf_center as W


class TinyCenter(nn.Module):
    def __init__(self):
        super().__init__();self.features=nn.Conv2d(3,4,1);self.center=nn.Conv2d(4,1,1);self.size=nn.Linear(4,2)
        y,x=torch.meshgrid((torch.arange(4)+.5)/4,(torch.arange(4)+.5)/4,indexing='ij');self.register_buffer('grid',torch.stack((x,y),-1).reshape(-1,2))
    def forward(self,x):
        f=self.features(x).tanh();logits=self.center(f).flatten(1);uv=logits.softmax(1)@self.grid
        size,shape=self.size(f.mean((2,3))).unbind(1);return logits,uv,size,shape


def affine(x,y,rng):
    values=torch.as_tensor(rng.uniform(-1,1,(len(x),3)),dtype=x.dtype)
    zoom,shift=1+.15*values[:,0],.06*values[:,1:];theta=torch.zeros(len(x),2,3)
    theta[:,0,0]=theta[:,1,1]=1/zoom;theta[:,:,2]=-2*shift/zoom[:,None]
    target=y.clone();target[:,:2]=(target[:,:2]-.5)*zoom[:,None]+.5+shift;target[:,2:4]+=zoom.log()[:,None]
    return F.grid_sample(x,F.affine_grid(theta,x.shape,align_corners=False),align_corners=False),target


def matched(x,clean,y,rng):
    other=np.random.default_rng();other.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
    a,b=affine(x,y,rng);c,bb=affine(clean,y,other);assert torch.equal(b,bb);return a,c,b


def fake_native():
    def write(path,value):
        with Path(path).open('x') as f:json.dump(value,f,allow_nan=False)
    return SimpleNamespace(matched_affine=matched,shared=SimpleNamespace(METHODS=('eventgenesis','v2e'),affine=affine,write=write,sha=W.sha))


class BafCenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_actual_contract_rejects_direction_budget_loss_and_real_changes(self):
        c=W.read(Path(__file__).with_name('contract.json'));W.validate_contract(c)
        self.assertEqual(W.sha(Path(__file__).with_name('contract.json')),W.CONTRACT_SHA)
        for key,value in [('new_center_optimizer_updates',3456),('candidate_initialization','C1_finetune'),('normal_teacher_weight',0),('pair_weight',2),('filter_threshold_us',3000),('real_data_scoring_allowed',True)]:
            wrong=copy.deepcopy(c);wrong[key]=value
            with self.assertRaises(ValueError):W.validate_contract(wrong)
        self.assertEqual(W.sha(Path(__file__).with_name('pose_priority_policy.py')),W.POLICY_SHA)

    def test_full288_replay_is_input_independent_but_npz_label_exact(self):
        torch.manual_seed(12);x=torch.rand(1529,3,4,4);y=torch.rand(1529,9)
        aux=torch.cat((x*.7,x*.1));ay=torch.cat((y,y));clean=torch.cat((x,x));f=SimpleNamespace(shared=fake_native().shared,native=fake_native())
        before=[t.clone() for t in (x,y,aux,ay,clean)]
        h,rows=W.replay(f,x,y,aux,ay,clean,2)
        raw_h,raw_rows=W.replay(f,x*0+1,y,aux*0+.5,ay,clean*0+2,2)
        self.assertEqual(len(rows),288);self.assertEqual(h,raw_h);self.assertEqual(rows,raw_rows)
        self.assertEqual(rows[0]['normal_indices'],np.random.default_rng(71072).permutation(1529)[:32].tolist())
        altered=y.clone();altered[0,0]+=1e-5
        self.assertNotEqual(W.replay(f,x,altered,aux,ay,clean,2)[0],h)
        for a,b in zip((x,y,aux,ay,clean),before):self.assertTrue(torch.equal(a,b))

    def test_two_actual_center_updates_recover_adam_freeze_size_and_match_replay(self):
        torch.manual_seed(19);x=torch.rand(64,3,4,4);y=torch.rand(64,9)
        aux=torch.cat((x*.7,x*.1));ay=torch.cat((y,y));clean=torch.cat((x,x));state=copy.deepcopy(TinyCenter().state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            f=W.BoundedFitter(fake_native(),tmp,True);f.order=[(m,0,'physical_supervised','center') for m in ('eventgenesis','v2e')]
            expected_hash,draws=W.replay(f,x,y,aux,ay,clean,0,smoke=True)
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(2):
                    model,_=f.fit_center(SimpleNamespace(CenterScale=TinyCenter),state,x,y,aux,ay,0,True,clean,0)
                    self.assertTrue(any(not torch.equal(v,state[k]) for k,v in model.state_dict().items() if k.startswith('features.')))
                    self.assertTrue(all(torch.equal(v,state[k]) for k,v in model.state_dict().items() if k.startswith('size.')))
            for r in f.records:
                self.assertEqual(r['labels_sha256'],expected_hash);self.assertEqual(r['updates'],1)
                self.assertTrue(r['teacher_unchanged'] and r['independent_model_optimizer_rebuild'] and r['optimizer_state_equal'])
                tag=f"{r['method']}_seed0_physical_supervised_center.jsonl"
                self.assertEqual(W.trajectory(W.read_log(Path(tmp)/'training_evidence'/tag)),draws)

    def test_per_episode_adapter_preserves_order_and_limits_416_lifetime(self):
        sizes=[]
        def encode(t,x,y,p,start,end,width,height,count_scale):
            out=np.zeros((3,height,width),np.float32);inside=(t>start)&(t<=end)
            np.add.at(out[0],(y[inside],x[inside]),1);return out
        def adapter(bank,files,method,callback):
            self.assertEqual(len(files),1);rid=files[0].stem;t=np.array([.003,.003001,.012]);x=np.array([1,2,3]);y=np.array([1,1,3]);p=np.ones(3,np.uint8)
            return [dict(id=rid+f'/{i:03}',rid=rid,image=callback(t,x,y,p,a,b,4,4,8.),uv=np.zeros((9,2)),K=np.eye(3)) for i,(a,b) in enumerate(((.002,.004),(.01,.013)))]
        def tensors(rows,device):sizes.append(len(rows));return torch.as_tensor(np.stack([r['image'] for r in rows]))
        ctx=SimpleNamespace(encode=encode,fresh=SimpleNamespace(adapter=adapter),shared=SimpleNamespace(tensors=tensors,labels=lambda s,r,v,d:torch.zeros(len(r),9)),device=torch.device('cpu'),scale=None,vertices=None)
        result=W.episode_inputs(ctx,baf,Path('/synthetic'),['b','a'],'eventgenesis')
        self.assertEqual([r['id'] for r in result['rows']],['b/000','b/001','a/000','a/001'])
        self.assertEqual(sizes,[2,2,2,2]);self.assertEqual(result['filtered'][1].sum(),0)
        self.assertTrue(all('image' not in r for r in result['rows']))

    def test_stage36_keeps_per_seed_thresholds_and_does_not_claim_global_pass(self):
        def arm(epe,success):return dict(center_epe_px=epe,pose=dict(summary=dict(success=success)))
        results=[dict(method=m,seed=s,arms=dict(raw_C1={c:arm(10,.5) for c in ('clean','half','eighth')},baf_center={c:arm(9,.5) for c in ('clean','half','eighth')})) for m in ('eventgenesis','v2e') for s in (0,1,2)]
        gate=W.stage_gate(results);self.assertEqual(len(gate['checks']),36);self.assertTrue(gate['passed']);self.assertIn('not_global30',gate['scope'])
        results[0]['arms']['baf_center']['eighth']['center_epe_px']=10
        gate=W.stage_gate(results);self.assertFalse(gate['passed']);self.assertEqual(sum(r['passed'] for r in gate['checks']),35)
        results[0]['arms']['baf_center']['clean']['pose']['summary']['success']=.479
        self.assertEqual(sum(r['passed'] for r in W.stage_gate(results)['checks']),34)

    def test_smoke_schema_requires_two_rebuilt_updates_and_six_admissions(self):
        records=[dict(method=m,seed=0,branch='center',updates=1,new_optimizer_updates=1,teacher_unchanged=True,independent_model_optimizer_rebuild=True,optimizer_state_equal=True) for m in ('eventgenesis','v2e')]
        s=dict(schema='pose_baf_center_adaptation_v1',status='complete_synthetic_only',phase='smoke',new_optimizer_updates=2,contract_sha256=W.CONTRACT_SHA,source_manifest_sha256='manifest',real_scores_computed=False,canonical_candidate='fixed_baf_center',records=records,full288_admissions=[dict(full288_replay_exact=True) for _ in range(6)])
        W.verify_smoke(s,'manifest')
        for key in ('real_scores_computed','records','full288_admissions'):
            wrong=copy.deepcopy(s);wrong[key]=True if key=='real_scores_computed' else wrong[key][1:]
            with self.assertRaises(ValueError):W.verify_smoke(wrong,'manifest')

    def test_input_lineage_checks_actual_bytes_not_just_cached_source_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'events.npy';p.write_bytes(b'unchanged-event-input');r=W.record(p)
            W.verify_input_lineage([r])
            p.write_bytes(b'changedxx-event-input')
            with self.assertRaises(ValueError):W.verify_input_lineage([r])
            with self.assertRaises(ValueError):W.verify_input_lineage([])


if __name__=='__main__':unittest.main(verbosity=2)
