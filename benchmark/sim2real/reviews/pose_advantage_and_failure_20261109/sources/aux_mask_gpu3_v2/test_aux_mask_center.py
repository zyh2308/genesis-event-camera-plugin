"""CPU numerical and actual-update contracts; no server or real data."""
import copy,json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import aux_center_policy as A
import legacy_center_bridge as P
import native_paired_source as N
import pose_light_robustness as H
from pose_priority_policy import BoundedFitter,state_sha
from run_aux_mask_center import validate_contract,verify_smoke,stage_gates


class TinyCenter(nn.Module):
    def __init__(self):
        super().__init__();self.features=nn.Sequential(nn.Conv2d(3,4,3,stride=8,padding=1),nn.SiLU());self.center=nn.Conv2d(4,1,1);self.size=nn.Linear(8,2)
        y,x=torch.meshgrid((torch.arange(16)+.5)/16,(torch.arange(16)+.5)/16,indexing='ij');self.register_buffer('grid',torch.stack((x,y),-1).reshape(-1,2))
    def forward(self,x):
        f=self.features(x);logits=self.center(f).flatten(1);p=logits.softmax(1);v=f.flatten(2);local=(v*p[:,None]).sum(2);s=self.size(torch.cat((local,v.mean(2)),1))
        return logits,p@self.grid,s[:,0],s[:,1]


def tensors(n=33):
    g=torch.Generator().manual_seed(814);x=torch.rand(n,3,128,128,generator=g);aux=torch.cat((x*.5,x*.125));clean=torch.cat((x,x));y=torch.zeros(n,9);y[:,:2]=.5;y[:,2]=-2;y[:,3]=-.5;y[:,4]=1
    m=torch.zeros(n,1,128,128);m[:,:,48:66,50:70]=1
    return x,y,aux,torch.cat((y,y)),clean,m,torch.cat((m,m))


class AuxMaskContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_balanced_fractional_loss_and_absent_classes_are_explicit(self):
        logits=torch.zeros(3,1,2,2,dtype=torch.float64,requires_grad=True);m=torch.zeros_like(logits);m[0,0,0,0]=.01;m[1]=1
        loss,r=A.balanced_mask_bce(logits,m);expected=np.log(2)*(1+.5+.5)/3
        self.assertAlmostEqual(float(loss.detach()),expected);self.assertEqual(r['empty_foreground_samples'],1);self.assertEqual(r['empty_background_samples'],1)
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all());self.assertGreater(float(logits.grad[2].sum()),0);self.assertLess(float(logits.grad[1].sum()),0)

    def test_balanced_loss_gradcheck_and_invalid_labels_reject(self):
        x=torch.tensor([[[[-.3,.4],[.1,.8]]]],dtype=torch.float64,requires_grad=True);m=torch.tensor([[[[0.,.1],[.9,1.]]]],dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda v:A.balanced_mask_bce(v,m)[0],(x,)))
        with self.assertRaises(ValueError):A.balanced_mask_bce(x,m+2)

    def test_mask_transform_matches_image_and_consumes_only_original_rng(self):
        x,y,aux,ay,clean,m,am=tensors(3);x=x.clone();x[:,0:1]=m;oldx=x.clone();oldm=m.clone();rng=np.random.default_rng(44);reference=np.random.default_rng(44)
        xx,yy,mm=A.paired_mask_affine(H,x,y,m,rng);xr,yr=H.affine(x,y,reference)
        self.assertTrue(torch.equal(xx,xr) and torch.equal(yy,yr));self.assertTrue(torch.equal(mm,F.interpolate(xx[:,:1],(16,16),mode='area')))
        self.assertEqual(rng.bit_generator.state,reference.bit_generator.state);self.assertTrue(torch.equal(oldx,x) and torch.equal(oldm,m))

    def test_auxiliary_head_initialization_does_not_advance_original_rng(self):
        torch.manual_seed(351);before=torch.get_rng_state().clone();a=A.auxiliary_head(0,4,'cpu');self.assertTrue(torch.equal(before,torch.get_rng_state()))
        b=A.auxiliary_head(0,4,'cpu');self.assertEqual(state_sha(a.state_dict()),state_sha(b.state_dict()))

    def test_auxiliary_initialization_never_calls_cuda_seed_all(self):
        with patch('torch.cuda.manual_seed_all',side_effect=AssertionError('CUDA RNG must not be seeded')):
            A.auxiliary_head(0,4,'cpu')

    def test_actual_old_and_new_smoke_tapes_model_size_teacher_and_adam_rebuild(self):
        x,y,aux,ay,clean,m,am=tensors();torch.manual_seed(998);state=TinyCenter().state_dict();before=state_sha(state);inputs=[v.clone() for v in (x,y,aux,ay,clean,m,am)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);control=root/'control';control.mkdir();trial=root/'trial';trial.mkdir();scale=SimpleNamespace(CenterScale=TinyCenter)
            fitter=BoundedFitter(N,control,True);fitter.order=[('eventgenesis',0,'physical_supervised','center')]
            fitter.fit_center(scale,state,x,y,aux,ay,0,True,clean,0)
            old=P.read_log(control/'training_evidence/eventgenesis_seed0_physical_supervised_center.jsonl')
            model,head,history,r=A.fit_center(N,scale,state,x,y,aux,ay,clean,m,am,0,True,trial,'eventgenesis',old)
            log=P.read_log(trial/'training_evidence/eventgenesis_seed0_aux_visible_mask_center.jsonl')
            self.assertEqual(P.trajectory(old),P.trajectory(log));self.assertEqual(r['labels_sha256'],fitter.records[0]['labels_sha256']);self.assertEqual(r['updates'],1)
            self.assertTrue(r['first_native_loss_oracle']['passed'] and r['teacher_unchanged'] and r['size_unchanged'] and r['optimizer_state_equal'] and r['independent_model_optimizer_rebuild'])
            self.assertNotEqual(state_sha(model.state_dict()),before);self.assertEqual(state_sha(state),before)
            self.assertTrue(all(torch.equal(model.state_dict()[k],state[k]) for k in state if k.startswith('size.')));self.assertEqual(log[0]['auxiliary_finite_gradient_tensors'],2)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip((x,y,aux,ay,clean,m,am),inputs)))

    def test_full288_label_tape_is_deterministic_and_no_real_data_required(self):
        n=1529;normal=torch.zeros(n,3,1,1);aux=torch.zeros(n*2,3,1,1);y=torch.zeros(n,9);y[:,:2]=.5;ay=torch.cat((y,y));tape=SimpleNamespace(shared=H,native=N)
        a=P.replay(tape,normal,y,aux,ay,aux,0);b=P.replay(tape,normal,y,aux,ay,aux,0)
        self.assertEqual(a,b);self.assertEqual(len(a[1]),288);self.assertEqual([r['update'] for r in a[1]],list(range(1,289)))

    def test_registered_contract_rejects_teacher_input_or_objective_drift(self):
        p=Path(__file__).parent/'MASK_AUX_CONTRACT.json';c=json.loads(p.read_text()) if p.exists() else json.loads((p.parent/'MASK_AUX_CONTRACT_TEMPLATE.json').read_text());validate_contract(c)
        for key,value in [('teacher_input','BAF'),('auxiliary_loss_coefficient',.1),('new_center_optimizer_updates',288),('real_data_scoring_allowed',True)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):validate_contract(bad)

    def test_stage36_and_matched12_cannot_hide_one_regression(self):
        def arm():return {k:dict(center_epe_px=2.,pose=dict(summary=dict(success=.5))) for k in ('clean','half','eighth','matched_normal')}
        rows=[dict(method=m,seed=s,arms=dict(raw_C1=arm(),aux_visible_mask_center=arm())) for m in ('eventgenesis','v2e') for s in range(3)]
        for r in rows:r['arms']['aux_visible_mask_center']['eighth']['center_epe_px']=1.
        g=stage_gates(rows);self.assertEqual(len(g['center_stage36']['checks']),36);self.assertEqual(len(g['matchednormal12']['checks']),12);self.assertTrue(g['center_stage36']['passed'] and g['matchednormal12']['passed'])
        rows[0]['arms']['aux_visible_mask_center']['matched_normal']['pose']['summary']['success']=.49
        self.assertFalse(stage_gates(rows)['matchednormal12']['passed'])


if __name__=='__main__':unittest.main(verbosity=2)
