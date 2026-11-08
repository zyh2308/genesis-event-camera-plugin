"""Real CPU micro-updates, C1 center reuse and checkpoint reconstruction contracts."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from test_pose_priority_policy import affine,matched_affine,fake_supervised,write
from pose_priority_policy import BoundedFitter
from pose_teacher_projection_policy import TeacherProjectionFitter,state_sha
from reuse_controls import PriorControls,sha
from run_pose_teacher_projection import validate_candidate_contract,verify_smoke
import projection


class Small24(nn.Module):
    def __init__(self):
        super().__init__();layers=[];width=3
        for _ in range(3):
            layers.extend([nn.Conv2d(width,4,3,padding=1,bias=False),nn.GroupNorm(2,4),nn.SiLU(),nn.Conv2d(4,4,3,padding=1,bias=False),nn.GroupNorm(2,4),nn.SiLU()]);width=4
        self.features=nn.Sequential(*layers);self.center=nn.Conv2d(4,1,1)
        self.size=nn.Sequential(nn.Linear(8,6),nn.SiLU(),nn.Linear(6,2))
        coords=(torch.arange(16)+.5)/16;yy,xx=torch.meshgrid(coords,coords,indexing='ij')
        self.register_buffer('grid',torch.stack((xx,yy),-1).reshape(-1,2))

    def forward(self,x):
        f=self.features(x).flatten(2);logits=self.center(f.reshape(len(x),4,16,16)).flatten(1)
        attention=logits.softmax(1);uv=attention@self.grid
        size,shape=self.size(torch.cat(((f*attention[:,None]).sum(2),f.mean(2)),1)).unbind(1)
        return logits,uv,size,shape


def fixture():
    torch.manual_seed(22);normal=torch.rand(64,3,16,16)
    target=torch.zeros(64,9);target[:,:2]=.45;target[:,2]=-.2;target[:,4]=.3
    clean=torch.cat((normal,normal));relit=torch.cat((normal*.7,normal*.1));labels=torch.cat((target,target))
    state=copy.deepcopy(Small24().state_dict())
    native=SimpleNamespace(shared=SimpleNamespace(METHODS=('eventgenesis','v2e'),affine=affine,supervised=fake_supervised,write=write,sha=sha),matched_affine=matched_affine)
    return native,state,normal,target,clean,relit,labels


def run_fits(fitter,data):
    native,state,normal,target,clean,relit,labels=data;outputs={}
    for method in native.shared.METHODS:
        for arm in ('clean_control','physical_supervised','paired_consistency'):
            for branch in ('depth','center'):
                fn=fitter.fit if branch=='depth' else fitter.fit_center
                outputs[(method,arm,branch)]=fn(SimpleNamespace(CenterScale=Small24),state,normal,target,clean if arm=='clean_control' else relit,labels,0,True,clean,2 if arm=='paired_consistency' else 0)
    return outputs


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_ten_actual_smoke_updates_and_exact_C1_center_reuse(self):
        data=fixture();native,state,*_=data
        self.assertEqual(len(tuple(Small24().parameters())),24)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            fitter=TeacherProjectionFitter(native,Path(tmp),True)
            original=projection.project_teacher_gradients
            with patch('projection.project_teacher_gradients',wraps=original) as spy:outputs=run_fits(fitter,data)
            self.assertEqual(spy.call_count,2)
            self.assertEqual(len(fitter.records),12);self.assertEqual(sum(r['new_optimizer_updates'] for r in fitter.records),10)
            self.assertEqual(len({r['labels_sha256'] for r in fitter.records}),1)
            self.assertTrue(all(r['teacher_unchanged'] and r['independent_model_optimizer_rebuild'] and r['optimizer_state_equal'] for r in fitter.records))
            for method in native.shared.METHODS:
                c1=outputs[(method,'physical_supervised','center')][0];candidate=outputs[(method,'paired_consistency','center')][0]
                self.assertEqual(state_sha(c1.state_dict()),state_sha(candidate.state_dict()))
                self.assertEqual(outputs[(method,'physical_supervised','center')][1],outputs[(method,'paired_consistency','center')][1])
                r=next(r for r in fitter.records if (r['method'],r['arm'],r['branch'])==(method,'paired_consistency','center'))
                self.assertEqual(r['new_optimizer_updates'],0);self.assertTrue(r['reused_C1_center'] and r['candidate_center_state_equal_C1'])
                self.assertFalse(r['candidate_center_optimizer_used'])
                log=Path(tmp)/'training_evidence'/f'{method}_seed0_paired_consistency_depth.jsonl'
                row=json.loads(log.read_text().strip());self.assertEqual(row['projection']['parameter_tensors'],24)
                self.assertFalse(row['raw_scalar_loss_is_applied_gradient_potential']);self.assertEqual(row['finite_gradient_tensors'],24)
                cp=torch.load(Path(tmp)/'training_evidence'/f'{method}_seed0_paired_consistency_depth_training_state.pt',weights_only=False)
                self.assertEqual(cp['canonical_arm'],'teacher_projection');self.assertEqual(cp['update'],1)
                self.assertTrue(all(int(v['step'])==1 for v in cp['optimizer']['state'].values()))
            self.assertEqual(state_sha(state),fitter.records[0]['source_state_sha256'])
            smoke=Path(tmp)/'smoke.json'
            value=dict(status='complete_synthetic_only',phase='smoke',real_scores_computed=False,
                source_manifest_sha256='fixture_manifest',contract_sha256='fixture_contract',
                canonical_candidate='teacher_projection',new_optimizer_updates=10,records=fitter.records)
            write(smoke,value);verify_smoke(smoke,sha(smoke),'fixture_manifest','fixture_contract',sha)
            wrong=copy.deepcopy(value)
            next(r for r in wrong['records'] if r['arm']=='paired_consistency' and r['branch']=='center')['new_optimizer_updates']=1
            bad=Path(tmp)/'invalid_smoke.json';write(bad,wrong)
            with self.assertRaises(AssertionError):verify_smoke(bad,sha(bad),'fixture_manifest','fixture_contract',sha)

    def test_reused_controls_and_candidate_center_have_unique_immutable_evidence(self):
        data=fixture();native,state,*_=data
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            old=Path(tmp)/'old';old.mkdir();new=Path(tmp)/'new';new.mkdir()
            original=BoundedFitter(native,old,True);outputs=run_fits(original,data);results=[]
            for method in native.shared.METHODS:
                run=dict(method=method,seed=0,train_loss={},localization_train_loss={})
                for arm in ('clean_control','physical_supervised','paired_consistency'):
                    for branch in ('depth','center'):
                        model,history=outputs[(method,arm,branch)];suffix='_localizer' if branch=='center' else ''
                        torch.save(model.state_dict(),old/f'{method}_seed0_{arm}{suffix}.pt')
                        run['localization_train_loss' if branch=='center' else 'train_loss'][arm]=history
                results.append(run)
            write(old/'completion.json',dict(status='complete',real_data_used=False,results=results))
            write(old/'priority_execution_receipt.json',dict(status='complete_synthetic_only',real_scores_computed=False,records=original.records))
            before={str(p):sha(p) for p in old.rglob('*') if p.is_file()}
            prior=PriorControls(old,Path(tmp)/'unused',strict=False);fitter=TeacherProjectionFitter(native,new,True,prior)
            new_outputs=run_fits(fitter,data)
            self.assertEqual(sum(r['new_optimizer_updates'] for r in fitter.records),2)
            self.assertEqual(sum(bool(r.get('reused_C1_center')) for r in fitter.records),2)
            self.assertTrue(all(r['labels_and_draws_identical_to_C1'] for r in fitter.records if r['arm']=='paired_consistency' and r['branch']=='depth'))
            for method in native.shared.METHODS:
                self.assertEqual(state_sha(new_outputs[(method,'paired_consistency','center')][0].state_dict()),state_sha(outputs[(method,'physical_supervised','center')][0].state_dict()))
            self.assertEqual(before,{str(p):sha(p) for p in old.rglob('*') if p.is_file()})
            self.assertEqual(len(list((new/'training_evidence').glob('*physical_supervised_center_reused_receipt.json'))),2)
            self.assertEqual(len(list((new/'training_evidence').glob('*paired_consistency_center_C1_reuse_receipt.json'))),2)
            self.assertFalse(hasattr(prior,'check_candidate'))

    def test_contract_rejects_scope_expansion_and_lr_or_center_changes(self):
        c=dict(schema='pose_depth_teacher_projection_v1',candidate_name='teacher_projection',candidate_center='reuse_C1_unchanged',new_depth_optimizer_updates=1728,
               updates_per_depth_branch=288,epochs=6,batch_size=32,half_probability=.75,feature_learning_rate=.00015,head_learning_rate=.00015,
               normal_teacher_weight=2,auxiliary_teacher_weight=0,methods=['eventgenesis','v2e'],seeds=[0,1,2])
        c.update(timeout_seconds=1200,phase_runtime_bounds_seconds={'smoke':1200,'study':3600})
        validate_candidate_contract(c)
        for key,value in [('feature_learning_rate',.000015),('candidate_center','train'),('new_depth_optimizer_updates',3456),('auxiliary_teacher_weight',2),('phase_runtime_bounds_seconds',{'smoke':1200,'study':99999})]:
            bad=dict(c);bad[key]=value
            with self.assertRaises(AssertionError):validate_candidate_contract(bad)
        with self.assertRaises(ValueError):verify_smoke(None,None,'sha','contract',sha)


if __name__=='__main__':unittest.main(verbosity=2)
