"""Meaningful CPU tests; all fixtures are local to this source bundle."""
import contextlib
import copy
import io
import json
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from test_pose_priority_policy import TinyCenterScale,affine,matched_affine,fake_supervised,write
from pose_priority_policy import BoundedFitter
from pose_feature_retention_policy import FeatureRetentionFitter,state_sha
from reuse_controls import PriorControls,sha
from feature_retention import (protected_feature_loss,per_sample_relative_feature_loss,common_coefficient)
from calibrate_feature_retention import one_calibration
from calibration_contract import verify_calibration,verify_smoke


def native_fixture():
    return SimpleNamespace(shared=SimpleNamespace(METHODS=('eventgenesis','v2e'),affine=affine,
                           supervised=fake_supervised,write=write,sha=sha),matched_affine=matched_affine)


def call_all(fitter,state,normal,target,clean,relit,aux_target):
    output={}
    for method in fitter.shared.METHODS:
        for arm in ('clean_control','physical_supervised','paired_consistency'):
            for branch in ('depth','center'):
                fn=fitter.fit if branch=='depth' else fitter.fit_center
                output[(method,arm,branch)]=fn(SimpleNamespace(CenterScale=TinyCenterScale),state,normal,target,
                    clean if arm=='clean_control' else relit,aux_target,0,True,clean,2 if arm=='paired_consistency' else 0)
    return output


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_feature_formula_double_gradcheck_teacher_detached_and_dark_zero(self):
        torch.manual_seed(6)
        student=torch.randn(2,3,2,2,dtype=torch.double,requires_grad=True)
        teacher=torch.randn_like(student,requires_grad=True)
        expected=((student-teacher.detach()).square().mean((1,2,3))/teacher.detach().square().mean((1,2,3))).mean()
        actual=per_sample_relative_feature_loss(student,teacher).mean()
        torch.testing.assert_close(actual,expected)
        self.assertTrue(torch.autograd.gradcheck(lambda x:per_sample_relative_feature_loss(x,teacher).mean(),(student,)))
        actual.backward(); self.assertIsNone(teacher.grad)
        model=nn.Module();model.features=nn.Linear(3,3,bias=False).double()
        frozen=copy.deepcopy(model).requires_grad_(False)
        with torch.no_grad():model.features.weight.add_(.1)
        normal=torch.randn(2,3,dtype=torch.double,requires_grad=True)
        auxiliary=torch.randn(2,3,dtype=torch.double,requires_grad=True)
        loss=protected_feature_loss(model,frozen,normal,auxiliary,torch.tensor([True,False]))
        loss.backward();torch.testing.assert_close(auxiliary.grad[1],torch.zeros(3,dtype=torch.double),rtol=0,atol=0)
        self.assertGreater(float(auxiliary.grad[0].norm()),0)
        self.assertTrue(all(p.grad is None for p in frozen.parameters()))

    def test_calibration_uses_drifted_C1_and_common_twelve_branch_scale(self):
        torch.manual_seed(30);model=TinyCenterScale();initial=copy.deepcopy(model.state_dict())
        with torch.no_grad():model.features.weight.add_(.02)
        C1=copy.deepcopy(model.state_dict());original_hashes=(state_sha(initial),state_sha(C1));x=torch.rand(8,3,16,16);a=x*.7
        y=torch.zeros(8,9);y[:,:2]=.45;y[:,2]=-.2;y[:,4]=.3
        shared=native_fixture().shared
        rows=[]
        for method in shared.METHODS:
            for seed in range(3):
                for branch in ('depth','center'):
                    row=one_calibration(SimpleNamespace(CenterScale=TinyCenterScale),shared,initial,C1,x,y,a,y,torch.tensor([True]*6+[False]*2),branch)
                    row.update(method=method,seed=seed,branch=branch);rows.append(row)
        coefficient,ratios=common_coefficient(rows)
        self.assertAlmostEqual(statistics.median(ratios),.1,places=13)
        self.assertGreater(coefficient,0)
        self.assertTrue(all(r['source_initial_feature_retention']==0 and r['feature_retention']>0 for r in rows))
        self.assertEqual(original_hashes,(state_sha(initial),state_sha(C1)))
        bad=copy.deepcopy(rows);bad[0]['feature_retention_gradient_l2']=0
        with self.assertRaises(ValueError):common_coefficient(bad)
        with self.assertRaises(ValueError):common_coefficient(rows[:2])

    def test_new_micro_updates_reuse_controls_restore_and_match_labels(self):
        torch.manual_seed(22);normal=torch.rand(64,3,16,16)
        target=torch.zeros(64,9);target[:,:2]=.45;target[:,2]=-.2;target[:,4]=.3
        clean=torch.cat((normal,normal));relit=torch.cat((normal*.7,normal*.1));aux_target=torch.cat((target,target))
        state=copy.deepcopy(TinyCenterScale().state_dict());native=native_fixture()
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            old=Path(tmp)/'old';old.mkdir();new=Path(tmp)/'new';new.mkdir()
            old_fitter=BoundedFitter(native,old,True);outputs=call_all(old_fitter,state,normal,target,clean,relit,aux_target)
            results=[]
            for method in native.shared.METHODS:
                row=dict(method=method,seed=0,train_loss={},localization_train_loss={})
                for arm in ('clean_control','physical_supervised','paired_consistency'):
                    for branch in ('depth','center'):
                        model,history=outputs[(method,arm,branch)]
                        suffix='_localizer' if branch=='center' else ''
                        torch.save(model.state_dict(),old/f'{method}_seed0_{arm}{suffix}.pt')
                        row['localization_train_loss' if branch=='center' else 'train_loss'][arm]=history
                results.append(row)
            write(old/'completion.json',dict(status='complete',real_data_used=False,results=results))
            write(old/'priority_execution_receipt.json',dict(status='complete_synthetic_only',real_scores_computed=False,records=old_fitter.records))
            before={str(p):sha(p) for p in old.rglob('*') if p.is_file()}
            prior=PriorControls(old,Path(tmp)/'unused',strict=False)
            fitter=FeatureRetentionFitter(native,new,True,.3,prior)
            candidates=call_all(fitter,state,normal,target,clean,relit,aux_target)
            self.assertEqual(len(fitter.records),12);self.assertEqual(sum(r['new_optimizer_updates'] for r in fitter.records),4)
            self.assertEqual(len({r['labels_sha256'] for r in fitter.records}),1)
            for method in native.shared.METHODS:
                for branch in ('depth','center'):
                    candidate=candidates[(method,'paired_consistency',branch)][0]
                    self.assertTrue(all(p.requires_grad for p in candidate.features.parameters()))
                    self.assertTrue(any(not torch.equal(v,state[k]) for k,v in candidate.state_dict().items() if k.startswith('features.')))
                    # At source initialization feature loss is zero; first update must equal C1.
                    self.assertEqual(state_sha(candidate.state_dict()),state_sha(outputs[(method,'physical_supervised',branch)][0].state_dict()))
                    for arm in ('clean_control','physical_supervised'):
                        self.assertEqual(state_sha(candidates[(method,arm,branch)][0].state_dict()),state_sha(outputs[(method,arm,branch)][0].state_dict()))
            self.assertEqual(before,{str(p):sha(p) for p in old.rglob('*') if p.is_file()})
            self.assertTrue(all(r['independent_model_optimizer_rebuild'] and r['optimizer_state_equal'] and r['teacher_unchanged'] for r in fitter.records))
            self.assertTrue(all(r.get('labels_and_draws_identical_to_C1') for r in fitter.records if r['arm']=='paired_consistency'))

    def test_phase_dependencies_reject_missing_corrupt_and_unpassed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'calibration.json'
            rows=[dict(method=m,seed=s,branch=b,C1_objective_feature_gradient_l2=2.,feature_retention_gradient_l2=1.,
                       source_initial_feature_retention=0,optimizer_updates=0,source_and_C1_weights_unchanged=True)
                  for m in ('eventgenesis','v2e') for s in range(3) for b in ('depth','center')]
            value=dict(schema='pose_feature_retention_calibration_v1',status='passed',source_manifest_sha256='fixture',optimizer_updates=0,
                       real_data_used=False,validation_data_used=False,validation_metrics_used=False,train_only=True,
                       coefficient=.2,target_median_gradient_ratio=.1,records=rows)
            dependency=Path(tmp)/'dependency.py';dependency.write_text('# fixture')
            value['dependency_sources']={n:dict(path=str(dependency),size_bytes=dependency.stat().st_size,sha256=sha(dependency))
                                        for n in ('scale','pilot','fresh','rotation','causal','encoder')}
            write(p,value);accepted=verify_calibration(p,sha(p),'fixture')
            with self.assertRaises(AssertionError):verify_calibration(p,'0'*64,'fixture')
            with self.assertRaises(AssertionError):verify_smoke(None,None,accepted,sha(p),'fixture')
            value['records'][0]['method']='wrong';p.write_text(json.dumps(value))
            with self.assertRaises(AssertionError):verify_calibration(p,sha(p),'fixture')

    def test_fixed_original_control_metrics_cannot_be_weakened(self):
        prior=object.__new__(PriorControls)
        original={c:{a:{'half':{'summary':{'depth_mae_m':.03,'success':.4}}} for a in ('clean_control','physical_supervised')}
                  for c in ('arms','depth_only_arms','translation_only_arms')}
        prior.runs={('eventgenesis',0):original};result=dict(copy.deepcopy(original),method='eventgenesis',seed=0)
        prior.compare_control_summaries(result);result['arms']['clean_control']['half']['summary']['success']=.39
        with self.assertRaises(AssertionError):prior.compare_control_summaries(result)


if __name__=='__main__':unittest.main(verbosity=2)
