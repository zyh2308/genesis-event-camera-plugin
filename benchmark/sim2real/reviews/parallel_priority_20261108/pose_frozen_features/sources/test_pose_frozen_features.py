"""CPU micro-updates, immutable-control reuse, feature freeze and optimizer recovery."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from test_pose_priority_policy import TinyCenterScale, affine, matched_affine, fake_supervised, write
from pose_priority_policy import BoundedFitter
from pose_frozen_features_policy import FrozenFeaturesFitter, state_sha
from reuse_controls import PriorControls, sha


def native_fixture():
    shared=SimpleNamespace(METHODS=('eventgenesis','v2e'),affine=affine,supervised=fake_supervised,write=write,sha=sha)
    return SimpleNamespace(shared=shared,matched_affine=matched_affine)


def call_all(fitter,state,normal,target,clean,relit,aux_target):
    output={}
    for method in fitter.shared.METHODS:
        for arm in ('clean_control','physical_supervised','paired_consistency'):
            for branch in ('depth','center'):
                fn=fitter.fit if branch=='depth' else fitter.fit_center
                model,history=fn(SimpleNamespace(CenterScale=TinyCenterScale),state,normal,target,
                    clean if arm=='clean_control' else relit,aux_target,0,True,clean,2 if arm=='paired_consistency' else 0)
                output[(method,arm,branch)]=(model,history)
    return output


class FrozenFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_real_cpu_updates_reuse_freeze_optimizer_rebuild_and_identical_draws(self):
        torch.manual_seed(22)
        normal=torch.rand(64,3,16,16)
        target=torch.zeros(64,9);target[:,:2]=.45;target[:,2]=-.2;target[:,4]=.3
        clean=torch.cat((normal,normal)); relit=torch.cat((normal*.7,normal*.1))
        aux_target=torch.cat((target,target));state=copy.deepcopy(TinyCenterScale().state_dict())
        native=native_fixture()
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            old=Path(tmp)/'old';old.mkdir();new=Path(tmp)/'new';new.mkdir()
            old_fitter=BoundedFitter(native,old,True)
            old_outputs=call_all(old_fitter,state,normal,target,clean,relit,aux_target)
            results=[]
            for method in native.shared.METHODS:
                r=dict(method=method,seed=0,train_loss={},localization_train_loss={})
                for arm in ('clean_control','physical_supervised','paired_consistency'):
                    for branch in ('depth','center'):
                        model,history=old_outputs[(method,arm,branch)]
                        suffix='_localizer' if branch=='center' else ''
                        torch.save(model.state_dict(),old/f'{method}_seed0_{arm}{suffix}.pt')
                        r['localization_train_loss' if branch=='center' else 'train_loss'][arm]=history
                results.append(r)
            write(old/'completion.json',dict(status='complete',real_data_used=False,results=results))
            write(old/'priority_execution_receipt.json',dict(status='complete_synthetic_only',real_scores_computed=False,records=old_fitter.records))
            before={str(p):sha(p) for p in old.rglob('*') if p.is_file()}
            prior=PriorControls(old,Path(tmp)/'no_diagnostic_needed_for_fixture',strict=False)
            fitter=FrozenFeaturesFitter(native,new,True,prior)
            new_outputs=call_all(fitter,state,normal,target,clean,relit,aux_target)
            self.assertEqual(len(fitter.records),12)
            self.assertEqual(sum(r['new_optimizer_updates'] for r in fitter.records),4)
            self.assertEqual(len({r['labels_sha256'] for r in fitter.records}),1)
            self.assertEqual(sum(bool(r.get('reused_immutable_control')) for r in fitter.records),8)
            for method in native.shared.METHODS:
                for branch in ('depth','center'):
                    model,_=new_outputs[(method,'paired_consistency',branch)]
                    candidate=model.state_dict()
                    self.assertTrue(all(torch.equal(v,state[k]) for k,v in candidate.items() if k.startswith('features.')))
                    self.assertTrue(any(not torch.equal(v,state[k]) for k,v in candidate.items() if k.startswith('center.')))
                    if branch=='depth': self.assertTrue(any(not torch.equal(v,state[k]) for k,v in candidate.items() if k.startswith('size.')))
                    else: self.assertTrue(all(torch.equal(v,state[k]) for k,v in candidate.items() if k.startswith('size.')))
                    self.assertTrue(all(not p.requires_grad for p in model.features.parameters()))
                    old_model,_=old_outputs[(method,'physical_supervised',branch)]
                    self.assertTrue(any(not torch.equal(v,state[k]) for k,v in old_model.state_dict().items() if k.startswith('features.')))
                    for arm in ('clean_control','physical_supervised'):
                        self.assertEqual(state_sha(new_outputs[(method,arm,branch)][0].state_dict()),state_sha(old_outputs[(method,arm,branch)][0].state_dict()))
            self.assertEqual(before,{str(p):sha(p) for p in old.rglob('*') if p.is_file()})
            self.assertTrue(all(r['independent_model_optimizer_rebuild'] and r['teacher_unchanged'] for r in fitter.records))
            self.assertTrue(all(r.get('labels_and_draws_identical_to_C1') for r in fitter.records if r['arm']=='paired_consistency'))

    def test_original_control_metric_decline_is_rejected(self):
        prior=object.__new__(PriorControls)
        original={c:{a:{'half':{'summary':{'depth_mae_m':.03,'success':.4}}} for a in ('clean_control','physical_supervised')} for c in ('arms','depth_only_arms','translation_only_arms')}
        prior.runs={('eventgenesis',0):original}
        result=dict(copy.deepcopy(original),method='eventgenesis',seed=0)
        prior.compare_control_summaries(result)
        result['arms']['clean_control']['half']['summary']['success']=.39
        with self.assertRaises(AssertionError): prior.compare_control_summaries(result)

    def test_changed_original_completion_and_empty_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'completion.json').write_text('{}')
            (root/'priority_execution_receipt.json').write_text('{}')
            with self.assertRaises(AssertionError): PriorControls(root,root/'diagnostic.json')

    def test_candidate_freezes_batchnorm_buffers_as_well_as_weights(self):
        class WithBN(TinyCenterScale):
            def __init__(self):
                super().__init__()
                self.features=torch.nn.Sequential(torch.nn.Conv2d(3,4,1),torch.nn.BatchNorm2d(4))
        torch.manual_seed(123)
        normal=torch.rand(32,3,16,16);target=torch.zeros(32,9)
        target[:,:2]=.4;target[:,2]=-.2;target[:,4]=.3
        clean=torch.cat((normal,normal));relit=torch.cat((normal*.7,normal*.1));labels=torch.cat((target,target))
        state=copy.deepcopy(WithBN().state_dict())
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            fitter=FrozenFeaturesFitter(native_fixture(),Path(tmp),True)
            fitter.order=[('eventgenesis',0,'paired_consistency','depth')]
            model,_=fitter.fit(SimpleNamespace(CenterScale=WithBN),state,normal,target,relit,labels,0,True,clean,2)
            self.assertTrue(all(torch.equal(v,state[k]) for k,v in model.state_dict().items() if k.startswith('features.')))
            self.assertEqual(int(model.features[1].num_batches_tracked),0)
            self.assertTrue(fitter.records[0]['features_weights_and_buffers_unchanged'])


if __name__=='__main__': unittest.main(verbosity=2)
