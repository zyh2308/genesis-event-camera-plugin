"""CPU-only optimizer/trajectory fixtures; no real pose model or data training."""
import contextlib,copy,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from test_pose_priority_policy import TinyCenterScale,affine,matched_affine,fake_supervised,write
from pose_priority_policy import BoundedFitter
from pose_slow_features_policy import SlowFeaturesFitter,state_sha
from reuse_controls import PriorControls,sha
from slow_feature_optimizer import make_optimizer,validate_optimizer,FEATURES_LR,HEADS_LR
from smoke_contract import verify_smoke


def native_fixture():
    return SimpleNamespace(shared=SimpleNamespace(METHODS=('eventgenesis','v2e'),affine=affine,supervised=fake_supervised,write=write,sha=sha),matched_affine=matched_affine)
def call_all(fitter,state,normal,target,clean,relit,aux_target):
    output={}
    for method in fitter.shared.METHODS:
        for arm in ('clean_control','physical_supervised','paired_consistency'):
            for branch in ('depth','center'):
                fn=fitter.fit if branch=='depth' else fitter.fit_center
                output[(method,arm,branch)]=fn(SimpleNamespace(CenterScale=TinyCenterScale),state,normal,target,
                    clean if arm=='clean_control' else relit,aux_target,0,True,clean,2 if arm=='paired_consistency' else 0)
    return output


class SlowFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_parameter_groups_exact_no_omission_overlap_or_frozen_size(self):
        for center in (False,True):
            model=TinyCenterScale()
            if center:model.size.requires_grad_(False)
            opt,signature=make_optimizer(model,True)
            self.assertEqual([g['lr'] for g in signature['groups']],[1.5e-5,1.5e-4])
            names=[n for g in signature['groups'] for n in g['parameter_names']]
            self.assertEqual(len(names),len(set(names)))
            self.assertEqual(set(names),{n for n,p in model.named_parameters() if p.requires_grad})
            self.assertTrue(all(n.startswith('features.') for n in signature['groups'][0]['parameter_names']))
            self.assertTrue(all(not n.startswith('features.') for n in signature['groups'][1]['parameter_names']))
            if center:self.assertFalse(any(n.startswith('size.') for n in names))
            opt.param_groups[0]['lr']=HEADS_LR
            with self.assertRaises(ValueError):validate_optimizer(model,opt,True)

    def test_both_groups_update_with_fixed_one_tenth_step_and_independent_Adam_restore(self):
        class EqualGradient(torch.nn.Module):
            def __init__(self):
                super().__init__();self.features=torch.nn.Linear(1,1,bias=False,dtype=torch.float64);self.head=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
                torch.nn.init.ones_(self.features.weight);torch.nn.init.ones_(self.head.weight)
            def forward(self,x):return self.features(x)+self.head(x)
        model=EqualGradient();opt,signature=make_optimizer(model,True)
        model(torch.ones(1,1,dtype=torch.float64)).sum().backward();opt.step()
        df=1-float(model.features.weight.detach());dh=1-float(model.head.weight.detach())
        self.assertGreater(df,0);self.assertGreater(dh,0);self.assertAlmostEqual(df/dh,.1,places=9)
        rebuilt=EqualGradient();rebuilt.load_state_dict(model.state_dict());other,sig=make_optimizer(rebuilt,True)
        self.assertEqual(signature,sig);other.load_state_dict(copy.deepcopy(opt.state_dict()))
        self.assertEqual(validate_optimizer(rebuilt,other,True),signature)
        self.assertEqual(len(other.param_groups),2)
        for a,b in zip(opt.state.values(),other.state.values()):
            self.assertEqual(a.keys(),b.keys());self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
        # Check a genuine next update after independently reconstructing both groups.
        for m,o in ((model,opt),(rebuilt,other)):
            o.zero_grad();m(torch.ones(1,1,dtype=torch.float64)).sum().backward();o.step()
        self.assertEqual(state_sha(model.state_dict()),state_sha(rebuilt.state_dict()))

    def test_actual_CPU_policy_reuses_controls_and_matches_C1_labels_draws(self):
        torch.manual_seed(22);normal=torch.rand(64,3,16,16)
        target=torch.zeros(64,9);target[:,:2]=.45;target[:,2]=-.2;target[:,4]=.3
        clean=torch.cat((normal,normal));relit=torch.cat((normal*.7,normal*.1));aux_target=torch.cat((target,target))
        state=copy.deepcopy(TinyCenterScale().state_dict());native=native_fixture()
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            old=Path(tmp)/'old';old.mkdir();new=Path(tmp)/'new';new.mkdir()
            original=BoundedFitter(native,old,True);old_outputs=call_all(original,state,normal,target,clean,relit,aux_target)
            results=[]
            for method in native.shared.METHODS:
                r=dict(method=method,seed=0,train_loss={},localization_train_loss={})
                for arm in ('clean_control','physical_supervised','paired_consistency'):
                    for branch in ('depth','center'):
                        model,history=old_outputs[(method,arm,branch)];suffix='_localizer' if branch=='center' else ''
                        torch.save(model.state_dict(),old/f'{method}_seed0_{arm}{suffix}.pt')
                        r['localization_train_loss' if branch=='center' else 'train_loss'][arm]=history
                results.append(r)
            write(old/'completion.json',dict(status='complete',real_data_used=False,results=results))
            write(old/'priority_execution_receipt.json',dict(status='complete_synthetic_only',real_scores_computed=False,records=original.records))
            before={str(p):sha(p) for p in old.rglob('*') if p.is_file()}
            prior=PriorControls(old,Path(tmp)/'synthetic_no_diagnostic',strict=False)
            fitter=SlowFeaturesFitter(native,new,True,prior);outputs=call_all(fitter,state,normal,target,clean,relit,aux_target)
            self.assertEqual(len(fitter.records),12);self.assertEqual(sum(r['new_optimizer_updates'] for r in fitter.records),4)
            self.assertEqual(sum(bool(r.get('reused_immutable_control')) for r in fitter.records),8)
            self.assertEqual(len({r['labels_sha256'] for r in fitter.records}),1)
            for method in native.shared.METHODS:
                for branch in ('depth','center'):
                    model,_=outputs[(method,'paired_consistency',branch)]
                    self.assertTrue(all(p.requires_grad for p in model.features.parameters()))
                    self.assertTrue(any(not torch.equal(v,state[k]) for k,v in model.state_dict().items() if k.startswith('features.')))
                    self.assertTrue(any(not torch.equal(v,state[k]) for k,v in model.state_dict().items() if k.startswith('center.')))
                    if branch=='center':self.assertTrue(all(torch.equal(v,state[k]) for k,v in model.state_dict().items() if k.startswith('size.')))
                    for arm in ('clean_control','physical_supervised'):
                        self.assertEqual(state_sha(outputs[(method,arm,branch)][0].state_dict()),state_sha(old_outputs[(method,arm,branch)][0].state_dict()))
            candidates=[r for r in fitter.records if r['arm']=='paired_consistency']
            self.assertEqual(len(candidates),4)
            for r in candidates:
                self.assertEqual(r['canonical_arm'],'slow_features');self.assertEqual(r['feature_KD_weight'],0.)
                self.assertEqual(r['actual_half_teacher_output_weight'],0);self.assertEqual(r['normal_teacher_weight'],2)
                self.assertTrue(r['labels_and_draws_identical_to_C1']);self.assertTrue(r['optimizer_groups_independently_rebuilt'])
                self.assertEqual([g['lr'] for g in r['optimizer_group_signature']['groups']],[FEATURES_LR,HEADS_LR])
            self.assertEqual(before,{str(p):sha(p) for p in old.rglob('*') if p.is_file()})

    def test_controls_cannot_be_lowered_to_make_candidate_pass(self):
        prior=object.__new__(PriorControls)
        original={c:{a:{'half':{'summary':{'depth_mae_m':.03,'success':.4}}} for a in ('clean_control','physical_supervised')} for c in ('arms','depth_only_arms','translation_only_arms')}
        prior.runs={('eventgenesis',0):original};result=dict(copy.deepcopy(original),method='eventgenesis',seed=0)
        prior.compare_control_summaries(result)
        result['arms']['clean_control']['half']['summary']['success']=.39
        with self.assertRaises(AssertionError):prior.compare_control_summaries(result)

    def test_actual_smoke_required_not_just_CPU_fixture_name(self):
        with self.assertRaises(ValueError):verify_smoke(None,None,'f'*64)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fake.json';path.write_text(json.dumps(dict(status='complete_synthetic_only',phase='study',source_manifest_sha256='f'*64)))
            with self.assertRaises((ValueError,KeyError)):verify_smoke(path,sha(path),'f'*64)

    def test_smoke_receipt_checks_all_twelve_updates_and_two_group_restore(self):
        torch.manual_seed(13);normal=torch.rand(32,3,16,16);target=torch.zeros(32,9)
        target[:,:2]=.4;target[:,2]=-.2;target[:,4]=.3
        clean=torch.cat((normal,normal));relit=torch.cat((normal*.7,normal*.1));labels=torch.cat((target,target))
        state=copy.deepcopy(TinyCenterScale().state_dict())
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);fitter=SlowFeaturesFitter(native_fixture(),root,True)
            call_all(fitter,state,normal,target,clean,relit,labels)
            receipt=dict(status='complete_synthetic_only',phase='smoke',source_manifest_sha256='f'*64,
                canonical_candidate='slow_features',real_scores_computed=False,feature_KD_weight=0.,
                actual_half_teacher_weight=0,features_frozen=False,records=fitter.records,new_optimizer_updates=12)
            path=root/'artificial_CPU_fixture_not_real_GPU_smoke.json';path.write_text(json.dumps(receipt))
            self.assertEqual(verify_smoke(path,sha(path),'f'*64)['new_optimizer_updates'],12)
            changed=copy.deepcopy(receipt);changed['records'].pop();path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):verify_smoke(path,sha(path),'f'*64)
            changed=copy.deepcopy(receipt);candidate=next(r for r in changed['records'] if r['arm']=='paired_consistency')
            candidate['optimizer_group_signature']['groups'][0]['lr']=HEADS_LR;path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):verify_smoke(path,sha(path),'f'*64)


if __name__=='__main__':unittest.main(verbosity=2)
