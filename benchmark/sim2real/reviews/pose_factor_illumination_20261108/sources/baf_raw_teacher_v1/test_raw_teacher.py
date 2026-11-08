"""Keep the frozen previous15 tests and add actual paired-teacher CPU checks."""
import contextlib,copy,importlib,io,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import run_raw_teacher as W
import test_policy_derivation as DERIVATION

C=W.read(Path(__file__).with_name('contract.json'))
PREVIOUS=Path(os.environ.get('POSE_PREVIOUS_BAF_SOURCES',C['previous_baf_sources']))
if W.sha(PREVIOUS/'source_manifest.json')!=W.PREVIOUS_SOURCE_SHA:raise ValueError('Original15 test package unregistered')
for name,digest in W.read(PREVIOUS/'source_manifest.json').items():
    if W.sha(PREVIOUS/name)!=digest:raise ValueError('Original test/source changed: '+name)
sys=__import__('sys');sys.path.insert(0,str(PREVIOUS))
OLD=importlib.import_module('test_baf_center');RULE=importlib.import_module('test_baf');P=importlib.import_module('run_baf_center')


class RawTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_contract_keeps_single_teacher_route_and_equal_budget(self):
        self.assertEqual(W.sha(Path(__file__).with_name('contract.json')),W.CONTRACT_SHA);W.validate_contract(C)
        for key,value in [('candidate_teacher_input','BAF_normal'),('new_center_optimizer_updates',3456),('normal_teacher_weight',0),('half_probability',.5),('real_data_scoring_allowed',True)]:
            wrong=copy.deepcopy(C);wrong[key]=value
            with self.assertRaises(ValueError):W.validate_contract(wrong)

    def test_paired_normal_augmentation_consumes_exactly_original_rng_once(self):
        torch.manual_seed(91);raw=torch.rand(13,3,4,4);filtered=raw*.2;y=torch.rand(13,9);originals=[v.clone() for v in (raw,filtered,y)]
        a=np.random.default_rng(720);b=np.random.default_rng(720)
        x,target=OLD.affine(filtered,y,a);paired,teacher_x,paired_y=OLD.matched(filtered,raw,y,b)
        self.assertTrue(torch.equal(x,paired) and torch.equal(target,paired_y))
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)
        self.assertFalse(torch.equal(teacher_x,paired))
        for before,after in zip(originals,(raw,filtered,y)):self.assertTrue(torch.equal(before,after))

    def test_all288_new_replay_matches_original_c1_draws_labels_and_inputs_unchanged(self):
        torch.manual_seed(31);raw=torch.rand(1529,3,4,4);normal=raw*.3;y=torch.rand(1529,9)
        aux=torch.cat((normal*.7,normal*.1));ay=torch.cat((y,y));clean=torch.cat((normal,normal));native=OLD.fake_native()
        f=SimpleNamespace(native=native,shared=native.shared,raw_normal=raw)
        a,draws=W.replay(f,normal,y,aux,ay,clean,1);b,original=P.replay(f,normal,y,aux,ay,clean,1)
        self.assertEqual(len(draws),288);self.assertEqual(a,b);self.assertEqual(draws,original)

    def test_actual_two_updates_teacher_sees_raw_matched_view_not_filtered_and_restores(self):
        class TraceCenter(OLD.TinyCenter):
            teacher_inputs=[]
            def forward(self,images):
                if not self.training:self.teacher_inputs.append(images.detach().clone())
                return super().forward(images)
        torch.manual_seed(52);raw=torch.rand(64,3,4,4);normal=raw*.15;y=torch.rand(64,9)
        aux=torch.cat((normal*.7,normal*.1));ay=torch.cat((y,y));clean=torch.cat((normal,normal));state=copy.deepcopy(TraceCenter().state_dict())
        snapshots=[t.clone() for t in (raw,normal,y,aux,ay,clean)]
        rng=np.random.default_rng(71070);idx=rng.permutation(64)[:32];W.draw_auxiliary(rng,64,32)
        expected_filtered,expected_raw,_=OLD.matched(normal[idx],raw[idx],y[idx],rng)
        with tempfile.TemporaryDirectory() as temporary:
            fitter=W.BoundedFitter(OLD.fake_native(),temporary,True);fitter.raw_normal=raw
            fitter.order=[(m,0,'physical_supervised','center') for m in ('eventgenesis','v2e')]
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(2):
                    TraceCenter.teacher_inputs=[];model,_=fitter.fit_center(SimpleNamespace(CenterScale=TraceCenter),state,normal,y,aux,ay,0,True,clean,0)
                    self.assertTrue(torch.equal(TraceCenter.teacher_inputs[-1],expected_raw));self.assertFalse(torch.equal(TraceCenter.teacher_inputs[-1],expected_filtered))
                    self.assertTrue(all(torch.equal(v,state[k]) for k,v in model.state_dict().items() if k.startswith('size.')))
            self.assertEqual(len(fitter.records),2)
            for record in fitter.records:
                self.assertTrue(record['teacher_unchanged'] and record['independent_model_optimizer_rebuild'] and record['optimizer_state_equal'])
                self.assertEqual(record['labels_sha256'],W.replay(fitter,normal,y,aux,ay,clean,0,True)[0])
        for a,b in zip((raw,normal,y,aux,ay,clean),snapshots):self.assertTrue(torch.equal(a,b))

    def test_raw_cache_rejects_labels_ids_source_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'source.npz';path.write_bytes(b'original-NPZ');lineage=[P.record(path)]
            data=dict(normal=torch.zeros(3,3,4,4),normal_ids=['a','b','c'],y=torch.ones(3,9))
            value=dict(contract_sha256=W.CONTRACT_SHA,source_manifest_sha256='source',data=dict(raw_normal=torch.ones(3,3,4,4),labels=data['y'].clone(),normal_ids=data['normal_ids'],source_records=lineage,role='RAW_teacher_only_BAF_student_unchanged'))
            before=data['normal'].clone();W.check_raw(P,value,data,'source');self.assertTrue(torch.equal(before,data['normal']))
            for key,bad in [('labels',torch.zeros(3,9)),('normal_ids',['c','b','a']),('raw_normal',torch.full((3,3,4,4),float('nan')))]:
                wrong=copy.deepcopy(value);wrong['data'][key]=bad
                with self.assertRaises(ValueError):W.check_raw(P,wrong,data,'source')
            path.write_bytes(b'modified-NPZ')
            with self.assertRaises(ValueError):W.check_raw(P,value,data,'source')

    def test_smoke_requires_actual_raw_teacher_pair_and_cache_admission(self):
        records=[dict(method=m,seed=0,branch='center',updates=1,teacher_unchanged=True,independent_model_optimizer_rebuild=True,optimizer_state_equal=True) for m in ('eventgenesis','v2e')]
        value=dict(schema='pose_baf_raw_teacher_adaptation_v1',status='complete_synthetic_only',phase='smoke',contract_sha256=W.CONTRACT_SHA,source_manifest_sha256='source',new_optimizer_updates=2,
                   real_scores_computed=False,canonical_candidate='raw_teacher_baf_center',records=records,full288_admissions=[dict(raw_teacher_paired_replay_exact=True) for _ in range(6)],raw_teacher_caches=[{},{}])
        W.verify_smoke(value,'source')
        for key,bad in [('canonical_candidate','fixed_baf_center'),('new_optimizer_updates',1),('raw_teacher_caches',[])]:
            wrong=copy.deepcopy(value);wrong[key]=bad
            with self.assertRaises(ValueError):W.verify_smoke(wrong,'source')


def load_tests(loader,tests,pattern):
    combined=unittest.TestSuite()
    for module in (RULE,OLD,DERIVATION):combined.addTests(loader.loadTestsFromModule(module))
    combined.addTests(loader.loadTestsFromTestCase(RawTeacherTests));return combined


if __name__=='__main__':unittest.main(verbosity=2)
