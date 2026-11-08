import ast,copy,hashlib,tempfile,unittest
from pathlib import Path
import prepare_confirmation as P

class PreparationTests(unittest.TestCase):
    def registration(self):
        return dict(schema='pose_frozen_c1_advantage_confirmation32_v1',frozen_core=P.CORE,physical_seed=P.BASE,rollouts=32,root=P.ROOT,
            methods=['eventgenesis','v2e'],seeds=[0,1,2],conditions={'normal':1.,'half':.5,'eighth':.125},primary_arm='C1_RAW',optimizer_updates=0,real_data_used=False,
            primary_conditions=['normal','half'],bootstrap_draws=10000,bootstrap_seed=P.BOOTSTRAP_SEED,per_condition_ci_level=.975,all_three_seed_deltas_positive=True,
            fresh_source={'sha256':P.FRESH_SHA},relight_source={'sha256':P.RELIGHT_SHA},source_records=[dict(path=str(Path(__file__).resolve()),sha256='0'*64,size_bytes=1)],campaign=str(Path(__file__).resolve().parent))
    def test_scope_rejects_wrong_core_or_model_or_count(self):
        P.validate_registration(self.registration())
        for key,value in [('frozen_core','oldcore'),('rollouts',8),('primary_arm','C0'),('optimizer_updates',1),('real_data_used',True),('conditions',{'normal':1.,'half':.5}),('per_condition_ci_level',.95)]:
            c=self.registration();c[key]=value
            with self.assertRaises(ValueError):P.validate_registration(c)
    def test_prefix_is_exact_original_statements_and_original_return(self):
        fixture=Path(__file__).parent/'pose_fresh_confirm_original.py'
        if not fixture.exists():fixture=Path(__file__).parent.parent/'pose_optimization_20261106/pose_fresh_confirm.py'
        source=fixture.read_text()
        fn=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='generate')
        original=copy.deepcopy(fn);original.name='_frozen_physical_prefix';original.body=original.body[:-2]+[original.body[-1]]
        compiled,digest=P.physical_function(source,{})
        self.assertEqual(digest,hashlib.sha256(ast.dump(original,include_attributes=False).encode()).hexdigest())
        self.assertEqual(compiled.__name__,'_frozen_physical_prefix')
        with self.assertRaises(ValueError):P.physical_function(source.replace("['replay_new_events.py', 'replay_v2e_parallel.py']","['replay_v2e_parallel.py', 'replay_new_events.py']"),{})
    def test_source_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'frozen';p.write_bytes(b'original');r=P.record(p);P.verify_records([r]);p.write_bytes(b'mutated!')
            with self.assertRaises(ValueError):P.verify_records([r])
    def test_cpu_stage_refuses_gpu_before_loading_any_dependency(self):
        import os
        old=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']='GPU-wrong'
        try:
            with self.assertRaisesRegex(ValueError,'externally hidden CUDA'):P.physical({},Path(P.ROOT))
        finally:
            if old is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:os.environ['CUDA_VISIBLE_DEVICES']=old

if __name__=='__main__':unittest.main()
