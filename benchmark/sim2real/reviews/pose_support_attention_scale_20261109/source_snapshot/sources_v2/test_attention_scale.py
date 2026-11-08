import copy,hashlib,importlib.util,math,os,unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from unittest.mock import patch
from types import SimpleNamespace

import attention_scale_metrics as M
import observe_native as O
import probe_attention_scale as P

NATIVE_SHA='7d0245708c02e13df316f81c89af82f4083844d19b01963fc23a172364607866'


def native_module():
    p=Path(os.environ.get('POSE_NATIVE_SCALE_SOURCE',str(Path(__file__).parents[1]/'source_facts/center_scale_source.py')))
    if hashlib.sha256(p.read_bytes()).hexdigest()!=NATIVE_SHA:raise ValueError('actual native scale source SHA differs')
    spec=importlib.util.spec_from_file_location('cpu_actual_center_scale',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_row(condition='normal',branch='depth',rid='r0',sample='r0/000',shift=0):
    scale=M.scale_statistics(-2+shift,-2+shift,0,-2,1,1) if branch=='depth' else None
    return dict(method='eventgenesis',seed=0,arm='C1',branch=branch,input_view='raw',rid=rid,sample_id=sample,condition=condition,
                labels=dict(gt_log_size=-2,gt_z_m=1,focal_log=0),scale=scale,
                features=dict(local_feature=[0.]*64,context_feature=[0.]*64),
                attention=dict(endpoint=dict(center_error_px=0,region_attention_mass=.2,prediction_in_region=True,peak_in_region=True)))


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.scale=native_module()

    def test_original_forward_hook_reconstruction_and_immutability(self):
        torch.manual_seed(32711);model=self.scale.CenterScale().eval().requires_grad_(False)
        original={k:v.clone() for k,v in model.state_dict().items()}
        images=torch.rand(2,3,128,128);before=images.clone();K=[np.diag([300.,300.,1.])]*2
        with torch.inference_mode():baseline=model(images)
        observed=O.observe_native(model,images,self.scale,K)
        for native,key in zip(baseline,('attention_logits_unused','uv','log_size','log_shape')):
            if key!='attention_logits_unused':self.assertTrue(torch.equal(native,observed[key]))
        self.assertTrue(torch.equal(images,before));self.assertEqual(len(model.features._forward_hooks),0);self.assertEqual(len(model.size._forward_pre_hooks),0)
        self.assertTrue(all(torch.equal(v,original[k]) for k,v in model.state_dict().items()))
        diagonal=O.hybrid_scale(model,observed['features_map'],observed['attention'],observed['focal'])
        self.assertTrue(all(torch.equal(diagonal[k],observed[k]) for k in diagonal))

    def test_hybrid_feature_and_attention_factors_are_separate(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.size=nn.Linear(128,2,bias=False)
                with torch.no_grad():self.size.weight.zero_();self.size.weight[0,0]=1;self.size.weight[1,64]=1
        model=Toy().eval();features=torch.zeros(1,64,256);features[:,0,:128]=2
        first=torch.zeros(1,256);first[:,0]=1;last=torch.zeros(1,256);last[:,-1]=1
        a=O.hybrid_scale(model,features,first,torch.zeros(1));b=O.hybrid_scale(model,features,last,torch.zeros(1))
        self.assertAlmostEqual(float(a['log_size']-b['log_size']),2)
        self.assertEqual(float(a['log_shape']),float(b['log_shape']))
        doubled=O.hybrid_scale(model,features*2,last,torch.zeros(1))
        self.assertAlmostEqual(float(doubled['log_shape']-b['log_shape']),1)

    def test_nominal_soft_heat_is_reference_not_hard_instance_target(self):
        model=self.scale.CenterScale().eval();gt=torch.tensor([[.5,.5],[.223,.820]])
        target=O.nominal_center_target(model.grid,gt)
        self.assertTrue(torch.allclose(target['attention'].sum(1),torch.ones(2)))
        self.assertFalse(target['outside_grid_range'].any())
        # The native FP32 Gaussian has broad support but can underflow to zero
        # at distant cells; never replace it with a hard instance target.
        self.assertTrue(((target['attention']>0).sum(1)>9).all())
        self.assertLess(float(torch.linalg.vector_norm(target['uv'][0]-gt[0])),1e-6)

    def test_raw_reference_guard_rejects_wrong_branch_or_drift(self):
        obs=dict(uv=np.array([.5,.5]),log_size=-2.,log_shape=-2.,z=1.)
        ref=dict(pred_uv=[.5,.5],depth_model_uv=[.4,.5],log_size=-2.,log_shape=-2.,z_m=1.)
        self.assertEqual(len(P.check_raw(obs,ref,'center')),1)
        with self.assertRaises(ValueError):P.check_raw(obs,ref,'depth')
        ref['depth_model_uv']=[.5,.5];ref['z_m']=1.001
        with self.assertRaises(ValueError):P.check_raw(obs,ref,'depth')


class MetricTests(unittest.TestCase):
    def test_mass_enrichment_uniform_and_empty_null(self):
        a=np.ones((16,16))/256;region=np.zeros((16,16));region[:4]=1
        r=M.attention_statistics(a,region,[.5,.5],[.5,.5],True,False)
        self.assertEqual(r['region_attention_mass'],.25);self.assertEqual(r['region_mass_enrichment'],1)
        empty=M.attention_statistics(a,region*0,[.5,.5],[.5,.5],None,None)
        self.assertIsNone(empty['region_attention_mass']);self.assertIsNone(empty['region_mass_enrichment']);self.assertIsNone(empty['prediction_in_region'])

    def test_scale_error_cancellation_and_exp_identity(self):
        r=M.scale_statistics(-1,-1,0,-2,1,1)
        self.assertEqual(r['signed_log_size_error'],1);self.assertEqual(r['signed_log_shape_error'],1)
        self.assertEqual(r['signed_log_depth_error'],0);self.assertEqual(r['size_shape_error_product'],1)
        r=M.scale_statistics(-2,-1,0,-2,1,math.e)
        self.assertAlmostEqual(r['signed_log_depth_error'],1)

    def test_cross_light_alignment_and_zero_feature_cosine_is_undefined(self):
        rows=[test_row(c,shift=s) for c,s in [('normal',0),('half',.1),('eighth',.4)]]
        pairs=M.paired_light_deltas(rows)
        self.assertEqual(len(pairs),2)
        self.assertTrue(all(p['delta_log_depth']==0 for p in pairs))
        self.assertTrue(all(p['feature_context']['cosine'] is None for p in pairs))
        bad=copy.deepcopy(rows);bad[-1]['labels']['gt_log_size']+=1
        with self.assertRaises(ValueError):M.paired_light_deltas(bad)
        with self.assertRaises(ValueError):M.paired_light_deltas(rows[:-1])
        with self.assertRaises(ValueError):M.paired_light_deltas(rows+[rows[0]])

    def test_center_branch_scale_is_not_misreported_as_pose_depth(self):
        rows=[test_row(c,branch='center') for c in ('normal','half','eighth')]
        pairs=M.paired_light_deltas(rows)
        self.assertTrue(all('delta_log_size' not in p for p in pairs))
        summaries=M.scalar_rollout_summaries(rows)
        self.assertTrue(all(not any(k.startswith('signed_log') for k in r['equal_rollout_summary']) for r in summaries))

    def test_macro_is_equal_rollout_and_scatter_not_bias(self):
        rows=[test_row(rid='a',sample='a/0')]+[test_row(rid='b',sample=f'b/{i}') for i in range(9)]
        for row in rows:row['scale']['signed_depth_error_m']=0 if row['rid']=='a' else 10
        summary=M.scalar_rollout_summaries(rows)[0]
        self.assertEqual(summary['equal_rollout_summary']['signed_depth_error_m/bias'],5)
        self.assertEqual(summary['window_count'],10)
        self.assertEqual(summary['equal_rollout_summary']['signed_depth_error_m/centered_scatter'],0)

    def test_hybrid_summaries_keep_rollout_then_seed_order(self):
        rows=[]
        for seed in range(3):
            for rid,n in [('a',1),('b',9)]:
                for i in range(n):
                    rows.append(dict(method='eventgenesis',seed=seed,arm='C1',paired_light='half',feature_source='normal',attention_source='half',
                                     rid=rid,sample_id=f'{rid}/{i}',gt_z_m=1,z_m=1 if rid=='a' else 1.02,
                                     log_size=-2,gt_log_size=-2,log_shape=-2,gt_log_shape=-2))
        table=M.hybrid_summaries(rows)
        self.assertEqual(len(table['per_seed']),3)
        self.assertEqual(table['three_seed_means'][0]['seeds'],[0,1,2])
        self.assertAlmostEqual(table['three_seed_means'][0]['metrics']['depth_mae_m'],.01)
        with self.assertRaises(ValueError):M.hybrid_summaries(rows+[rows[0]])


class DeviceImportTests(unittest.TestCase):
    def test_prepare_device_alias_is_normalized_without_implicit_copy(self):
        raw=SimpleNamespace(device=torch.device('cuda:0'));filtered=SimpleNamespace(device=torch.device('cuda:0'))
        P.check_prepared_devices(raw,filtered,torch.device('cuda'),0)
        P.check_prepared_devices(raw,filtered,torch.device('cuda:0'),0)
        with self.assertRaises(ValueError):P.check_prepared_devices(raw,SimpleNamespace(device=torch.device('cpu')),torch.device('cuda'),0)
        with self.assertRaises(ValueError):P.check_prepared_devices(raw,filtered,torch.device('cuda'),1)
        cpu=SimpleNamespace(device=torch.device('cpu'));P.check_prepared_devices(cpu,cpu,torch.device('cpu'))

    def test_cpu_module_side_effect_restores_same_original_assignment(self):
        for assigned in (P.ASSIGNED_GPU_UUID,'4'):
            with self.subTest(assigned=assigned),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':assigned}):
                marker=object()
                def loader():os.environ['CUDA_VISIBLE_DEVICES']='';return marker
                self.assertIs(P.import_cpu_dependency_preserving_device(loader,'cuda',P.ASSIGNED_GPU_UUID,lambda:False),marker)
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],assigned)

    def test_wrong_assignment_or_contract_refused_without_import(self):
        called=[]
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(ValueError,'assigned GPU4'):P.import_cpu_dependency_preserving_device(lambda:called.append(1),'cuda',P.ASSIGNED_GPU_UUID,lambda:False)
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'4'}):
            with self.assertRaisesRegex(ValueError,'assigned GPU4'):P.import_cpu_dependency_preserving_device(lambda:called.append(1),'cuda','GPU-wrong',lambda:False)
        self.assertEqual(called,[])

    def test_initialized_cuda_refused_before_dependency(self):
        called=[]
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'4'}):
            with self.assertRaisesRegex(ValueError,'uninitialized before'):P.import_cpu_dependency_preserving_device(lambda:called.append(1),'cuda',P.ASSIGNED_GPU_UUID,lambda:True)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'4')
        self.assertEqual(called,[])

    def test_import_cannot_initialize_cuda_or_continue_to_gpu_guard(self):
        state={'initialized':False}
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'4'}):
            def loader():os.environ['CUDA_VISIBLE_DEVICES']='';state['initialized']=True
            with self.assertRaisesRegex(ValueError,'unexpectedly initialized'):P.import_cpu_dependency_preserving_device(loader,'cuda',P.ASSIGNED_GPU_UUID,lambda:state['initialized'])

    def test_cpu_mode_stays_hidden_and_failed_import_restores_uninitialized_env(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            marker=object();self.assertIs(P.import_cpu_dependency_preserving_device(lambda:marker,'cpu',P.ASSIGNED_GPU_UUID,lambda:False),marker)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'')
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'4'}):
            def loader():os.environ['CUDA_VISIBLE_DEVICES']='';raise RuntimeError('fixture-import-failure')
            with self.assertRaisesRegex(RuntimeError,'fixture-import-failure'):P.import_cpu_dependency_preserving_device(loader,'cuda',P.ASSIGNED_GPU_UUID,lambda:False)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'4')


if __name__=='__main__':unittest.main(verbosity=2)
