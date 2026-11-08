"""CPU fixture tests for factor export, never loads project data or a GPU."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import cv2
import numpy as np
import torch
from torch import nn
import export_pose_factors as E


class ToyCenterScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.GroupNorm(2,4),nn.SiLU())
        self.center=nn.Conv2d(4,1,1)
        self.size=nn.Sequential(nn.Linear(8,6),nn.SiLU(),nn.Linear(6,2))
        coords=(torch.arange(8)+.5)/8;yy,xx=torch.meshgrid(coords,coords,indexing='ij')
        self.register_buffer('grid',torch.stack((xx,yy),-1).reshape(-1,2))

    def forward(self,x):
        f=self.features(x).flatten(2);logits=self.center(f.reshape(len(x),-1,8,8)).flatten(1)
        attention=logits.softmax(1);uv=attention@self.grid
        local=(f*attention[:,None]).sum(2);context=f.mean(2)
        size,shape=self.size(torch.cat((local,context),1)).unbind(1)
        return logits,uv,size-2.3,shape-2.3


def independent_encoder(t,x,y,p,start,end,width,height,count_scale):
    selected=(t>start)&(t<=end);pixel=y[selected]*width+x[selected];polarity=p[selected]
    on=np.bincount(pixel[polarity==1],minlength=height*width).reshape(height,width).astype(np.float32)
    off=np.bincount(pixel[polarity==0],minlength=height*width).reshape(height,width).astype(np.float32)
    scale=np.float32(np.log1p(count_scale))
    return np.ascontiguousarray(np.stack([cv2.resize(np.minimum(np.log1p(v)/scale,1.),(8,8),interpolation=cv2.INTER_LINEAR) for v in (on,off,on+off)]),dtype=np.float32)


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_contract_rejects_wrong_scope_and_unfrozen_sources(self):
        c=dict(schema='pose_factor_conflict_diagnostic_v1',frozen_core=E.CORE,gpu_uuid=E.GPU4,
            timeout_seconds=1200,methods=['eventgenesis','v2e'],seeds=[0,1,2],arm_mapping=E.ARM_MAPPING,
            native_source_sha256=E.NATIVE_SHA,prior_source_manifest_sha256=E.PRIOR_MANIFEST,
            campaign='/R',prior_source_dir='/R/prior',slow_study='/R/slow',radiance_train='/R/train',
            radiance_validation='/R/val',data_audit_path='/R/audit.json',
            source_dependencies={f'/R/source{i}.py':'a'*64 for i in range(6)},
            **{k:'b'*64 for k in ('slow_source_manifest_sha256','slow_receipt_sha256','slow_completion_sha256','data_audit_sha256')})
        E.validate_contract(c)
        for key,value in [('gpu_uuid','wrong'),('frozen_core','old'),('seeds',[0]),('timeout_seconds',99999),('native_source_sha256','a'*64)]:
            wrong=copy.deepcopy(c);wrong[key]=value
            with self.assertRaises(AssertionError):E.validate_contract(wrong)

    def test_gain_histograms_match_native_and_preserve_support_and_inputs(self):
        t=np.array([0.]+[.01]*9+[.005]*2+[.02]);x=np.array([0]+[4]*9+[2]*2+[1]);y=np.array([0]+[5]*9+[3]*2+[1]);p=np.array([1]*10+[0,0,1],np.uint8)
        originals=[v.copy() for v in (t,x,y,p)]
        base,gains=E.raw_count_gain_images(independent_encoder,8,t,x,y,p,0.,.01,12,10,8.)
        np.testing.assert_array_equal(gains[1.],independent_encoder(t,x,y,p,0.,.01,12,10,8.))
        self.assertEqual(set(gains),{.5,1.,2.})
        for gain,frame in gains.items():
            np.testing.assert_array_equal(frame>0,base>0);self.assertTrue(np.isfinite(frame).all())
        for before,after in zip(originals,(t,x,y,p)):np.testing.assert_array_equal(before,after)
        def bad(*args):return independent_encoder(*args)+.01
        with self.assertRaises(AssertionError):E.raw_count_gain_images(bad,8,t,x,y,p,0.,.01,12,10,8.)

    def test_decomposed_forward_and_identity_global_swap_exact_without_mutation(self):
        torch.manual_seed(44);model=ToyCenterScale().eval();images=torch.rand(35,3,8,8)
        scale=SimpleNamespace(focal=lambda K:float(.3));intrinsics=[np.eye(3)]*len(images)
        state=E.state_sha(model.state_dict());out=E.predict_raw(model,images,scale,intrinsics)
        identity=E.swapped_global(model,out,out,scale,intrinsics,torch.device('cpu'))
        for key in ('log_size','log_shape','z'):np.testing.assert_array_equal(identity[key],out[key])
        donor=dict(out);donor['global']=out['global']+.1
        swapped=E.swapped_global(model,out,donor,scale,intrinsics,torch.device('cpu'))
        self.assertGreater(float(np.max(np.abs(swapped['log_size']-out['log_size']))),1e-7)
        self.assertEqual(E.state_sha(model.state_dict()),state);self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(np.all(out['normalized_entropy']<=1.000001) and np.all(out['normalized_entropy']>=0))

    def test_peak_local3_uses_prediction_not_ground_truth(self):
        model=ToyCenterScale().eval();model.features=nn.Identity();model.center=nn.Conv2d(3,1,1,bias=False);model.size=nn.Linear(6,2)
        with torch.no_grad():model.center.weight.zero_();model.center.weight[0,0,0,0]=1
        image=torch.zeros(1,3,8,8);image[:,0]=-20;image[0,0,1,1]=10;image[0,0,6,6]=9.8
        result=E.predict_raw(model,image,SimpleNamespace(focal=lambda K:0.),[np.eye(3)])
        np.testing.assert_allclose(result['argmax_uv'][0],[1.5/8,1.5/8],rtol=0,atol=0)
        np.testing.assert_allclose(result['local3_uv'][0],result['argmax_uv'][0],rtol=0,atol=1e-6)
        self.assertGreater(float(result['uv'][0,0]),.4)

    def test_checkpoint_state_and_file_sha_bound_and_json_full_numpy(self):
        torch.manual_seed(4);model=ToyCenterScale();state=model.state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'eventgenesis_seed0_clean_control.pt';torch.save(state,path)
            row=dict(method='eventgenesis',seed=0,arm='clean_control',branch='depth',final_state_sha256=E.state_sha(state))
            ctx=SimpleNamespace(study=root,receipt=dict(records=[row]),scale=SimpleNamespace(CenterScale=ToyCenterScale),device=torch.device('cpu'),source_records=[])
            loaded,record,actual=E.load_branch(ctx,'eventgenesis',0,'clean_control','depth')
            self.assertEqual(record,row);self.assertEqual(actual,path);self.assertEqual(E.state_sha(loaded.state_dict()),row['final_state_sha256'])
            fixture=dict(K=np.eye(3),T_gt=np.eye(4),anchor=np.ones(3),uv_gt=np.array([.4,.5],np.float32),pred_uv=np.array([.3,.7],np.float32),
                z_m=np.float32(.6),R_pred=np.eye(3),large=np.int64(2**60+1),diagnostic_only=True)
            stream=io.StringIO();E.write_line(stream,fixture);parsed=json.loads(stream.getvalue());self.assertEqual(parsed['large'],2**60+1)
            file=root/'data.json';E.write_json(file,fixture)
            with self.assertRaises(FileExistsError):E.write_json(file,fixture)
            with self.assertRaises(ValueError):E.write_line(io.StringIO(),dict(value=np.float32(np.nan)))
            changed=copy.deepcopy(state);changed['center.weight'].add_(.1);torch.save(changed,path)
            with self.assertRaises(AssertionError):E.load_branch(ctx,'eventgenesis',0,'clean_control','depth')

    def test_parity_requires_all_summary_fields_and_finite_values(self):
        E.compare_summary({'success':.3,'depth':.04},{'success':.3,'depth':.04})
        with self.assertRaises(AssertionError):E.compare_summary({'success':.31},{'success':.3})
        with self.assertRaises(AssertionError):E.compare_summary({'success':float('nan')},{'success':.3})


if __name__=='__main__':unittest.main(verbosity=2)
