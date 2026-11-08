"""CPU original-source structure tests; no project runs, models, or GPU."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from unittest.mock import patch
import native_pose_shared as N
import softfocus_probability as M
import pose_adapter as A
import run_readout as D


class TinyCenter(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,1,1);self.register_buffer('grid',torch.as_tensor(M.cell_center_grid().reshape(256,2),dtype=torch.float32))
    def forward(self,x):
        logits=torch.nn.functional.adaptive_avg_pool2d(self.conv(x),(16,16)).flatten(1);uv=logits.softmax(1)@self.grid
        return logits,uv,uv[:,0]*0,uv[:,0]*0


class FakeRotation:
    def __init__(self):self.crops=[];self.ray_centers=[]
    def ray_frame(self,centers,K):
        self.ray_centers.append(np.array(centers,copy=True));rays=[];axes=[]
        for (u,v),k in zip(centers,K):
            rays.append([(u*346-k[0,2])/k[0,0],(v*260-k[1,2])/k[1,1],1.]);t=u-.5;c,s=np.cos(t),np.sin(t);axes.append([[c,0,s],[0,1,0],[-s,0,c]])
        return np.asarray(rays),np.asarray(axes)
    def crop(self,images,centers,sizes,aspect,angle):
        self.crops.append((centers.clone(),sizes.clone()));out=images.clone();out[:,0]=centers[:,0,None,None];out[:,1]=centers[:,1,None,None];return out,None


class FakeHead(nn.Module):
    def __init__(self):super().__init__();self.register_buffer('fixed',torch.tensor(0.))
    def forward(self,images):
        t=images[:,0,0,0]-.5;c,s=t.cos(),t.sin();z=torch.zeros_like(t);o=torch.ones_like(t)
        return torch.stack((c,-s,z,s,c,z,z,z,o),1).reshape(-1,3,3)


def scene():
    vertices=np.array([[x,y,z] for x in (-.01,.01) for y in (-.01,.01) for z in (-.01,.01)])
    keys=np.vstack((np.zeros(3),vertices));K=np.array([[100.,0,173],[0,100,130],[0,0,1.]])
    T=np.eye(4);T[2,3]=2.;rows=[dict(rid='fixture',id=f'fixture/{i:03}',T=T,keys=keys,K=K,uv=np.array([[.5,.5]])) for i in range(2)]
    rotation=FakeRotation();ctx=SimpleNamespace(shared=N,rotation=rotation,vertices=vertices,threshold=.01)
    return ctx,rows,cKDTree(vertices)


class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_native_fp32_uv_and_fixed_numpy_candidate_keep_inputs_states_and_batch_order(self):
        torch.manual_seed(7);model=TinyCenter().eval();images=torch.rand(33,3,32,32);before=images.clone();state=copy.deepcopy(model.state_dict())
        out=A.center_readouts(model,images,M)
        with torch.inference_mode():native=torch.cat([model(images[i:i+32])[1] for i in range(0,33,32)]).numpy()
        np.testing.assert_array_equal(out['control_uv'],native);self.assertEqual(out['candidate_uv'].shape,(33,2));self.assertEqual(out['weights'].dtype,np.float64)
        self.assertTrue(torch.equal(images,before));self.assertTrue(all(torch.equal(v,state[k]) for k,v in model.state_dict().items()))
        for i in (0,31,32):np.testing.assert_array_equal(out['candidate_uv'][i],M.softfocus_predicted_peak(out['attention'][i])['uv'])

    def test_full_original_pose_chain_recomputes_both_crop_and_ray_frame_with_frozen_z_size(self):
        self.assertEqual(D.sha(Path(N.__file__)),D.SHARED_SHA);ctx,rows,tree=scene();head=FakeHead().eval();images=torch.zeros(2,3,4,4)
        uv0=np.full((2,2),.5,np.float32);uv1=uv0.astype(float)+.01;z=np.full(2,2.);size=np.full(2,.1);originals=(z.copy(),size.copy());gt=np.full(2,np.log(.1))
        ref=N.score(ctx.rotation,rows,uv0,z,size,gt,np.tile(np.eye(3),(2,1,1)),ctx.vertices,tree,ctx.threshold)
        ctx.rotation.ray_centers=[];out=A.pose_pair(ctx,head,images,rows,uv0,uv1,z,size,gt,tree,ref)
        self.assertEqual(len(ctx.rotation.crops),2);self.assertEqual(len(ctx.rotation.ray_centers),4)
        np.testing.assert_allclose(ctx.rotation.crops[0][0].numpy(),uv0);np.testing.assert_allclose(ctx.rotation.crops[1][0].numpy(),uv1)
        self.assertFalse(np.array_equal(out['control']['R'],out['candidate']['R']));self.assertEqual(out['control']['score']['summary']['success'],1.)
        self.assertLess(out['candidate']['score']['summary']['success'],1.);np.testing.assert_array_equal(z,originals[0]);np.testing.assert_array_equal(size,originals[1])

    def test_bad_control_pose_parity_stops_before_candidate_rotation(self):
        ctx,rows,tree=scene();images=torch.zeros(2,3,4,4);uv=np.full((2,2),.5);z=np.full(2,2.);size=np.full(2,.1);gt=np.full(2,np.log(.1))
        ref=N.score(ctx.rotation,rows,uv,z,size,gt,np.tile(np.eye(3),(2,1,1)),ctx.vertices,tree,ctx.threshold);ref['summary']['success']=0
        with self.assertRaises(ValueError):A.pose_pair(ctx,FakeHead().eval(),images,rows,uv,uv+.01,z,size,gt,tree,ref)
        self.assertEqual(len(ctx.rotation.crops),1)

    def test_full_requires_actual_baseline_bound_to_source_math_and_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'completion.json';value=dict(schema='pose_readout_control_driver_v1',status='complete',mode='baseline',optimizer_updates=0,real_data_used=False,
                registration_sha256=D.REG_SHA,contract_sha256=D.CONTRACT_SHA,source_manifest_sha256='source',math_manifest_sha256=D.MATH_MANIFEST,
                paired_cases=384,score_rows=12,all_reference_checks_passed=True,all_models_unchanged=True)
            D.write(p,value);D.baseline_admission(p,D.sha(p),'source')
            for key,bad in [('paired_cases',383),('all_reference_checks_passed',False),('math_manifest_sha256','wrong'),('real_data_used',True)]:
                wrong=dict(value,**{key:bad});q=Path(tmp)/(key+'.json');D.write(q,wrong)
                with self.assertRaises(ValueError):D.baseline_admission(q,D.sha(q),'source')

    def test_actual_registration_math_contract_and_native_source_sha_are_fixed(self):
        p=Path(__file__).parent;self.assertEqual(D.sha(p/'registration.json'),D.REG_SHA);self.assertEqual(D.sha(p/'softfocus_probability.py'),D.MATH_SHA)
        r=D.read(p/'registration.json');self.assertEqual((r['input'],r['arm'],r['legacy_arm']),('RAW','C1','physical_supervised'));self.assertEqual(r['sigma_normalized'],1/16)
        self.assertFalse(r['real_data_used']);self.assertEqual(r['optimizer_updates'],0)


if __name__=='__main__':unittest.main(verbosity=2)
