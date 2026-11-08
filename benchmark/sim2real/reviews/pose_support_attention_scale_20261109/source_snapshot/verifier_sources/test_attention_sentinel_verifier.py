"""CPU fixtures only. Root may run these; no project inputs or GPU needed."""
import os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import verify_attention_sentinels as V

class NumericTests(unittest.TestCase):
    def test_original64_forward_keeps_two_batch32_and_only_subselects_after(self):
        import torch
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.features=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((16,16)),torch.nn.Conv2d(3,64,1));self.center=torch.nn.Conv2d(64,1,1)
                y,x=torch.meshgrid((torch.arange(16)+.5)/16,(torch.arange(16)+.5)/16,indexing='ij');self.register_buffer('grid',torch.stack((x,y),-1).reshape(-1,2));self.batch_sizes=[]
            def forward(self,x):
                self.batch_sizes.append(len(x));logits=self.center(self.features(x)).flatten(1);return logits,logits.softmax(1)@self.grid,None,None
        torch.manual_seed(9);model=Toy().eval();images=torch.rand(64,3,128,128);before=images.clone()
        with torch.inference_mode():chunks=list(V.original_chunks(model,images,torch))
        self.assertEqual(model.batch_sizes,[32,32]);self.assertEqual([(c[0],c[1]) for c in chunks],[(0,32),(32,32)]);self.assertTrue(torch.equal(images,before))
        self.assertEqual(len(model.features._forward_hooks),0);self.assertEqual(len(model.center._forward_hooks),0)

    def test_uniform_attention_mass_enrichment_and_empty_null(self):
        mask=np.zeros((260,346),bool);mask[:130]=True
        f=V.independent_fraction(mask);a=np.full((16,16),1/256,np.float32)
        r=V.independent_metrics(mask,f,a,[.5,.25],[.5,.75])
        self.assertAlmostEqual(r['region_attention_mass'],f[1],places=12);self.assertAlmostEqual(r['region_mass_enrichment'],1,places=12)
        self.assertTrue(r['prediction_in_region']);self.assertFalse(r['peak_in_region'])
        empty=V.independent_metrics(mask&False,V.independent_fraction(mask&False),a,[.5,.5],[.5,.5])
        self.assertFalse(empty['available']);self.assertIsNone(empty['region_attention_mass']);self.assertIsNone(empty['region_mass_enrichment']);self.assertIsNone(empty['prediction_in_region'])

    def test_native_pixel_boundary_membership_and_discrete_failure(self):
        mask=np.zeros((260,346),bool);mask[10,20]=True
        self.assertTrue(V.member(mask,[20.25/346,10.25/260]));self.assertFalse(V.member(mask,[21.25/346,10.25/260]));self.assertFalse(V.member(mask,[1,1]))
        r=V.independent_metrics(mask,V.independent_fraction(mask),np.ones((16,16))/256,[20.25/346,10.25/260],[.5,.5]);wrong=dict(r,peak_in_region=not r['peak_in_region'])
        self.assertFalse(V.compare_region(r,wrong)[0]);self.assertTrue(V.compare_region(r,r)[0])

    def test_direct_h5_dynamic_slot_and_all21_validation(self):
        import h5py
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'r.h5';target=np.tile(np.eye(4),(21,1,1));target[:,2,3]=.7
            objects=np.tile(np.eye(4),(21,4,1,1));objects[:,3]=target
            seg=np.zeros((21,260,346),np.int32);seg[:,5:10,7:11]=37
            with h5py.File(path,'w') as f:
                g=f.create_group('episodes/r')
                for name,value in dict(target_pose=target,object_to_world=objects,object_segmentation_link_ids=np.array([2,9,15,37]),segmentation_link_idx=seg,depth_m=np.ones_like(seg,dtype=np.float32),target_visible_pixels=np.full(21,20),frame_timestamps_s=np.arange(21)*.01,camera_intrinsics_K_px=np.eye(3),cad_keypoints=np.zeros((9,3)),camera_to_world_cv=np.tile(np.eye(4),(21,1,1))).items():g[name]=value
            row=dict(K=np.eye(3),keys=np.zeros((9,3)),T=target[13])
            masks,meta=V.direct_regions(path,'r',row)
            self.assertEqual(meta['target_slot'],3);self.assertEqual(meta['target_id'],37);self.assertEqual(int(masks['endpoint'].sum()),20)
            with h5py.File(path,'r+') as f:f['episodes/r/target_visible_pixels'][0]=19
            with self.assertRaisesRegex(ValueError,'All21'):V.direct_regions(path,'r',row)

class ImportBoundaryTests(unittest.TestCase):
    def test_assignment_restored_before_original_guard_and_failure(self):
        fake=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False))
        for original in (V.GPU4,'4'):
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':original}):
                def loader(*_args):os.environ['CUDA_VISIBLE_DEVICES']='';return object()
                with patch.object(V,'load',side_effect=loader):V.safe_cpu_helper_import(fake,'module',Path('fixture.py'),'cuda')
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],original)
                def bad(*_args):os.environ['CUDA_VISIBLE_DEVICES']='';raise RuntimeError('fixture')
                with patch.object(V,'load',side_effect=bad),self.assertRaises(RuntimeError):V.safe_cpu_helper_import(fake,'module',Path('fixture.py'),'cuda')
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],original)

    def test_initialized_or_wrong_assignment_rejected_without_import(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'4'}),patch.object(V,'load') as load:
            with self.assertRaises(ValueError):V.safe_cpu_helper_import(SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True)),'module',Path('fixture.py'),'cuda')
            load.assert_not_called()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),patch.object(V,'load') as load:
            with self.assertRaises(ValueError):V.safe_cpu_helper_import(SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False)),'module',Path('fixture.py'),'cuda')
            load.assert_not_called()

    def test_helper_initializes_cuda_is_refused_without_context_repair(self):
        state={'initialized':False};fake=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:state['initialized']))
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'4'}):
            def loader(*_args):os.environ['CUDA_VISIBLE_DEVICES']='';state['initialized']=True
            with patch.object(V,'load',side_effect=loader),self.assertRaises(ValueError):V.safe_cpu_helper_import(fake,'module',Path('fixture.py'),'cuda')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'')

if __name__=='__main__':unittest.main(verbosity=2)
