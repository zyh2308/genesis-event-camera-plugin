"""Portable CPU correctness tests using byte-pinned native source fixtures."""
import ast,hashlib,json,unittest
from pathlib import Path
import numpy as np
import torch
import photometric_math as M
import pinned_compat as C
import pinned_multiscaleloss as L
import run_photometric_alignment as W

class Correctness(unittest.TestCase):
    def test_controlled_translation_direction_and_interventions(self):
        r=M.controls(C.loss_warp,torch.device('cpu'));self.assertTrue(r['clean_required_controls_passed']);self.assertEqual(len(r['cases']),4)
    def test_analytic_original_warp_sign(self):
        x=torch.arange(7).float()[None,None,None].expand(1,1,6,7);f=torch.zeros((1,2,6,7));f[:,0]=2
        y=C.loss_warp(x,f);self.assertTrue(torch.equal(y[:,:,:,:5],x[:,:,:,2:]));self.assertTrue(torch.equal(y[:,:,:,-2:],torch.zeros_like(y[:,:,:,-2:])))
    def test_unit_scaling_and_invalid_interpolation(self):
        flow=torch.ones((1,2,8,8));flow[:,0]*=4;flow[:,1]*=-2
        valid=torch.ones((1,1,8,8));valid[:,:,0,0]=0
        target,strict,nearest,active,sensor=M.target_at_scale(flow,valid,valid,torch.ones_like(valid),(4,4))
        self.assertTrue(torch.all(target[:,0]==2));self.assertTrue(torch.all(target[:,1]==-1));self.assertFalse(bool(strict[0,0,0]));self.assertTrue(bool(strict[0,1,1]))
    def test_native_charbonnier_formula(self):
        for value in (0.,1e-6,.1,1.):
            r=torch.full((1,1,4,4),value);field=torch.zeros((1,2,4,4))
            actual=M.native_map(torch.zeros_like(r),r,field,C.loss_warp).sum()
            self.assertTrue(torch.equal(L.charbonnier_loss(r),actual))
    def test_mask_only_is_not_new_learning_signal(self):
        gray=torch.rand((1,1,16,16));p=torch.ones((1,2,16,16))*2.25;target=p*.5;valid=torch.ones((1,16,16),dtype=torch.bool)
        row=M.analyze_scale(gray,gray,p,target,valid,valid,valid,valid,C.loss_warp)
        self.assertEqual(row['mask_only_diagnostic']['max_abs_gradient_difference'],0.)
    def test_invalid_GT_never_has_oracle(self):
        gray=torch.rand((1,1,8,8));pred=torch.zeros((1,2,8,8));valid=torch.zeros((1,8,8),dtype=torch.bool);sensor=~valid
        row=M.analyze_scale(gray,gray,pred,pred,valid,valid,valid,sensor,C.loss_warp)['regions']['invalid_geometry_or_visibility']
        self.assertIsNone(row['photo_mean_GT_valid_only']);self.assertIsNone(row['gradient_toward_GT_cosine']);self.assertEqual(row['oracle_valid_pixels'],0)
    def test_native_source_fixture_hashes(self):
        p=Path(__file__).parent
        self.assertEqual(hashlib.sha256((p/'pinned_multiscaleloss.py').read_bytes()).hexdigest(),'ff401317e49ce37ae0f8e9bfd6b82f133549b4a0271be35095f74bdc50a439c0')
        self.assertEqual(hashlib.sha256((p/'pinned_compat.py').read_bytes()).hexdigest(),'0f7290aa45b8ba020c03d2d26b64a0cc213ccec7536e386bce186e4d0446393e')
    def test_no_optimizer_or_parameter_update(self):
        t=ast.parse((Path(__file__).parent/'run_photometric_alignment.py').read_text());attrs={n.attr for n in ast.walk(t) if isinstance(n,ast.Attribute)}
        self.assertFalse({'optim','backward','step','zero_grad'}&attrs);self.assertIn('grad',attrs)
    def test_contract_fixed_scope_and_oracle_only(self):
        c=json.loads((Path(__file__).parent/'contract.json').read_text());self.assertEqual(c['optimizer_updates'],0);self.assertFalse(c['automatic_training']);self.assertEqual(c['checkpoint_source'],'complete8910_last')
        self.assertEqual(c['validation_indices'],[0,22,43,65]);self.assertEqual(c['training_sample_indices'],[11,44])
    def test_parameter_cosine_keeps_unshared_gradient_norm(self):
        r=W.gradient_agreement([torch.ones(1),torch.ones(1)],[torch.ones(1),None],['a','b'])
        self.assertAlmostEqual(r['cosine'],2**-.5);self.assertEqual(r['GT_directional_derivative_for_negative_photo_gradient'],-1.)
    def test_parameter_inventory_is82_but_involved_gradient70(self):
        c=json.loads((Path(__file__).parent/'contract.json').read_text());oracle=c['parameter_oracle']
        self.assertEqual(oracle['named_parameter_tensors'],82);self.assertEqual(oracle['trainable_parameter_tensors'],82)
        self.assertEqual(oracle['expected_native_photo_nonNone'],70);self.assertTrue(oracle['allow_unused'])
        text=(Path(__file__).parent/'run_photometric_alignment.py').read_text()
        self.assertIn('parameter_inventory_sha256',text);self.assertNotIn('need(len(named)==70',text)

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
