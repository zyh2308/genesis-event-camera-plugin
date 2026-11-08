"""Targeted CPU loss/units/unused/projection controls; no models or data loaded."""
import types,unittest
import numpy as np
import torch
from torch.nn import functional as F
import probe_pose_training_losses as P

class Formula(unittest.TestCase):
    def setUp(self):torch.manual_seed(11)
    def test_components_match_native_supervision_formula_and_gradient(self):
        grid=torch.tensor([[.25,.25],[.75,.25],[.25,.75],[.75,.75]])
        model=types.SimpleNamespace(grid=grid);y=torch.rand((3,9));y[:,2:5]-=1
        pred=tuple(torch.randn(shape,requires_grad=True) for shape in ((3,4),(3,2),(3,),(3,)))
        parts=P.depth_components(model,pred,y);actual=sum(parts.values());logits,uv,s,q=pred
        heat=(-.5*(grid[None]-y[:,None,:2]).square().sum(-1)*16**2).softmax(1)
        smooth=lambda a,b:F.smooth_l1_loss(a,b,beta=.1)
        expected=-(heat*logits.log_softmax(1)).sum(1).mean()+4*smooth(uv,y[:,:2])+2*smooth(s,y[:,2])+smooth(q,y[:,4]-y[:,3]+y[:,2])+2*smooth(q+y[:,3]-s,y[:,4])
        self.assertTrue(torch.equal(actual,expected))
        ga=torch.autograd.grad(actual,pred,retain_graph=True);gb=torch.autograd.grad(expected,pred)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(ga,gb)))
    def test_shape_size_depth_targets_are_algebraically_consistent(self):
        y=torch.rand((4,9));s=y[:,2];q=y[:,4]-y[:,3]+s
        self.assertTrue(torch.allclose(q+y[:,3]-s,y[:,4],atol=1e-7))
    def test_unused_and_frozen_parameters_are_explicit_zero(self):
        model=torch.nn.Module();model.features=torch.nn.Linear(2,2);model.center=torch.nn.Linear(2,1);model.size=torch.nn.Linear(2,2);model.size.requires_grad_(False)
        loss=model.center(model.features(torch.ones((3,2)))).square().mean();v,m=P.collect(loss,model)
        self.assertEqual(float(v['size'].norm()),0.);self.assertEqual(m['named_parameters'],6);self.assertEqual(m['trainable_parameters'],4)
        self.assertTrue(all(x.grad is None for x in model.parameters()))
    def test_projection_uses_one_full_space_coefficient_and_reports_cancellation(self):
        def v(a):
            t=torch.tensor(a,dtype=torch.float64);return dict(full=t,features=t[:1],center=t[1:],size=torch.zeros(0,dtype=torch.float64))
        data={c+'/supervised':v([1.,0.]) for c in P.CONDITIONS};data['normal_teacher']=v([-1.,.5])
        result=P.mixture_diagnostics(data);self.assertAlmostEqual(result['projected_teacher_diagnostic']['full_space_removed_negative_coefficient'],-2.)
        self.assertAlmostEqual(result['comparisons']['actual_expected_mixture_vs_half']['full']['cosine'],-2**-.5)
        self.assertGreater(result['comparisons']['projected_teacher_mixture_vs_half']['full']['cosine'],0.)
        self.assertLess(result['cancellation_ratio']['full'],1.)
    def test_fixed32_train_selection_has_no_error_based_choice(self):
        ids=[f'train_{i:03d}' for i in range(8)];rows=[dict(rid=r,id=f'{r}/{k:03d}') for r in ids for k in range(16)]
        selected=P.select_cohort(rows,ids);self.assertEqual(len(selected),32);self.assertEqual([x['id'] for x in selected[:4]],['train_000/000','train_000/005','train_000/010','train_000/015'])
        with self.assertRaises(ValueError):P.select_cohort(rows[:-14],ids)

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
