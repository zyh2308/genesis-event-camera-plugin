"""Synthetic CPU controls; no learned model, CP, training, GPU, real data or SSH."""
import copy
import json
import math
import unittest

import torch

import soft_census as c

torch.set_num_threads(2)


def analytic_pair(transform='identity'):
    yy, xx = torch.meshgrid(torch.arange(32, dtype=torch.float32), torch.arange(40, dtype=torch.float32), indexing='ij')
    def image(x):
        return .45 + .15*torch.sin(.41*x) + .1*torch.cos(.37*yy) + .08*torch.sin(.17*x+.29*yy)
    previous=image(xx)[None,None];following=image(xx-1)[None,None]
    if transform=='affine':following=.7*following+.1
    if transform=='gamma':following=following.pow(.7)
    return previous,following


def objective_at(previous,following,displacement):
    d=torch.tensor(displacement,dtype=torch.float32,requires_grad=True)
    x=torch.ones_like(previous[:,0])*d
    flow=torch.stack((x,torch.zeros_like(x)),dim=1)
    loss,details=c.census_loss_sum(previous,following,flow)
    derivative=torch.autograd.grad(loss,d)[0]
    return float(loss.detach()),float(derivative),float(details['loss_support'].mean())


class CensusControls(unittest.TestCase):
    def test_author_transform_and_identity_pixel_sum(self):
        p,_=analytic_pair();p=p.double();flow=torch.zeros(1,2,*p.shape[-2:],dtype=torch.float64,requires_grad=True)
        loss,details=c.census_loss_sum(p,p,flow)
        self.assertLess(float(details['soft_hamming'].detach().abs().max()),1e-8)
        self.assertAlmostEqual(float(loss.detach()),float(details['loss_support'].sum())*(.01**.4),places=3)
        self.assertFalse(details['loss_support'].requires_grad)
        self.assertFalse(details['predicted_inbounds'].requires_grad)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss,flow)[0]).all())

    def test_translation_and_monotonic_photometry_gradients(self):
        self.oracle_rows=[]
        for transform in ('identity','affine','gamma'):
            p,n=analytic_pair(transform)
            left=objective_at(p,n,.5);correct=objective_at(p,n,1.);right=objective_at(p,n,1.5);zero=objective_at(p,n,0.)
            with self.subTest(transform=transform):
                self.assertLess(correct[0],left[0]);self.assertLess(correct[0],right[0]);self.assertLess(correct[0],zero[0])
                self.assertLess(left[1],0.);self.assertGreater(right[1],0.)
            self.oracle_rows.append(dict(transform=transform,flow_half=left,flow_true_one=correct,flow_one_half=right,flow_zero=zero))

    def test_outbound_zero_and_constant_patch_unobservable(self):
        p,n=analytic_pair();flow=torch.full((1,2,*p.shape[-2:]),100.,requires_grad=True)
        loss,details=c.census_loss_sum(p,n,flow)
        self.assertEqual(float(loss.detach()),0.);self.assertEqual(float(details['loss_support'].sum()),0.)
        self.assertEqual(float(torch.autograd.grad(loss,flow)[0].abs().max()),0.)
        p=torch.full((1,1,32,40),.3);n=torch.full_like(p,.7)
        flow=torch.zeros(1,2,32,40,requires_grad=True);loss,details=c.census_loss_sum(p,n,flow)
        self.assertEqual(float((details['soft_hamming']*details['loss_support']).detach().abs().max()),0.)
        self.assertEqual(float(torch.autograd.grad(loss,flow)[0].abs().max()),0.)

    def test_four_scale_native_sum_reduction_and_batch_duplication(self):
        p,_=analytic_pair();p=torch.nn.functional.interpolate(p,size=(64,64),mode='bilinear',align_corners=False)
        outputs=[torch.zeros(1,2,n,n,requires_grad=True) for n in (64,32,16,8)]
        one,_=c.multiscale_census_sum(outputs,p,p)
        two,_=c.multiscale_census_sum([torch.cat([x,x]) for x in outputs],torch.cat([p,p]),torch.cat([p,p]))
        self.assertAlmostEqual(float(two.detach()),2*float(one.detach()),places=3)
        with self.assertRaises(ValueError):c.multiscale_census_sum(outputs[:3],p,p)
        with self.assertRaises(ValueError):c.native_gray_resize(p,(20,30),outputs[0])

    def test_input_and_no_gt_mask_contract(self):
        p,n=analytic_pair();flow=torch.zeros(1,2,*p.shape[-2:])
        with self.assertRaises(TypeError):c.census_loss_sum(p,n,flow,GT_mask=torch.ones_like(p))
        with self.assertRaises(ValueError):c.census_loss_sum(p*255,n,flow)
        bad=p.clone();bad[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):c.census_loss_sum(bad,n,flow)

    def test_one_shared_train_only_gradient_calibration(self):
        rows=[]
        for method in ('eventgenesis','v2e'):
            for i,wid in enumerate(('train_a_1','train_b_3')):
                rows.append(dict(method=method,window_id=wid,split='train',optimizer_updates=0,GT_used=False,real_data_used=False,
                    gradient_space='four_native_flow_outputs',model_state_sha256='a'*64,
                    physical_label_and_augmentation_sha256=str(i)*64,native_photo_gradient_l2=4.+i*2,census_photo_gradient_l2=2.+i))
        r=c.calibrate_shared_weight(rows,['train_a_1','train_b_3']);self.assertEqual(r['shared_census_weight'],2.)
        json.dumps(r,allow_nan=False)
        for key,value in [('split','validation'),('GT_used',True),('real_data_used',True),('optimizer_updates',1),
                          ('census_photo_gradient_l2',float('nan')),('physical_label_and_augmentation_sha256','wrong')]:
            v=copy.deepcopy(rows);v[0][key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):c.calibrate_shared_weight(v,['train_a_1','train_b_3'])
        with self.assertRaises(ValueError):c.calibrate_shared_weight(rows[:-1],['train_a_1','train_b_3'])


if __name__=='__main__':unittest.main(verbosity=2)
