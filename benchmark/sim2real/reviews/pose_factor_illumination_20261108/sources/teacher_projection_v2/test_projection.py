"""Meaningful CPU algebra, full-space, dtype, zero, and mutation controls."""
import json,unittest
import torch
from projection import project_teacher_gradients

def case(p,t,dtype=torch.float64):
    parameters=[torch.nn.Parameter(torch.tensor([0.],dtype=dtype)) for _ in range(24)]
    def grads(values):return [None if v is None else torch.tensor([v],dtype=dtype) for v in values+[0.]*(24-len(values))]
    return parameters,grads(p),grads(t)

class Projection(unittest.TestCase):
    def test_negative_teacher_removes_only_parallel_component(self):
        ps,p,t=case([1.,0.],[-3.,4.]);r=project_teacher_gradients(ps,p,t)
        self.assertEqual(r['diagnostics']['coefficient'],-3.)
        self.assertEqual([float(x) for x in r['projected_teacher_grads'][:2]],[0.,4.])
        self.assertEqual([float(x) for x in r['combined_grads'][:2]],[1.,4.])
        self.assertEqual(r['diagnostics']['physical_projected_teacher_dot'],0.)
        self.assertEqual(r['diagnostics']['physical_combined_dot'],r['diagnostics']['physical_gradient_norm_squared'])
    def test_one_global_coefficient_is_not_per_tensor_clipping(self):
        ps,p,t=case([1.,1.],[-3.,1.]);r=project_teacher_gradients(ps,p,t)
        self.assertEqual(r['diagnostics']['coefficient'],-1.)
        self.assertEqual([float(x) for x in r['projected_teacher_grads'][:2]],[-2.,2.])
        self.assertEqual(r['diagnostics']['physical_projected_teacher_dot'],0.)
    def test_positive_or_orthogonal_teacher_is_bitwise_unchanged(self):
        for p,t in [([1.,2.],[2.,1.]),([1.,0.],[0.,4.])]:
            ps,p,t=case(p,t,dtype=torch.float32);r=project_teacher_gradients(ps,p,t)
            self.assertFalse(r['diagnostics']['projection_applied']);self.assertEqual(r['diagnostics']['coefficient'],0.)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(t,r['projected_teacher_grads'])))
            self.assertTrue(all(a.data_ptr()!=b.data_ptr() for a,b in zip(t,r['projected_teacher_grads'])))
    def test_zero_physical_and_unused_slots_preserve_teacher(self):
        ps,p,t=case([None,None],[2.,-4.]);r=project_teacher_gradients(ps,p,t)
        self.assertEqual(r['diagnostics']['reason'],'zero_physical_norm');self.assertIsNone(r['diagnostics']['physical_teacher_cosine'])
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(t,r['combined_grads'])))
        ps,p,t=case([1.,None],[-2.,None]);r=project_teacher_gradients(ps,p,t)
        self.assertEqual(r['diagnostics']['physical_unused_slots'],[1]);self.assertEqual(r['diagnostics']['teacher_unused_slots'],[1])
        self.assertEqual(float(r['combined_grads'][1]),0.);self.assertEqual(len(r['combined_grads']),24)
    def test_no_parameter_input_or_grad_field_mutation_and_no_graph(self):
        ps,p,t=case([1.,1.],[-3.,1.],dtype=torch.float32)
        for param in ps:param.grad=torch.full_like(param,17.)
        old=[(x.detach().clone(),x.grad,x.grad.clone()) for x in ps];pp=[x.clone() for x in p];tt=[x.clone() for x in t]
        p[0].requires_grad_(True);t[0].requires_grad_(True)
        r=project_teacher_gradients(ps,p,t)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(p,pp)));self.assertTrue(all(torch.equal(x,y) for x,y in zip(t,tt)))
        for param,(value,old_grad,grad_value) in zip(ps,old):
            self.assertTrue(torch.equal(param,value));self.assertIs(param.grad,old_grad);self.assertTrue(torch.equal(param.grad,grad_value))
        self.assertIsNone(p[0].grad);self.assertIsNone(t[0].grad)
        self.assertTrue(all(not x.requires_grad and x.grad_fn is None for x in r['combined_grads']+r['projected_teacher_grads']))
        json.dumps(r['diagnostics'],allow_nan=False)
    def test_fp64_projection_does_not_cast_huge_coefficient_to_fp32(self):
        ps,p,t=case([1e-40],[-1.],dtype=torch.float32);r=project_teacher_gradients(ps,p,t)
        self.assertTrue(r['diagnostics']['projection_applied']);self.assertLess(r['diagnostics']['coefficient'],-1e39)
        self.assertTrue(all(torch.isfinite(x).all() for x in r['combined_grads']))
        self.assertAlmostEqual(float(r['projected_teacher_grads'][0]),0.,places=6)
    def test_bad_scope_shapes_and_nonfinite_fail_without_clipping(self):
        ps,p,t=case([1.],[-1.])
        with self.assertRaises(ValueError):project_teacher_gradients(ps[:-1],p[:-1],t[:-1])
        bad=t.copy();bad[0]=torch.ones(2,dtype=torch.float64)
        with self.assertRaises(ValueError):project_teacher_gradients(ps,p,bad)
        bad=t.copy();bad[0]=torch.tensor([float('nan')],dtype=torch.float64)
        with self.assertRaises(ValueError):project_teacher_gradients(ps,p,bad)
        with self.assertRaises(ValueError):project_teacher_gradients(ps,p,t,parameter_names=['same']*24)

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
