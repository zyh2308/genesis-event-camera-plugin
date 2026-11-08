"""CPU fixtures for immutable internal swaps and gradient-only measurements."""
import ast
from pathlib import Path
import unittest
import torch
from torch import nn

from pose_forgetting_diagnostic import (swap_state,state_sha,summarize_gradients,gradient_vectors,fixed_subset)


class DiagnosticTests(unittest.TestCase):
    def test_swaps_are_copy_only_and_all_modules_recover_control(self):
        control={name:torch.tensor([float(i)]) for i,name in enumerate(('features.weight','center.weight','size.weight','grid'))}
        candidate={name:value+10 if name!='grid' else value.clone() for name,value in control.items()}
        before=state_sha(candidate)
        mixed=swap_state(candidate,control,['features.'])
        torch.testing.assert_close(mixed['features.weight'],control['features.weight'])
        torch.testing.assert_close(mixed['size.weight'],candidate['size.weight'])
        self.assertEqual(state_sha(candidate),before)
        self.assertEqual(state_sha(swap_state(candidate,control,['features.','center.','size.'])),state_sha(control))
        mixed['features.weight'].add_(5)
        self.assertEqual(state_sha(candidate),before)
        wrong={**control,'grid':torch.tensor([-1.])}
        with self.assertRaises(AssertionError):swap_state(candidate,wrong,['features.'])

    def test_known_condition_conflict_and_undefined_zero_gradient(self):
        vectors={key:{'features.':torch.tensor(value)} for key,value in {
            'normal':[1.,0.],'half':[1.,0.],'eighth':[-4.,0.],
            'normal_teacher':[0.,0.],'half_teacher':[-1.,0.]}.items()}
        out=summarize_gradients(vectors)['features.']
        self.assertAlmostEqual(out['half_eighth_cosine'],-1.)
        self.assertAlmostEqual(out['legacy_mixture_half_cosine'],-1.)
        self.assertAlmostEqual(out['priority_mixture_half_cosine'],1.)
        self.assertAlmostEqual(out['teacher_half_vs_half_GT_cosine'],-1.)
        self.assertIsNone(out['teacher_normal_vs_half_GT_cosine'])

    def test_autograd_measurement_never_updates_state_or_parameter_grad(self):
        model=nn.Module();model.features=nn.Linear(2,2);model.center=nn.Linear(2,1);model.size=nn.Linear(2,1)
        before=state_sha(model.state_dict())
        x=model.features(torch.tensor([[.3,.7]]));loss=(model.center(x).square()+model.size(x).square()).mean()
        vectors=gradient_vectors(loss,model)
        self.assertEqual(set(vectors),{'features.','center.','size.'})
        self.assertTrue(all(torch.isfinite(v).all() for v in vectors.values()))
        self.assertEqual(state_sha(model.state_dict()),before)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_fixed_subset_is_bounded_and_worker_has_no_optimizer(self):
        rows=[dict(rid=rid,id=f'{rid}/{i}') for rid in ('b','a') for i in range(16)]
        selected=fixed_subset(rows,4)
        self.assertEqual(len(selected),8);self.assertEqual(len({r['id'] for r in selected}),8)
        self.assertEqual([r['id'] for r in selected[:4]],['a/0','a/5','a/10','a/15'])
        source=Path(__file__).with_name('pose_forgetting_diagnostic.py').read_text()
        ast.parse(source)
        self.assertNotIn('torch.optim',source)
        self.assertNotIn('.step(',source)
        self.assertNotIn('.backward(',source)


if __name__=='__main__':unittest.main(verbosity=2)
