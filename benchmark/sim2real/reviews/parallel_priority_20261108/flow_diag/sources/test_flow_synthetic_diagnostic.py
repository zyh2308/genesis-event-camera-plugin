"""Small local CPU tests, no server connection/model/checkpoint/public dataset."""
import json
import unittest
import numpy as np
import torch
import torch.nn.functional as F

import flow_synthetic_diagnostic as d


def warp(image, flow):
    b, _, h, w = image.shape
    yy, xx = torch.meshgrid(torch.arange(h, dtype=flow.dtype), torch.arange(w, dtype=flow.dtype), indexing="ij")
    grid = torch.stack((xx, yy), 0)[None] + flow
    grid[:, 0] = 2 * grid[:, 0] / (w - 1) - 1
    grid[:, 1] = 2 * grid[:, 1] / (h - 1) - 1
    sample_grid = grid.permute(0, 2, 3, 1)
    output = F.grid_sample(image, sample_grid, align_corners=True)
    valid = F.grid_sample(torch.ones_like(image), sample_grid, align_corners=True) >= .9999
    return output * valid


class MetricsTests(unittest.TestCase):
    def test_bins_support_zero_flow_and_invalid_distinction(self):
        gt = np.zeros((2, 2, 3), dtype=np.float32)
        gt[0] = 2
        valid = np.ones((2, 3), dtype=np.uint8)
        support = np.zeros((2, 3), bool); support[:, 0] = True
        result = d.region_summary(np.zeros_like(gt), gt, valid, support)
        self.assertEqual(result["all_physical_valid"]["epe_px"], 2)
        self.assertEqual(result["common_event_active_valid"]["pixels"], 2)
        self.assertEqual(result["event_inactive_valid"]["pixels"], 4)
        self.assertEqual(result["motion_2_5px"]["pixels"], 6)
        valid[0, 0] = 0
        result = d.region_summary(None, gt, valid, support)
        self.assertEqual(result["invalid_geometry_or_visibility"]["pixels"], 1)
        self.assertNotIn("epe_px", result["invalid_geometry_or_visibility"])
        json.dumps(result, allow_nan=False)

    def test_events_not_shared_labels(self):
        events = np.zeros((2, 3, 4, 5), dtype=np.uint8)
        events[0, 1, 2, 4] = 1
        self.assertEqual(d.active(events).sum(), 1)
        self.assertTrue(d.active(events)[1, 2])

    def test_gradient_probe_is_finite_and_never_updates_field(self):
        # Test diagnostic plumbing with a transparent CPU fixture objective;
        # not a claim of upstream CUDA objective equivalence or model quality.
        torch.set_num_threads(2); torch.manual_seed(20261108)
        class FixtureLoss:
            pass
        FixtureLoss.warp = staticmethod(warp)
        class FixtureTrainer:
            @staticmethod
            def objective_dt1(fields, previous, following, event, loss):
                photo = sum((warp(F.interpolate(following, size=x.shape[-2:], mode="bilinear", align_corners=False), x)
                    - F.interpolate(previous, size=x.shape[-2:], mode="bilinear", align_corners=False)).square().add(1e-6).pow(.45).sum() for x in fields)/len(fields)
                smooth = sum((x[..., 2:, :] - 2*x[..., 1:-1, :] + x[..., :-2, :]).abs().mean()/(2**i) for i,x in enumerate(fields))
                return photo + 10*smooth, photo, smooth
        gray=torch.rand(1,2,256,256)
        flow=torch.zeros(1,2,256,256);valid=torch.ones(1,1,256,256)
        common=torch.ones(1,1,256,256,dtype=torch.bool)
        fields=tuple(torch.randn(1,2,n,n)*.01 for n in [256,128,64,32])
        snapshots=[x.clone() for x in fields]
        result=d.gradient_probe(FixtureTrainer,FixtureLoss,fields,gray,torch.zeros(1,2,256,256,5),flow,valid,common)
        self.assertEqual(len(result['per_scale']),4)
        self.assertFalse(result['GT_used_in_optimizer'])
        self.assertGreater(result['photo_gradient_l2'],0)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(fields,snapshots)))
        self.assertTrue(all(x.grad is None for x in fields))
        json.dumps(result,allow_nan=False)


if __name__=="__main__":unittest.main(verbosity=2)
