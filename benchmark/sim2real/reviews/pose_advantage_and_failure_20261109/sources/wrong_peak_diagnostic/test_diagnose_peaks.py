import copy,unittest
from types import SimpleNamespace
import numpy as np
import diagnose_peaks as d


def fixture():
    p=np.zeros((16,16));p[8,8]=1;uv=d.grid(16)[8*16+8]
    case=dict(method='eventgenesis',seed=0,condition='eighth',rid='validation_000',sample_id='validation_000/004',gt_uv=[.5,.5],control_uv=uv,peak_uv=uv.tolist())
    part=dict(raw_events=10,baf_events=5,raw_occupancy_ratio=.1,event_retention_ratio=.5)
    region_support=dict(foreground=part,background_region=dict(part,raw_events=20,baf_events=10))
    sup=dict(pred_uv=uv,peak_uv=uv.tolist(),attention={n:dict(peak_in_region=False,prediction_in_region=False,region_attention_mass=.3,region_mass_enrichment=3) for n in ('endpoint','endpoint_union_approx')},
        nominal_GT_heat_reference=dict(endpoint=dict(region_attention_mass=.4),GT_anchor_outside_grid_range=False,target_decoded_center_bias_px=0),
        geometric_sanity=dict(GT_anchor_uv_px=[173,130],GT_anchor_visible_membership=dict(inside=True),GT_anchor_to_visible_centroid_distance_px=2),
        visible_geometry=dict(endpoint=dict(bbox_xyxy=[170,127,180,136],area_px=90,visible_mask_touches_border=False)),
        support=dict(regions=dict(endpoint=region_support)),motion=dict(max_projected_corner_displacement_px=2,projected_anchor_displacement_px=1),
        features=dict(local_l2=1,context_l2=2))
    mask=np.zeros((260,346),bool);mask[127:136,170:180]=True
    regions={n:SimpleNamespace(mask=mask.copy()) for n in ('endpoint','endpoint_union_approx')}
    pyramids={n:dict(available=True,attention16=np.zeros((16,16))) for n in ('endpoint','endpoint_union_approx')}
    return case,p,sup,regions,pyramids


class FullPeakControls(unittest.TestCase):
    def test_native_point_outside_but_cell_overlap_is_not_background(self):
        c,p,s,r,py=fixture();py['endpoint']['attention16'][8,8]=.1;py['endpoint_union_approx']['attention16'][8,8]=.2
        result=d.peak_diagnostics(c,p,s,r,py)['metrics']
        self.assertEqual(result['peak_category'],'native_point_outside_endpoint_but_peak_cell_has_target_support')
        self.assertFalse(result['endpoint_peak_point_hit']);self.assertTrue(result['endpoint_peak_cell_has_support'])
        self.assertAlmostEqual(result['peak_error_above_hard16_floor_px'],0)
        self.assertLess(result['hard_peak_grid128_floor_px'],result['hard_peak_grid16_floor_px'])

    def test_temporal_union_separate_and_native_mismatch_rejected(self):
        c,p,s,r,py=fixture();py['endpoint_union_approx']['attention16'][8,8]=.1
        result=d.peak_diagnostics(c,p,s,r,py)['metrics']
        self.assertEqual(result['peak_category'],'no_endpoint_cell_support_but_endpoint_union_cell_support')
        py['endpoint_union_approx']['attention16'][8,8]=0
        self.assertEqual(d.peak_diagnostics(c,p,s,r,py)['metrics']['peak_category'],'no_target_support_in_endpoint_or_union_peak_cell')
        s['attention']['endpoint']['peak_in_region']=True
        with self.assertRaises(ValueError):d.peak_diagnostics(c,p,s,r,py)

    def test_alignment_finite_probabilities_and_model_prediction_cannot_drift(self):
        c,p,s,r,py=fixture();p[1,1]=np.nan
        with self.assertRaises(ValueError):d.peak_diagnostics(c,p,s,r,py)
        c,p,s,r,py=fixture();s['pred_uv']=np.asarray(s['pred_uv'])+.01
        with self.assertRaises(ValueError):d.peak_diagnostics(c,p,s,r,py)
        self.assertIsNone(d.cosine([0,0],[1,0]))


if __name__=='__main__':unittest.main(verbosity=2)
