import copy
import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

import diagnose_rotation_crops as d

f=d.load_helper(Path(__file__).parent)


def fixture():
    verts=np.array([[-.03,-.02,-.01],[.03,-.02,-.01],[-.03,.02,.01],[.03,.02,.01]])
    keys=np.vstack(([.01,.0,.0],verts))
    assets=f.FactorAssets(verts,keys,.08)
    K=np.array([[300,0,173],[0,300,130],[0,0,1.]])
    T=np.eye(4);T[2,3]=.5
    anchor=keys[0];cam=anchor+T[:3,3];px=K@cam;uv=px[:2]/px[2]/[346,260]
    bad=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    flat=[]
    for arm in ('C0','C1','slow'):
        row=dict(method='eventgenesis',seed=0,arm=arm,condition='normal',rid='r0',sample_id='r0/000',
                 K=K.tolist(),T_gt=T.tolist(),anchor=anchor.tolist(),uv_gt=uv.tolist(),gt_log_size=math.log(.1),gt_log_shape=0.,
                 pred_uv=uv.tolist(),z_m=.5,size=.1,R_pred=bad.tolist(),R_crop_gt_center=bad.tolist(),
                 R_crop_gt_size=np.eye(3).tolist(),R_crop_gt_center_and_size=np.eye(3).tolist())
        flat.append(row)
    records=f.group_flat_predictions(flat)
    macro=f.aggregate_rollout_equal(list(f.evaluate_records(records,assets)))
    native=[dict(method=r['method'],seed=r['seed'],arm=r['variant'].split('/')[1],condition=r['condition'],
                 score=dict(summary={k:r['summary'][k] for k in f.NATIVE_METRICS},
                            by_rollout={rid:{k:v[k] for k in f.NATIVE_METRICS} for rid,v in r['by_rollout'].items()}))
            for r in macro if r['variant'].startswith('actual/')]
    return flat,assets,native,macro


class CropTests(unittest.TestCase):
    def test_crop_rescue_and_both_numerical_parities(self):
        flat,assets,native,macro=fixture()
        result=d.analyze(f,flat,assets,native,macro,strict_inventory=False)
        self.assertEqual(len(result['scored']),8)
        self.assertEqual(result['native_parity']['status'],'passed')
        self.assertEqual(result['gttranslation_reference_parity']['status'],'passed')
        self.assertTrue(result['gtboth_rotation_consistency']['all_bitwise_equal'])
        table={r['rotation_factor']:r['metrics'] for r in result['summary'] if r['translation_condition']=='GTtranslation'}
        self.assertGreater(table['predR']['rotation_error_deg'],80)
        self.assertEqual(table['GTboth_crop_R']['success_percent'],100)
        self.assertLess(table['GTboth_crop_R']['rotation_error_deg'],1e-10)

    def test_gtboth_arm_mismatch_recorded_never_substituted(self):
        flat,assets,native,macro=fixture()
        # C0 GT-both differs; every C1 result stays unchanged.
        flat[0]['R_crop_gt_center_and_size']=flat[0]['R_pred']
        result=d.analyze(f,flat,assets,native,macro,strict_inventory=False)
        consistency=result['gtboth_rotation_consistency']
        self.assertFalse(consistency['all_bitwise_equal'])
        self.assertFalse(consistency['all_within_tolerance'])
        self.assertEqual(consistency['max_abs'],1.)
        row=next(r for r in result['summary'] if r['translation_condition']=='GTtranslation' and r['rotation_factor']=='GTboth_crop_R')
        self.assertEqual(row['metrics']['success_percent'],100)

    def test_missing_crop_and_duplicate_or_wrong_label_rejected(self):
        flat,assets,native,macro=fixture()
        wrong=copy.deepcopy(flat);del wrong[1]['R_crop_gt_size']
        with self.assertRaises(KeyError):d.analyze(f,wrong,assets,native,macro,strict_inventory=False)
        with self.assertRaisesRegex(ValueError,'duplicate'):d.analyze(f,flat+[flat[0]],assets,native,macro,strict_inventory=False)
        wrong=copy.deepcopy(flat);wrong[1]['gt_log_size']+=1
        with self.assertRaisesRegex(ValueError,'label mismatch'):d.analyze(f,wrong,assets,native,macro,strict_inventory=False)

    def test_reference_formula_change_detected(self):
        flat,assets,native,macro=fixture()
        ref=copy.deepcopy(macro)
        next(r for r in ref if r['variant']=='oracle/C1/u1z1r0')['summary']['auc']+=.01
        result=d.analyze(f,flat,assets,native,ref,strict_inventory=False)
        self.assertEqual(result['gttranslation_reference_parity']['status'],'failed')

    def test_formal_inventory_cannot_silently_use_subset(self):
        flat,assets,native,macro=fixture()
        with self.assertRaisesRegex(ValueError,'inventory'):d.analyze(f,flat,assets,native,macro)


if __name__=='__main__':unittest.main(verbosity=2)
