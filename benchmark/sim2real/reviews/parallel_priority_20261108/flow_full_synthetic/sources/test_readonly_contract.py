"""CPU-only tests for real masks, aggregation, scope and no optimizer path."""
import ast,copy,json,unittest
from pathlib import Path
import numpy as np
import evaluate_frozen_synthetic as E
import flow_synthetic_diagnostic as D

class Readonly(unittest.TestCase):
    def test_fixed_common_mask_and_motion_bins(self):
        gt=np.zeros((2,2,3),np.float32);gt[0,0]=[0,.1,2]
        valid=np.ones((2,3),bool);eg=np.zeros((2,2,3,5));v2=eg.copy()
        eg[0,0,1,0]=1;v2[1,0,1,1]=1;eg[0,1,2,0]=3
        common=D.active(eg)&D.active(v2);self.assertEqual(int(common.sum()),1)
        rr=D.region_summary(np.zeros_like(gt),gt,valid,common)
        self.assertEqual(rr['common_event_active_valid']['pixels'],1)
        self.assertAlmostEqual(rr['common_event_active_valid']['epe_px'],.1)
        self.assertEqual(rr['motion_2_5px']['pixels'],1)
    def test_empty_support_is_null(self):
        rr=D.region_summary(np.zeros((2,2,2)),np.zeros((2,2,2)),np.ones((2,2),bool),np.zeros((2,2),bool))
        rr['common_event_active_valid']['zero_epe_px']=None
        result=E.aggregate([rr])['common_event_active_valid'];self.assertIsNone(result['epe_px']);self.assertEqual(result['supported_members'],0)
    def test_equal_window_then_rollout_not_pixel_weighted(self):
        rows=[{'region':{'epe_px':v,'zero_epe_px':2.,'pixels':p}} for v,p in [(1.,1),(3.,99)]]
        self.assertEqual(E.aggregate(rows)['region']['epe_px'],2.)
        self.assertEqual(E.aggregate([E.aggregate(rows),{'region':{'epe_px':4.}}])['region']['epe_px'],3.)
    def test_contract_rejects_wrong_device_scope_or_partial_seed(self):
        c=json.loads((Path(__file__).parent/'contract.json').read_text());device=c['shards']['seed01']['gpu_uuid']
        self.assertEqual(E.validate(c,[0,1],E.ROOT/'seed01',device),'seed01')
        for seeds,out,gpu in [([0],E.ROOT/'seed01',device),([0,1],E.ROOT/'seed01','bad'),([2],E.ROOT/'seed01',device)]:
            with self.assertRaises(ValueError):E.validate(c,seeds,out,gpu)
        bad=copy.deepcopy(c);bad['real_data_used']=True
        with self.assertRaises(ValueError):E.validate(bad,[0,1],E.ROOT/'seed01',device)
    def test_no_optimizer_or_backward(self):
        tree=ast.parse((Path(__file__).parent/'evaluate_frozen_synthetic.py').read_text())
        attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
        self.assertFalse({'optim','backward','step','zero_grad'}&attrs)
        self.assertIn('inference_mode',attrs)
    def test_full_scope_checkpoint_hashes_and_paired_manifests(self):
        c=json.loads((Path(__file__).parent/'contract.json').read_text());self.assertEqual(len(c['checkpoint_binding']),15)
        self.assertEqual(len({b['selected_checkpoint_sha256'] for b in c['checkpoint_binding'].values()}),15)
        for m in E.METHODS:self.assertEqual(len({c['checkpoint_binding'][f'{m}/seed{s}']['data_manifest_sha256'] for s in range(3)}),1)
    def test_reused_region_functions_byte_exact(self):
        p=Path(__file__).resolve().parent;self.assertEqual((p/'flow_synthetic_diagnostic.py').read_bytes(),(p.parent/'flow_priority_20261108/flow_synthetic_diagnostic.py').read_bytes())

if __name__=='__main__':unittest.main()
