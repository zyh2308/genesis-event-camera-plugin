import hashlib,json,unittest
from pathlib import Path
import torch
import torch.nn.functional as F
import strict_support as S

class Strict(unittest.TestCase):
    def test_2x2_affine_zero_sentinel_leaks_through_nearest(self):
        valid=torch.tensor([[[[1.,0.],[0.,0.]]]])
        flow=valid.repeat(1,2,1,1)*2
        grid=torch.tensor([[[[-.25,-.25]]]])
        nearest=F.grid_sample(valid,grid,mode='nearest',align_corners=False)
        sampled=F.grid_sample(flow,grid,mode='bilinear',align_corners=False)
        strict,weight=S.strict_support(valid,grid,torch.ones_like(nearest),nearest)
        self.assertTrue(bool(nearest.item()));self.assertFalse(bool(strict.item()))
        self.assertAlmostEqual(float(weight),.5625);self.assertAlmostEqual(float(sampled[0,0]),1.125)
    def test_full_valid_supported_and_crop_excluded(self):
        valid=torch.ones((1,1,2,2));grid=torch.tensor([[[[-.25,-.25]]]]);nn=torch.ones((1,1,1,1))
        strict,_=S.strict_support(valid,grid,nn,nn);self.assertTrue(bool(strict))
        strict,_=S.strict_support(valid,grid,torch.zeros_like(nn),nn);self.assertFalse(bool(strict))
    def test_strict_is_subset_and_statistics(self):
        nn=torch.ones((1,1,2,2));strict=nn.bool();strict[0,0,0,0]=False
        row=S.removal_summary(nn,strict,torch.ones((1,2,2,2)),torch.ones((1,2,2,2)),torch.ones((1,2,2)),torch.ones((1,1,2,2)))
        self.assertEqual(row['removed_pixels'],1);self.assertEqual(row['removed_fraction_of_NN_valid'],.25)
        self.assertEqual(row['removed_fraction_of_NN_valid_photo_gradient_energy'],.25)
    def test_original151_failure_provenance_preserved(self):
        p=Path(__file__).parent;c=json.loads((p/'contract.json').read_text())
        self.assertEqual(c['preserved_failure']['source_manifest_sha256'],'26d7601446088f45d5194a9eef058caf90d2e8fd50f4e21e71fb9dd99a7f2dee')
        self.assertEqual(c['parameter_oracle']['oracle_masks'],['native_nearest','raw_bilinear_strict'])
    def test_no_optimizer_and_no_real_scope(self):
        p=Path(__file__).parent;c=json.loads((p/'contract.json').read_text());self.assertEqual(c['optimizer_updates'],0);self.assertFalse(c['automatic_training']);self.assertFalse(c['real_data_used'])
        self.assertNotIn('torch.optim',(p/'run_photometric_alignment.py').read_text())

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
