import unittest
import numpy as np
from decompose_tail import decompose
class Algebra(unittest.TestCase):
    def test_tail_pull_and_signed_cancellation(self):
        p=np.zeros((16,16));p[4,4]=.7;p[4,12]=.3;f=np.zeros_like(p);f[4,4]=1
        fg=np.array([(4.5)/16*346,4.5/16*260]);g=fg+np.array([5.,0.]);r=decompose(p,f,g)
        self.assertAlmostEqual(r['mass_fg'],.7);self.assertAlmostEqual(r['signed_foreground_projection_px'],-5.)
        self.assertGreater(r['signed_outside_tail_projection_px'],0);self.assertLess(r['identity_residual_px'],1e-10)
        self.assertAlmostEqual(r['signed_foreground_projection_px']+r['signed_outside_tail_projection_px'],r['total_error_length_px'])
    def test_zero_mass_and_zero_error_keep_null(self):
        p=np.zeros((16,16));p[4,4]=1;g=np.array([4.5/16*346,4.5/16*260]);r=decompose(p,p*0,g)
        self.assertEqual(r['total_error_length_px'],0);self.assertIsNone(r['e_fg_native_px']);self.assertIsNone(r['outside_tail_native_px']);self.assertIsNone(r['signed_outside_tail_projection_px'])
        r=decompose(p,np.ones_like(p),g);self.assertEqual(r['mass_bg'],0);self.assertIsNone(r['outside_tail_native_px']);self.assertIsNone(r['signed_foreground_projection_px'])
    def test_fractional_partition_and_sum_rounding_are_explicit(self):
        p=np.full((16,16),np.float32(1/256))*np.float32(1+1e-6);f=np.full_like(p,.25);r=decompose(p,f,[173.,130.])
        self.assertNotEqual(r['original_attention_sum'],1);self.assertAlmostEqual(r['mass_fg']+r['mass_bg'],1);self.assertLess(r['identity_residual_px'],1e-10)
if __name__=='__main__':unittest.main(verbosity=2)
