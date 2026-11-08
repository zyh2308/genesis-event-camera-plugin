import inspect
import math
import unittest
import numpy as np
import softfocus_probability as s


class SoftFocusProbabilityTests(unittest.TestCase):
    def test_uniform_retains_first_peak_without_favorable_fallback(self):
        p = np.full((16, 16), 1/256, dtype=np.float64)
        out = s.softfocus_predicted_peak(p)
        self.assertEqual(out['peak']['flat_index'], 0)
        self.assertEqual(out['numerical_diagnostics']['peak_tie_count'], 256)
        axis_weight = np.exp(-.5*np.arange(16, dtype=float)**2)
        expected = np.sum(((np.arange(16)+.5)/16)*axis_weight)/axis_weight.sum()
        np.testing.assert_allclose(out['uv'], [expected, expected], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(out['original_fp64_uv'], [.5, .5])
        self.assertLess(out['uv'][0], .1)
        self.assertFalse(out['numerical_diagnostics']['flat_fallback_applied'])

    def test_onehot_corners_and_edges_remain_exact(self):
        for row, col in [(0,0),(0,15),(15,0),(15,15),(0,8),(15,7),(9,0),(6,15)]:
            with self.subTest(row=row,col=col):
                p = np.zeros((16,16), dtype=np.float64); p[row,col] = 1
                out = s.softfocus_predicted_peak(p)
                self.assertEqual((out['peak']['row'],out['peak']['column']), (row,col))
                np.testing.assert_array_equal(out['uv'], [(col+.5)/16,(row+.5)/16])
                np.testing.assert_array_equal(out['weights'], p)

    def test_exact_gaussian_product_has_known_discrete_variance(self):
        row, col = 6, 9
        yy, xx = np.mgrid[:16,:16]
        p = np.exp(-.5*((xx-col)**2+(yy-row)**2)); p /= p.sum()
        out = s.softfocus_predicted_peak(p)
        expected_map = np.exp(-((xx-col)**2+(yy-row)**2)); expected_map /= expected_map.sum()
        np.testing.assert_allclose(out['weights'], expected_map, atol=2e-16, rtol=1e-15)
        axis = (np.arange(16)+.5)/16
        expected_uv = [np.sum(axis*np.exp(-(np.arange(16)-center)**2))/np.sum(np.exp(-(np.arange(16)-center)**2)) for center in (col,row)]
        np.testing.assert_allclose(out['uv'], expected_uv, atol=1e-15, rtol=0)

    def test_far_tail_is_softly_suppressed_never_hard_deleted(self):
        p = np.zeros((16,16), dtype=np.float64); p[8,8] = .51; p[0,0] = .49
        out = s.softfocus_predicted_peak(p)
        np.testing.assert_allclose(out['original_fp64_uv'], [.28625,.28625], atol=1e-15, rtol=0)
        np.testing.assert_allclose(out['uv'], [.53125,.53125], atol=1e-15, rtol=0)
        self.assertGreater(out['weights'][0,0], 0)
        self.assertLess(out['weights'][0,0], 1e-25)
        self.assertEqual(out['numerical_diagnostics']['input_nonzero_count'], out['numerical_diagnostics']['output_nonzero_count'])

    def test_tied_peak_is_row_major_even_for_fortran_and_strided_input(self):
        p = np.zeros((16,16)); p[2,10] = .5; p[3,1] = .5
        for array in (p,np.asfortranarray(p),np.stack([p,p],axis=-1)[...,0]):
            out = s.softfocus_predicted_peak(array)
            self.assertEqual(out['peak']['flat_index'],42)
            self.assertEqual(out['peak']['uv'],[10.5/16,2.5/16])

    def test_original_probability_bytes_and_readonly_views_unchanged(self):
        rng=np.random.default_rng(20261109); p=rng.random((16,16)).astype(np.float32);p/=p.sum()
        original=p.tobytes(); p.setflags(write=False)
        out=s.softfocus_predicted_peak(p)
        self.assertEqual(p.tobytes(),original)
        self.assertFalse(np.shares_memory(out['weights'],p))
        self.assertEqual(out['weights'].dtype,np.float64)
        self.assertEqual(out['uv'].dtype,np.float64)
        out['weights'][0,0]=9
        self.assertEqual(p.tobytes(),original)

    def test_map_sum_and_original_control_are_not_silently_renormalized(self):
        p=np.full((16,16),(1+1e-6)/256,dtype=np.float64)
        out=s.softfocus_predicted_peak(p)
        self.assertAlmostEqual(out['map_sum'],1+1e-6,places=14)
        expected=(p[...,None]*s.cell_center_grid()).sum((0,1))
        np.testing.assert_array_equal(out['original_fp64_uv'],expected)
        self.assertGreater(out['original_fp64_uv'][0],.5)
        self.assertLess(out['numerical_diagnostics']['normalization_abs_error'],1e-15)

    def test_nonfinite_negative_wrong_shape_dtype_and_sum_rejected(self):
        base=np.full((16,16),1/256,dtype=float)
        invalid=[]
        for value in (math.nan,math.inf,-math.inf,-1e-20):
            a=base.copy();a[2,3]=value;invalid.append(a)
        invalid += [np.zeros((16,16)),base*2,np.ones((16,16),dtype=np.int64),np.ones((16,16),dtype=bool),
                    np.ones((256,))/256,np.ones((1,16,16))/256,np.ones((15,16))/240]
        for value in invalid:
            with self.subTest(shape=value.shape,dtype=value.dtype):
                with self.assertRaises(ValueError):s.softfocus_predicted_peak(value)

    def test_api_has_no_GT_or_tunable_sigma_grid_parameter(self):
        self.assertEqual(list(inspect.signature(s.softfocus_predicted_peak).parameters),['attention'])
        p=np.full((16,16),1/256)
        for kwargs in ({'gt_uv':[.5,.5]},{'sigma':.1},{'grid':s.cell_center_grid()},{'mask':np.ones((16,16))}):
            with self.assertRaises(TypeError):s.softfocus_predicted_peak(p,**kwargs)
        self.assertEqual(s.SIGMA,1/16)


if __name__=='__main__':unittest.main(verbosity=2)
