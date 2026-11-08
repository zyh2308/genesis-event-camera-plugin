import itertools,math,unittest
import numpy as np
from scipy.spatial import cKDTree
import verify_readout as v


class IndependentAuditFixtures(unittest.TestCase):
    def test_independent_probability_corner_tail_tie_and_invalid(self):
        p=np.zeros((16,16),np.float32);p[8,8]=.51;p[0,0]=.49
        q,uv,old,peak=v.probability(p)
        np.testing.assert_allclose(uv,[8.5/16]*2,atol=1e-14,rtol=0)
        self.assertGreater(q[0,0],0);np.testing.assert_array_equal(peak,[8.5/16]*2)
        p=np.zeros((16,16));p[0,15]=1
        np.testing.assert_array_equal(v.probability(p)[1],[15.5/16,.5/16])
        p=np.ones((16,16))/256;self.assertTrue((v.probability(p)[1]<.1).all())
        for bad in (p*0,p*2,p.reshape(256),np.full((16,16),np.nan)):
            with self.assertRaises(ValueError):v.probability(bad)

    def test_anchor_aware_geometry_dtype_and_invalid_frontality(self):
        vertices=np.asarray(list(itertools.product([-.02,.02],repeat=3)));keys=np.vstack(([.01,.015,.03],vertices));tree=cKDTree(vertices)
        R=np.array([[0,0,1],[0,1,0],[-1,0,0.]]);T=np.eye(4);T[:3,:3]=R;T[:3,3]=[.02,-.01,.5]
        K=np.array([[300,0,173],[0,301,130],[0,0,1.]]);anchor=keys[0];point=R@anchor+T[:3,3];pix=K@point;uv=pix[:2]/pix[2]/[346,260]
        case=dict(K=K,T_gt=T,anchor=anchor,z_m=np.float32(point[2]),size=np.float32(.1),gt_uv=uv.astype(np.float32),
                  control_uv=uv.astype(np.float32),candidate_uv=uv,R_control=R,R_candidate=R)
        self.assertNotEqual(float(point[2]),T[2,3])
        for arm in ('control','candidate'):
            metrics,extra=v.measure(case,arm,vertices,keys,tree,.07)
            self.assertEqual(metrics['success'],1.);self.assertLess(extra['adds_m'],1e-7)
        case['candidate_uv']=uv+[.2,0]
        self.assertEqual(v.measure(case,'candidate',vertices,keys,tree,.07)[0]['success'],0.)
        badkeys=np.vstack((keys,[2,0,0]))
        metrics,extra=v.measure(case,'control',vertices,badkeys,tree,.07)
        self.assertFalse(extra['valid']);self.assertIsNone(extra['adds_m']);self.assertEqual(metrics['success'],0.)
        # Explicitly expose the native arithmetic distinction, rather than
        # pretending a common FP64 subtraction reproduces the FP32 control.
        case['gt_uv']=[.5,.5];case['control_uv']=[.5+1e-8,.5];case['candidate_uv']=[.5+1e-8,.5]
        self.assertEqual(v.measure(case,'control',vertices,keys,tree,.07)[0]['center_px'],0.)
        self.assertGreater(v.measure(case,'candidate',vertices,keys,tree,.07)[0]['center_px'],0.)


if __name__=='__main__':unittest.main(verbosity=2)
