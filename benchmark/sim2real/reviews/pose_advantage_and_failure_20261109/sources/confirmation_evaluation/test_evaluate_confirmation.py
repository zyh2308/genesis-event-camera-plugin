import copy,unittest
import numpy as np
from evaluate_confirmation import assess_scores,compare_score

def matrix():
    rows=[]
    for method in ('eventgenesis','v2e'):
        for seed in (0,1,2):
            for condition in ('normal','half','eighth'):
                by={f'validation_{i:03d}':dict(success=(.6 if method=='eventgenesis' else .5),center_px=float(i+1)) for i in range(32)}
                rows.append(dict(method=method,seed=seed,condition=condition,score=dict(summary=dict(success=.6 if method=='eventgenesis' else .5,center_px=16.5),by_rollout=by)))
    return rows

class EvaluationTests(unittest.TestCase):
    def test_bonferroni_ci_and_paired_seeds_exact_constant_case(self):
        d=assess_scores(matrix());self.assertTrue(d['passed']);self.assertEqual(len(d['checks']),2)
        for c in d['checks']:
            np.testing.assert_allclose(c['seed_mean_delta'],[.1]*3,atol=1e-15,rtol=0)
            np.testing.assert_allclose(c['paired_trajectory_ci'],[.1,.1],atol=1e-15,rtol=0)
            self.assertEqual(c['ci_level'],.975)
    def test_mean_advantage_cannot_hide_one_seed_loss(self):
        x=matrix()
        r=next(r for r in x if r['method']=='eventgenesis' and r['seed']==2 and r['condition']=='normal')
        for row in r['score']['by_rollout'].values():row['success']=.49
        d=assess_scores(x);self.assertGreater(d['checks'][0]['mean_delta'],0);self.assertFalse(d['passed'])
    def test_missing_duplicate_or_wrong_trajectory_rejected(self):
        for kind in ('missing','duplicate','trajectory'):
            x=matrix()
            if kind=='missing':x.pop()
            elif kind=='duplicate':x[-1]=copy.deepcopy(x[0])
            else:x[-1]['score']['by_rollout'].pop('validation_031')
            with self.assertRaises(ValueError):assess_scores(x)
    def test_dark_negative_kept_not_reclassified_as_primary(self):
        x=matrix()
        for r in x:
            if r['method']=='eventgenesis' and r['condition']=='eighth':
                for v in r['score']['by_rollout'].values():v['success']=0
        d=assess_scores(x);self.assertTrue(d['passed']);self.assertEqual(d['mandatory_negative_condition'],'eighth');self.assertFalse(d['real_scoring_authorized'])
    def test_native_baseline_field_and_numeric_changes_rejected(self):
        a=matrix()[0]['score'];self.assertEqual(compare_score(a,copy.deepcopy(a)),0)
        b=copy.deepcopy(a);b['by_rollout']['validation_031']['success']+=.01
        with self.assertRaises(ValueError):compare_score(a,b)
        b=copy.deepcopy(a);del b['summary']['center_px']
        with self.assertRaises(ValueError):compare_score(a,b)

if __name__=='__main__':unittest.main()
