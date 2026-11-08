import copy,unittest
from assess_advance import assess_advance,assess_driver_outputs,METHODS,SEEDS,LIGHTS,RIDS
def fixture():
    rows=[]
    for m in METHODS:
        for s in SEEDS:
            rows.append(dict(method=m,seed=s,conditions={l:dict(summary=dict(center_px=10.,success=.5),by_rollout={r:dict(center_px=10.,success=.5) for r in RIDS},window_evidence=[dict(sample_id=f'{r}/{k:03}',rid=r,finite_prediction=True) for r in RIDS for k in range(4,20)]) for l in LIGHTS}))
    candidate=copy.deepcopy(rows)
    for r in candidate:
        if r['method']=='eventgenesis':
            c=r['conditions']['eighth'];c['summary']=dict(center_px=9.,success=.6);c['by_rollout']={rid:dict(center_px=9.,success=.6) for rid in RIDS}
    return rows,candidate
def flat_fixture(a,b):
    scores=[];cases=[]
    for arm,runs in [('control',a),('candidate',b)]:
        for r in runs:
            for light,c in r['conditions'].items():
                scores.append(dict(method=r['method'],seed=r['seed'],condition=light,arm=arm,readout='native_global' if arm=='control' else 'predicted_peak_softfocus',score={k:c[k] for k in ('summary','by_rollout')},window_ids=[x['sample_id'] for x in c['window_evidence']],windows=64))
                if arm=='control':
                    for x in c['window_evidence']:cases.append(dict(method=r['method'],seed=r['seed'],condition=light,rid=x['rid'],sample_id=x['sample_id'],control_uv=[.5,.5],candidate_uv=[.5,.5],z_m=1.,size=.1,R_control=[[1.,0,0],[0,1.,0],[0,0,1.]],R_candidate=[[1.,0,0],[0,1.,0],[0,0,1.]],finite_prediction_control=True,finite_prediction_candidate=True))
    return scores,cases
class Advance(unittest.TestCase):
    def test_all36_exact_and_ties_do_not_satisfy_EG_dark_improvement(self):
        a,b=fixture();r=assess_advance(a,b);self.assertEqual(r['passed_checks'],36);self.assertTrue(r['advance_to_independent_synthetic_confirmation']);self.assertFalse(r['real_scoring_authorized'])
        scores,cases=flat_fixture(a,b);self.assertEqual(assess_driver_outputs(scores,cases),r)
        cases[0]['R_candidate'][0][0]=float('nan')
        with self.assertRaises(ValueError):assess_driver_outputs(scores,cases)
        r=assess_advance(a,copy.deepcopy(a));self.assertEqual(r['failed_checks'],6)
    def test_harming_control_or_healthy_light_is_rejected_without_mean_hiding(self):
        a,b=fixture();c=next(x for x in b if x['method']=='v2e' and x['seed']==1)['conditions']['eighth'];c['summary']['center_px']=11.;c['by_rollout']={rid:dict(center_px=11.,success=.5) for rid in RIDS}
        r=assess_advance(a,b);self.assertFalse(r['advance_to_independent_synthetic_confirmation']);self.assertEqual(r['failed_checks'],1);self.assertEqual(len(r['per_rollout_negative_changes']),4)
    def test_missing_case_window_or_nonfinite_is_fail_closed(self):
        a,b=fixture()
        with self.assertRaises(ValueError):assess_advance(a,b[:-1])
        b[0]['conditions']['normal']['window_evidence'].pop()
        with self.assertRaises(ValueError):assess_advance(a,b)
        a,b=fixture();b[0]['conditions']['normal']['window_evidence'][0]['finite_prediction']=False
        with self.assertRaises(ValueError):assess_advance(a,b)
if __name__=='__main__':unittest.main(verbosity=2)
