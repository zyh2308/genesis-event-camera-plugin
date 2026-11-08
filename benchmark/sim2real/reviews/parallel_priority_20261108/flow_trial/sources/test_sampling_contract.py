"""Meaningful CPU schedule/gate/optimizer-input rejection tests; no CUDA."""
import ast,copy,unittest
from pathlib import Path
import numpy as np
from sampling import STRATA,schedules

class Replay(unittest.TestCase):
    def setUp(self):self.rows={f's{g}_{i:02d}':STRATA[g] for g in range(4) for i in range(6)}
    def test_native_control_replay(self):
        r=schedules(self.rows);expected=np.random.default_rng(1000003*45).permutation(1584).tolist()
        self.assertEqual(r['arms']['control'][:1584],expected);self.assertEqual(len(set(expected)),1584)
        self.assertEqual(r['augmentation_seed_sequences'][1584],[0,46,0,0])
    def test_target_and_only_static_replacements(self):
        r=schedules(self.rows);tags=[STRATA.index(self.rows[rid]) for rid,i in r['samples']]
        self.assertEqual(np.bincount([tags[i] for i in r['arms']['motion']],minlength=4).tolist(),[480,160,480,480])
        for a,b in zip(r['arms']['control'],r['arms']['motion']):
            if tags[a]!=1:self.assertEqual(a,b)
        self.assertGreater(r['preserved_fraction'],.83)
    def test_deterministic_pairing(self):self.assertEqual(schedules(self.rows),schedules(dict(reversed(list(self.rows.items())))))
    def test_changed_bank_rejected(self):
        self.rows.pop(next(iter(self.rows)))
        with self.assertRaises(ValueError):schedules(self.rows)
    def test_budget_rejected(self):
        with self.assertRaises(ValueError):schedules(self.rows,300)
    def test_objective_has_no_gt_or_mask_argument(self):
        tree=ast.parse((Path(__file__).parent/'flow_sampling_short.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='objective_dt1']
        self.assertEqual(len(calls),1)
        used={n.id for a in calls[0].args for n in ast.walk(a) if isinstance(n,ast.Name)}
        self.assertFalse({'flow','valid','gt'}&used)
    def test_static_class_regression_blocks_gate(self):
        # Extract only the pure quality function, not torch/data/model imports.
        tree=ast.parse((Path(__file__).parent/'flow_sampling_short.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='quality')
        scope=dict(STRATA=STRATA);exec(compile(ast.Module(body=[fn],type_ignores=[]),'<quality_test>','exec'),scope)
        rows=[]
        for m in ['eventgenesis','v2e']:
            for arm,value in [('control',1.),('motion',.9)]:
                v=dict(regions={r:{'epe_px':value} for r in ['all_physical_valid','common_event_active_valid']},
                       strata={s:{'all_physical_valid':{'epe_px':value}} for s in STRATA})
                rows.append(dict(method=m,arm=arm,validation=v))
        self.assertTrue(scope['quality'](rows)['passed'])
        rows[-1]['validation']['strata'][STRATA[1]]['all_physical_valid']['epe_px']=1.06
        self.assertFalse(scope['quality'](rows)['passed'])

if __name__=='__main__':unittest.main()
