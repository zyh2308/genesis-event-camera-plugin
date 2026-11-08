"""Development media schema checks. No synthetic scene images or clips."""
from pathlib import Path
import sys,tempfile,unittest
sys.dont_write_bytecode=True
import make_development_demo as demo


class DevelopmentMediaContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='contract_',dir=Path(__file__).parent)
        self.root=Path(self.temp.name);self.path=self.root/'DEVELOPMENT_DEMO_PLAN.json'
        self.plan=dict(schema=demo.SCHEMA,actual_episodes=16,outcome_selection=False,locked_frontends=demo.EXPECTED,
            blocks=[dict(seed=i,panels=[dict(endpoint_hz=r,method=m,frontend_id=demo.EXPECTED[m])
                for r in demo.RATES for m in demo.fair.METHODS]) for i in range(4)],artifacts=[])
    def tearDown(self):self.temp.cleanup()
    def seal(self):
        demo.shared.write_json(self.path,self.plan)
        demo.shared.write_json(self.root/'PLAN_SHA256.json',dict(plan_sha256=demo.fair.sha(self.path)))
    def test_complete_fixed_layout(self):
        self.seal();self.assertEqual(sum(len(b['panels']) for b in demo.read_plan(self.path)['blocks']),16)
    def test_missing_panel_rejected_even_after_reseal(self):
        self.plan['blocks'][0]['panels'].pop();self.seal()
        with self.assertRaises(demo.fair.DataError):demo.read_plan(self.path)
    def test_changed_frontend_rejected(self):
        self.plan['blocks'][0]['panels'][0]['frontend_id']='w10_fit1_gap50';self.seal()
        with self.assertRaises(demo.fair.DataError):demo.read_plan(self.path)
    def test_unsealed_changed_plan_rejected(self):
        self.seal();self.plan['actual_episodes']=15;demo.shared.write_json(self.path,self.plan)
        with self.assertRaises(demo.fair.DataError):demo.read_plan(self.path)
    def test_missing_transfer_receipt_creates_no_demo(self):
        self.seal();output=self.root/'must_not_exist'
        with self.assertRaises(demo.fair.DataError):demo.render(self.path,self.root/'missing.json',self.root,output,demo.shared.DEFAULT_FONT)
        self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
