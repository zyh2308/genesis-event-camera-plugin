import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import diagnose_centers as D


class DiagnosticTests(unittest.TestCase):
    def test_zero_input_control_uses_no_gt_and_preserves_raw_filtered_tensors(self):
        class FrozenCenter(torch.nn.Module):
            def forward(self,x):
                mean=x.mean((1,2,3));return torch.stack((.5+mean,.25-mean),1)
        model=FrozenCenter().eval();raw=torch.ones(3,3,4,4);filtered=raw*.2
        snapshots=(raw.clone(),filtered.clone());seen=[]
        def estimate(net,value):
            seen.append(value.clone());return net(value).numpy()
        a,b,zero=D.predict_three(estimate,model,raw,filtered)
        np.testing.assert_array_equal(zero,np.tile([.5,.25],(3,1)))
        self.assertEqual(torch.count_nonzero(seen[2]),0)
        self.assertTrue(torch.equal(raw,snapshots[0]) and torch.equal(filtered,snapshots[1]))
        self.assertFalse(np.array_equal(a,zero));self.assertFalse(np.array_equal(b,zero))
        def mutate(net,value):value.add_(1);return net(value).numpy()
        with self.assertRaises(ValueError):D.predict_three(mutate,model,raw,filtered)

    def test_fixed_train_selection_matches_original_visible_linspace(self):
        rows=[dict(rid=f'train_{rid:03}',id=f'train_{rid:03}/{k:03}') for rid in range(8) for k in range(4,20)]
        indices=D.select_indices(rows,[f'train_{i:03}' for i in range(8)])
        self.assertEqual(indices[:4],[0,5,10,15]);self.assertEqual(len(indices),32)
        with self.assertRaises(ValueError):D.select_indices(rows[:2],['train_000'])

    def test_raw_prediction_parity_fixed_tolerance_gt_and_finite(self):
        row=dict(id='validation_000/004',rid='validation_000',uv=np.array([[.5,.25]],np.float32))
        key=('eventgenesis',0,'half',row['id'])
        reference={key:dict(rid=row['rid'],uv_gt=[.5,.25],pred_uv=[.5,.25])}
        self.assertEqual(D.check_reference(np.array([[.5,.25]]),[row],reference,*key[:3]),0.)
        for wrong in ([.5+2.01e-6,.25],[float('nan'),.25]):
            with self.assertRaises(ValueError):D.check_reference(np.array([wrong]),[row],reference,*key[:3])
        reference[key]['uv_gt']=[.4,.25]
        with self.assertRaises(AssertionError):D.check_reference(np.array([[.5,.25]]),[row],reference,*key[:3])

    def test_complete_source_and_contract_pin_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'export_pose_factors.py').write_text('identity=42\n');(p/'contract.json').write_text('{}')
            manifest={n:D.sha(p/n) for n in ('export_pose_factors.py','contract.json')}
            (p/'source_manifest.json').write_text(json.dumps(manifest))
            with patch.object(D,'DIAGNOSTIC_MANIFEST',D.sha(p/'source_manifest.json')),patch.object(D,'EXPORT_SHA',manifest['export_pose_factors.py']):
                module,records=D.load_export(p,p/'contract.json');self.assertEqual(module.identity,42);self.assertEqual(len(records),3)
                (p/'contract.json').write_text('{"changed":true}')
                with self.assertRaises(ValueError):D.load_export(p,p/'contract.json')

    def test_reference_complete_inventory_and_artifact_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=root/'sources';sources.mkdir();out=root/'export';out.mkdir()
            prediction=out/'predictions.jsonl'
            rows=[dict(method=m,seed=s,arm='C1',condition=c,sample_id=f'validation_{i//16:03}/{i%16+4:03}',
                       rid=f'validation_{i//16:03}',uv_gt=[.5,.5],pred_uv=[.4,.6])
                  for m in ('eventgenesis','v2e') for s in (0,1,2) for c in D.CONDITIONS for i in range(64)]
            prediction.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            value=dict(schema='pose_factor_forward_export_v1',status='complete',optimizer_updates=0,real_data_used=False,
                       native_parity_passed=True,contract_sha256='contracthash',source_script_sha256=D.EXPORT_SHA,
                       artifacts=[D.record(prediction)])
            completion=out/'completion.json';completion.write_text(json.dumps(value))
            with patch.object(D,'FORWARD_COMPLETION_SHA',D.sha(completion)):
                reference,_=D.load_reference(sources,'contracthash');self.assertEqual(len(reference),1152)
                with self.assertRaises(ValueError):D.load_reference(sources,'wrongcontract')
                prediction.write_text(prediction.read_text()+'{}\n')
                with self.assertRaises(ValueError):D.load_reference(sources,'contracthash')

    def test_summary_rollout_equal_empty_retention_and_does_not_select_winners(self):
        def row(rid,error):return dict(method='eventgenesis',seed=0,split='train',condition='half',rid=rid,
                                      raw_center_error_px=error,filtered_center_error_px=2*error,zero_center_error_px=3*error,center_shift_px=error,
                                      raw_events=0,retained_events=0)
        values=[row('a',2.),row('b',10.),row('b',10.)]
        result=D.summarize(values)[0]
        self.assertEqual(result['raw_center_error_px'],6.)
        self.assertEqual(result['filtered_center_error_px'],12.)
        self.assertEqual(result['zero_center_error_px'],18.)
        self.assertIsNone(result['event_retention_ratio'])
        self.assertEqual(len(values),3)


if __name__=='__main__':unittest.main(verbosity=2)
