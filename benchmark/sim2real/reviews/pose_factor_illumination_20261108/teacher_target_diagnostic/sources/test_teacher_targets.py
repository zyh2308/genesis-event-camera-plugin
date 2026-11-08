import json,math,tempfile,unittest
from pathlib import Path
import probe_teacher_targets as P


class TeacherTargetTests(unittest.TestCase):
    def test_pixel_units_zero_shift_and_known_3_4_5_geometry(self):
        raw=[.5,.5];filtered=[.5+3/346,.5+4/260]
        result=P.metrics(raw,filtered,raw)
        self.assertEqual(result['raw_teacher_center_error_px'],0.)
        self.assertAlmostEqual(result['baf_teacher_center_error_px'],5.)
        self.assertAlmostEqual(result['teacher_target_shift_px'],5.)
        self.assertEqual(P.metrics(raw,raw,raw)['teacher_target_shift_px'],0.)

    def test_nonfinite_and_wrong_shapes_stop_instead_of_changing_scores(self):
        for bad in ([math.nan,.5],[.5,math.inf],[.5],[]):
            with self.assertRaises(ValueError):P.metrics(bad,[.5,.5],[.5,.5])

    def test_rollout_macro_and_scope_keep_train_val_method_seed_separate(self):
        def row(rid,error,**extra):return dict(method='eventgenesis',seed=0,split='train',condition='normal',rid=rid,raw_teacher_center_error_px=error,baf_teacher_center_error_px=2*error,teacher_target_shift_px=error,**extra)
        items=[row('a',2),row('b',10),row('b',10)]
        val=dict(items[0],split='validation',raw_teacher_center_error_px=90)
        result=P.summarize(items+[val]);train=next(r for r in result if r['split']=='train');validation=next(r for r in result if r['split']=='validation')
        self.assertEqual(train['raw_teacher_center_error_px'],6.)
        self.assertEqual(train['baf_teacher_center_error_px'],12.)
        self.assertEqual(validation['raw_teacher_center_error_px'],90.)
        self.assertEqual(train['windows'],3);self.assertEqual(train['rollouts'],2)

    def test_frozen_manifest_and_actual_source_bytes_must_both_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'baf.py';source.write_text('fixed-rule')
            manifest=root/'source_manifest.json';manifest.write_text(json.dumps({'baf.py':P.sha(source)}));digest=P.sha(manifest)
            records=P.checked_sources(root,digest);self.assertEqual(len(records),2)
            source.write_text('other-rule')
            with self.assertRaises(ValueError):P.checked_sources(root,digest)
            with self.assertRaises(ValueError):P.checked_sources(root,'0'*64)


if __name__=='__main__':unittest.main(verbosity=2)
