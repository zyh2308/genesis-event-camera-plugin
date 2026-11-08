"""Actual forward/TRAIN emitter schema contracts; CPU metadata fixtures only."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from run_pose_teacher_projection import verify_diagnostic_completions


def actual_schema_fixtures():
    # Required keys exactly follow the two pinned diagnostic emitters. Numeric
    # values are CPU metadata fixtures, not fabricated server measurements.
    forward=dict(schema='pose_factor_forward_export_v1',status='complete',optimizer_updates=0,
                 real_data_used=False,models_and_sources_unchanged=True,native_parity_passed=True,native_parity_checks=54)
    train=dict(schema='pose_train_factor_gradient_diagnostic_v1',status='complete',optimizer_updates=0,
               real_scores_read=False,validation_gradients_used=False,real_gradients_used=False,
               models=36,results=[dict(fixture_endpoint=i) for i in range(36)])
    assert 'real_data_used' not in train
    return forward,train


def reproduce_v1_failure():
    source=Path(__file__).with_name('V1_RUNNER_REFERENCE.py')
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='6534e65766de497ac993b5521a774849e623c93533303ac09681996515423454'
    tree=ast.parse(source.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.iter,ast.Tuple)
              and [getattr(x,'value',None) for x in n.iter.elts]==['forward_completion','train_probe_completion'])
    forward,train=actual_schema_fixtures()
    with tempfile.TemporaryDirectory() as tmp:
        f,t=Path(tmp)/'forward.json',Path(tmp)/'train.json';f.write_text(json.dumps(forward));t.write_text(json.dumps(train))
        try:exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),dict(json=json,Path=Path,c=dict(forward_completion=str(f),train_probe_completion=str(t))))
        except KeyError as error:
            if error.args!=('real_data_used',):raise
            return dict(schema='pose_projection_v1_predeployment_admission_failure_fixture_v1',status='reproduced',
                        scope='Local CPU metadata fixture executing original v1 admission AST; no server or GPU failure claimed',
                        error_type='KeyError',error_key='real_data_used',original_runner_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        original_admission_ast_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),
                        original_v1_zip_sha256='7ecea137e64e757e8229a77d3e96fb6f07c2ded67ddff39fe6bbc094aaaa3516',
                        original_v1_source_and_zip_preserved=True)
    raise AssertionError('Original v1 schema bug was not reproduced')


class Admission(unittest.TestCase):
    def test_actual_forward_and_train_schema_positive(self):
        forward,train=actual_schema_fixtures();self.assertNotIn('real_data_used',train)
        self.assertEqual(verify_diagnostic_completions(forward,train)['train'],'passed_explicit_train_gradient_schema')

    def test_each_missing_or_true_isolation_field_rejected(self):
        forward,train=actual_schema_fixtures()
        for key in ('real_scores_read','validation_gradients_used','real_gradients_used'):
            absent=copy.deepcopy(train);del absent[key]
            with self.assertRaises(KeyError):verify_diagnostic_completions(forward,absent)
            enabled=copy.deepcopy(train);enabled[key]=True
            with self.assertRaises(AssertionError):verify_diagnostic_completions(forward,enabled)
        absent=copy.deepcopy(forward);del absent['real_data_used']
        with self.assertRaises(KeyError):verify_diagnostic_completions(absent,train)
        enabled=copy.deepcopy(forward);enabled['real_data_used']=True
        with self.assertRaises(AssertionError):verify_diagnostic_completions(enabled,train)
        wrong=copy.deepcopy(train);wrong['schema']='forward'
        with self.assertRaises(AssertionError):verify_diagnostic_completions(forward,wrong)

    def test_original_v1_failure_is_reproducible_and_not_hidden(self):
        self.assertEqual(reproduce_v1_failure()['error_key'],'real_data_used')


if __name__=='__main__':unittest.main(verbosity=2)
