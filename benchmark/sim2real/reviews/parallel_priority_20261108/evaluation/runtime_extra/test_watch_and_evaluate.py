"""Small CPU fixtures for dependency, authorization and no-GPU-before-ready guards."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).with_name('watch_and_evaluate.py')
spec = importlib.util.spec_from_file_location('watcher_contract_fixture', PATH)
watcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(watcher)


def plan():
    return dict(schema='recon_selected_full_plan_v1', root=str(watcher.TRAINING),
                methods=watcher.METHODS, seeds=watcher.SEEDS, real_sequences=watcher.SEQUENCES,
                training_budget_updates=12000, epochs=200, final_activation='sigmoid',
                loss=dict(lpips_vgg=1., temporal=2., auxiliary_l1=0.),
                real_result_scope=watcher.SCOPE, real_scoring_enabled=False,
                no_real_feedback_to_training_or_selection=True, original_hqf14_results_preserved=True,
                source_user_instruction='EXPLICIT_STANDING_SELECTED3_FIXTURE_NOT_A_NEW_SIGNATURE',
                lanes=[dict(seed=s, host='g3', gpu_index=index, gpu_uuid='GPU-FIXTURE-' + str(s),
                            gpu_model='NVIDIA GeForce RTX 4090') for s, index in enumerate((2, 5, 6))])


def matrix():
    return dict(schema='recon_selected_frozen_checkpoints_v1', status='frozen_all15_before_real_scoring',
                methods=watcher.METHODS, seeds=watcher.SEEDS, real_sequences=watcher.SEQUENCES,
                result_scope=watcher.SCOPE, runs=[dict(method=m, seed=s, completed_updates=12000)
                                                for m in watcher.METHODS for s in watcher.SEEDS])


def completion():
    return dict(status='full15_training_complete_selected_eval_pending', training_runs=15,
                evaluated_sequence_runs=0, real_scoring_enabled=False, real_result_scope=watcher.SCOPE,
                full14_historical_results_unchanged=True)


class WatcherTests(unittest.TestCase):
    def test_full_freeze_admitted_but_partial_or_early_real_rejected(self):
        watcher.frozen_contract(matrix(), completion())
        bad = matrix(); bad['runs'].pop()
        with self.assertRaises(ValueError): watcher.frozen_contract(bad, completion())
        bad = matrix(); bad['runs'][0]['completed_updates'] = 11999
        with self.assertRaises(ValueError): watcher.frozen_contract(bad, completion())
        bad = completion(); bad['evaluated_sequence_runs'] = 1
        with self.assertRaises(ValueError): watcher.frozen_contract(matrix(), bad)

    def test_lanes_are_only_original_three_and_training_stays_closed(self):
        self.assertEqual([r['gpu_index'] for r in watcher.plan_contract(plan())], [2, 5, 6])
        bad = plan(); bad['lanes'][0]['gpu_index'] = 1
        with self.assertRaises(ValueError): watcher.plan_contract(bad)
        bad = plan(); bad['real_scoring_enabled'] = True
        with self.assertRaises(ValueError): watcher.plan_contract(bad)

    def test_derived_authorization_binds_protocol_and_preserves_standing_instruction(self):
        p = plan(); template = dict(schema='recon_selected_hqf3_development_authorization_v1', authorized=False,
            task='reconstruction', result_scope=watcher.SCOPE, protocol_sha256='a' * 64,
            no_feedback_to_training_or_selection=True, all15_frozen_before_any_real_prediction=True)
        before = copy.deepcopy(template)
        actual = watcher.derive_authorization(template, 'a' * 64, p, 'Automated CPU guards passed under standing scope')
        self.assertTrue(actual['authorized']); self.assertEqual(template, before)
        self.assertEqual(actual['source_user_instruction'], p['source_user_instruction'])
        self.assertIn('no new human signature', actual['authorization_derivation'])
        with self.assertRaises(ValueError): watcher.derive_authorization(template, 'b' * 64, p, 'note')
        with self.assertRaises(ValueError): watcher.derive_authorization(template, 'a' * 64, p, '')

    def test_source_order_has_no_gpu_acquisition_until_cpu_admission(self):
        tree = ast.parse(PATH.read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        text = ast.get_source_segment(PATH.read_text(encoding='utf-8'), fn)
        order = [text.index(x) for x in ("'waiting_for_all15_training_freeze_no_gpu'", "'prepare'",
                 "'verify'", "'metric_assets'", 'derive_authorization(', 'acquire_one(', "'evaluate_all45'")]
        self.assertEqual(order, sorted(order))
        self.assertIn('CUDA_VISIBLE_DEVICES=\'\'', text)
        self.assertIn("CUDA_VISIBLE_DEVICES=state['gpu_uuid']", text)
        self.assertFalse(any(isinstance(n, ast.Import) and any(x.name in ('torch', 'cupy') for x in n.names)
                             for n in ast.walk(tree)))


if __name__ == '__main__': unittest.main(verbosity=2)
