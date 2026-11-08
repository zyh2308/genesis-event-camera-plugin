"""CPU-only contract checks; these are not measured experiment cost results."""
import importlib.util
from pathlib import Path
import unittest
from cost_ledger import WallLedger


class TimingContracts(unittest.TestCase):
    def test_nested_spans_do_not_double_count_worker_wait(self):
        t=[0]
        ledger=WallLedger(clock=lambda:t[0],cpu_clock=lambda:t[0])
        ledger.start_hot()
        with ledger.span('capture'):
            t[0]=4
            with ledger.span('worker_wait'):
                t[0]=9
            t[0]=13
        ledger.stop_hot()
        result=ledger.summary()
        self.assertEqual(result['hot_wall_s'],13/1e9)
        self.assertEqual(result['hot_stages']['capture']['exclusive_wall_s'],8/1e9)
        self.assertEqual(result['hot_stages']['worker_wait']['exclusive_wall_s'],5/1e9)
        self.assertAlmostEqual(result['measured_parent_exclusive_sum_s'],13/1e9)

    def test_cold_initialization_is_not_warm_compute(self):
        t=[0]
        ledger=WallLedger(clock=lambda:t[0],cpu_clock=lambda:t[0])
        with ledger.span('cold_model'):
            t[0]=100
        ledger.start_hot()
        with ledger.span('hot_packet'):
            t[0]=120
        ledger.stop_hot()
        self.assertNotIn('cold_model',ledger.summary()['hot_stages'])
        self.assertEqual(ledger.summary()['hot_wall_s'],20/1e9)
        self.assertIsNone(ledger.summary()['gpu_cumulative_time_s'])

    def test_end_horizon_cannot_silently_stop_an_open_span(self):
        ledger=WallLedger()
        ledger.start_hot()
        with ledger.span('physics'):
            with self.assertRaises(RuntimeError):
                ledger.stop_hot()
        ledger.stop_hot()

    def test_disabled_archive_creates_no_fake_hdr_file(self):
        path=Path(__file__).with_name('run_fixed_trajectory_cost.py')
        spec=importlib.util.spec_from_file_location('isolated_cost_launcher_contract',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        archive=module.NullArchive()
        dataset=archive.create_dataset('radiance',shape=(0,2,2,3))
        dataset.resize(1,axis=0)
        dataset[0]='no pixel stored by an explicit disabled archive'
        self.assertEqual(dataset.shape,(1,2,2,3))
        self.assertIn('radiance',archive)
        self.assertIsNone(archive.close())

    def test_prewarm_snapshot_does_not_advance_execution_noise_or_mutate_actions(self):
        import numpy as np
        from types import SimpleNamespace
        path=Path(__file__).with_name('run_fixed_trajectory_cost.py')
        spec=importlib.util.spec_from_file_location('isolated_cost_diagnostic_contract',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env=SimpleNamespace(Dcmm=SimpleNamespace(data_arm=SimpleNamespace(qpos=np.arange(6.))),
            action_buffer={'arm':[]})
        live={'arm':np.array([.1,.2],dtype=np.float32)}
        recorded={'arm':live['arm'].copy()}
        np.random.seed(783)
        before=np.random.get_state()
        snapshot=module.prewarm_physics_snapshot(env,live,recorded,np)
        after=np.random.get_state()
        self.assertEqual(before[0],after[0])
        np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(before[2:],after[2:])
        np.testing.assert_array_equal(live['arm'],recorded['arm'])
        self.assertEqual(snapshot['actual_actor_vs_archived_decoded_max_abs']['arm'],0.)
        self.assertEqual(snapshot['actual_actor_action_dtypes']['arm'],'float32')

    def test_public_reference_requires_identical_actions_and_full_actual_states(self):
        import copy
        path=Path(__file__).with_name('run_fixed_trajectory_cost.py')
        spec=importlib.util.spec_from_file_location('isolated_public_reference_contract',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        controls=[dict(action=[.125],t_before=0.,t=.04,qpos=[1.],qvel=[0.])]
        expected=dict(original_action_trace_sha256='archive',recording_host='same_registered_host')
        value=dict(schema=module.PHYSICS_SCHEMA,status=module.RECORDED,utility_result=False,
            cost_measurement=False,**expected,controls=[dict(step=0,action=[.125],t_before=0.,t=.04,qpos=[1.001],qvel=[.02])])
        self.assertEqual(module.validate_public_physics_reference(value,controls,expected),value['controls'])
        for change in ('action','qpos','t'):
            broken=copy.deepcopy(value)
            broken['controls'][0][change]={'action':[.126],'qpos':[float('nan')],'t':.08}[change]
            with self.assertRaises(ValueError):module.validate_public_physics_reference(broken,controls,expected)
        broken=copy.deepcopy(value);broken['recording_host']='a_different_host'
        with self.assertRaises(ValueError):module.validate_public_physics_reference(broken,controls,expected)

    def test_reference_recording_never_exports_as_formal_cost(self):
        import json
        from types import SimpleNamespace
        path=Path(__file__).with_name('build_cost_table.py')
        spec=importlib.util.spec_from_file_location('isolated_cost_export_contract',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        record=SimpleNamespace(read_text=lambda **kw:json.dumps(dict(status='PHYSICS_REFERENCE_RECORDED',
            utility_result=False,cost_only_shadow=True,cost_measurement=False)))
        with self.assertRaises(ValueError):module.row_from_result(record)


if __name__=='__main__':
    unittest.main()
