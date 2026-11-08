"""Small endpoint-only grid/variant/statistics checks; no physical runs or fake figures."""
from pathlib import Path
from copy import deepcopy
import sys,tempfile,unittest
sys.dont_write_bytecode=True
import collect_endpoint_ablation as collect
import plot_endpoint_ablation as plot


def registered():
    env=dict(CATCHIT_EVENT_OBSERVATION_WINDOW_MS='10',CATCHIT_EVENT_FLIGHT_FIT='0',CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS='100')
    base=dict(core_files={str(i):'a'*64 for i in range(17)},checkpoint_sha256='b'*64,profile_sha256='c'*64,
        frontend_selection_sha256='d'*64,bank_sha256='e'*64,frontend_by_method={'ours':env},
        runtime_files={'sparse_event_sensor.py':'old_sensor','run_original.py':'old_runner','genesis_adapter.py':'unchanged'})
    cases=[dict(case_id=f'b{i}_hz{r}',physical_case_id=f'b{i}',seed=1000+i,endpoint_hz=r,
        method_order=['ours'],launch={'position':[i/100,1.,1.]},lighting='normal') for i in range(32) for r in collect.RATES]
    main=dict(spec=deepcopy(base),rows={collect.key(c):{} for c in cases},cases={collect.key(c):c for c in cases})
    spec=deepcopy(base);spec.update(event_reconstruction_mode='endpoint_only',cases=deepcopy(cases))
    spec['runtime_files'].update({'sparse_event_sensor.py':'new_sensor','run_original.py':'new_runner','endpoint_event_worker.py':'new_worker'})
    return spec,main


class EndpointContracts(unittest.TestCase):
    def test_exact128_grid_accepts_without_requiring_missing_v2e(self):
        spec,main=registered();self.assertEqual(len(collect.verify_grid(spec,main)),128)
    def test_missing_repeated_or_other_actor_rejected(self):
        spec,main=registered();spec['cases'].pop()
        with self.assertRaises(ValueError):collect.verify_grid(spec,main)
        spec,main=registered();spec['cases'][-1]=deepcopy(spec['cases'][0])
        with self.assertRaises(ValueError):collect.verify_grid(spec,main)
        spec,main=registered();spec['cases'][0]['method_order']=['ours','v2e']
        with self.assertRaises(ValueError):collect.verify_grid(spec,main)
    def test_changed_launch_core_policy_frontend_or_shared_runtime_rejected(self):
        for field in ['launch','core_files','checkpoint_sha256','frontend','runtime']:
            spec,main=registered()
            if field=='launch':spec['cases'][0]['launch']['position'][0]+=.01
            elif field=='frontend':spec['frontend_by_method']['ours']['CATCHIT_EVENT_FLIGHT_FIT']='1'
            elif field=='runtime':spec['runtime_files']['genesis_adapter.py']='changed'
            elif field=='core_files':spec[field]['0']='changed'
            else:spec[field]='changed'
            with self.assertRaises(ValueError,msg=field):collect.verify_grid(spec,main)
    def test_worker_and_every_packet_need_zero_intermediate_evidence(self):
        with tempfile.TemporaryDirectory(prefix='endpoint_contract_',dir=Path(__file__).parent) as folder:
            root=Path(folder);sensor=root/'migration_audit/sensor_episode_000';sensor.mkdir(parents=True)
            collect.write(sensor/'worker_start.json',dict(reconstruction_mode='endpoint_only',intermediate_frames=0))
            collect.write(sensor/'packet_audit.json',[dict(reconstruction_mode='endpoint_only',intermediate_frames=0,endpoint_reads=1,dvs_update_calls=0)])
            self.assertEqual(collect.variant_proof(root,dict(event_reconstruction_mode='endpoint_only'),{} )['status'],'PASS')
            collect.write(sensor/'packet_audit.json',[dict(reconstruction_mode='endpoint_only',intermediate_frames=0,endpoint_reads=1,dvs_update_calls=1)])
            with self.assertRaises(ValueError):collect.variant_proof(root,dict(event_reconstruction_mode='endpoint_only'),{})
            collect.write(sensor/'packet_audit.json',[dict(reconstruction_mode='full',intermediate_frames=4)])
            with self.assertRaises(ValueError):collect.variant_proof(root,dict(event_reconstruction_mode='endpoint_only'),{})
    def test_statistics_keep32_blocks_and_all384_actual_actor_rows(self):
        _,main=registered();rows={}
        for logical in main['rows']:
            for method in plot.METHODS:
                fail=method=='endpoint_only'
                rows[logical,method]=dict(physical_case_id=logical[0],seed=logical[1],method='ours' if fail else method,
                    frontend_id=collect.FRONTEND,verified_catch=not fail,failure_class='no_contact' if fail else 'none',
                    perception=dict(queries=0 if fail else 12,control_state_coverage=0. if fail else 1.,
                        penalized_position_error_mean_m=.5 if fail else .01,penalized_velocity_error_mean_m_s=2. if fail else .1))
        blocks=sorted({(p,s) for p,s,r in main['rows']});groups,paired,mechanism=plot.tables(dict(rows=rows,blind={'blocks':blocks}))
        self.assertEqual(len(groups),12);self.assertEqual(len(paired),4);self.assertEqual(len(mechanism),384)
        self.assertTrue(all(r['n']==32 for r in groups));self.assertTrue(all(r['paired_blocks']==32 for r in paired))
        self.assertTrue(all(r['mean_penalized_position_error_mm']==500 and r['successes']==0 for r in groups if r['comparison_method']=='endpoint_only'))


if __name__=='__main__':unittest.main()
