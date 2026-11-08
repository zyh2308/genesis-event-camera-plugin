"""Assemble the immutable 32-block experiment only after development selection."""
from pathlib import Path
import argparse, datetime, hashlib, json

BANK_SHA = 'a38be34e36a70b710e711b91fabd4673b244b2107e7c9d2b8a14c5f0784e6b72'
VALUES = {'CATCHIT_EVENT_OBSERVATION_WINDOW_MS': {'0','10'},
          'CATCHIT_EVENT_FLIGHT_FIT': {'0','1'},
          'CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS': {'50','100'}}


def main():
    p=argparse.ArgumentParser()
    for k in ['base-runtime','selection','bank','output']:
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--root',required=True)
    a=p.parse_args()
    assert hashlib.sha256(a.bank.read_bytes()).hexdigest()==BANK_SHA
    bank=json.loads(a.bank.read_text(encoding='utf-8-sig'))
    selection=json.loads(a.selection.read_text(encoding='utf-8-sig'))
    assert selection['status']=='LOCKED'
    assert selection['no_blind_outcome_used'] and selection['single_global_configuration_per_source']
    assert selection['actual_primary_episodes']==48 and selection['physical_seed_blocks']==4
    validation=Path(selection['validation_root'])
    for field,name in [('validation_results_sha256','FAIR_STAGE_RESULTS.json'),
                       ('validation_completion_sha256','FAIR_COLLECTION_COMPLETE.json'),
                       ('registered_protocol_sha256','EVALUATION_PROTOCOL.json')]:
        assert hashlib.sha256((validation/name).read_bytes()).hexdigest()==selection[field]
    completion=json.loads((validation/'FAIR_COLLECTION_COMPLETE.json').read_text())
    assert completion['status']=='PASS' and completion['results_sha256']==selection['validation_results_sha256']
    frontends=selection['frontend_by_method']
    assert set(frontends)=={'ours','v2e'}
    for source,env in frontends.items():
        assert set(env)==set(VALUES)
        assert all(str(env[k]) in VALUES[k] for k in VALUES)
    spec=json.loads(a.base_runtime.read_text())
    parent=Path(spec['root'])
    spec.update(root=a.root,stage='locked_frontend_wider_blind32',cases=[],
        actual_actor_budget=320,independent_throwing_blocks=32,
        independent_sample_unit='32 preregistered throwing blocks; four source rates and two generators are paired repeated conditions.',
        frontend_by_method=frontends,
        frontend_selection_sha256=hashlib.sha256(a.selection.read_bytes()).hexdigest(),
        frontend_selection_file=str(a.selection),
        frontend_lock=selection,
        bank_sha256=BANK_SHA,
        bank_parameters_sha256=bank['parameters_sha256'],
        registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        no_blind_outcome_used_for_configuration=True,
        shared_reference_reuse='One init_only and one noisy GT state_reference per block, explicitly shared across four source rates.',
        catch_definition='Original task reaches horizon without failure, real hand touch occurred, and hand touch exists in the final40ms control step. No continuous0.5s hold claim.')
    spec['v2e_parity_receipts']={str(hz):str(parent/'native_parity'/('shard'+str(shard))/str(hz)/'V2E_PARITY_RECEIPT.json')
        for shard,rates in [(0,[100,50]),(1,[25,12.5])] for hz in rates}
    a.output.mkdir(exist_ok=False,parents=True)
    launchroot=a.output/'launch_configs';launchroot.mkdir()
    for block in bank['cases']:
        shard=block['block_index']-1
        assert 0<=shard<32
        sequence=list(block['primary_condition_order'])+[
            {'endpoint_hz':25,'method':method} for method in block['reference_method_order']]
        assert len(sequence)==10
        for index,condition in enumerate(sequence):
            method=condition['method'];hz=float(condition['endpoint_hz'])
            if hz.is_integer():hz=int(hz)
            caseid=block['case_id']+'_hz'+str(hz).replace('.','p')+'_'+method
            case=dict(case_id=caseid,physical_case_id=block['case_id'],block_index=block['block_index'],
                seed=block['seed'],shard=shard,launch=block['launch'],stratum=block['stratum'],
                condition='wide_original_distribution_blind32',lighting='normal',endpoint_hz=hz,
                method_order=[method],condition_order_within_block=index,
                selected_frontend_id='w{w}_fit{f}_gap{g}'.format(
                    w=frontends[method]['CATCHIT_EVENT_OBSERVATION_WINDOW_MS'],
                    f=frontends[method]['CATCHIT_EVENT_FLIGHT_FIT'],
                    g=frontends[method]['CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS']) if method in frontends else 'shared_reference')
            spec['cases'].append(case)
            (launchroot/(caseid+'.json')).write_text(json.dumps(block['launch'],indent=2),encoding='utf-8',newline='\n')
    assert len(spec['cases'])==320
    assert len({(c['case_id'],c['method_order'][0]) for c in spec['cases']})==320
    protocol=a.output/'EVALUATION_PROTOCOL.json'
    protocol.write_text(json.dumps(spec,indent=2),encoding='utf-8',newline='\n')
    queue=a.output/'workqueue';queue.mkdir()
    (queue/'PLAN.json').write_text(json.dumps(dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        ordered_shards=list(range(32)),actual_actor_budget=320,no_idle_wait=True),indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='PASS',cases=320,blocks=32,protocol=str(protocol),
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest())))


if __name__=='__main__':
    main()
