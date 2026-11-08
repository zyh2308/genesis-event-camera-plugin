"""Register the single-component ablation on the original 32 throwing blocks."""
from pathlib import Path
import argparse, copy, datetime, hashlib, json

MAIN_SHA='a5fa1e8d3d8cb448f2b798e5c8627df9e908eacd1c5464adead9ffa78e5cb766'
BASE_COMMIT='4931807d1cb229a55a96961b5c4d99b054ef4fb0'
MODE='endpoint_only'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(main, runtime, output, main_path, runtime_path, main_sha):
    assert main_sha==MAIN_SHA and main['code_commit']==BASE_COMMIT
    assert main['actual_actor_budget']==320 and main['independent_throwing_blocks']==32
    assert runtime['event_reconstruction_mode']==MODE
    assert runtime['base_source']==main['source'] and runtime['base_code_commit']==BASE_COMMIT
    for key in ('core_root','core_files','profile_file','profile_sha256','checkpoint','checkpoint_sha256'):
        assert runtime[key]==main[key],key
    changed={name for name,digest in main['runtime_files'].items() if runtime['runtime_files'].get(name)!=digest}
    added=set(runtime['runtime_files'])-set(main['runtime_files'])
    assert changed=={'sparse_event_sensor.py','run_original.py'} and added=={'endpoint_event_worker.py'}
    assert not (set(main['runtime_files'])-set(runtime['runtime_files']))
    assert main['frontend_lock']['status']=='LOCKED' and main['frontend_lock']['no_blind_outcome_used']
    selected=main['frontend_by_method']['ours']
    assert selected=={'CATCHIT_EVENT_OBSERVATION_WINDOW_MS':'10','CATCHIT_EVENT_FLIGHT_FIT':'0',
                      'CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS':'100'}
    output=Path(output)
    spec=copy.deepcopy(main)
    spec.update(root=str(output),source=runtime['source'],runtime_files=runtime['runtime_files'],
        code_commit=runtime['code_commit'],template=list(runtime['template']),
        stage='endpoint_only_ablation_after_locked_blind',event_reconstruction_mode=MODE,variant=MODE,
        actual_actor_budget=128,independent_throwing_blocks=32,
        primary_methods=['ours'],reference_methods=[],frontend_by_method={'ours':selected},
        paired_main_protocol=str(main_path),paired_main_protocol_sha256=main_sha,
        paired_main_root=main['root'],ablation_runtime_file=str(runtime_path),
        ablation_runtime_sha256=sha(runtime_path),
        endpoint_source_delta_receipt=runtime['endpoint_source_delta_receipt'],
        endpoint_source_delta_sha256=runtime['endpoint_source_delta_sha256'],
        required_job_env={'CATCHIT_EVENT_RECONSTRUCTION_MODE':MODE},
        registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        no_ablation_outcome_used_for_configuration=True,
        source_delta={'changed':sorted(changed),'added':sorted(added)},
        ablation_definition='One accurate HDR endpoint per DVS update; no intermediate image/warp. Frozen pixel response, its internal event timestamps, source cadence and parent end+10ms availability are preserved.',
        order='Only after the original full320 blind audit and complete24 cost repeats pass. Within each physical block retain the original Ours rate order.',
        independent_sample_unit='32 original physical throwing blocks; four paired cadence repetitions are correlated.',
        shared_reference_reuse='Use the original320 study Ours/V2E/reference results without repeating or inflating their denominators.')
    cases=[]
    for original in main['cases']:
        if original['method_order']!=['ours']:continue
        case=copy.deepcopy(original)
        case.update(case_id='endpoint_'+original['case_id'],paired_full_case_id=original['case_id'],
                    event_reconstruction_mode=MODE,variant=MODE,
                    condition='same_bank_endpoint_only_ablation',selected_frontend_id='w10_fit0_gap100')
        cases.append(case)
    keys={(c['physical_case_id'],c['seed'],float(c['endpoint_hz'])) for c in cases}
    assert len(cases)==len(keys)==128 and len({c['physical_case_id'] for c in cases})==32
    assert {float(c['endpoint_hz']) for c in cases}=={100.,50.,25.,12.5}
    assert {c['shard'] for c in cases}==set(range(32))
    assert all(sum(x['shard']==i for x in cases)==4 for i in range(32))
    spec['cases']=cases
    return spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main-protocol',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();parent=read(a.main_protocol);runtime=read(a.runtime)
    spec=build(parent,runtime,a.output,a.main_protocol,a.runtime,sha(a.main_protocol))
    assert not a.output.exists(),'Fresh ablation root required'
    a.output.mkdir(parents=True);launch=a.output/'launch_configs';launch.mkdir()
    for case in spec['cases']:
        (launch/(case['case_id']+'.json')).write_text(json.dumps(case['launch'],indent=2),encoding='utf-8')
    path=a.output/'EVALUATION_PROTOCOL.json';path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    queue=a.output/'workqueue';queue.mkdir()
    (queue/'PLAN.json').write_text(json.dumps(dict(protocol_sha256=sha(path),ordered_shards=list(range(32)),
        actual_actor_budget=128,no_idle_wait=True,event_reconstruction_mode=MODE),indent=2),encoding='utf-8')
    print(json.dumps(dict(status='REGISTERED_NOT_STARTED',actors=128,blocks=32,root=str(a.output),
                         protocol_sha256=sha(path),mode=MODE)))


if __name__=='__main__':main()
