"""Join unchanged blind320 with independent endpoint-only128; no weakened blind guard."""
from pathlib import Path,PurePosixPath
import argparse,csv,json,math,sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'figures_tool'))
import make_fair_figures as fig
import collect_endpoint_ablation as collector

METHODS=('ours','v2e','endpoint_only')
NAMES=dict(ours='Ours (full)',v2e='V2E',endpoint_only='Ours (endpoint only)')
COLORS=dict(ours='#218d88',v2e='#c68049',endpoint_only='#777799')
METRICS=('penalized_position_error_mean_m','penalized_velocity_error_mean_m_s','control_state_coverage')


def joined_data(main_root,endpoint_root,selection_path=None):
    main_root=Path(main_root);endpoint_root=Path(endpoint_root)
    main=collector.main_reference(main_root);main_stage=fig.load_stage(main_root,'blind')
    selection=Path(selection_path) if selection_path else main_root/'FRONTEND_SELECTION.json'
    blind=fig.validate_blind(main_stage,selection,32,collector.RATES)
    endpoint=fig.load_stage(endpoint_root,'ablation');data=collector.read(endpoint_root/'FAIR_STAGE_RESULTS.json')
    spec=collector.read(endpoint_root/'EVALUATION_PROTOCOL.json');collector.verify_grid(spec,main)
    marker=collector.read(endpoint_root/'FAIR_COLLECTION_COMPLETE.json');audit=collector.read(endpoint_root/'FINAL_AUDIT.json')
    require=collector.require
    require(marker.get('actual_episodes')==128 and data['metadata'].get('variant')=='endpoint_only'
        and data['metadata'].get('main_blind_reference')==main['hashes'] and marker.get('main_blind_reference')==main['hashes'],
        'Ablation completion does not bind the unchanged complete main blind reference')
    require(collector.sha(endpoint_root/'SHA256SUMS')==marker['raw_manifest_sha256']
        and collector.sha(endpoint_root/'FINAL_AUDIT.json')==marker['final_audit_sha256']
        and audit.get('status')=='PASS' and audit.get('episodes')==128 and audit.get('raw_event_contract_and_packet_counts_pass') is True,
        'Endpoint-only full raw audit is missing or inconsistent')
    proof=collector.read(endpoint_root/'ENDPOINT_VARIANT_PROOFS.json')
    delta=collector.read(endpoint_root/'ENDPOINT_SOURCE_DELTA_PROOF.json')
    require(collector.sha(endpoint_root/'ENDPOINT_SOURCE_DELTA_PROOF.json')==marker['source_delta_proof_sha256']
        and delta.get('status')=='PASS' and delta.get('control_loop_byte_identical') is True,
        'Endpoint-only source changes do not have the reviewed immutable control-loop receipt')
    require(collector.sha(endpoint_root/'ENDPOINT_VARIANT_PROOFS.json')==marker['variant_proofs_sha256']
        and proof.get('status')=='PASS' and proof.get('actual_actors')==128 and len(proof.get('records',[]))==128,
        'Endpoint-only actual worker/packet variant proof is incomplete')
    raw=data['rows'];indexed={collector.key(r):r for r in raw}
    require(len(raw)==len(indexed)==128 and set(indexed)==set(main['rows'])
        and all(r['method']=='ours' and r.get('variant')=='endpoint_only' and r['frontend_id']==collector.FRONTEND for r in raw),
        'Endpoint-only results do not contain exactly the registered128 lockedB actors')
    with (endpoint_root/'FAIR_EPISODES.csv').open(encoding='utf-8-sig',newline='') as stream:
        csv_rows=list(csv.DictReader(stream))
    require(len(csv_rows)==128 and all(r.get('variant')=='endpoint_only' for r in csv_rows),'CSV variant labels differ from verified endpoint-only mode')
    proof_by_key={(str(r['physical_case_id']),int(r['seed']),float(r['endpoint_hz'])):r for r in proof['records']}
    require(len(proof_by_key)==128 and set(proof_by_key)==set(indexed),'Endpoint-only variant evidence has missing/repeated physical cells')
    for key,row in indexed.items():
        witness=proof_by_key[key]
        require(witness['status']==('NO_GENERATION_INITIALIZATION_FAILURE' if not row['initialization_valid'] else 'PASS'),
            'Endpoint-only mode proof does not retain the actual initialization outcome')
        if row['initialization_valid']:require(witness.get('intermediate_frames')==0 and witness['worker']['reconstruction_mode']=='endpoint_only'
            and witness.get('per_endpoint_dvs_counter_pass') is True and witness.get('endpoint_reads')==witness['packets']
            and witness.get('dvs_update_calls')==witness['packets']-1,'Actual endpoint-only worker did not prove zero intermediate reconstruction and per-endpoint DVS calls')
    merged={}
    for row in main['data']['rows']:
        if row['method'] in ('ours','v2e'):merged[collector.key(row),row['method']]=row
    for key,row in indexed.items():merged[key,'endpoint_only']=row
    expected={(logical,method) for logical in main['rows'] for method in METHODS}
    require(set(merged)==expected and len(merged)==384,'Joined three-source ablation is incomplete; never omit failures')
    return dict(main=main,main_stage=main_stage,endpoint_stage=endpoint,endpoint_spec=spec,blind=blind,rows=merged,
        selection=selection,endpoint_root=endpoint_root,main_root=main_root)


def tables(join):
    groups=[];paired=[];mechanism=[]
    blocks=sorted(join['blind']['blocks'])
    for rate in collector.RATES:
        for method in METHODS:
            rows=[join['rows'][(physical,seed,rate),method] for physical,seed in blocks]
            values={metric:[] for metric in METRICS}
            for row in rows:
                metrics=row.get('perception')
                collector.require(isinstance(metrics,dict),'Actual controller-state error metrics are missing')
                for metric in METRICS:
                    value=float(metrics[metric]);collector.require(math.isfinite(value),'Nonfinite mechanism metric');values[metric].append(value)
                collector.require(0<=values[METRICS[0]][-1]<=.5+1e-8 and 0<=values[METRICS[1]][-1]<=2+1e-8
                    and 0<=values[METRICS[2]][-1]<=1,'Mechanism metrics do not use the existing failure-penalized contract')
                mechanism.append(dict(physical_case_id=row['physical_case_id'],seed=row['seed'],endpoint_hz=rate,
                    comparison_method=method,actor_method=row['method'],variant=row.get('variant','full'),frontend_id=row['frontend_id'],
                    verified_catch=row['verified_catch'],failure_class=row['failure_class'],queries=metrics['queries'],
                    **{metric:metrics[metric] for metric in METRICS}))
            groups.append(dict(endpoint_hz=rate,comparison_method=method,**fig.counts(rows),
                mean_penalized_position_error_mm=1000*sum(values[METRICS[0]])/32,
                mean_penalized_velocity_error_m_s=sum(values[METRICS[1]])/32,
                mean_controller_state_coverage=sum(values[METRICS[2]])/32))
        cells=dict(both_success=0,full_ours_only=0,endpoint_only_only=0,both_fail=0)
        for physical,seed in blocks:
            a=join['rows'][(physical,seed,rate),'ours']['verified_catch'];b=join['rows'][(physical,seed,rate),'endpoint_only']['verified_catch']
            cells['both_success' if a and b else 'full_ours_only' if a else 'endpoint_only_only' if b else 'both_fail']+=1
        paired.append(dict(endpoint_hz=rate,paired_blocks=32,**cells,
            full_minus_endpoint_success_difference=(cells['full_ours_only']-cells['endpoint_only_only'])/32,
            mcnemar_exact_two_sided=fig.exact_mcnemar(cells['full_ours_only'],cells['endpoint_only_only'])))
    for row,p in zip(paired,fig.holm([r['mcnemar_exact_two_sided'] for r in paired])):row['holm_across_four_ablation_rates']=p
    return groups,paired,mechanism


def media_interface(join,main_media_plan=None):
    blocks=join['blind']['blocks'];by_block={}
    for case in join['main']['spec']['cases']:by_block[case['physical_case_id'],int(case['seed'])]=int(case['block_index'])
    if main_media_plan:
        source=collector.read(main_media_plan)
        collector.require(any(item.get('sha256')==join['main']['hashes']['results_sha256'] for item in source.get('inputs',[])),
            'Main media plan does not reference these unchanged blind outcomes')
        selected=[(str(pair['physical_case_id']),int(pair['seed']),float(pair['endpoint_hz'])) for pair in source['pairs']]
        selection='Exactly reuse main blind representative pair keys; endpoint-only results never select examples'
    else:
        physical,seed=min(blocks,key=lambda b:by_block[b]);selected=[(physical,seed,rate) for rate in collector.RATES]
        selection='Fixed first registered block at every rate, independent of endpoint-only outcomes'
    manifests={}
    for label,folder in [('main',join['main_root']),('endpoint',join['endpoint_root'])]:
        entries={}
        for line in (folder/'SHA256SUMS').read_text(encoding='utf-8-sig').splitlines():
            digest,relative=line.split(None,1);entries[relative.lstrip('*').strip()]=digest
        manifests[label]=entries
    examples=[]
    for key in selected:
        collector.require(key in join['main']['rows'],'Unknown representative physical cell')
        actors=[]
        for method in METHODS:
            row=join['rows'][key,method];root=PurePosixPath(row['output'])
            source='endpoint' if method=='endpoint_only' else 'main'
            protocol=join['endpoint_spec'] if source=='endpoint' else join['main']['spec']
            artifact_paths={kind:str(root/relative) for kind,relative in [('trace','episode_000/trace.json'),
                ('result','episode_000/result.json'),('actor_protocol','protocol.json')]}
            artifact_paths['video' if row['initialization_valid'] else 'initial_image']=str(root/(
                'episode_000/genesis.mp4' if row['initialization_valid'] else 'migration_audit/sensor_episode_000/initial_rgb.png'))
            artifacts={}
            for kind,remote in artifact_paths.items():
                try:label=str(PurePosixPath(remote).relative_to(PurePosixPath(protocol['root'])))
                except ValueError:label=remote
                collector.require(label in manifests[source],'Actual representative source file is not in its completed raw manifest: '+remote)
                artifacts[kind]=dict(remote=remote,sha256=manifests[source][label])
            actors.append(dict(display_method=method,actor_method=row['method'],variant=row.get('variant','full'),
                endpoint_hz=key[2],frontend_id=row['frontend_id'],verified_catch=row['verified_catch'],
                seed=key[1],checkpoint_sha256=protocol['checkpoint_sha256'],sim_time_s=row['sim_time_s'],
                raw_success=row['raw_success'],terminated=row['terminated'],hand_contact_steps=row['hand_contact_steps'],
                initialization_valid=row['initialization_valid'],actor_output=str(root),artifacts=artifacts,
                actual_scene=str(root/'episode_000/genesis.mp4') if row['initialization_valid'] else str(root/'migration_audit/sensor_episode_000/initial_rgb.png'),
                trace=str(root/'episode_000/trace.json'),result=str(root/'episode_000/result.json'),protocol=str(root/'protocol.json'),
                reconstruction_mode='endpoint_only' if method=='endpoint_only' else 'main_unchanged',
                renderer_contract='Use actual frames; reset+trace-step frame mapping; hold ended episodes with an explicit label'))
        examples.append(dict(physical_case_id=key[0],seed=key[1],endpoint_hz=key[2],actors=actors))
    return dict(schema='endpoint_ablation_true_actor_media_join_v1',status='INTERFACE_ONLY_NO_RENDER',selection=selection,
        no_new_physical_images=True,no_outcome_based_endpoint_example_selection=True,examples=examples,
        main_blind_reference=join['main']['hashes'])


def plot(join,output,main_media_plan=None):
    groups,paired,mechanism=tables(join);output=Path(output)
    collector.require(not output.exists(),'Use a fresh ablation figure directory')
    output.mkdir(parents=True)
    plt=fig.pyplot();figure,axis=plt.subplots(figsize=(9.2,5.8))
    for method,marker in zip(METHODS,['o','s','^']):
        values=[r for r in groups if r['comparison_method']==method];x=[r['endpoint_hz'] for r in values]
        axis.plot(x,[r['success_rate'] for r in values],marker=marker,color=COLORS[method],label=NAMES[method],linewidth=2)
        axis.fill_between(x,[r['wilson95_low'] for r in values],[r['wilson95_high'] for r in values],color=COLORS[method],alpha=.10)
    axis.set_xscale('log',base=2);axis.set_xticks(collector.RATES,[f'{r:g}' for r in collector.RATES]);axis.set_ylim(-.04,1.04)
    axis.set_xlabel('Source image rate (Hz)');axis.set_ylabel('Verified catch success');axis.grid(alpha=.15);axis.legend(frameon=False)
    figure.suptitle('Ablation: contribution of intermediate reconstruction',fontsize=15,fontweight='bold')
    figure.subplots_adjust(bottom=.24)
    figure.text(.04,.045,'Same32 physical seed blocks at every rate; full Ours/V2E reused unchanged from main blind results.\nEndpoint-only:128 additional actors, same locked Ours frontend and response. Shading: pointwise Wilson95% intervals.',fontsize=9)
    artifacts=fig.save_figure(figure,output,'endpoint_ablation_success');plt.close(figure)
    figure,axes=plt.subplots(1,3,figsize=(13.2,4.8))
    fields=['mean_penalized_position_error_mm','mean_penalized_velocity_error_m_s','mean_controller_state_coverage']
    labels=['Penalized position error (mm)','Penalized velocity error (m/s)','Controller-state availability']
    for axis,field,label in zip(axes,fields,labels):
        for method in METHODS:
            values=[r for r in groups if r['comparison_method']==method]
            axis.plot([r['endpoint_hz'] for r in values],[r[field] for r in values],marker='o',color=COLORS[method],label=NAMES[method])
        axis.set_xscale('log',base=2);axis.set_xticks(collector.RATES,[f'{r:g}' for r in collector.RATES]);axis.set_ylabel(label);axis.set_xlabel('Source image rate (Hz)');axis.grid(alpha=.15)
    axes[-1].legend(frameon=False,fontsize=9);figure.suptitle('Actual controller-state errors before first physical contact',fontweight='bold')
    figure.subplots_adjust(left=.07,right=.99,wspace=.34,bottom=.28)
    figure.text(.03,.06,'Episode-macro means over the same32 blocks; all failures included, missing/invalid states penalized at0.5m/2m/s.\nThese are on-policy closed-loop diagnostics: physical trajectories may diverge after different actions.',fontsize=9)
    artifacts+=fig.save_figure(figure,output,'endpoint_ablation_mechanism');plt.close(figure)
    for name,rows in [('endpoint_ablation_groups.csv',groups),('full_vs_endpoint_paired_by_rate.csv',paired),('all384_actor_mechanism.csv',mechanism)]:
        path=output/name;fig.write_csv(path,rows);artifacts.append(path)
    path=output/'ENDPOINT_DEMO_JOIN_PLAN.json';collector.write(path,media_interface(join,main_media_plan));artifacts.append(path)
    inputs=join['main_stage']['inputs']+join['endpoint_stage']['inputs']+[join['main_root']/'EVALUATION_PROTOCOL.json',
        join['endpoint_root']/'EVALUATION_PROTOCOL.json',join['endpoint_root']/'ENDPOINT_VARIANT_PROOFS.json',
        join['endpoint_root']/'ENDPOINT_SOURCE_DELTA_PROOF.json',join['selection'],Path(__file__),Path(collector.__file__)]
    if main_media_plan:inputs.append(Path(main_media_plan))
    fig.manifest(output,'endpoint_ablation',inputs,artifacts,dict(independent_physical_blocks=32,new_endpoint_actors=128,
        reused_main_event_actors=256,all_four_rates_shown=True,main_reference=join['main']['hashes'],
        no_unchanged_blind_guard_relaxed=True,ablation_frontend_retuned=False,
        paired_test='Full Ours vs endpoint-only separately per rate; four tests Holm adjusted; rates not independent new samples',
        mechanism_scope='On-policy episode-macro actual controller states; not same-state image or event fidelity proof'))
    return dict(status='PASS',main_actors_reused=256,endpoint_actors=128,independent_blocks=32,output=str(output.resolve()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--main-blind-root',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--selection',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--main-media-plan',type=Path);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    try:
        join=joined_data(args.main_blind_root,args.root,args.selection)
        if args.check_only:
            groups,paired,mechanism=tables(join);result=dict(status='PASS',new_endpoint_actors=128,reused_main_actors=256,
                groups=len(groups),paired_rate_tests=len(paired),mechanism_actor_rows=len(mechanism),plots_written=0)
        else:result=plot(join,args.output,args.main_media_plan)
        print(json.dumps(result))
    except (ValueError,fig.DataError,AssertionError,KeyError,OSError,TypeError) as error:
        parser.exit(2,'NOT_READY_ENDPOINT_ABLATION: '+str(error)+'\nNo incomplete/selected subset is plotted.\n')


if __name__=='__main__':main()
