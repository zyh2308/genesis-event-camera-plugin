"""Meaningful stdlib-only rejection fixtures, no real data/model/Torch/GPU."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,subprocess,sys

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
out=a.out.resolve();assert not out.exists();out.mkdir(parents=True)
source=Path(__file__).with_name('stage9_hqf_bridge.py')
spec=importlib.util.spec_from_file_location('stage9_hqf_bridge_fixture',source)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')

def fixture(name):
    root=out/name;recipe=root/'sources/training_recipe.json'
    write(recipe,dict(frozen_simulator_core=b.CORE,seeds=list(b.SEEDS),reconstruction={'expected_optimizer_updates':12000,'model_args':{'fixture_only':True}}))
    for seed in b.SEEDS:
        for method in b.METHODS:
            run=root/'training/reconstruction'/f'seed{seed}'/method;run.mkdir(parents=True)
            cp=run/'synthetic_best.pt';cp.write_bytes(f'NOT_A_MODEL fixture only {method} seed{seed}'.encode())
            write(run/'completion.json',dict(task='reconstruction',method=method,seed=seed,status='complete',
              completed_updates=12000,recipe_sha256=b.sha(recipe),real_data_used_in_training=False,
              real_scoring_performed=False,selected_checkpoint=str(cp),selected_checkpoint_sha256=b.sha(cp),
              initial_state_sha256=hashlib.sha256(f'shared_seed{seed}'.encode()).hexdigest()))
    return root,recipe

results=[]
def reject(name,mutate,expected):
    root,recipe=fixture(name);mutate(root,recipe)
    try:b.collect_training(root,recipe)
    except (RuntimeError,FileNotFoundError) as exc:
        assert expected in str(exc),(name,str(exc));results.append(dict(test=name,passed=True,rejection=str(exc)))
    else:raise AssertionError('Invalid training state accepted: '+name)

root,recipe=fixture('valid_full_matrix');runs=b.collect_training(root,recipe)
assert len(runs)==15;results.append(dict(test='valid_full_matrix',passed=True,runs=len(runs)))
def change(root,recipe,key,value):
    f=root/'training/reconstruction/seed0/v2e/completion.json';j=b.read(f);j[key]=value;write(f,j)
reject('short_budget',lambda r,p:change(r,p,'completed_updates',11999),'Incomplete formal budget')
reject('wrong_recipe',lambda r,p:change(r,p,'recipe_sha256','f'*64),'Recipe fingerprint mismatch')
reject('real_training_used',lambda r,p:change(r,p,'real_data_used_in_training',True),'real-data guard')
reject('identity_mismatch',lambda r,p:change(r,p,'method','eventgenesis'),'identity mismatch')
reject('paired_initialization_mismatch',lambda r,p:change(r,p,'initial_state_sha256','e'*64),'Paired initialization differs')
def tamper(root,recipe):
    cp=root/'training/reconstruction/seed0/v2e/synthetic_best.pt';cp.write_bytes(cp.read_bytes()+b'tampered')
reject('checkpoint_bytes_changed',tamper,'checkpoint SHA mismatch')
def wrong_selected(root,recipe):
    cp=root/'training/reconstruction/seed0/v2e/last.pt';cp.write_bytes(b'NOT_A_MODEL last checkpoint fixture')
    change(root,recipe,'selected_checkpoint',str(cp));change(root,recipe,'selected_checkpoint_sha256',b.sha(cp))
reject('last_checkpoint_substitution',wrong_selected,'synthetic_best.pt is eligible')
def missing(root,recipe):
    f=root/'training/reconstruction/seed2/rife_esim/completion.json';f.rename(f.with_name('not_a_completion.json'))
reject('one_missing_of15',missing,'Missing formal training receipts')
cli_root,cli_recipe=fixture('prepare_missing_run_cli');missing(cli_root,cli_recipe)
cli_repo=out/'empty_repository';cli_repo.mkdir();cli_out=out/'must_not_prepare_partial_matrix'
command=[sys.executable,str(source),'prepare','--repo',str(cli_repo),'--campaign',str(cli_root),
    '--hqf-root',str(out/'nonexistent_REAL_DATA'),'--model-root',str(out/'nonexistent_MODEL'),
    '--evreal-root',str(out/'nonexistent_EVREAL'),'--out',str(cli_out)]
failed=subprocess.run(command,text=True,capture_output=True)
assert failed.returncode!=0 and 'Missing formal training receipts' in failed.stderr
assert not cli_out.exists()
results.append(dict(test='prepare_cli_refuses_missing_run_before_data_or_output',passed=True,exit_code=failed.returncode))

# Verify protocol/sidecar and source binding without loading any real dataset.
bound={}
for name in ('inherited_protocol','data_manifest'):
    f=root/(name+'.json');write(f,{'fixture':True});bound[name]=b.record(f)
source_roots={}
pins=[]
for name in ('EVREAL','event_cnn_minimal'):
    tree=root/'fake_sources'/name;tree.mkdir(parents=True)
    payload=tree/'fixture_model_source.py';payload.write_text('# NOT A MODEL. Fixture source only.\n')
    pin=b.sha(payload);pins.append({'repo':name,'commit':pin});source_roots[name]=tree
bundle=root/'downstream_source_bundle.json';write(bundle,{'repositories':pins})
bound['downstream_source_bundle']=b.record(bundle)
verifier=root/'fake_verify_pinned_source.py'
verifier.write_text('''from pathlib import Path
import hashlib
def verify_pinned_source(root,name,commit,bundle):
    actual=hashlib.sha256((Path(root)/'fixture_model_source.py').read_bytes()).hexdigest()
    if actual!=commit:raise RuntimeError('Downstream fixture source changed: '+name)
    return {'repo':name,'commit':commit,'verified':True}
''')
protocol=dict(schema='stage9_hqf15_frozen_protocol_v1',status='prepared_unscored',real_scores_computed=False,
    methods=list(b.METHODS),seeds=list(b.SEEDS),expected_runs=15,expected_sequence_runs=210,
    campaign=str(root),runs=runs,training_recipe=b.record(recipe),bridge_source=b.record(source),
    evaluation_sources={'verify_pinned_source.py':b.record(verifier)},
    model_root=str(source_roots['event_cnn_minimal']),evreal_root=str(source_roots['EVREAL']),**bound)
protocol_path=out/'fixture_protocol.json';write(protocol_path,protocol)
protocol_path.with_suffix('.json.sha256').write_text(b.sha(protocol_path)+'\n')
b.verify_protocol(protocol_path,full_data=False)
results.append(dict(test='protocol_binding_valid',passed=True))

# Complete metadata-only inference contract fixture. No image/event files exist.
inf_p=copy.deepcopy(protocol);inf_p['hqf_root']=str(root/'fake_hqf')
inf_p['source_verification']={name:{'commit':x['commit']} for name in source_roots for x in pins if x['repo']==name}
data_hashes={name:{'sha256':hashlib.sha256(name.encode()).hexdigest()} for name in b.FILES}
inf_data=root/'inference_fake_data_manifest.json'
write(inf_data,{'sequences':[{'sequence':'fake_sequence','evreal_memmap':{'files':data_hashes}}]})
inf_p['data_manifest']=b.record(inf_data);row=runs[0]
fake_inf=dict(status=b.INFERENCE_STATUS,self_test={'is_synthetic_fixture':False},
    model=dict(checkpoint_path=row['checkpoint']['path'],checkpoint_sha256=row['checkpoint']['sha256'],
        checkpoint_generator_metadata=row['method'],checkpoint_model_args={'fixture_only':True},checkpoint_state_dict_key='model',
        public_EVREAL_weights_loaded=False,reference_frame_pixels_used_as_input=False,real_GT_scoring_performed=False,
        input_contract='event-only; [1,5,H,W]',recurrent_reset_calls=1,
        reset_policy='one reset at the initial anchor; first model call uses events from (t0,t1]; no reset inside sequence',
        output_contract_after_crop=[1,1,180,240],source_verification=inf_p['source_verification'],model_source_path=inf_p['model_root']),
    sequence=dict(name='fake_sequence',path=str((Path(inf_p['hqf_root'])/'fake_sequence').resolve()),
        num_frames_in_memmap=3,num_samples_processed=2,num_inferable_event_endpoints=2,
        initial_anchor_endpoint_excluded_from_model_and_metrics=True,timestamps_strictly_increasing=True,
        samples=[{'sample_index':1},{'sample_index':2}]),
    registered_inference=dict(save_every_n_samples=1,save_float_arrays=True,saved_sample_indices=[1,2],
        saved_images=[{'sample_index':1},{'sample_index':2}]),
    provenance=dict(data_files=data_hashes,frozen_simulator_core_commit=b.LEGACY_INFER_CORE))
inf_path=root/'fake_inference.json';write(inf_path,fake_inf)
b.validate_inference_contract(inf_p,row,'fake_sequence',inf_path)
results.append(dict(test='complete_inference_metadata_contract',passed=True))
def reject_inference(name,section,key,value,expected):
    bad=copy.deepcopy(fake_inf);bad[section][key]=value;path=root/(name+'.json');write(path,bad)
    try:b.validate_inference_contract(inf_p,row,'fake_sequence',path)
    except RuntimeError as exc:
        assert expected in str(exc),(name,str(exc));results.append(dict(test=name,passed=True))
    else:raise AssertionError('Invalid inference accepted: '+name)
reject_inference('inference_wrong_checkpoint','model','checkpoint_sha256','0'*64,'wrong checkpoint')
reject_inference('inference_wrong_method','model','checkpoint_generator_metadata','v2e','wrong generator')
reject_inference('inference_real_reference_input','model','reference_frame_pixels_used_as_input',True,'isolation failed')
reject_inference('inference_public_weights','model','public_EVREAL_weights_loaded',True,'public weights')
reject_inference('inference_wrong_reset','model','recurrent_reset_calls',2,'reset count')
reject_inference('inference_missing_endpoint','registered_inference','saved_sample_indices',[1],'incomplete saved')
reject_inference('inference_unrecognized_legacy_core','provenance','frozen_simulator_core_commit','wrong','core metadata')
bad_sources=copy.deepcopy(inf_p['source_verification']);bad_sources['EVREAL']['commit']='wrong'
reject_inference('inference_wrong_source_commit','model','source_verification',bad_sources,'source commit')
bad_hashes=copy.deepcopy(data_hashes);bad_hashes['images.npy']['sha256']='wrong'
reject_inference('inference_wrong_data_hash','provenance','data_files',bad_hashes,'data hash differs')

for name in ('EVREAL','event_cnn_minimal'):
    payload=source_roots[name]/'fixture_model_source.py';original=payload.read_bytes()
    payload.write_bytes(original+b'# changed after prepare\n')
    try:b.verify_protocol(protocol_path,full_data=False)
    except RuntimeError as exc:
        assert 'Downstream fixture source changed: '+name in str(exc)
        results.append(dict(test='verify_rechecks_full_source_'+name,passed=True))
    else:raise AssertionError('Changed downstream source accepted: '+name)
    payload.write_bytes(original)
authorized=out/'authorized_fixture_only.json'
write(authorized,dict(schema='stage9_final_real_evaluation_authorization_v1',authorized=True,
    task='reconstruction',protocol_sha256=b.sha(protocol_path),
    source_user_instruction='NOT REAL AUTHORIZATION: fixture only',
    research_verification_note='NOT REAL AUTHORIZATION: fixture only',no_feedback_to_training_or_selection=True))
payload=source_roots['EVREAL']/'fixture_model_source.py';original=payload.read_bytes()
payload.write_bytes(original+b'# changed before launch\n')
try:b.run_evaluation(protocol_path,authorized,out/'source_changed_must_not_be_created','cuda:0')
except RuntimeError as exc:
    assert 'Downstream fixture source changed: EVREAL' in str(exc)
    assert not (out/'source_changed_must_not_be_created').exists()
    results.append(dict(test='launch_rechecks_source_before_output_and_data',passed=True))
else:raise AssertionError('Launch accepted changed downstream source')
payload.write_bytes(original)

auth=out/'no_authorization.json';write(auth,dict(schema='stage9_final_real_evaluation_authorization_v1',authorized=False,task='reconstruction'))
try:b.run_evaluation(protocol_path,auth,out/'must_not_be_created','cuda:0')
except RuntimeError as exc:
    assert 'not authorized' in str(exc);assert not (out/'must_not_be_created').exists()
    results.append(dict(test='real_launch_refused_without_authorization_before_output_creation',passed=True))
else:raise AssertionError('Unauthorized real evaluation accepted')
protocol_path.write_text(protocol_path.read_text()+' ')
try:b.verify_protocol(protocol_path,full_data=False)
except RuntimeError as exc:
    assert 'sidecar SHA mismatch' in str(exc);results.append(dict(test='tampered_protocol_sidecar_refused',passed=True))
else:raise AssertionError('Protocol tampering accepted')

result=dict(status='pass',tests=results,real_data_opened=False,model_loaded=False,Torch_or_GPU_called=False,
            evidence_scope='contract and meaningful rejection fixtures, not real inference/metric correctness')
write(out/'contract_test_results.json',result);print(json.dumps(result,indent=2))
