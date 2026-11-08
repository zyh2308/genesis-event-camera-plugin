"""Frozen32 synthetic confirmation preparation, split CPU physical / GPU replay.

No training, optimizer, real evaluation, sampling search, or model selection.
The CPU prefix is extracted from the pinned original generate() without changing
its statements; only its final two replay calls are deferred to the events phase.
An external resource parent owns admission and the bounded process group.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,importlib.util,json,os,sys,traceback
from pathlib import Path
from types import SimpleNamespace

CORE='62da4f35ce3c5c96f04268bd1687a27657af9238'
BASE=202611090137
COUNT=32
ROOT='/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261109-pose-advantage-and-failure-v1/advantage_confirmation'
FRESH_SHA='d1b16345bbf174f535453fc4c8e9657afe2d154f4e98a11e20d7940e8463446c'
RELIGHT_SHA='0526adbb228a1787cbd1fb35ded40173775e4034a1516f7561a26bbea901068e'
SCIENCE_SHA='cdb5a83e5c80798a700f5c001b93d7994ab2c1b272212e6437f80342f07c3c52'
BOOTSTRAP_SEED=20261109137
METHODS=('eventgenesis','v2e')

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def record(p):
    p=Path(p);return dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def validate_registration(c):
    need(c['schema']=='pose_frozen_c1_advantage_confirmation32_v1','Unknown confirmation schema')
    need(c['frozen_core']==CORE and c['physical_seed']==BASE and c['rollouts']==COUNT,'Core/physical scope differs')
    need(c['root']==ROOT and c['methods']==list(METHODS) and c['seeds']==[0,1,2],'Scope/root differs')
    need(c['conditions']=={'normal':1.,'half':.5,'eighth':.125},'Relight scope differs')
    need(c['primary_arm']=='C1_RAW' and c['optimizer_updates']==0 and c['real_data_used'] is False,'Scientific boundary differs')
    need(c['primary_conditions']==['normal','half'] and c['bootstrap_draws']==10000 and c['bootstrap_seed']==BOOTSTRAP_SEED,'Statistics registration differs')
    need(c['per_condition_ci_level']==.975 and c['all_three_seed_deltas_positive'] is True,'Primary confirmation rule differs')
    need(c['fresh_source']['sha256']==FRESH_SHA and c['relight_source']['sha256']==RELIGHT_SHA,'Frozen reuse sources differ')
    need(isinstance(c['source_records'],list) and len(c['source_records'])>0,'No source freeze inventory')
    for r in c['source_records']:
        need(Path(r['path']).is_absolute() and len(r['sha256'])==64,'Bad source record')
    need(Path(c['campaign']).is_absolute() and Path(c['root']).is_absolute(),'Absolute roots required')

def verify_records(records):
    for r in records:
        p=Path(r['path']);need(p.is_file() and ('size_bytes' not in r or p.stat().st_size==r['size_bytes']) and sha(p)==r['sha256'],'Pinned input/source changed: '+str(p))

def physical_function(source,namespace):
    """Compile the original statements before the original method replay loop."""
    tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate']
    need(len(nodes)==1,'Original generate function missing')
    fn=copy.deepcopy(nodes[0]);need(len(fn.body)>=3 and isinstance(fn.body[-2],ast.For) and isinstance(fn.body[-1],ast.Return),'Original generate control flow differs')
    loop=fn.body[-2]
    need(ast.dump(loop.target)==ast.dump(ast.parse('for method, script in []:\n pass').body[0].target),'Original replay loop differs')
    need(isinstance(loop.iter,ast.Call) and isinstance(loop.iter.func,ast.Name) and loop.iter.func.id=='zip','Original replay dispatch differs')
    expected=ast.parse("zip(METHODS, ['replay_new_events.py', 'replay_v2e_parallel.py'], strict=True)",mode='eval').body
    need(ast.dump(loop.iter)==ast.dump(expected),'Original replay scripts/order differ')
    fn.name='_frozen_physical_prefix';fn.body=fn.body[:-2]+[fn.body[-1]]
    new=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));scope=dict(namespace)
    exec(compile(new,'<pinned-generate-physical-prefix>','exec'),scope)
    return scope[fn.name],hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()

def check_inventory(c):
    r=c['seed_inventory'];verify_records([r]);d=read(r['path'])
    need(d['schema']=='registered_pose_seed_inventory_v1' and d['passed'] and d['overlap']==[],'Registered seed inventory failed')
    need(d['proposed_base_seed']==BASE and d['proposed_count']==COUNT and d['protocol_count']==len(d['protocols'])>=3 and d['observed_H5_files']==len(d['actual_H5'])>=226,'Registered inventory scope differs')
    old=set();verify_records(d['protocols'])
    for protocol in d['protocols']:
        values=read(protocol['path']);need(values['physical_seed']==protocol['base_seed'] and values['train_rollouts']==protocol['train_rollouts'] and values['validation_rollouts']==protocol['validation_rollouts'],'Actual bank protocol differs')
        old.update(protocol['base_seed']+1009*i for i in range(protocol['train_rollouts']+protocol['validation_rollouts']))
    import h5py
    for item in d['actual_H5']:
        with h5py.File(item['path'],'r') as h:
            groups=list(h['episodes'].values());need(len(groups)==1,'Old bank episode inventory differs');g=groups[0]
            need(int(g.attrs['scene_seed'])==item['scene_seed'] and str(g.attrs['split'])==item['split'],'Old bank actual scene seed differs')
        old.add(item['scene_seed'])
    new={BASE+1009*i for i in range(COUNT)}
    need(new==set(d['proposed_scene_seeds']) and len(old)==d['unique_registered_or_observed_scene_seeds'],'Reconstructed inventory scope differs')
    need(not old.intersection(new),'New physical seeds overlap already used bank')
    need({20261106017+1009*i for i in range(96)}<=old,'Known previously seen96 bank omitted')
    need({20261106017+400000+1009*i for i in range(2)}<=old,'Known smoke bank omitted')
    return d

def check_science_and_core(c):
    verify_records([c['scientific_registration'],c['core_integrity']])
    need(c['scientific_registration']['sha256']==SCIENCE_SHA,'Scientific preregistration differs')
    science=read(c['scientific_registration']['path']);guard=read(c['core_integrity']['path'])
    need(science['schema']=='pose_condition_advantage_confirmation_v1' and science['frozen_core']==CORE and science['physical_rollouts']==COUNT and science['physical_seed']==BASE,'Preregistered physical scope differs')
    need(science['methods']==list(METHODS) and science['seeds']==[0,1,2] and science['model_arm']=='RAW_C1_physical_supervised' and science['softfocus_disabled'],'Preregistered models differ')
    need(science['optimizer_updates']==0 and not science['real_scoring'] and not science['long_training'],'Preregistered scientific isolation differs')
    need(science['seed_inventory_sha256']==c['seed_inventory']['sha256'] and science['frozen_core_guard_sha256']==c['core_integrity']['sha256'],'Actual inventory/core receipts differ')
    need(science['radiance_multipliers']==c['conditions'] and science['window_indices']==list(range(4,20)),'Registered radiance/windows differ')
    need(science['statistics']['bootstrap_seed']==c['bootstrap_seed'] and science['statistics']['bootstrap_draws']==c['bootstrap_draws'],'Registered statistics differ')
    need(guard['schema']=='generation_frozen_core_integrity_v1' and guard['passed'] and guard['release_commit']==CORE and len(guard['core_files'])==17,'Core integrity receipt failed')
    verify_records([dict(path=guard['baseline_metadata_path'],sha256=guard['baseline_metadata_sha256']),dict(path=guard['task_sensor_profile_path'],sha256=guard['task_sensor_profile_sha256'])])
    for item in guard['core_files']:
        raw=Path(item['path']).read_bytes();need(hashlib.sha256(raw).hexdigest()==item['sha256'] and item['passed'],'Frozen core bytes changed')
        need(hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==item['canonical_LF_sha256']==item['expected_canonical_LF_sha256'],'Frozen core canonical LF differs')

def common(c):
    validate_registration(c);verify_records(c['source_records']);check_science_and_core(c)
    for name in ('fresh_source','relight_source'):verify_records([c[name]])
    recipe=Path(c['campaign'])/'sources/training_recipe.json'
    need(read(recipe)['frozen_simulator_core']==CORE,'Campaign recipe core differs')
    need(any(Path(r['path'])==recipe for r in c['source_records']),'Recipe not source-bound')
    return load('confirmation_frozen_fresh',c['fresh_source']['path'])

def physical(c,out):
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU physical phase requires externally hidden CUDA')
    need(out==Path(ROOT) and not out.exists(),'Unique registered new confirmation root required')
    F=common(c);check_inventory(c);need(not F.torch.cuda.is_initialized(),'CPU helper initialized CUDA')
    out.mkdir();normal=out/'normal';normal.mkdir()
    fn,prefix_sha=physical_function(Path(c['fresh_source']['path']).read_text(),F.__dict__)
    bank,files=fn(SimpleNamespace(out=normal,campaign=Path(c['campaign'])),COUNT,BASE)
    need(len(files)==COUNT and not (bank/'events').exists(),'CPU phase unexpectedly replayed events')
    rows=[]
    for i,p in enumerate(files):
        with F.np.errstate(all='raise'),__import__('h5py').File(p,'r') as h:
            g=h['episodes'][p.stem];need(g.attrs['scene_seed']==BASE+1009*i,'Actual scene seed changed')
        rows.append(record(p))
    verify_records(c['source_records']);verify_records([c['fresh_source'],c['relight_source']]);check_inventory(c);check_science_and_core(c)
    write(out/'physical_completion.json',dict(schema='pose_confirm32_physical_v1',status='complete',optimizer_updates=0,real_data_used=False,
        count=COUNT,physical_seed=BASE,physical_prefix_ast_sha256=prefix_sha,physical_files=rows,normal_bank=str(bank),
        bank_manifest=record(bank/'bank_manifest.json'),source_records=c['source_records'],seed_inventory=c['seed_inventory'],cuda_initialized=F.torch.cuda.is_initialized(),registration_sha256=c['_registration_sha256']))

def events(c,out,receipt_sha):
    F=common(c);p=out/'physical_completion.json';need(sha(p)==receipt_sha,'Externally bound physical receipt differs');q=read(p)
    need(q['status']=='complete' and q['optimizer_updates']==0 and q['real_data_used'] is False and q['count']==COUNT and q['registration_sha256']==c['_registration_sha256'],'Physical phase incomplete/mismatched registration')
    need(not (out/'event_completion.json').exists(),'Event phase already has completion')
    verify_records(q['physical_files']+[q['bank_manifest']]);verify_records([c['export_source'],c['base_contract']])
    E=load('confirmation_bound_gpu_guard',c['export_source']['path']);base=read(c['base_contract']['path']);E.check_gpu(base)
    need(not (out/'normal/pose_bank/events').exists(),'Normal event outputs already exist; never overwrite/retry here')
    src=out/'normal/sources';R=load('confirmation_frozen_relight',c['relight_source']['path'])
    tone_path=Path(c['campaign'])/'sources/benchmark/sim2real/scripts/generate_stage4_flow_bank.py'
    need(any(Path(r['path'])==tone_path for r in c['source_records']),'Canonical tone source not bound')
    tone=R.canonical_tone(tone_path);banks={};relit=[]
    for condition,gain in c['conditions'].items():
        folder=out/condition;bank=folder/'pose_bank'
        if condition!='normal':
            need(not folder.exists(),'Relit output already exists');folder.mkdir();(folder/'sources').symlink_to(src,target_is_directory=True);(bank/'physical').mkdir(parents=True)
            rows=[R.relight(Path(r['path']),bank/'physical'/Path(r['path']).name,Path(r['path']).stem,gain,tone) for r in q['physical_files']]
            write(bank/'bank_manifest.json',dict(rollouts=rows,source_manifest_sha256=sha(out/'normal/pose_bank/bank_manifest.json'),real_data_used=False));relit.extend(rows)
        for method,script in zip(METHODS,('replay_new_events.py','replay_v2e_parallel.py'),strict=True):
            E.check_gpu(base);F.call(src/script,'--campaign',folder,'--kind','pose','--workers',1)
            manifest=read(bank/'events'/method/'manifest.json')
            need(manifest['physical_manifest_sha256']==sha(bank/'bank_manifest.json') and len(manifest['rollouts'])==COUNT and not manifest['real_data_used'],'Replay manifest pairing/scope differs')
        banks[condition]=str(bank)
    verify_records(c['source_records']);verify_records(q['physical_files']);verify_records([c['fresh_source'],c['relight_source'],c['export_source'],c['base_contract']]);check_science_and_core(c)
    artifacts=[record(p) for folder in c['conditions'] for p in (out/folder/'pose_bank').rglob('*') if p.is_file()]
    write(out/'event_completion.json',dict(schema='pose_confirm32_events_v1',status='complete',optimizer_updates=0,real_data_used=False,
        physical_receipt=record(out/'physical_completion.json'),banks=banks,relit=relit,artifacts=artifacts,registration_sha256=c['_registration_sha256']))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('physical','events'))
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--registration-sha256',required=True);p.add_argument('--physical-receipt-sha256')
    a=p.parse_args();need(sha(a.registration)==a.registration_sha256,'External registration SHA differs');c=read(a.registration);need('_registration_sha256' not in c,'Reserved runtime key');c['_registration_sha256']=a.registration_sha256;out=Path(c['root'])
    try:
        if a.phase=='physical':physical(c,out)
        else:events(c,out,a.physical_receipt_sha256)
    except BaseException as e:
        if out.exists():
            dest=out/(a.phase+'_failure.json')
            if not dest.exists():write(dest,dict(status='failed',phase=a.phase,error=repr(e),traceback=traceback.format_exc(),optimizer_updates=0,real_data_used=False))
        raise

if __name__=='__main__':main()
