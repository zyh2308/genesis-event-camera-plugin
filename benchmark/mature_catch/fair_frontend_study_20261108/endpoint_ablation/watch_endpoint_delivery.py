"""Bounded local delivery watcher using the existing read-only g1 relay only.

Import and --dry-run start nothing. No SSH, GPU run, training, server mutation,
or automation is launched; actual128 completion is required before publication.
"""
from pathlib import Path
import argparse,datetime,json,shutil,sys,time,uuid
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
import watch_blind_delivery as relay
import plot_endpoint_ablation as paired

MAIN_FILES=['EVALUATION_PROTOCOL.json','FRONTEND_SELECTION.json','FAIR_COLLECTION_COMPLETE.json',
    'FAIR_STAGE_RESULTS.json','FAIR_EPISODES.csv','FINAL_AUDIT.json','DATA_COMPLETION_RECEIPT.json','SHA256SUMS','PAIRED_PORTABLE.json']
ENDPOINT_FILES=['EVALUATION_PROTOCOL.json','FAIR_COLLECTION_COMPLETE.json','FAIR_STAGE_RESULTS.json','FAIR_EPISODES.csv',
    'FINAL_AUDIT.json','DATA_COMPLETION_RECEIPT.json','SHA256SUMS','PAIRED_PORTABLE.json',
    'ENDPOINT_VARIANT_PROOFS.json','ENDPOINT_SOURCE_DELTA_PROOF.json','MAIN_STUDY_COMPLETION_GATE.json']


def snapshot_code(remote):
    return '''from pathlib import Path
import json,hashlib,time
root=Path(%r);main=root/'blind32';endpoint=root/'endpoint_only32_20261109a'
def value(path):
 if not path.is_file():return None
 try:return json.loads(path.read_text())
 except Exception as error:return {'read_error':str(error)}
def info(path,name):
 if not path.is_file():return None
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return {'remote':str(path),'name':name,'size':path.stat().st_size,'sha256':h.hexdigest()}
out={'observed_unix_s':time.time(),'advance':value(root/'endpoint_only32_20261109a_ADVANCE_ENDPOINT_STATUS.json'),
 'main_complete':value(main/'FAIR_COLLECTION_COMPLETE.json'),'endpoint_complete':value(endpoint/'FAIR_COLLECTION_COMPLETE.json')}
if (out['endpoint_complete'] or {}).get('status')=='PASS' and (out['main_complete'] or {}).get('status')=='PASS':
 out['main_files']=[];spec=value(main/'EVALUATION_PROTOCOL.json') or {}
 for name in %r:
  path=main/name
  if name=='FRONTEND_SELECTION.json' and not path.is_file():path=Path(spec.get('frontend_selection_file',str(root/'frontend_validation/FRONTEND_SELECTION.json')))
  out['main_files'].append(info(path,name))
 out['endpoint_files']=[info(endpoint/name,name) for name in %r]
 for folder,key in [(main,'main_files'),(endpoint,'endpoint_files')]:
  for name in ['INFRASTRUCTURE_RECOVERY.json','SCHEDULING_COST_V4_CHECKPOINT.json','RECOVERY_MAP.json']:
   if (folder/name).is_file():out[key].append(info(folder/name,name))
 out['endpoint_files'] += [info(root/'cost_v4_matched/COMPLETE.json','MAIN_COST_COMPLETE.json'),
  info(root/'BLIND_COST_V4_CHECKPOINT_COST_RECEIPT.json','MAIN_COST_OUTER_RECEIPT.json')]
print(json.dumps(out))
'''%(remote,MAIN_FILES,ENDPOINT_FILES)


def verify(main,endpoint):
    require=relay.require;join=paired.joined_data(main,endpoint,main/'FRONTEND_SELECTION.json')
    paired.collector.main_reference(main,portable=True)
    marker=relay.read(endpoint/'FAIR_COLLECTION_COMPLETE.json');receipt=relay.read(endpoint/'DATA_COMPLETION_RECEIPT.json')
    portable=relay.read(endpoint/'PAIRED_PORTABLE.json');gate=relay.read(endpoint/'MAIN_STUDY_COMPLETION_GATE.json')
    require(marker.get('stage')=='ablation' and marker.get('actual_episodes')==128
        and receipt.get('stage')=='ablation' and receipt.get('valid_completed_unique_episodes')==128
        and receipt.get('actual_planned_actors')==128,'Exactly128 completed endpoint actor executions required')
    require(portable.get('complete') is True and not portable.get('violations') and len(portable.get('records',[]))==128,
        'Endpoint portable source record is incomplete')
    require(gate.get('status')=='PASS' and relay.sha(main/'FAIR_COLLECTION_COMPLETE.json')==gate['blind_marker_sha256']
        and relay.sha(endpoint/'MAIN_COST_COMPLETE.json')==gate['cost_marker_sha256']
        and relay.sha(endpoint/'MAIN_COST_OUTER_RECEIPT.json')==gate['outer_cost_sha256'],
        'Ablation did not bind the original completed main+cost gates')
    cost=relay.read(endpoint/'MAIN_COST_COMPLETE.json');outer=relay.read(endpoint/'MAIN_COST_OUTER_RECEIPT.json')
    require(cost.get('status')=='PASS' and cost.get('formal_primary_runs')==24 and outer.get('status')=='PASS'
        and outer.get('actual_job_result',{}).get('status')=='completed'
        and outer.get('actual_job_result',{}).get('exit_code')==0,'Original cost closure is not verified')
    return join


def publish(main,endpoint,config,work):
    verify(main,endpoint);plots=work/'figures';plan=work/'media_plan';media=work/'media'
    if not (plots/'endpoint_ablation_MANIFEST.json').exists():
        relay.command([config['python'],HERE/'plot_endpoint_ablation.py','--main-blind-root',main,
            '--root',endpoint,'--selection',main/'FRONTEND_SELECTION.json','--output',plots],work/'FIGURES.log')
    if not (plan/'DEMO_DOWNLOAD_PLAN.json').exists():
        relay.command([config['python'],HERE/'make_endpoint_demo.py','plan','--main-blind-root',main,
            '--root',endpoint,'--selection',main/'FRONTEND_SELECTION.json','--output',plan],work/'MEDIA.log')
    relay.command([config['python'],HERE.parent/'fetch_many_artifacts.py','--plan',plan/'DOWNLOAD_FETCH_PLAN.json',
        '--prefix',config['id']+'_media','--queue',config['queue']],work/'TRANSFER.log')
    if not (media/'DEMO_PROVENANCE.json').exists():
        relay.command([config['python'],HERE/'make_endpoint_demo.py','render','--plan',plan/'DEMO_DOWNLOAD_PLAN.json',
            '--output',media],work/'MEDIA.log')
    proof=relay.read(media/'DEMO_PROVENANCE.json')
    relay.require(proof.get('status')=='PASS' and proof.get('actual_pairs',99)<=16,'Actual paired media is incomplete')
    destination=Path(config['desktop'])/'04_无中间重建对照'/config['id']
    relay.require(not destination.exists(),'Desktop delivery already exists; refuse overwrite')
    destination.mkdir(parents=True)
    relay.copy_fresh(main,destination/'原320主实验凭据');relay.copy_fresh(endpoint,destination/'新增128端点对照凭据')
    relay.copy_fresh(plots,destination/'指标图表');relay.copy_fresh(media,destination/'真实双列Demo')
    shutil.copy2(plan/'ALL128_PAIRED_OUTCOMES.csv',destination/'全部128组配对结果.csv')
    shutil.copy2(plan/'DEMO_DOWNLOAD_PLAN.json',destination/'代表回合选取与来源.json')
    (destination/'阅读说明.md').write_text(
        'Full 重建与仅端点对照均使用 Ours、同一前端 B、原权重和冻结像素响应。只改变中间图像重建。\n\n'
        '复用原320条主实验结果，新增128条端点对照；独立来球仍为原32组，四档帧率重复使用同一组来球。全部成功和失败保留在配对表和逐集表。\n\n'
        'Demo按每档帧率的Full单独成功、端点单独成功、双方成功、双方失败四类，固定选最小注册block；无该类则不造案例，最多16对。两列是完整真实25fps录像；提前结束后有明确末帧定格标注。代表录像不代替整体统计。原完整录像继续保留服务器；本次不生成或重渲物理场景。\n',encoding='utf-8')
    relay.tree_manifest(destination)
    return str(destination)


def run(config):
    work=Path(config['work']);work.mkdir(parents=True,exist_ok=False)
    relay.fresh(work/'CONFIG.json',config);queue=Path(config['queue'])
    deadline=time.monotonic()+config['max_hours']*3600;relay.DEADLINE=deadline
    poll=0;pending=None;state=dict(status='RUNNING',started_utc=relay.now(),delivery=None)
    try:
        relay.require(queue.is_dir(),'Existing g1 relay missing; this watcher never starts SSH')
        while time.monotonic()<deadline:
            tick=time.monotonic()
            try:
                if pending is None:
                    pending=(config['id']+'_poll%04d'%poll,snapshot_code(config['remote']));poll+=1
                snapshot=relay.queue_snapshot(queue,*pending);pending=None
                state.update(observed_utc=relay.now(),latest_status=snapshot,phase='WAIT_COMPLETE128')
                relay.mutable(work/'WATCH_STATUS.json',state)
                if (snapshot.get('advance') or {}).get('stage')=='STOPPED':
                    raise RuntimeError('Actual endpoint coordinator stopped: '+json.dumps(snapshot['advance'],ensure_ascii=False))
                if (snapshot.get('endpoint_complete') or {}).get('status')=='PASS':
                    relay.require((snapshot.get('main_complete') or {}).get('status')=='PASS','Main320 must remain complete')
                    relay.download(snapshot.get('main_files'),work/'main_data',config['id']+'_main',config,work)
                    relay.download(snapshot.get('endpoint_files'),work/'endpoint_data',config['id']+'_endpoint',config,work)
                    state['delivery']=publish(work/'main_data',work/'endpoint_data',config,work)
                    state.update(status='DELIVERED',finished_utc=relay.now())
                    relay.mutable(work/'WATCH_STATUS.json',state);relay.fresh(work/'DELIVERY_COMPLETE.json',state)
                    return 0
            except relay.Pending as error:
                state.update(phase='WAIT_EXISTING_RELAY',waiting=str(error));relay.mutable(work/'WATCH_STATUS.json',state)
            time.sleep(max(0.,min(config['interval_s']-(time.monotonic()-tick),deadline-time.monotonic())))
        raise TimeoutError('Bounded watcher timed out before complete ablation delivery')
    except BaseException as error:
        failure=dict(status='FAIL',utc=relay.now(),error=type(error).__name__+': '+str(error),state=state)
        relay.fresh(work/'FAIL.json',failure);relay.mutable(work/'WATCH_STATUS.json',failure);return 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--remote-root',default=relay.DEFAULT_REMOTE)
    p.add_argument('--queue',type=Path,default=HERE.parent/'remote_queue_g1');p.add_argument('--python',default=sys.executable)
    p.add_argument('--desktop',type=Path,default=Path('C:/Users/zhaoyuhan/Desktop/CatchIt_公平对照_20261109'))
    p.add_argument('--id',default='endpoint_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    p.add_argument('--work',type=Path);p.add_argument('--interval-s',type=float,default=60);p.add_argument('--max-hours',type=float,default=24)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    relay.require(a.interval_s>=60 and 0<a.max_hours<=24,'Use>=60s reads and at most24hours')
    relay.require(a.id.replace('_','').replace('-','').isalnum(),'Unsafe relay identifier')
    config=dict(id=a.id,remote=a.remote_root,queue=str(a.queue.resolve()),python=a.python,
        desktop=str(a.desktop.resolve()),work=str((a.work or HERE.parent/'delivery_local'/a.id).resolve()),
        interval_s=a.interval_s,max_hours=a.max_hours)
    if a.dry_run:
        print(json.dumps(dict(status='NOT_STARTED',config=config,read_only_remote_code=snapshot_code(config['remote']),
            starts_ssh=False,starts_physics=False,starts_automation=False),indent=2,ensure_ascii=False));return
    raise SystemExit(run(config))


if __name__=='__main__':main()
