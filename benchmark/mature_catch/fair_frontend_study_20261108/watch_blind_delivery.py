"""Read existing g1 relay, then publish only completed actual blind/cost artifacts.

This script never launches SSH, physical runs, training, or a scheduled task.
The already-running relay must remain available. Nothing starts on import.
"""
from pathlib import Path,PurePosixPath
import argparse,csv,datetime,hashlib,json,shutil,subprocess,sys,time,uuid

HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE/'figures_tool'))
import make_fair_figures as figures
DEFAULT_REMOTE='/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/fair_frontend_blind_20261108b'
BLIND_FILES=['EVALUATION_PROTOCOL.json','FRONTEND_SELECTION.json','FAIR_COLLECTION_COMPLETE.json',
    'FAIR_STAGE_RESULTS.json','FAIR_EPISODES.csv','FINAL_AUDIT.json','DATA_COMPLETION_RECEIPT.json','SHA256SUMS']
COST_FILES=['COMPLETE.json','FAIR_COST_RUNS.csv','FAIR_COST_RUNS.manifest.json','BUDGET_RECEIPT.json','COST_STUDY_PLAN.json','SHA256SUMS']
DEADLINE=None


class Pending(RuntimeError):pass


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)


def fresh(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,ensure_ascii=False)


def mutable(path,value):
    temporary=Path(str(path)+'.pending');temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


def snapshot_code(remote_root,want_blind=True,want_cost=True):
    """Literal read-only Python sent to the existing relay; no shell commands."""
    return '''from pathlib import Path
import hashlib,json,time
root=Path(%r);blind=root/'blind32';cost=root/'cost_v4_matched'
def value(p):
 if not p.is_file():return None
 try:return json.loads(p.read_text())
 except Exception as e:return {'read_error':type(e).__name__+': '+str(e)}
def info(p,label=None):
 if not p.is_file():return None
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return {'remote':str(p),'name':label or p.name,'size':p.stat().st_size,'sha256':h.hexdigest()}
out={'observed_unix_s':time.time(),'advance':value(root/'ADVANCE_V3_STATUS.json'),
 'checkpoint':value(root/'BLIND_COST_V4_CHECKPOINT_STATUS.json'),
 'outer_cost':value(root/'BLIND_COST_V4_CHECKPOINT_COST_RECEIPT.json'),
 'blind_complete':value(blind/'FAIR_COLLECTION_COMPLETE.json'),
 'cost_complete':value(cost/'COMPLETE.json'),'cost_stopped':value(cost/'STOPPED.json')}
if %r and (out['blind_complete'] or {}).get('status')=='PASS':
 files=[];protocol=value(blind/'EVALUATION_PROTOCOL.json') or {}
 for name in %r:
  p=blind/name
  if name=='FRONTEND_SELECTION.json' and not p.is_file():p=Path(protocol.get('frontend_selection_file',str(p)))
  files.append(info(p,name))
 for name in ['INFRASTRUCTURE_RECOVERY.json','SCHEDULING_COST_V4_CHECKPOINT.json','RECOVERY_MAP.json']:
  if (blind/name).is_file():files.append(info(blind/name,name))
 out['blind_files']=files
if %r and (out['cost_complete'] or {}).get('status')=='PASS' and not out['cost_stopped']:
 files=[]
 for name in %r:files.append(info(cost/name,name))
 seen={str(cost/name) for name in %r}
 for p in sorted(cost.rglob('*.json')):
  if str(p) not in seen:files.append(info(p,str(p.relative_to(cost))))
 out['cost_files']=files
print(json.dumps(out))
'''%(remote_root,want_blind,BLIND_FILES,want_cost,COST_FILES,COST_FILES)


def queue_snapshot(queue,identifier,code,timeout_s=55.):
    response=queue/(identifier+'.response.json');request=queue/(identifier+'.request.py')
    if not request.exists():
        draft=queue/(identifier+'.draft');draft.write_text(code,encoding='utf-8');draft.rename(request)
    else:require(request.read_text(encoding='utf-8')==code,'Existing relay request has different read-only content')
    if DEADLINE is not None:timeout_s=min(timeout_s,max(0.,DEADLINE-time.monotonic()))
    end=time.monotonic()+timeout_s
    while not response.exists() and time.monotonic()<end:time.sleep(.4)
    if not response.exists():raise Pending('Existing relay has not returned '+identifier+'; no new SSH is started')
    envelope=read(response);require(envelope.get('ok') is True,'Relay read failed: '+str(envelope.get('error',envelope)))
    return json.loads(envelope['output'])


def command(argv,log,timeout=7200):
    if DEADLINE is not None:timeout=min(timeout,max(.01,DEADLINE-time.monotonic()))
    start=Path(log).stat().st_size if Path(log).exists() else 0
    with Path(log).open('a',encoding='utf-8') as stream:
        stream.write('\n'+now()+' '+json.dumps([str(a) for a in argv],ensure_ascii=False)+'\n');stream.flush()
        result=subprocess.run([str(a) for a in argv],stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:
        with Path(log).open('rb') as stream:stream.seek(start);text=stream.read().decode('utf-8',errors='replace')[-10000:]
        if 'TimeoutError' in text and 'remains queued; rerun to resume' in text:raise Pending('Artifact transfer is waiting on the existing relay; resumable files retained')
        raise RuntimeError('Local command failed with exit '+str(result.returncode)+': '+text)


def download(items,directory,prefix,config,work):
    require(items and all(item is not None for item in items),'Completed stage is missing required real files')
    require(len(items)<=2500 and sum(item['size'] for item in items)<=512*1024**2,'Small-artifact bundle exceeds the bounded512MiB budget')
    directory.mkdir(parents=True,exist_ok=True);plan=directory/'FETCH_PLAN.json'
    rows=[]
    for item in items:
        relative=PurePosixPath(item['name']);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe artifact name')
        rows.append(dict(remote=item['remote'],local=str(directory/Path(*relative.parts)),expected_sha256=item['sha256']))
    if plan.exists():require(read(plan)==rows,'Completed download bundle unexpectedly changed')
    else:fresh(plan,rows)
    command([config['python'],HERE/'fetch_many_artifacts.py','--plan',plan,'--prefix',prefix,'--queue',config['queue']],work/'TRANSFER.log')
    for item,row in zip(items,rows):require(sha(row['local'])==item['sha256'],'Downloaded artifact disagrees with observed final SHA: '+item['name'])


def verify_blind(directory):
    stage=figures.load_stage(directory,'blind');blind=figures.validate_blind(stage,directory/'FRONTEND_SELECTION.json',32,figures.BLIND_RATES)
    marker=read(directory/'FAIR_COLLECTION_COMPLETE.json');audit=read(directory/'FINAL_AUDIT.json');receipt=read(directory/'DATA_COMPLETION_RECEIPT.json')
    require(sha(directory/'SHA256SUMS')==marker['raw_manifest_sha256'] and sha(directory/'FINAL_AUDIT.json')==marker['final_audit_sha256'],'Blind raw/final audit hash mismatch')
    require(audit.get('status')=='PASS' and audit.get('episodes')==320 and audit.get('raw_event_contract_and_packet_counts_pass') is True,
        'Blind full raw audit is not passed for all320 actors')
    require(receipt.get('status')=='PASS' and receipt.get('actual_planned_actors')==320 and receipt.get('valid_completed_unique_episodes')==320
        and receipt.get('frozen_source_policy_profile_hashes_verified') is True,'Blind actual completion receipt is incomplete')
    protocol=read(directory/'EVALUATION_PROTOCOL.json')
    require(sha(directory/'FRONTEND_SELECTION.json')==protocol['frontend_selection_sha256'],'Downloaded lock is not the original registered lock')
    return blind


def tree_manifest(root):
    path=root/'SHA256SUMS'
    with path.open('x',encoding='utf-8') as stream:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p!=path:stream.write(sha(p)+'  '+str(p.relative_to(root))+'\n')


def copy_fresh(source,destination):
    require(not destination.exists(),'Refuse to overwrite an existing delivery: '+str(destination))
    shutil.copytree(source,destination)


def publish_blind(data,config,work):
    verify_blind(data);plots=work/'blind_figures';media_plan=work/'blind_media_plan';media=work/'blind_media_rendered'
    if not (plots/'blind_success_vs_rate_MANIFEST.json').exists():
        command([config['python'],HERE/'figures_tool/make_fair_figures.py','--blind',data,'--selection',data/'FRONTEND_SELECTION.json',
            '--bootstrap-repetitions','10000','--output',plots],work/'FIGURES.log')
    summary=HERE/'write_blind_summary.py'
    require(summary.is_file(),'Expected local Chinese summary CLI is missing: '+str(summary))
    if not (work/'blind_summary').exists():
        command([config['python'],summary,'--stage',data,'--figures',plots,'--output',work/'blind_summary'],work/'SUMMARY.log')
    if not (media_plan/'DEMO_DOWNLOAD_PLAN.json').exists():
        command([config['python'],HERE/'media_tool/make_blind_demo.py','plan','--stage',data,'--selection',data/'FRONTEND_SELECTION.json','--output',media_plan],work/'MEDIA.log')
    command([config['python'],HERE/'fetch_many_artifacts.py','--plan',media_plan/'DOWNLOAD_FETCH_PLAN.json',
        '--prefix',config['id']+'_media','--queue',config['queue']],work/'TRANSFER.log')
    if not (media/'DEMO_PROVENANCE.json').exists():
        command([config['python'],HERE/'media_tool/make_blind_demo.py','render','--plan',media_plan/'DEMO_DOWNLOAD_PLAN.json','--output',media],work/'MEDIA.log')
    destination=Path(config['desktop'])/'02_新来球盲测'/config['id']
    require(not destination.exists(),'Blind desktop delivery already exists; do not overwrite')
    destination.mkdir(parents=True)
    copy_fresh(data,destination/'逐集结果与凭据');copy_fresh(plots,destination/'图表');copy_fresh(media,destination/'代表回合实录')
    if (work/'blind_summary').exists():copy_fresh(work/'blind_summary',destination/'中文汇总')
    (destination/'阅读说明.md').write_text('32个新来球的完整盲测，四档输入帧率重复使用同一来球。全部成功和失败保留。\n\n图表含逐档配对检验；代表录像按预先固定的最小block规则选取，保留V2E有利例，不能用代表录像代替完整成功率。两种参考各只计一次/来球。\n',encoding='utf-8')
    tree_manifest(destination);return str(destination)


def verify_cost(directory,blind):
    complete=read(directory/'COMPLETE.json');manifest=read(directory/'FAIR_COST_RUNS.manifest.json');budget=read(directory/'BUDGET_RECEIPT.json')
    require(complete.get('status')=='PASS' and complete.get('validation_cells')==8 and complete.get('formal_primary_runs')==24,'Formal cost is not the full8-cell/24-run completed study')
    require(sha(directory/'FAIR_COST_RUNS.csv')==complete['cost_csv_sha256']==manifest['table_sha256']
        and sha(directory/'BUDGET_RECEIPT.json')==complete['budget_receipt_sha256'] and sha(directory/'COST_STUDY_PLAN.json')==complete['plan_sha256'],
        'Formal cost table/plan/budget hashes mismatch')
    require(budget.get('status')=='PASS' and len(budget.get('formal_runs',[]))==24 and len(budget.get('validation_cells',[]))==8,
        'Formal cost actual budget receipt does not cover all validations/repeats')
    result=figures.validate_cost(directory/'FAIR_COST_RUNS.csv',blind)
    require(len(result['accepted'])==24 and not result['excluded'] and all(r['repeats']==3 for r in result['summaries'])
        and manifest['rows']==manifest['primary_rows']==24,'Formal cost CSV is not exactly24 eligible actual native repeats')
    for row in budget['formal_runs']:
        remote=PurePosixPath(row['path'])/'COST_RESULT.json';relative=remote.relative_to(PurePosixPath(configured_cost_root(directory)))
        require(sha(directory/Path(*relative.parts))==row['cost_result_sha256'],'Actual cost result is absent or disagrees with the budget receipt')
    return result


def configured_cost_root(directory):
    plan=read(directory/'FETCH_PLAN.json')
    entry=next(row for row in plan if Path(row['local']).name=='COMPLETE.json' and PurePosixPath(row['remote']).name=='COMPLETE.json')
    return str(PurePosixPath(entry['remote']).parent)


def publish_cost(data,blind_data,config,work):
    blind=verify_blind(blind_data);verify_cost(data,blind);plots=work/'cost_figures'
    if not (plots/'blind_success_vs_cost_MANIFEST.json').exists():
        command([config['python'],HERE/'figures_tool/make_fair_figures.py','--blind',blind_data,'--selection',blind_data/'FRONTEND_SELECTION.json',
            '--cost',data/'FAIR_COST_RUNS.csv','--output',plots],work/'COST_FIGURES.log')
    destination=Path(config['desktop'])/'03_成本对照'/config['id'];destination.mkdir(parents=True,exist_ok=False)
    copy_fresh(data,destination/'完整成本凭据');copy_fresh(plots,destination/'图表')
    (destination/'阅读说明.md').write_text('成本来自同一固定物理轨迹0.32–1.12秒完整窗口，同一GPU；两来源四档各3次实测。成功率来自完整32来球盲测。\n\n功能等价要求完整原策略输入/输出精确一致，原始事件是否精确一致单独保留。不用提前失败回合的墙钟时间代替成本。\n',encoding='utf-8')
    tree_manifest(destination);return str(destination)


def publish_cost_failure(snapshot,config,reason):
    destination=Path(config['desktop'])/'03_成本对照'/(config['id']+'_未出图')
    destination.mkdir(parents=True,exist_ok=False)
    fresh(destination/'FAIL_成本未通过或尚未闭合.json',dict(status='NO_FORMAL_COST_FIGURE',reason=reason,observed_utc=now(),actual_status=snapshot))
    (destination/'阅读说明.md').write_text('正式成本尚未满足完整24次通过条件，因此没有生成成本图。具体失败或未完成原因见同目录JSON。盲测结果独立保留，不以成本故障抹掉任务结果。\n',encoding='utf-8')
    tree_manifest(destination);return str(destination)


def run(config):
    global DEADLINE
    work=Path(config['work']);work.mkdir(parents=True,exist_ok=False);fresh(work/'CONFIG.json',config)
    queue=Path(config['queue'])
    deadline=time.monotonic()+config['max_hours']*3600;blind_done=False;cost_done=False;blind_delivered_at=None
    DEADLINE=deadline
    poll=0;pending=None;state=dict(status='RUNNING',started_utc=now(),blind_delivery=None,cost_delivery=None)
    try:
        require(queue.is_dir(),'Existing relay queue directory is missing; no relay/SSH is started automatically')
        while time.monotonic()<deadline:
            tick=time.monotonic()
            try:
                if pending is None:
                    pending=(config['id']+'_poll%04d'%poll,snapshot_code(config['remote'],not blind_done,not cost_done));poll+=1
                snapshot=queue_snapshot(queue,*pending);pending=None
                state.update(observed_utc=now(),latest_status=snapshot,phase='READ_ACTUAL_COMPLETION')
                mutable(work/'WATCH_STATUS.json',state)
                if not blind_done and (snapshot.get('blind_complete') or {}).get('status')=='PASS':
                    download(snapshot.get('blind_files'),work/'blind_data',config['id']+'_blind',config,work)
                    state['blind_delivery']=publish_blind(work/'blind_data',config,work);blind_done=True;blind_delivered_at=time.monotonic()
                    fresh(work/'BLIND_DELIVERED.json',dict(status='PASS',utc=now(),path=state['blind_delivery']))
                if blind_done and not cost_done:
                    stopped=snapshot.get('cost_stopped');outer=snapshot.get('outer_cost') or {}
                    if (stopped or {}).get('status')=='STOPPED' or outer.get('status')=='FAILED':
                        state['cost_delivery']=publish_cost_failure(snapshot,config,'New cost_v4 driver or its actual managed job failed');cost_done=True
                    elif (snapshot.get('cost_complete') or {}).get('status')=='PASS' and outer.get('status')=='PASS':
                        actual=outer.get('actual_job_result') or {}
                        require(actual.get('status')=='completed' and actual.get('exit_code')==0,'Outer cost PASS lacks an actual completed0 job receipt')
                        download(snapshot.get('cost_files'),work/'cost_data',config['id']+'_cost',config,work)
                        state['cost_delivery']=publish_cost(work/'cost_data',work/'blind_data',config,work);cost_done=True
                    elif time.monotonic()-blind_delivered_at>=config['cost_wait_hours']*3600:
                        state['cost_delivery']=publish_cost_failure(snapshot,config,'Cost did not close within the configured one-hour post-blind wait');cost_done=True
                if blind_done and cost_done:
                    state.update(status='DELIVERED',finished_utc=now());mutable(work/'WATCH_STATUS.json',state);fresh(work/'DELIVERY_COMPLETE.json',state);return 0
                if not blind_done and (snapshot.get('advance') or {}).get('stage')=='STOPPED':
                    raise RuntimeError('Blind coordinator stopped before its real completion: '+json.dumps(snapshot['advance'],ensure_ascii=False))
            except Pending as error:state.update(phase='WAIT_EXISTING_RELAY',waiting=str(error));mutable(work/'WATCH_STATUS.json',state)
            time.sleep(max(0.,min(config['interval_s']-(time.monotonic()-tick),deadline-time.monotonic())))
        raise TimeoutError('24-hour bounded watcher deadline reached before actual blind/cost closure')
    except BaseException as error:
        failure=dict(status='FAIL',utc=now(),error=type(error).__name__+': '+str(error),state=state)
        fresh(work/'FAIL.json',failure);mutable(work/'WATCH_STATUS.json',failure);return 2


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--remote-root',default=DEFAULT_REMOTE)
    parser.add_argument('--queue',type=Path,default=HERE/'remote_queue_g1');parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--desktop',type=Path,default=Path('C:/Users/zhaoyuhan/Desktop/CatchIt_公平对照_20261109'))
    parser.add_argument('--work',type=Path);parser.add_argument('--id',default='auto_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    parser.add_argument('--interval-s',type=float,default=60);parser.add_argument('--max-hours',type=float,default=24)
    parser.add_argument('--cost-wait-hours',type=float,default=1);parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    require(args.interval_s>=60 and 0<args.max_hours<=24 and 0<args.cost_wait_hours<=1,'Use>=60s reads,<=24h total and<=1h post-blind cost wait')
    config=dict(remote=args.remote_root,queue=str(args.queue.resolve()),python=args.python,desktop=str(args.desktop.resolve()),
        id=args.id,work=str((args.work or HERE/'delivery_local'/args.id).resolve()),interval_s=args.interval_s,max_hours=args.max_hours,cost_wait_hours=args.cost_wait_hours)
    require(config['id'].replace('_','').replace('-','').isalnum(),'Unsafe relay/delivery identifier')
    if args.dry_run:
        print(json.dumps(dict(status='NOT_STARTED',config=config,read_only_remote_code=snapshot_code(config['remote']),
            will_start_ssh=False,will_launch_physics=False,will_register_automation=False),indent=2,ensure_ascii=False));return
    raise SystemExit(run(config))


if __name__=='__main__':main()
